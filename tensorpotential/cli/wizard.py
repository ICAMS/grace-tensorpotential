from __future__ import annotations

import os
import sys
import glob
import json
from dataclasses import dataclass
from typing import Any
from importlib import resources

try:
    import questionary
    from questionary import Style as QStyle

    _QUESTIONARY_STYLE = QStyle(
        [
            ("qmark", "fg:#00d7ff bold"),
            ("question", "bold"),
            ("answer", "fg:#00d7ff bold"),
            ("pointer", "fg:#00d7ff bold"),
            ("highlighted", "fg:#00d7ff bold"),
            ("selected", "fg:#00d7ff"),
            ("separator", "fg:#6c6c6c"),
            ("instruction", "fg:#6c6c6c"),
            ("text", ""),
            ("disabled", "fg:#858585 italic"),
        ]
    )
    _HAS_QUESTIONARY = True
except ImportError:
    _HAS_QUESTIONARY = False

try:
    from rich.console import Console as RichConsole
    from rich.panel import Panel
    from rich.text import Text as RichText
    from rich import box as rich_box

    _rich_console = RichConsole()
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

try:
    import readline

    no_readline = False
except ImportError:
    no_readline = True

from tensorpotential.potentials import (
    get_preset,
    get_public_preset_list,
    get_default_preset_name,
    get_preset_settings,
)

from tensorpotential.calculator.foundation_models import (
    MODELS_METADATA,
    CHECKPOINT_URL_KEY,
    DESCRIPTION_KEY,
)

try:
    from tensorpotential.extra.gen_tensor.wizard import run_gen_tensor_wizard

    _HAS_GEN_TENSOR = True
except ImportError:
    _HAS_GEN_TENSOR = False


# ---------------------------------------------------------------------------
# Plain input fallbacks
# ---------------------------------------------------------------------------


def input_choice(query, choices, default_choice):
    """Input from stdin with list of available choices and default choice."""
    choices = list(choices)
    assert default_choice in choices
    while True:
        choice = (
            input(
                query
                + f", available options: {', '.join(choices)} (default = {default_choice}): "
            )
            or default_choice
        )
        if choice in choices:
            break
    return choice


def input_with_default(query, default_choice: Any = None):
    """Input from stdin with default value."""
    choice = input(query + ": ") or default_choice
    return choice


def input_no_default(query):
    """Input from stdin without default."""
    while True:
        choice = input(query)
        if choice:
            break
    return choice


# ---------------------------------------------------------------------------
# Rich / Questionary helpers
# ---------------------------------------------------------------------------


def _print_header():
    if _HAS_RICH:
        _rich_console.print()
        _rich_console.print(
            Panel(
                RichText.assemble(
                    ("GRACEmaker ", "bold cyan"),
                    ("input.yaml", "bold white"),
                    (" wizard", "bold cyan"),
                ),
                subtitle="[dim]navigate with arrow keys · confirm with Enter[/dim]",
                border_style="cyan",
                box=rich_box.ROUNDED,
                expand=False,
            )
        )
        _rich_console.print()
    else:
        print("=" * 50)
        print("  GRACEmaker input.yaml wizard")
        print("=" * 50)


def _info(msg: str):
    if _HAS_RICH:
        _rich_console.print(f"  [dim]→[/dim] {msg}")
    else:
        print(f"  -> {msg}")


def _success(msg: str):
    if _HAS_RICH:
        _rich_console.print(f"  [bold green]✓[/bold green] {msg}")
    else:
        print(f"  OK: {msg}")


def _section(title: str):
    if _HAS_RICH:
        _rich_console.print(f"\n[bold cyan]── {title}[/bold cyan]")
    else:
        print(f"\n-- {title}")


def _tip_path_input():
    """Print a one-line hint explaining how to enter file paths."""
    if _HAS_QUESTIONARY:
        tip = "Tab ↹ autocompletes path  ·  ↑↓ navigates history"
    elif not no_readline:
        tip = "↑↓ for command history  ·  Tab may autocomplete (readline)"
    else:
        tip = "Type or paste the full file path"
    if _HAS_RICH:
        _rich_console.print(f"  [dim italic]{tip}[/dim italic]")
    else:
        print(f"  ({tip})")


def _ask_select(message, choices, default=None):
    if _HAS_QUESTIONARY:
        choice_list = list(choices)
        # choices may be questionary.Choice objects; find matching default
        def_val = default
        if def_val not in choice_list:
            # try matching by .value for Choice objects
            def_val = next(
                (c for c in choice_list if getattr(c, "value", c) == default),
                choice_list[0],
            )
        result = questionary.select(
            message, choices=choice_list, default=def_val, style=_QUESTIONARY_STYLE
        ).ask()
        if result is None:
            raise KeyboardInterrupt
        return result
    else:
        # extract string values from Choice objects if needed
        str_choices = [getattr(c, "value", c) for c in choices]
        return input_choice(
            message, str_choices, default if default in str_choices else str_choices[0]
        )


def _ask_text(message, default=None):
    if _HAS_QUESTIONARY:
        result = questionary.text(
            message,
            default=str(default) if default is not None else "",
            style=_QUESTIONARY_STYLE,
        ).ask()
        if result is None:
            raise KeyboardInterrupt
        return result if result else default
    else:
        return input_with_default(message, default)


def _ask_path(message):
    if _HAS_QUESTIONARY:
        result = questionary.path(message, style=_QUESTIONARY_STYLE).ask()
        if result is None:
            raise KeyboardInterrupt
        return result
    else:
        return input_no_default(message + " ")


def _ask_confirm(message, default=True):
    if _HAS_QUESTIONARY:
        result = questionary.confirm(
            message, default=default, style=_QUESTIONARY_STYLE
        ).ask()
        if result is None:
            raise KeyboardInterrupt
        return result
    else:
        ans = input_choice(message, ["yes", "no"], "yes" if default else "no")
        return ans == "yes"


# ---------------------------------------------------------------------------
# Wizard state
# ---------------------------------------------------------------------------


@dataclass
class _WizardState:
    # Dataset
    train_filename: str = ""
    use_separate_test: bool = False
    test_filename: str = ""
    test_size: float = 0.05
    # Model
    fit_type: str = "finetune"
    finetune_model: bool = False
    foundation_model_name: str = ""
    finetune_mode: str = "naive"  # "naive" or "frozen"
    preset_name: str = ""
    preset_complexity: str = "medium"
    cutoff: float = 6.0
    preset_kwargs_str: str = "{}"
    learning_rate: float = 0.008
    eval_init_stats: bool = False
    checkpoint_name: str = ""
    reset_epoch_and_step: bool = False
    # Optimizer
    optimizer: str = "Adam"
    bfgs_maxcor: int = 100
    # Loss
    loss_type: str = "huber"
    huber_delta: float = 0.01
    energy_loss_weight: int = 16
    force_loss_weight: int = 32
    use_stress: bool = False
    stress_loss_weight: float = 128.0
    use_switch: bool = False
    switch_after: Any = 0.75  # fraction of maxiter, or int epoch, or "auto"
    new_lr: float = None  # kept for legacy; prefer lr_reduction_factor
    lr_reduction_factor: float = 0.1  # new LR = current_LR * factor at switch point
    new_energy_w: int = 128
    new_force_w: int = 32
    new_stress_w: float = 32.0
    # Weighting & batch
    weighting_scheme: str = "uniform"
    batch_size: int = 32
    target_total_updates: int = 10000


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _is_fs_model(s: _WizardState) -> bool:
    if s.preset_name == "FS":
        return True
    if s.foundation_model_name and "-FS-" in s.foundation_model_name:
        return True
    return False


# ---------------------------------------------------------------------------
# Hierarchical foundation model selection
# ---------------------------------------------------------------------------

# Static tree: tier -> dataset -> size -> [model_name]
# ft-AM variants of OMAT models are grouped under OAM because they are
# fine-tuned on sAlex+MPTraj (AM data), identical to what "OAM" denotes.
_FOUNDATION_MODEL_TREE = {
    "FS": {
        "OAM": {"standard": ["GRACE-FS-OAM"]},
        "OMAT": {"standard": ["GRACE-FS-OMAT"]},
    },
    "1L": {
        "OAM": {
            "standard": ["GRACE-1L-OAM"],
            "medium": ["GRACE-1L-OMAT-medium-ft-AM"],
            "large": ["GRACE-1L-OMAT-large-ft-AM"],
        },
        "OMAT": {
            "standard": ["GRACE-1L-OMAT"],
            "medium": ["GRACE-1L-OMAT-medium-base", "GRACE-1L-OMAT-medium-ft-E"],
            "large": ["GRACE-1L-OMAT-large-base", "GRACE-1L-OMAT-large-ft-E"],
        },
        "SMAX-OMAT": {
            "large": ["GRACE-1L-SMAX-OMAT-large"],
        },
    },
    "2L": {
        "OAM": {
            "standard": ["GRACE-2L-OAM"],
            "medium": ["GRACE-2L-OMAT-medium-ft-AM"],
            "large": ["GRACE-2L-OMAT-large-ft-AM"],
        },
        "OMAT": {
            "standard": ["GRACE-2L-OMAT"],
            "medium": ["GRACE-2L-OMAT-medium-base", "GRACE-2L-OMAT-medium-ft-E"],
            "large": ["GRACE-2L-OMAT-large-base", "GRACE-2L-OMAT-large-ft-E"],
        },
        "SMAX-OMAT": {
            "large": ["GRACE-2L-SMAX-OMAT-large"],
            "medium": ["GRACE-2L-SMAX-OMAT-medium"],
        },
    },
}

_TIER_DESC = {
    "FS": "FS  — Finnis-Sinclair-like, fastest",
    "1L": "1L  — single message-passing layer",
    "2L": "2L  — two message-passing layers (most accurate)",
}
_DS_DESC = {
    "OAM": "OAM       — OMat24 + sAlex + MPTraj (fine-tuned on AM data)",
    "OMAT": "OMAT      — OMat24 only (base / ft-E variants)",
    "SMAX-OMAT": "SMAX-OMAT — MaxEntropy (SMAX) and OMat24 variants",
}
_SIZE_DESC = {
    "standard": "standard  — original model",
    "medium": "medium    — larger capacity",
    "large": "large     — largest capacity",
    "r5": "r5  — cutoff 5 Å",
    "r6": "r6  — cutoff 6 Å",
}
_VARIANT_DESC = {
    "base": "base      — pre-trained only",
    "ft-E": "ft-E      — fine-tuned on energies",
    "ft-AM": "ft-AM     — fine-tuned on energies + forces",
    "SMAX-OMAT": "SMAX-OMAT — trained on MaxEntropy (SMAX) + OMat24",
}


def _ask_foundation_model() -> str:
    def _choice(value, desc_map):
        label = desc_map.get(value, value)
        return questionary.Choice(label, value=value) if _HAS_QUESTIONARY else value

    # 1. Tier
    tier = _ask_select(
        "Model tier:",
        [_choice(t, _TIER_DESC) for t in _TIER_DESC],
        default="1L",
    )

    # 2. Dataset
    datasets = _FOUNDATION_MODEL_TREE[tier]
    default_ds = "OAM" if "OAM" in datasets else next(iter(datasets))
    dataset = _ask_select(
        "Training dataset:",
        [_choice(ds, _DS_DESC) for ds in _DS_DESC if ds in datasets],
        default=default_ds,
    )

    # 3. Size (skip if only one bucket)
    sizes = datasets[dataset]
    if len(sizes) == 1:
        size = next(iter(sizes))
    else:
        default_size = next(
            (s for s in ("medium", "r6") if s in sizes), next(iter(sizes))
        )
        size = _ask_select(
            "Model size:",
            [_choice(s, _SIZE_DESC) for s in _SIZE_DESC if s in sizes],
            default=default_size,
        )

    models_in_size = sizes[size]

    # 4. Variant (skip if only one model)
    if len(models_in_size) == 1:
        return models_in_size[0]

    def _variant(name):
        for v in ("ft-AM", "ft-E", "base"):
            if name.endswith("-" + v):
                return v
        # SMAX-OMAT family: dataset variant embedded in the name
        if "-SMAX-OMAT-" in name:
            return "SMAX-OMAT"
        return name

    return _ask_select(
        "Fine-tuning variant:",
        [_choice(n, {n: _VARIANT_DESC.get(_variant(n), n)}) for n in models_in_size],
        default=next(
            (n for n in models_in_size if n.endswith("-ft-E")), models_in_size[0]
        ),
    )


# ---------------------------------------------------------------------------
# Wizard sections  (each takes + returns _WizardState)
# ---------------------------------------------------------------------------


def _section_dataset(s: _WizardState) -> _WizardState:
    _section("Dataset")
    _tip_path_input()
    s.train_filename = _ask_path("Training dataset file (e.g. data.pkl.gz):")
    if not s.train_filename:
        _info("No filename entered.")
        sys.exit(1)
    _success(f"Train file: {s.train_filename}")

    s.use_separate_test = _ask_confirm(
        "Use a separate test dataset file?", default=False
    )
    if s.use_separate_test:
        s.test_filename = _ask_path("Test dataset file:")
        _success(f"Test file: {s.test_filename}")
    else:
        s.test_size = float(
            _ask_text("Test set fraction (split from train)", default=0.05)
        )
    _success(f"Test fraction: {s.test_size}")
    return s


def _section_fit_type(s: _WizardState) -> _WizardState:
    _section("Fit type")
    base_choices = ["finetune foundation model", "fit from scratch", "continue fit"]
    if _HAS_GEN_TENSOR:
        base_choices.append("Generic Tensors (L<=2)")
    s.fit_type = _ask_select(
        "Fit type:",
        choices=base_choices,
        default="finetune foundation model",
    )
    # Map to internal logic
    if s.fit_type == "finetune foundation model":
        s.finetune_model = True
    elif s.fit_type == "continue fit":
        s.finetune_model = True
    else:
        s.finetune_model = False
    return s


def _section_model(s: _WizardState) -> _WizardState:
    _section("Model Details")

    if s.finetune_model:
        if s.fit_type == "continue fit":
            s.foundation_model_name = _ask_path(
                "Model config to continue from (e.g. model.yaml):"
            )

            # Look for checkpoints
            base_dir = os.path.dirname(s.foundation_model_name)
            cp_dir = os.path.join(base_dir, "checkpoints")

            indices = []
            if os.path.exists(cp_dir):
                indices = glob.glob(os.path.join(cp_dir, "checkpoint*.index"))

            if indices:
                _success(f"Found {len(indices)} checkpoints in {cp_dir}")
                # Strip .index and keep just the prefix
                choices = [os.path.basename(i).replace(".index", "") for i in indices]
                choices = sorted(list(set(choices)))

                # Add 'auto' choice and manual choice
                choice_list = [
                    questionary.Choice("Auto (best test or latest)", value=""),
                    questionary.Choice("Browse manually...", value="__manual__"),
                ] + [questionary.Choice(c, value=c) for c in choices]

                ans = _ask_select(
                    "Which checkpoint to load?",
                    choices=choice_list,
                    default="",
                )

                if ans == "__manual__":
                    s.checkpoint_name = _ask_path(
                        "Path to checkpoint index file (*.index):"
                    )
                elif ans == "":
                    # Auto: prefer best_test_loss checkpoint, fall back to latest
                    best_test = os.path.join(cp_dir, "checkpoint.best_test_loss")
                    latest = os.path.join(cp_dir, "checkpoint")
                    if os.path.exists(f"{best_test}.index"):
                        s.checkpoint_name = best_test
                        _info("Auto-selected: checkpoint.best_test_loss")
                    elif os.path.exists(f"{latest}.index"):
                        s.checkpoint_name = latest
                        _info("Auto-selected: checkpoint (latest)")
                    else:
                        s.checkpoint_name = ""
                else:
                    # try_load_checkpoint joins checkpoint_dir + checkpoint_name
                    # but if we provide an explicit checkpoint_name in input.yaml,
                    # gracemaker uses it directly.
                    # So we should provide the full path to the checkpoint (without .index)
                    s.checkpoint_name = os.path.join(cp_dir, ans)
            else:
                _info(f"No regular checkpoints found in {cp_dir}.")
                s.checkpoint_name = _ask_path(
                    "Select checkpoint index file (*.index) manually:"
                )

            if s.checkpoint_name.endswith(".index"):
                s.checkpoint_name = s.checkpoint_name.replace(".index", "")
        else:
            s.foundation_model_name = _ask_foundation_model()
        label = "Previous model" if s.fit_type == "continue fit" else "Foundation model"
        _success(f"{label}: {s.foundation_model_name}")
        if s.checkpoint_name:
            _success(f"Checkpoint: {s.checkpoint_name}")

        # Finetuning mode selection
        if s.fit_type == "finetune foundation model":
            s.finetune_mode = _ask_select(
                "Finetuning mode:",
                choices=[
                    "naive finetuning (all model parameters will be updated)",
                    "frozen weights (only some parameters will be updated)",
                ],
                default="naive finetuning (all model parameters will be updated)",
            )
            # Normalise to short key
            if s.finetune_mode.startswith("frozen"):
                s.finetune_mode = "frozen"
            else:
                s.finetune_mode = "naive"
            _success(f"Finetuning mode: {s.finetune_mode}")

        s.learning_rate = 0.001
        s.eval_init_stats = True
        if s.fit_type == "continue fit":
            s.reset_epoch_and_step = True
            _info(
                "Note: reset_epoch_and_step is set to True (training will start from epoch 0)"
            )
    else:
        def_preset = get_default_preset_name()
        s.preset_name = _ask_select(
            "Model preset:",
            choices=get_public_preset_list(),
            default=def_preset if def_preset is not None else "GRACE_1LAYER",
        )
        _success(f"Preset: {s.preset_name}")

        avail = get_preset_settings(s.preset_name)
        if avail is not None:
            s.preset_complexity = _ask_select(
                "Model complexity:",
                choices=list(avail.keys()),
                default="medium",
            )
            _success(f"Complexity: {s.preset_complexity}")
            kwargs = avail[s.preset_complexity].copy()
            def_cutoff = kwargs.pop("rcut")
            s.preset_kwargs_str = (
                json.dumps(kwargs).strip().replace('"', "").replace("'", "")
            )
            s.cutoff = float(_ask_text("Cutoff radius (Å)", default=def_cutoff))
        else:
            s.preset_kwargs_str = "{}"
            s.cutoff = 6.0
        _success(f"Cutoff: {s.cutoff} Å")
        s.learning_rate = 0.008
        s.eval_init_stats = False
    return s


def _section_optimizer(s: _WizardState) -> _WizardState:
    _section("Optimizer")
    is_fs = _is_fs_model(s)

    if is_fs:
        if s.fit_type == "fit from scratch":
            _info(
                "FS from scratch: BFGS (full Hessian) is recommended for small/medium models."
            )
            _info(
                "If your FS model has many parameters (large lmax/order), prefer L-BFGS-B instead."
            )
            default_opt = "BFGS"
        else:  # finetune or continue
            _info(
                "FS fine-tuning: L-BFGS-B is recommended (limited-memory, scales well from a warm start)."
            )
            default_opt = "L-BFGS-B"
    else:
        default_opt = "Adam"

    s.optimizer = _ask_select(
        "Optimizer:", choices=["Adam", "L-BFGS-B", "BFGS"], default=default_opt
    )
    _success(f"Optimizer: {s.optimizer}")

    if s.optimizer == "BFGS":
        _info(
            "Note: BFGS stores the full Hessian approximation — use only for small/medium FS models."
        )
    elif s.optimizer == "L-BFGS-B":
        _info("No learning-rate or scheduler needed for L-BFGS-B.")
    return s


def _section_loss(s: _WizardState) -> _WizardState:
    _section("Loss function")

    s.loss_type = _ask_select(
        "Loss type:", choices=["huber", "square"], default="huber"
    )
    _success(f"Loss type: {s.loss_type}")

    if s.loss_type == "huber":
        s.huber_delta = float(_ask_text("Huber delta", default=0.01))

    s.energy_loss_weight = int(_ask_text("Energy loss weight", default=16))
    _success(f"Energy weight: {s.energy_loss_weight}")

    s.force_loss_weight = int(_ask_text("Force loss weight", default=32))
    _success(f"Force weight: {s.force_loss_weight}")

    s.use_stress = _ask_confirm("Include stress in the loss?", default=False)
    if s.use_stress:
        s.stress_loss_weight = float(
            _ask_text("Stress loss weight", default=s.stress_loss_weight)
        )
        _success(f"Stress weight: {s.stress_loss_weight}")

    if s.optimizer != "Adam":
        _info("Loss-weight switching not supported for quasi-Newton optimizers.")
        s.use_switch = False
    else:
        s.use_switch = _ask_confirm("Switch E/F/S weights mid-training?", default=False)
        if s.use_switch:
            s.switch_after = 0.75
            s.lr_reduction_factor = 0.1
            s.new_energy_w = int(
                _ask_text(
                    f"Energy weight after switch (was {s.energy_loss_weight})",
                    default=128,
                )
            )
            s.new_force_w = int(
                _ask_text(
                    f"Force weight after switch (was {s.force_loss_weight})", default=32
                )
            )
            if s.use_stress:
                s.new_stress_w = float(
                    _ask_text(
                        f"Stress weight after switch (was {s.stress_loss_weight})",
                        default=s.stress_loss_weight,
                    )
                )
    return s


def _section_weighting(s: _WizardState) -> _WizardState:
    _section("Weighting & batch size")

    s.weighting_scheme = _ask_select(
        "Sample weighting scheme:", choices=["uniform", "energy"], default="uniform"
    )
    _success(f"Weighting: {s.weighting_scheme}")

    if s.optimizer in ("L-BFGS-B", "BFGS"):
        _info(
            "Batch size: less relevant for quasi-Newton (full dataset per step), but must be specified."
        )
    s.batch_size = int(_ask_text("Batch size", default=16))
    _success(f"Batch size: {s.batch_size}  (test: {4 * s.batch_size})")

    if s.optimizer in ("L-BFGS-B", "BFGS"):
        if s.fit_type == "fit from scratch":
            def_target = 500
        else:
            def_target = 100
        s.target_total_updates = int(
            _ask_text("Max iterations (epochs)", default=def_target)
        )
    else:
        if s.fit_type == "fit from scratch":
            def_target = 50000
        else:
            def_target = 10000
        s.target_total_updates = int(
            _ask_text("Target total updates", default=def_target)
        )
    _success(f"Total updates: {s.target_total_updates}")
    return s


# ---------------------------------------------------------------------------
# Review & template application
# ---------------------------------------------------------------------------


def _show_review(s: _WizardState):
    if _HAS_RICH:
        from rich.table import Table

        t = Table(box=rich_box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column("key", style="dim", no_wrap=True)
        t.add_column("value", style="bold")

        t.add_row("Fit type", s.fit_type)
        t.add_row("Train file", s.train_filename)
        t.add_row(
            "Test",
            s.test_filename if s.use_separate_test else f"split {s.test_size:.0%}",
        )
        t.add_row("Save dataset", "True" if not s.use_separate_test else "False")
        if s.finetune_model:
            label = (
                "Previous model" if s.fit_type == "continue fit" else "Foundation model"
            )
            t.add_row(label, s.foundation_model_name)
            if s.checkpoint_name:
                t.add_row("Checkpoint", os.path.basename(s.checkpoint_name))
        else:
            t.add_row(
                "Model",
                f"{s.preset_name} / {s.preset_complexity}  cutoff={s.cutoff} Å",
            )
        if s.finetune_model and s.fit_type == "finetune foundation model":
            t.add_row("Finetuning mode", s.finetune_mode)
        t.add_row("Eval initial stats", str(s.eval_init_stats))
        t.add_row("Reset epoch/step", str(s.reset_epoch_and_step))
        t.add_row("Optimizer", s.optimizer)
        stress_part = f"  S={s.stress_loss_weight}" if s.use_stress else ""
        t.add_row(
            "Loss",
            f"{s.loss_type}  E={s.energy_loss_weight}  F={s.force_loss_weight}{stress_part}",
        )
        t.add_row("Weighting", s.weighting_scheme)
        t.add_row("Batch size", str(s.batch_size))
        t.add_row("Total updates", str(s.target_total_updates))

        _rich_console.print()
        _rich_console.print(
            Panel(
                t,
                title="[bold]Review[/bold]",
                border_style="cyan",
                box=rich_box.ROUNDED,
            )
        )
    else:
        print("\n--- Review ---")
        print(f"  Fit type:     {s.fit_type}")
        print(f"  Train file:  {s.train_filename}")
        print(
            f"  Test:        {s.test_filename if s.use_separate_test else f'split {s.test_size}'}"
        )
        print(f"  Save dataset: {'True' if not s.use_separate_test else 'False'}")
        if s.finetune_model:
            label = (
                "Previous model" if s.fit_type == "continue fit" else "Foundation model"
            )
            print(
                f"  {label}:" + " " * (10 - len(label)) + f"{s.foundation_model_name}"
            )
            if s.checkpoint_name:
                print(f"  Checkpoint:  {os.path.basename(s.checkpoint_name)}")
        else:
            print(
                f"  Model:       {s.preset_name}/{s.preset_complexity}  cutoff={s.cutoff}"
            )
        if s.finetune_model and s.fit_type == "finetune foundation model":
            print(f"  Finetune:    {s.finetune_mode}")
        print(f"  Eval init:   {s.eval_init_stats}")
        print(f"  Reset epoch: {s.reset_epoch_and_step}")
        print(f"  Optimizer:   {s.optimizer}")
        print(f"  Loss:        {s.loss_type}  F={s.force_loss_weight}")
        print(f"  Weighting:   {s.weighting_scheme}  batch={s.batch_size}")
        print(f"  Total updates: {s.target_total_updates}")
        print("--------------")


def _apply_state(s: _WizardState, template: str) -> str:
    # Dataset
    template = template.replace("{{TRAIN_FILENAME}}", s.train_filename)
    if s.use_separate_test:
        template = template.replace(
            "{{TEST_DATA}}", f"test_filename: {s.test_filename}"
        )
        template = template.replace("{{SAVE_DATASET}}", "False")
    else:
        template = template.replace("{{TEST_DATA}}", f"test_size: {s.test_size}")
        template = template.replace("{{SAVE_DATASET}}", "True")

    # Model
    if s.finetune_model:
        if s.fit_type == "continue fit":
            potential_block = f"filename: {s.foundation_model_name}\n\n"
        else:
            potential_block = (
                f"finetune_foundation_model: {s.foundation_model_name}\n\n"
            )

        if s.checkpoint_name:
            potential_block += f"  checkpoint_name: {s.checkpoint_name}\n\n"

        if s.fit_type == "continue fit":
            potential_block += "  # model will be restored from checkpoint"
        else:
            potential_block += (
                "  reduce_elements: True  # reduce to elements present in dataset"
            )

        template = template.replace("{{CUTOFF}}", "n/a")
    else:
        potential_block = (
            f"preset: {s.preset_name}\n\n"
            "  ## For custom model from model.py::custom_model\n"
            "  #  custom: model.custom_model\n\n"
            "  # keyword-arguments passed to preset or custom function\n"
            f"  kwargs: {s.preset_kwargs_str}\n\n"
            "  #shift: False # True/False\n"
            "  scale: True  # False/True or float"
        )
        template = template.replace("{{CUTOFF}}", str(s.cutoff))

    template = template.replace("{{POTENTIAL_SETTINGS}}", potential_block)
    template = template.replace("{{eval_init_stats}}", str(s.eval_init_stats))
    template = template.replace("{{RESET_EPOCH}}", str(s.reset_epoch_and_step))

    # Frozen weights finetuning
    trainable_var_str = ""
    if s.finetune_mode == "frozen":
        if "2L" in s.foundation_model_name:
            trainable_var_str = (
                'trainable_variable_names: ["I2/reducing_", "rho/reducing_", "I1/reducing_"]'
            )
        else:
            trainable_var_str = 'trainable_variable_names: ["rho/reducing_"]'
    template = template.replace("{{TRAINABLE_VARIABLE_NAMES}}", trainable_var_str)

    # Optimizer block
    if s.optimizer in ("L-BFGS-B", "BFGS"):
        optimizer_block = (
            f"  optimizer: {s.optimizer}\n"
            f'  opt_params: {{"maxcor": {s.bfgs_maxcor}, "maxls": 20, "gtol": 1.e-8, "iprint": -1}}\n'
            f"  # No LR scheduler needed for quasi-Newton optimizers\n"
        )
    else:  # Adam
        min_lr = s.learning_rate / 12
        scheduler_name = "reduce_on_plateau" if s.use_switch else "cosine_decay"
        scheduler_params = (
            f'{{ "patience": 50, "reduction_factor": 0.8, "minimum_learning_rate": {min_lr}, "stop_at_min": True }}'
            if s.use_switch
            else f'{{ "minimal_learning_rate": {min_lr} }}'
        )
        optimizer_block = (
            f"  optimizer: Adam\n"
            f"  opt_params: {{\n"
            f"            learning_rate: {s.learning_rate},\n"
            f"            use_ema: True,\n"
            f"            ema_momentum: 0.99,\n"
            f"            weight_decay: null,\n"
            f"            clipnorm: 1.0,\n"
            f"        }}\n\n"
            f"  scheduler: {scheduler_name}\n"
            f"  scheduler_params: {scheduler_params}\n"
        )
    template = template.replace("{{OPTIMIZER_BLOCK}}", optimizer_block)

    # Loss
    template = template.replace("{{LOSS_TYPE}}", s.loss_type)
    extra_e = f", delta: {s.huber_delta}" if s.loss_type == "huber" else ""
    template = template.replace("{{EXTRA_E_ARGS}}", extra_e)
    template = template.replace("{{ENERGY_LOSS_WEIGHT}}", str(s.energy_loss_weight))
    template = template.replace("{{FORCE_LOSS_WEIGHT}}", str(s.force_loss_weight))

    if s.use_stress:
        stress_str = f"stress: {{ weight: {s.stress_loss_weight}, type: {s.loss_type}{extra_e} }},"
        template = template.replace("{{STRESS_LOSS}}", stress_str)
    else:
        template = template.replace("{{STRESS_LOSS}}", "")

    if s.use_switch:
        sw = (
            f"switch: {{ after_iter: {s.switch_after}, "
            f"learning_rate_reduction_factor: {s.lr_reduction_factor}, "
            f"energy: {{ weight: {s.new_energy_w} }}, "
            f"forces: {{ weight: {s.new_force_w} }}, "
        )
        if s.use_stress:
            sw += f"stress: {{ weight: {s.new_stress_w} }}, "
        sw += "}"
        template = template.replace("{{SWITCH_LOSS}}", sw)
    else:
        template = template.replace("{{SWITCH_LOSS}}", "")

    # Weighting
    if s.weighting_scheme == "energy":
        weighting = (
            "weighting: { type: energy_based, DElow: 1.0, DEup: 10.0, "
            "DFup: 50.0, DE: 1.0, DF: 1.0, wlow: 0.75, energy: convex_hull, seed: 42}"
        )
        template = template.replace("{{COMPUTE_CONVEX_HULL}}", "True")
    else:
        weighting = ""
        template = template.replace("{{COMPUTE_CONVEX_HULL}}", "False")
    template = template.replace("{{WEIGHTING_SCHEME}}", weighting)

    # Batch
    template = template.replace("{{BATCH_SIZE}}", str(s.batch_size))
    template = template.replace("{{TEST_BATCH_SIZE}}", str(4 * s.batch_size))
    template = template.replace("{{TARGET_TOTAL_UPDATES}}", str(s.target_total_updates))

    return template


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_template_input():
    _print_header()

    try:
        with resources.open_text(
            "tensorpotential.resources", "input_template.yaml"
        ) as f:
            original_template = f.read()

        sections = [
            ("Fit type", _section_fit_type),
            ("Dataset", _section_dataset),
            ("Model Details", _section_model),
            ("Optimizer", _section_optimizer),
            ("Loss function", _section_loss),
            ("Weighting & batch", _section_weighting),
        ]

        # Run fit type first so we can branch before loading the template
        s = _WizardState()
        s = _section_fit_type(s)

        if s.fit_type == "Generic Tensors (L<=2)":
            run_gen_tensor_wizard()
            return  # run_gen_tensor_wizard calls sys.exit itself

        # Run remaining sections once
        for _, fn in sections[1:]:
            s = fn(s)

        # Review + re-edit loop
        while True:
            _show_review(s)

            if _HAS_QUESTIONARY:
                redo_choices = [
                    questionary.Choice("Yes, write input.yaml", value="ok")
                ] + [
                    questionary.Choice(f"Re-do: {name}", value=i)
                    for i, (name, _) in enumerate(sections)
                ]
                answer = questionary.select(
                    "Looks good?", choices=redo_choices, style=_QUESTIONARY_STYLE
                ).ask()
                if answer is None:
                    raise KeyboardInterrupt
            else:
                print("Re-do a section? Enter number or press Enter to confirm:")
                for i, (name, _) in enumerate(sections):
                    print(f"  {i + 1}. {name}")
                raw = input("Section number (Enter = confirm): ").strip()
                answer = "ok" if not raw else int(raw) - 1

            if answer == "ok":
                break
            _, fn = sections[answer]
            s = fn(s)

        # Write output
        output = _apply_state(s, original_template)
        with open("input.yaml", "w") as f:
            print(output, file=f)

    except KeyboardInterrupt:
        if _HAS_RICH:
            _rich_console.print("\n[bold red]Interrupted.[/bold red] Exiting.")
        else:
            print("\nInterrupted. Exiting.")
        sys.exit(0)

    if _HAS_RICH:
        _rich_console.print()
        _rich_console.print(
            Panel(
                "[bold green]input.yaml[/bold green] written successfully!",
                border_style="green",
                box=rich_box.ROUNDED,
                expand=False,
            )
        )
        _rich_console.print()
    else:
        print("\nInput file written to `input.yaml`")
    sys.exit(0)
