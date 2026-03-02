"""gen_tensor wizard — called from gracemaker -t when user picks 'Generic Tensors (L<=2)'."""
from __future__ import annotations

import sys
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Wizard state
# ---------------------------------------------------------------------------


@dataclass
class _WizardStateTensor:
    # Dataset (duck-type-compatible with _WizardState for _section_dataset reuse)
    train_filename: str = ""
    use_separate_test: bool = False
    test_filename: str = ""
    test_size: float = 0.05
    # Tensor-specific
    tensor_components: list = None
    tensor_rank: int = 2
    per_structure: bool = False
    preset: str = "TENSOR_1L"
    cutoff: float = 6.0
    compute_energy: bool = False
    compute_forces: bool = False
    # Loss
    loss_type: str = "huber"
    tensor_delta: float = 0.1
    tensor_weight: float = 10.0
    energy_delta: float = 0.01
    energy_weight: float = 1.0
    forces_delta: float = 0.01
    forces_weight: float = 5.0
    # Batch & training
    batch_size: int = 10
    target_total_updates: int = 10000


# ---------------------------------------------------------------------------
# Wizard sections
# ---------------------------------------------------------------------------

_TENSOR_CHOICES_MAP = {
    "[1]": ([1], "[1]       — First-rank vector (e.g. Forces as tensor)"),
    "[2]": ([2], "[2]       — Symmetric, traceless L=2 (e.g. EFG)"),
    "[0, 2]": ([0, 2], "[0, 2]    — Symmetric, non-traceless (e.g. Stress)"),
    "[1, 2]": ([1, 2], "[1, 2]    — Antisymmetric, traceless"),
    "[0, 1, 2]": ([0, 1, 2], "[0, 1, 2] — General non-symmetric (e.g. BEC)"),
}


def _section_tensor_components(s: _WizardStateTensor) -> _WizardStateTensor:
    from tensorpotential.cli.wizard import _ask_select, _section, _success, _HAS_QUESTIONARY

    _section("Tensor type")

    if _HAS_QUESTIONARY:
        import questionary
        choices = [
            questionary.Choice(desc, value=key)
            for key, (_, desc) in _TENSOR_CHOICES_MAP.items()
        ]
    else:
        choices = list(_TENSOR_CHOICES_MAP.keys())

    chosen_key = _ask_select("Tensor type:", choices=choices, default="[0, 1, 2]")
    s.tensor_components = _TENSOR_CHOICES_MAP[chosen_key][0]
    s.tensor_rank = 1 if s.tensor_components == [1] else 2
    _success(f"Tensor components: {s.tensor_components}  rank={s.tensor_rank}")
    return s


def _section_tensor_per_structure(s: _WizardStateTensor) -> _WizardStateTensor:
    from tensorpotential.cli.wizard import _ask_select, _section, _success, _HAS_QUESTIONARY

    _section("Per-structure")

    _PER_STRUCTURE_DESCS = {
        "per-atom":      "per-atom      — one tensor per atom  (e.g. EFG, BEC, forces)",
        "per-structure": "per-structure — one tensor per structure  (e.g. stress × volume)",
    }

    if _HAS_QUESTIONARY:
        import questionary
        choices = [questionary.Choice(desc, value=key)
                   for key, desc in _PER_STRUCTURE_DESCS.items()]
    else:
        choices = list(_PER_STRUCTURE_DESCS.keys())

    answer = _ask_select(
        "Property granularity:", choices=choices, default="per-atom"
    )
    s.per_structure = answer == "per-structure"
    _success(f"Per structure: {s.per_structure}")
    return s


def _section_tensor_model(s: _WizardStateTensor) -> _WizardStateTensor:
    from tensorpotential.cli.wizard import _ask_select, _ask_text, _section, _success, _HAS_QUESTIONARY

    _section("Model")

    _PRESET_DESCS = {
        "TENSOR_1L": "TENSOR_1L — 1 message-passing layer, lighter (lmax=4, embedding=32)",
        "TENSOR_2L": "TENSOR_2L — 2 message-passing layers, more expressive (lmax=4, embedding=128)",
    }

    if _HAS_QUESTIONARY:
        import questionary
        choices = [questionary.Choice(desc, value=key) for key, desc in _PRESET_DESCS.items()]
    else:
        choices = list(_PRESET_DESCS.keys())

    s.preset = _ask_select("Model preset:", choices=choices, default="TENSOR_1L")
    _success(f"Preset: {s.preset}")

    default_cutoff = 6.0 if s.preset == "TENSOR_1L" else 5.0
    s.cutoff = float(_ask_text("Cutoff radius (Å)", default=default_cutoff))
    _success(f"Cutoff: {s.cutoff} Å")
    return s


def _section_tensor_energy(s: _WizardStateTensor) -> _WizardStateTensor:
    from tensorpotential.cli.wizard import _ask_confirm, _section, _success

    _section("Energy & forces")
    s.compute_energy = _ask_confirm("Fit energy alongside tensor?", default=False)
    _success(f"Compute energy: {s.compute_energy}")
    if s.compute_energy:
        s.compute_forces = _ask_confirm("Fit forces too?", default=True)
        _success(f"Compute forces: {s.compute_forces}")
    else:
        s.compute_forces = False
    return s


def _section_tensor_loss(s: _WizardStateTensor) -> _WizardStateTensor:
    from tensorpotential.cli.wizard import _ask_select, _ask_text, _section, _success

    _section("Loss")
    s.loss_type = _ask_select("Loss type:", choices=["huber", "square"], default="huber")
    _success(f"Loss type: {s.loss_type}")

    if s.loss_type == "huber":
        s.tensor_delta = float(_ask_text("Tensor Huber delta", default=0.1))

    s.tensor_weight = float(_ask_text("Tensor loss weight", default=10.0))
    _success(f"Tensor weight: {s.tensor_weight}")

    if s.compute_energy:
        s.energy_weight = float(_ask_text("Energy loss weight", default=1.0))
        _success(f"Energy weight: {s.energy_weight}")
        if s.loss_type == "huber":
            s.energy_delta = float(_ask_text("Energy/forces Huber delta", default=0.01))
        if s.compute_forces:
            s.forces_weight = float(_ask_text("Forces loss weight", default=5.0))
            _success(f"Forces weight: {s.forces_weight}")
    return s


def _section_tensor_batch(s: _WizardStateTensor) -> _WizardStateTensor:
    from tensorpotential.cli.wizard import _ask_text, _section, _success

    _section("Batch & training")
    s.batch_size = int(_ask_text("Batch size", default=10))
    _success(f"Batch size: {s.batch_size}  (test: {s.batch_size * 5})")
    s.target_total_updates = int(_ask_text("Target total updates", default=10000))
    _success(f"Total updates: {s.target_total_updates}")
    return s


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------


def _show_review_tensor(s: _WizardStateTensor):
    from tensorpotential.cli.wizard import _HAS_RICH

    if _HAS_RICH:
        from tensorpotential.cli.wizard import _rich_console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box as rich_box

        t = Table(box=rich_box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column("key", style="dim", no_wrap=True)
        t.add_column("value", style="bold")

        t.add_row("Tensor components", str(s.tensor_components))
        t.add_row("Tensor rank", str(s.tensor_rank))
        t.add_row("Per structure", str(s.per_structure))
        t.add_row("Preset", s.preset)
        t.add_row("Cutoff", f"{s.cutoff} Å")
        t.add_row("Compute energy", str(s.compute_energy))
        t.add_row("Compute forces", str(s.compute_forces))
        t.add_row("Loss type", s.loss_type)
        tensor_loss_str = f"weight={s.tensor_weight}"
        if s.loss_type == "huber":
            tensor_loss_str += f", delta={s.tensor_delta}"
        t.add_row("Tensor loss", tensor_loss_str)
        if s.compute_energy:
            ef_str = f"weight={s.energy_weight}"
            if s.loss_type == "huber":
                ef_str += f", delta={s.energy_delta}"
            t.add_row("Energy loss", ef_str)
        if s.compute_forces:
            t.add_row("Forces loss", f"weight={s.forces_weight}")
        t.add_row("Train file", s.train_filename)
        t.add_row(
            "Test",
            s.test_filename if s.use_separate_test else f"split {s.test_size:.0%}",
        )
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
        print(f"  Tensor components: {s.tensor_components}  rank={s.tensor_rank}")
        print(f"  Per structure:     {s.per_structure}")
        print(f"  Preset:            {s.preset}  cutoff={s.cutoff} Å")
        print(f"  Compute energy:    {s.compute_energy}")
        print(f"  Compute forces:    {s.compute_forces}")
        print(f"  Loss:              {s.loss_type}  tensor_w={s.tensor_weight}")
        print(f"  Train file:        {s.train_filename}")
        print(
            f"  Test:              {s.test_filename if s.use_separate_test else f'split {s.test_size}'}"
        )
        print(f"  Batch size:        {s.batch_size}")
        print(f"  Total updates:     {s.target_total_updates}")
        print("--------------")


# ---------------------------------------------------------------------------
# Dataset verification helpers
# ---------------------------------------------------------------------------

# ASE Voigt convention: [xx, yy, zz, yz, xz, xy]
# Expanded to full row-major 3×3: [xx, xy, xz, yx, yy, yz, zx, zy, zz]
_VOIGT6_TO_FULL9_IDX = [0, 5, 4, 5, 1, 3, 4, 3, 2]

# Which symmetry checks apply to each tensor_components choice
# (traceless, symmetric, antisymmetric)
_SYMMETRY_CHECKS = {
    str([1]):       (False, False, False),
    str([2]):       (True,  True,  False),
    str([0, 2]):    (False, True,  False),
    str([1, 2]):    (True,  False, True),
    str([0, 1, 2]): (False, False, False),
}


def _collect_tensor_arrays(df, col, max_rows=100):
    """Stack tensor_property from up to max_rows rows into one (M, C) array."""
    import numpy as np

    stacked = []
    for val in df[col].iloc[:max_rows]:
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
        stacked.append(arr)
    return np.vstack(stacked)


def _check_tensor_shapes(df, s, max_rows=50):
    """
    Verify each row's tensor_property has the expected shape.
    Returns (ok: bool, issues: list[str]).
    """
    import numpy as np

    expected_cols = 3 if s.tensor_rank == 1 else 9
    issues = []

    for idx, row in df.iloc[:max_rows].iterrows():
        atoms = row["ase_atoms"]
        n_expected = 1 if s.per_structure else len(atoms)
        try:
            arr = np.asarray(row["tensor_property"], dtype=float)
        except Exception as exc:
            issues.append(f"  row {idx}: cannot convert to array — {exc}")
            continue

        # Normalise to 2-D
        if arr.ndim == 1:
            arr2 = arr.reshape(1, -1)
        elif arr.ndim == 3:
            arr2 = arr.reshape(arr.shape[0], -1)
        else:
            arr2 = arr

        if arr2.shape != (n_expected, expected_cols):
            raw_shape = np.asarray(row["tensor_property"]).shape
            issues.append(
                f"  row {idx}: expected ({n_expected},{expected_cols}), "
                f"raw shape is {raw_shape}"
            )

    return len(issues) == 0, issues


def _check_tensor_symmetry(df, s, max_rows=100):
    """
    Check traceless/symmetric/antisymmetric constraints.
    Returns list of (label, max_violation, mean_violation, passed).
    """
    import numpy as np

    if s.tensor_rank == 1:
        return []

    key = str(s.tensor_components)
    do_traceless, do_symmetric, do_antisymmetric = _SYMMETRY_CHECKS.get(
        key, (False, False, False)
    )
    if not any([do_traceless, do_symmetric, do_antisymmetric]):
        return []

    T = _collect_tensor_arrays(df, "tensor_property", max_rows)  # (M, 9)
    if T.shape[1] != 9:
        return [("Shape mismatch — cannot check symmetry", 0.0, 0.0, False)]

    T3 = T.reshape(-1, 3, 3)
    scale = float(np.abs(T).max()) or 1.0
    results = []

    if do_traceless:
        trace = T3[:, 0, 0] + T3[:, 1, 1] + T3[:, 2, 2]
        mx = float(np.abs(trace).max())
        mn = float(np.abs(trace).mean())
        results.append(("Traceless |trace|", mx, mn, mx / scale < 1e-3))

    if do_symmetric:
        diff = np.abs(T3 - T3.transpose(0, 2, 1))
        mx = float(diff.max())
        mn = float(diff.mean())
        results.append(("Symmetric |T−Tᵀ|", mx, mn, mx / scale < 1e-3))

    if do_antisymmetric:
        spart = np.abs(T3 + T3.transpose(0, 2, 1))
        mx = float(spart.max())
        mn = float(spart.mean())
        results.append(("Antisymmetric |T+Tᵀ|", mx, mn, mx / scale < 1e-3))

    return results


def _detect_source_shape(df, col):
    """Return numpy shape of first row of col (None on error)."""
    import numpy as np

    try:
        return np.asarray(df[col].iloc[0]).shape
    except Exception:
        return None


def _generate_tensor_property_interactive(df, s):
    """
    Interactively ask for a source column, convert it to tensor_property,
    apply per-structure scaling if needed, and return the modified df.
    Returns None if user cancels or conversion fails.
    """
    import numpy as np
    from tensorpotential.cli.wizard import (
        _ask_select,
        _ask_confirm,
        _info,
        _success,
        _HAS_QUESTIONARY,
    )

    # Offer all columns except tensor_property itself and the non-numeric index column
    skip = {"ase_atoms", "tensor_property"}
    candidates = [c for c in df.columns if c not in skip]
    if not candidates:
        _info("No candidate columns found for tensor_property generation.")
        return None

    _info(f"Available columns: {', '.join(candidates)}")

    source_col = _ask_select(
        "Select source column for tensor_property:",
        choices=candidates,
        default=candidates[0],
    )

    shape = _detect_source_shape(df, source_col)
    if shape is None:
        _info("Cannot read source column.")
        return None
    _info(f"Source '{source_col}' per-row shape: {shape}")

    rank = s.tensor_rank

    # --- determine conversion ---
    def _infer_conversion(shape, rank, per_structure):
        if rank == 1:
            if shape == (3,):
                return "vec1d_to_1x3" if per_structure else "vec1d_to_Nx3"
            if len(shape) == 2 and shape[1] == 3:
                return "ok"
            return None
        else:  # rank 2
            if shape == (9,):
                return "flat9_to_1x9" if per_structure else "flat9_to_Nx9"
            if shape == (3, 3):
                return "mat_to_1x9"
            if len(shape) == 2 and shape[1] == 9:
                return "ok"
            if len(shape) == 3 and shape[1] == 3 and shape[2] == 3:
                return "Nx3x3_to_Nx9"
            if shape == (6,) or (len(shape) == 2 and shape[1] == 6):
                return "voigt6"
            return None

    conv = _infer_conversion(shape, rank, s.per_structure)

    if conv is None:
        _info(
            f"Cannot auto-convert shape {shape} for rank-{rank} tensor. "
            f"Expected one of: (N,{3 if rank==1 else 9}), "
            + ("(N,3,3), (3,3), (9,), (N,6), (6,)" if rank == 2 else "(3,)")
        )
        return None

    if conv == "voigt6":
        _info(
            "Detected 6-component format. Assuming ASE Voigt convention: "
            "[xx, yy, zz, yz, xz, xy] — symmetric expansion."
        )
        if not _ask_confirm("Proceed with this convention?", default=True):
            return None

    # --- scaling for per-structure ---
    scale_mode = "none"
    if s.per_structure:
        _info("Per-structure properties must be EXTENSIVE (total, not normalised).")
        scale_choices = [
            "Already extensive — no scaling needed",
            "Intensive per volume (e.g. eV/Å³) → multiply by cell volume",
            "Intensive per atom (e.g. eV/atom) → multiply by number of atoms",
        ]
        scale_mode = _ask_select(
            "Source data is:", choices=scale_choices, default=scale_choices[0]
        )

    # --- apply row-by-row ---
    _info(f"Generating tensor_property from '{source_col}'...")

    def _convert(row):
        val = np.asarray(row[source_col], dtype=float)
        atoms = row["ase_atoms"]

        # shape conversion
        if conv == "ok":
            tp = val
        elif conv in ("vec1d_to_1x3", "flat9_to_1x9"):
            tp = val.reshape(1, -1)
        elif conv in ("vec1d_to_Nx3", "flat9_to_Nx9"):
            # shape (C,) but per-atom — this shouldn't happen; reshape best-effort
            tp = val.reshape(1, -1)
        elif conv == "mat_to_1x9":
            tp = val.reshape(1, 9)
        elif conv == "Nx3x3_to_Nx9":
            tp = val.reshape(val.shape[0], 9)
        elif conv == "voigt6":
            v = val.reshape(-1, 6)
            tp = v[:, _VOIGT6_TO_FULL9_IDX]
        else:
            tp = val

        # scaling
        if scale_mode.startswith("Intensive per volume"):
            tp = tp * atoms.get_volume()
        elif scale_mode.startswith("Intensive per atom"):
            tp = tp * len(atoms)

        return tp

    try:
        df = df.copy()
        df["tensor_property"] = df.apply(_convert, axis=1)
        ex = df["tensor_property"].iloc[0]
        _success(
            f"Generated tensor_property  (example shape: {np.asarray(ex).shape})"
        )
    except Exception as exc:
        _info(f"Conversion error: {exc}")
        return None

    return df


def _verify_one_df(df, filename, s, *, label="dataset"):
    """
    Check one loaded DataFrame for required columns, tensor_property
    shape and symmetry.  Offers to generate tensor_property if absent.

    Returns (updated_filename, all_ok).
    updated_filename differs from filename only when a new file is saved.
    """
    from tensorpotential.cli.wizard import _ask_confirm, _ask_text, _info, _success

    all_ok = True

    # ── Required columns ────────────────────────────────────────────────────
    required = ["ase_atoms"]
    if s.compute_energy:
        required.append("energy")
    if s.compute_forces:
        required.append("forces")

    for col in required:
        if col in df.columns:
            _success(f"[{label}] '{col}': present")
        else:
            _info(f"[{label}] '{col}': MISSING (required for training)")
            all_ok = False

    # ── tensor_property ──────────────────────────────────────────────────────
    if "tensor_property" not in df.columns:
        _info(f"[{label}] 'tensor_property': MISSING")
        all_ok = False

        if _ask_confirm(
            f"Generate 'tensor_property' for {label} from an existing column?",
            default=True,
        ):
            df_new = _generate_tensor_property_interactive(df, s)
            if df_new is not None:
                df = df_new
                default_out = (
                    filename.replace(".pckl.gz", "_with_tensor.pckl.gz")
                    if ".pckl.gz" in filename
                    else filename + ".tensor.pckl.gz"
                )
                out_path = str(
                    _ask_text(f"Save modified {label} dataset to:", default=default_out)
                )
                try:
                    df.to_pickle(out_path)
                    _success(f"Saved {label} dataset to {out_path}")
                    filename = out_path
                    _info(f"Updated {label} filename in input.yaml.")
                    all_ok = True
                except Exception as exc:
                    _info(f"Could not save: {exc}")
    else:
        _success(f"[{label}] 'tensor_property': present")

        # ── Shape check ──────────────────────────────────────────────────────
        _info(f"[{label}] Checking tensor_property shapes …")
        ok, issues = _check_tensor_shapes(df, s)
        if ok:
            _success(f"[{label}] Shape: OK (checked up to 50 structures)")
        else:
            _info(f"[{label}] Shape issues:")
            for iss in issues[:5]:
                _info(iss)
            if len(issues) > 5:
                _info(f"  … and {len(issues) - 5} more")
            all_ok = False

        # ── Symmetry / traceless check ───────────────────────────────────────
        if s.tensor_rank == 2:
            sym_results = _check_tensor_symmetry(df, s)
            if sym_results:
                _info(f"[{label}] Checking symmetry properties …")
                for lbl, mx, mn, passed in sym_results:
                    if passed:
                        _success(f"[{label}] {lbl}: OK  (max={mx:.2e}, mean={mn:.2e})")
                    else:
                        _info(
                            f"[{label}] {lbl}: VIOLATION  (max={mx:.2e}, mean={mn:.2e})"
                        )
                        all_ok = False

    return filename, all_ok


def _section_verify_dataset(s: _WizardStateTensor) -> _WizardStateTensor:
    """
    Optional post-review step: verify train (and test) dataset files —
    checks columns, tensor shape and symmetry; offers to generate
    tensor_property if it is missing.
    """
    from tensorpotential.cli.wizard import (
        _ask_confirm,
        _section,
        _info,
        _success,
    )

    _section("Dataset verification")
    if not _ask_confirm(
        "Verify dataset now? (checks columns, tensor shape & symmetry)",
        default=False,
    ):
        return s

    try:
        import pandas as pd
    except ImportError:
        _info("pandas not available — skipping.")
        return s

    all_ok = True

    # ── Train ────────────────────────────────────────────────────────────────
    _info(f"Loading train: {s.train_filename} …")
    try:
        df_train = pd.read_pickle(s.train_filename)
        _success(f"Loaded {len(df_train)} structures (train)")
        new_train, ok_train = _verify_one_df(
            df_train, s.train_filename, s, label="train"
        )
        s.train_filename = new_train
        if not ok_train:
            all_ok = False
    except Exception as exc:
        _info(f"Could not load train dataset: {exc}")
        all_ok = False

    # ── Test (separate file only) ────────────────────────────────────────────
    if s.use_separate_test and s.test_filename:
        _info(f"Loading test: {s.test_filename} …")
        try:
            df_test = pd.read_pickle(s.test_filename)
            _success(f"Loaded {len(df_test)} structures (test)")
            new_test, ok_test = _verify_one_df(
                df_test, s.test_filename, s, label="test"
            )
            s.test_filename = new_test
            if not ok_test:
                all_ok = False
        except Exception as exc:
            _info(f"Could not load test dataset: {exc}")
            all_ok = False

    if all_ok:
        _success("All dataset checks passed!")
    else:
        _info("Issues found — please review before running gracemaker.")

    return s


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------


def _apply_state_tensor(s: _WizardStateTensor) -> str:
    # Test data line
    if s.use_separate_test:
        test_data_line = f"test_filename: {s.test_filename}"
        save_dataset = "False"
    else:
        test_data_line = f"test_size: {s.test_size}"
        save_dataset = "True"

    # Loss delta strings
    tensor_delta_str = f", delta: {s.tensor_delta}" if s.loss_type == "huber" else ""
    ef_delta_str = f", delta: {s.energy_delta}" if s.loss_type == "huber" else ""

    # Energy / forces loss lines
    energy_line = ""
    forces_line = ""
    if s.compute_energy:
        energy_line = (
            f"    energy: {{ type: {s.loss_type}, weight: {s.energy_weight}{ef_delta_str}}},\n"
        )
    if s.compute_forces:
        forces_line = (
            f"    forces: {{ type: {s.loss_type}, weight: {s.forces_weight}{ef_delta_str}}},\n"
        )

    return f"""\
cutoff: {s.cutoff}
seed: 1

data:
  filename: {s.train_filename}
  {test_data_line}
  reference_energy: 0.0
  save_dataset: {save_dataset}
  extra_components: {{
    ReferenceTensorDataBuilder: {{tensor_rank: {s.tensor_rank}, per_structure: {s.per_structure}}},
  }}


potential:
  preset: {s.preset}
  kwargs: {{
           compute_energy: {s.compute_energy},

           tensor_components: {s.tensor_components},
           }}
  scale: False
  shift: False

fit:
  compute_function: ComputeBatchEFTensor
  train_function: ComputeBatchEFTensor
  compute_function_config: {{tensor_components: {s.tensor_components},
                            per_structure: {s.per_structure},
                            compute_energy: {s.compute_energy},
                            compute_forces: {s.compute_forces}
  }}
  loss: {{
{energy_line}{forces_line}    extra_components: {{
          WeightedTensorLoss: {{weight: {s.tensor_weight}, type: {s.loss_type}{tensor_delta_str}}},
    }}
  }}

  target_total_updates: {s.target_total_updates}

  optimizer: Adam
  opt_params: {{learning_rate: 0.008, use_ema: True, ema_momentum: 0.99,
               weight_decay: null, clipnorm: 1.0}}
  scheduler: cosine_decay
  scheduler_params: {{minimal_learning_rate: 0.0006}}

  batch_size: {s.batch_size}
  test_batch_size: {s.batch_size * 5}

  jit_compile: True

  # train_max_n_buckets: auto
  # test_max_n_buckets: auto
  auto_bucket_max_padding: 0.3

  checkpoint_freq: 2
  progressbar: True
  train_shuffle: True
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_gen_tensor_wizard():
    """Entry point called from the main wizard when user picks 'Generic Tensors (L<=2)'."""
    from tensorpotential.cli.wizard import (
        _section_dataset,
        _HAS_QUESTIONARY,
        _HAS_RICH,
        _print_header,
    )

    _print_header()

    s = _WizardStateTensor()

    sections = [
        ("Dataset", _section_dataset),
        ("Tensor type", _section_tensor_components),
        ("Per-structure", _section_tensor_per_structure),
        ("Model", _section_tensor_model),
        ("Energy & forces", _section_tensor_energy),
        ("Loss", _section_tensor_loss),
        ("Batch & training", _section_tensor_batch),
    ]

    try:
        # Run all sections once
        for _, fn in sections:
            s = fn(s)

        # Review + re-edit loop
        while True:
            _show_review_tensor(s)

            if _HAS_QUESTIONARY:
                import questionary
                from tensorpotential.cli.wizard import _QUESTIONARY_STYLE

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

        # Optional dataset verification (may update s.train_filename)
        s = _section_verify_dataset(s)

        # Write output
        output = _apply_state_tensor(s)
        with open("input.yaml", "w") as f:
            f.write(output)

    except KeyboardInterrupt:
        if _HAS_RICH:
            from tensorpotential.cli.wizard import _rich_console
            _rich_console.print("\n[bold red]Interrupted.[/bold red] Exiting.")
        else:
            print("\nInterrupted. Exiting.")
        sys.exit(0)

    if _HAS_RICH:
        from tensorpotential.cli.wizard import _rich_console
        from rich.panel import Panel
        from rich import box as rich_box

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
