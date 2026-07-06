#!/usr/bin/env python
import os
import json
import argparse


# ---------------------------------------------------------------------------
# small rendering helpers
# ---------------------------------------------------------------------------
def _cap(val, ascii_mode):
    """Capability tri-state glyph: True / False / None(unknown)."""
    if ascii_mode:
        return {True: "Y", False: "N", None: "?"}[val]
    return {True: "✓", False: "✗", None: "?"}[val]


def _state(cached, applicable, ascii_mode):
    """Local-artifact 3-state glyph: cached / available-not-downloaded / n-a."""
    if not applicable:
        return "-" if ascii_mode else "–"
    if ascii_mode:
        return "Y" if cached else "."
    return "✓" if cached else "·"


def _model_paths(name, model_data):
    """(saved_model dir, checkpoint dir) for a model, honoring explicit overrides."""
    from tensorpotential.calculator.foundation_models import (
        FOUNDATION_CACHE_DIR,
        FOUNDATION_CHECKPOINTS_CACHE_DIR,
        MODEL_PATH_KEY,
        CHECKPOINT_PATH_KEY,
    )

    model_path = model_data.get(MODEL_PATH_KEY) or os.path.join(
        FOUNDATION_CACHE_DIR, name
    )
    checkpoint_path = model_data.get(CHECKPOINT_PATH_KEY) or os.path.join(
        FOUNDATION_CHECKPOINTS_CACHE_DIR, name
    )
    return model_path, checkpoint_path


def _row_cells(name, model_data, ascii_mode):
    """Compute the table cells for one model (capabilities + local cache state)."""
    from tensorpotential.calculator.foundation_models import (
        PRECISION_KEY,
        UQ_KEY,
        PARALLEL_KEY,
        KOKKOS_URL_KEY,
        CHECKPOINT_URL_KEY,
        CHECKPOINT_PATH_KEY,
    )

    model_path, checkpoint_path = _model_paths(name, model_data)

    prec = model_data.get(PRECISION_KEY, "?")
    uq = model_data.get(UQ_KEY)  # None if undeclared
    par = model_data.get(PARALLEL_KEY)

    model_cached = os.path.isdir(model_path)
    ckpt_applicable = (
        CHECKPOINT_URL_KEY in model_data or CHECKPOINT_PATH_KEY in model_data
    )
    ckpt_cached = os.path.isdir(checkpoint_path)
    kokkos_applicable = KOKKOS_URL_KEY in model_data
    kokkos_cached = os.path.isfile(os.path.join(model_path, "kokkos.npz"))
    uq_applicable = bool(uq)
    uq_art_cached = os.path.isfile(os.path.join(checkpoint_path, "gmm_artifacts.npz"))

    return [
        name,
        prec,
        _cap(uq, ascii_mode),
        _cap(par, ascii_mode),
        _state(model_cached, True, ascii_mode),
        _state(ckpt_cached, ckpt_applicable, ascii_mode),
        _state(kokkos_cached, kokkos_applicable, ascii_mode),
        _state(uq_art_cached, uq_applicable, ascii_mode),
    ]


# capability columns (PREC/UQ/PAR) then on-disk-state columns (MODEL/CKPT/KK/UQ-ART)
_HEADERS = ["NAME", "PREC", "UQ", "PAR", "MODEL", "CKPT", "KK", "UQ-ART"]


def _print_table(core_rows, extra_rows, ascii_mode):
    all_rows = core_rows + extra_rows
    widths = [len(h) for h in _HEADERS]
    for row in all_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    total_w = sum(widths) + 2 * (len(widths) - 1)

    def fmt(cells):
        # NAME left-justified, the rest centered for compact scanning
        out = [str(cells[0]).ljust(widths[0])]
        out += [str(c).center(widths[i + 1]) for i, c in enumerate(cells[1:])]
        return "  ".join(out)

    print(fmt(_HEADERS))
    print("-" * total_w)
    for row in core_rows:
        print(fmt(row))
    if extra_rows:
        print("-" * total_w)
        print("Extra models (local registry / experimental):")
        print("-" * total_w)
        for row in extra_rows:
            print(fmt(row))

    _print_legend(ascii_mode)


def _print_legend(ascii_mode):
    ok = "Y" if ascii_mode else "✓"
    no = "N" if ascii_mode else "✗"
    avail = "." if ascii_mode else "·"
    na = "-" if ascii_mode else "–"
    print()
    print("Legend:")
    print("  NAME    model name (bare = fp32 default; the '-fp64' suffix is the full-precision variant)")
    print("  Capabilities:")
    print("    PREC    weight precision: fp32 | fp64 | mixed | ? (undeclared)")
    print(f"    UQ      uncertainty quantification baked in (per-atom gamma): {ok} yes  {no} no  ? unknown")
    print(f"    PAR     multi-rank / domain-decomposition (compute_energy + partitioned fwd/bwd): {ok} yes  {no} no  ? unknown")
    print(f"  On disk ({ok} downloaded  {avail} available, not downloaded  {na} not offered):")
    print("    MODEL   SavedModel directory")
    print("    CKPT    checkpoint directory")
    print("    KK      separate LAMMPS-Kokkos export (kokkos.npz)")
    print("    UQ-ART  UQ artifact (gmm_artifacts.npz)")
    print("  (use `grace_models info <name>` for full paths + the downloaded artifact's introspected flags)")
    sep = " | " if ascii_mode else "  ·  "
    print()
    print(
        "Download missing parts:  grace_models download <name> [--kokkos]"
        + sep
        + "grace_models checkpoint <name>"
    )


def _print_verbose(name, model_data):
    """Today's multi-line block, augmented with the declared capability line."""
    from tensorpotential.calculator.foundation_models import (
        LICENSE_KEY,
        CHECKPOINT_URL_KEY,
        KOKKOS_URL_KEY,
        DESCRIPTION_KEY,
        PRECISION_KEY,
        UQ_KEY,
        PARALLEL_KEY,
    )

    model_path, checkpoint_path = _model_paths(name, model_data)
    msg = f"{name}"
    if DESCRIPTION_KEY in model_data:
        msg += "\n\tDESCRIPTION: " + model_data[DESCRIPTION_KEY].replace("\n", "\n\t")
    msg += (
        f"\n\tCAPABILITIES: precision={model_data.get(PRECISION_KEY, '?')}"
        f"  UQ={model_data.get(UQ_KEY, False)}"
        f"  parallel={model_data.get(PARALLEL_KEY, '?')}"
        f"  kokkos={'separate npz' if KOKKOS_URL_KEY in model_data else 'none'}"
    )
    if os.path.isdir(model_path):
        msg += f"\n\tPATH: {model_path}"
    else:
        msg += "\n\tPATH: [NOT DOWNLOADED]"
    if CHECKPOINT_URL_KEY in model_data:
        if os.path.isdir(checkpoint_path):
            msg += f"\n\tCHECKPOINT: {checkpoint_path}"
        else:
            msg += "\n\tCHECKPOINT: AVAILABLE, BUT NOT DOWNLOADED"
    msg += f"\n\tLICENSE: {model_data.get(LICENSE_KEY, 'not provided')}"
    print(msg)
    print("=" * 80)


def list_models(args):
    """List available models from MODELS_METADATA.

    Compact capability table by default; ``--verbose`` for the full per-model block.
    """
    from tensorpotential.calculator.foundation_models import (
        MODELS_METADATA,
        CORE_MODELS_NAME_LIST,
    )

    core_set = set(CORE_MODELS_NAME_LIST)
    ascii_mode = getattr(args, "ascii", False)
    if getattr(args, "verbose", False):
        print("Available models:")
        print("=" * 80)
        for name, model_data in MODELS_METADATA.items():
            _print_verbose(name, model_data)
        return

    core_rows, extra_rows = [], []
    for name, data in MODELS_METADATA.items():
        (core_rows if name in core_set else extra_rows).append(
            _row_cells(name, data, ascii_mode)
        )
    print(f"Available models ({len(core_rows) + len(extra_rows)}):")
    print()
    _print_table(core_rows, extra_rows, ascii_mode)


def _read_param_dtype(checkpoint_path):
    """Cheap line-scan of a checkpoint model.yaml for the stamped ``param_dtype``."""
    yaml_path = os.path.join(checkpoint_path, "model.yaml")
    if not os.path.isfile(yaml_path):
        return None
    with open(yaml_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("param_dtype:"):
                return stripped.split(":", 1)[1].strip()
    return None


def _fmt_cutoff(cutoff, cutoff_matrix):
    """Human-readable cutoff: flag bond-dependent (per-pair matrix) vs uniform."""
    if cutoff_matrix:
        flat = [v for row in cutoff_matrix for v in row]
        lo, hi = min(flat), max(flat)
        if lo != hi:
            return f"bond-dependent, per-pair {lo}–{hi} Å (global/neighbor {cutoff} Å)"
        return f"{lo} Å (per-pair matrix, uniform; global/neighbor {cutoff} Å)"
    return f"{cutoff} Å"


def _download_hints(name, model_data, model_path, checkpoint_path):
    """(label, command) for each part of ``name`` that is available but not cached."""
    from tensorpotential.calculator.foundation_models import (
        KOKKOS_URL_KEY,
        CHECKPOINT_URL_KEY,
        CHECKPOINT_PATH_KEY,
    )

    hints = []
    if not os.path.isdir(model_path):
        hints.append(("model", f"grace_models download {name}"))
    if (
        CHECKPOINT_URL_KEY in model_data or CHECKPOINT_PATH_KEY in model_data
    ) and not os.path.isdir(checkpoint_path):
        hints.append(("checkpoint", f"grace_models checkpoint {name}"))
    if KOKKOS_URL_KEY in model_data and not os.path.isfile(
        os.path.join(model_path, "kokkos.npz")
    ):
        hints.append(("kokkos.npz", f"grace_models download {name} --kokkos"))
    return hints


def info_model(args):
    """Show declared flags for a model and, if downloaded, the artifact's true flags.

    Introspection is TF-free: it reads the SavedModel ``metadata.json``
    (``has_uq`` / ``parallel_communication`` / ``cutoff``) and the checkpoint
    ``model.yaml`` (``param_dtype``), then flags any drift from the declared flags.
    """
    from tensorpotential.calculator.foundation_models import (
        MODELS_METADATA,
        MODELS_ALIASES_DICT,
        DESCRIPTION_KEY,
        LICENSE_KEY,
        PRECISION_KEY,
        UQ_KEY,
        PARALLEL_KEY,
        KOKKOS_URL_KEY,
    )

    name = MODELS_ALIASES_DICT.get(args.model_name) or args.model_name
    if name not in MODELS_METADATA:
        print(f"Model {args.model_name} not found in available models.")
        return
    md = MODELS_METADATA[name]
    model_path, checkpoint_path = _model_paths(name, md)

    print("=" * 80)
    print(name + (f"  (alias of {args.model_name})" if name != args.model_name else ""))
    print("=" * 80)
    if DESCRIPTION_KEY in md:
        print("DESCRIPTION: " + md[DESCRIPTION_KEY].replace("\n", " "))
    print(f"LICENSE:     {md.get(LICENSE_KEY, 'not provided')}")
    print()
    print("DECLARED:")
    print(f"  precision : {md.get(PRECISION_KEY, '?')}")
    print(f"  UQ        : {md.get(UQ_KEY, False)}")
    print(f"  parallel  : {md.get(PARALLEL_KEY, '?')}")
    print(f"  kokkos    : {'separate npz' if KOKKOS_URL_KEY in md else 'none'}")
    print(f"  model dir : {model_path}  [{'cached' if os.path.isdir(model_path) else 'not downloaded'}]")
    print(f"  ckpt dir  : {checkpoint_path}  [{'cached' if os.path.isdir(checkpoint_path) else 'not downloaded'}]")

    # ---- introspect the downloaded artifact (TF-free) ----
    meta_path = os.path.join(model_path, "metadata.json")
    has_uq = parallel_comm = cutoff = cutoff_matrix = n_elem = None
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            has_uq = meta.get("has_uq")
            parallel_comm = meta.get("parallel_communication")
            cutoff = meta.get("cutoff")
            cutoff_matrix = meta.get("cutoff_matrix")
            n_elem = len(meta.get("chemical_symbols", []) or []) or None
        except (json.JSONDecodeError, OSError):
            pass
    param_dtype = _read_param_dtype(checkpoint_path)
    gmm_present = os.path.isfile(os.path.join(checkpoint_path, "gmm_artifacts.npz"))
    kokkos_present = os.path.isfile(os.path.join(model_path, "kokkos.npz"))

    if os.path.isdir(model_path) or os.path.isdir(checkpoint_path):
        print()
        print("INTROSPECTED (downloaded artifact):")
        if os.path.isfile(meta_path):
            print(f"  has_uq            : {has_uq}")
            comm_channels = sorted(parallel_comm.keys()) if parallel_comm else []
            print(f"  parallel_comm     : {comm_channels if comm_channels else 'none (local model)'}")
            if cutoff is not None:
                print(f"  cutoff            : {_fmt_cutoff(cutoff, cutoff_matrix)}")
            if n_elem is not None:
                print(f"  n_elements        : {n_elem}")
        else:
            print("  metadata.json     : absent (legacy export — no introspectable flags)")
        print(f"  param_dtype       : {param_dtype or 'not stamped (full precision / legacy)'}")
        print(f"  gmm_artifacts.npz : {'present' if gmm_present else 'absent'}")
        print(f"  kokkos.npz        : {'present' if kokkos_present else 'absent'}")

        # ---- drift checks: declared vs actual ----
        warnings = []
        if has_uq is not None and bool(has_uq) != bool(md.get(UQ_KEY, False)):
            warnings.append(
                f"UQ: declared {bool(md.get(UQ_KEY, False))} but SavedModel has_uq={has_uq}"
            )
        if param_dtype:
            actual_prec = {"float32": "fp32", "float64": "fp64"}.get(param_dtype)
            declared_prec = md.get(PRECISION_KEY)
            if actual_prec and declared_prec not in (actual_prec, "mixed", None):
                warnings.append(
                    f"precision: declared {declared_prec} but checkpoint param_dtype={param_dtype}"
                )
        if warnings:
            print()
            print("  ⚠ DRIFT (declared vs downloaded):")
            for w in warnings:
                print(f"    - {w}")

    # ---- how to fetch whatever is not cached yet ----
    hints = _download_hints(name, md, model_path, checkpoint_path)
    print()
    if hints:
        print("TO DOWNLOAD MISSING PARTS:")
        for what, cmd in hints:
            print(f"  {what:<11}: {cmd}")
    else:
        print("All available parts are downloaded.")


def download_model(args):
    """Download the specified model using the get_or_download_model function.

    With ``--kokkos`` also fetch the LAMMPS-Kokkos export ``kokkos.npz`` into the
    model dir (distributed separately; on-demand — most ASE users never need it).
    """

    from tensorpotential.calculator.foundation_models import (
        get_or_download_model,
        get_or_download_kokkos,
        MODELS_METADATA,
        MODELS_NAME_LIST,
    )

    want_kokkos = getattr(args, "kokkos", False)
    names = MODELS_NAME_LIST if args.model_name == "all" else [args.model_name]
    if args.model_name != "all" and args.model_name not in MODELS_METADATA:
        print(f"Model {args.model_name} not found in available models.")
        return

    for model_name in names:
        print(f"Downloading model: {model_name}")
        get_or_download_model(model_name)
        if want_kokkos:
            print(f"Downloading kokkos.npz for {model_name}")
            get_or_download_kokkos(model_name)
        print(f"Model {model_name} downloaded successfully.")


def download_checkpoint(args):
    """Download the specified model's checkpoint"""

    from tensorpotential.calculator.foundation_models import (
        get_or_download_checkpoint,
        MODELS_METADATA,
        MODELS_NAME_LIST,
    )

    model_name = args.model_name

    if model_name in MODELS_METADATA:
        print(f"Downloading checkpoint for {model_name}")
        get_or_download_checkpoint(model_name)
        print(f"Checkpoint for {model_name} downloaded successfully.")
    elif model_name in ["all"]:
        print(f"Downloading ALL models: {MODELS_NAME_LIST}")
        for model_name in MODELS_NAME_LIST:
            print(f"Downloading model: {model_name}")
            get_or_download_checkpoint(model_name)
            print(f"Model {model_name} downloaded successfully.")
    else:
        print(f"Model {model_name} not found in available models.")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="grace_models", description="Download foundation GRACE models"
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Sub-command: list
    parser_list = subparsers.add_parser("list", help="List available models")
    parser_list.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="full per-model block (description, paths, license) instead of the table",
    )
    parser_list.add_argument(
        "--ascii",
        action="store_true",
        help="use ASCII glyphs (Y/N/./-) instead of unicode in the table",
    )
    parser_list.set_defaults(func=list_models)

    # Sub-command: info
    parser_info = subparsers.add_parser(
        "info", help="Show declared + downloaded-artifact capability flags for a model"
    )
    parser_info.add_argument(
        "model_name", type=str, help="Name (or alias) of the model to inspect"
    )
    parser_info.set_defaults(func=info_model)

    # Sub-command: download
    parser_download = subparsers.add_parser("download", help="Download a model")
    parser_download.add_argument(
        "model_name", type=str, help="Name of the model to download"
    )
    parser_download.add_argument(
        "--kokkos",
        action="store_true",
        help="also download the LAMMPS-Kokkos export (kokkos.npz) into the model dir",
    )
    parser_download.set_defaults(func=download_model)

    # Sub-command: checkpoint
    parser_checkpoint = subparsers.add_parser(
        "checkpoint", help="Download a checkpoint"
    )
    parser_checkpoint.add_argument(
        "model_name", type=str, help="Name of the model to download checkpoint for"
    )
    parser_checkpoint.set_defaults(func=download_checkpoint)

    return parser


def main(args=None):
    parser = build_parser()
    args_parse = parser.parse_args(args)

    if args_parse.command:
        args_parse.func(args_parse)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
