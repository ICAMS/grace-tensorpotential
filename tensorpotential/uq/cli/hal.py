"""grace_uq hal — run one uncertainty-biased (HAL) walker."""

from __future__ import annotations

import argparse
import json
import os
import warnings

# kept literal so --help does not import the HAL package; pinned by a test to
# tensorpotential.uq.hal.model_dir.SEARCH_ORDER_HELP
_MODEL_DIR_HELP = (
    "model.yaml; checkpoint: checkpoint.index, then checkpoints/checkpoint.best_test_loss.index, "
    "checkpoints/checkpoint.index; CALM artifact: gmm_artifacts.npz, UQ/gmm_artifacts.npz"
)

# literal for the same reason; pinned to tensorpotential.uq.hal.walkers.WALKERS_PER_GPU
_WALKERS_PER_GPU = 4


def _float_or_none(text: str):
    return None if text.lower() == "none" else float(text)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="grace_uq hal",
        description="Run one HAL walker (uncertainty-biased MD on the CALM grade), or "
                    "several independent ones in subprocesses with --repeats K (K walkers per setting).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("model and structure")
    g.add_argument("--model-dir", default=None,
                   help="directory holding the model; searched for (in this order) "
                        + _MODEL_DIR_HELP + ". A hit in two layouts is an error. "
                        "--model-yaml / --checkpoint / --artifact-path override what is found")
    g.add_argument("--model-yaml")
    g.add_argument("--checkpoint")
    g.add_argument("--artifact-path", help="training-set CALM artifact (frozen tower; live start by default)")
    g.add_argument("--live-start-artifact-path", default=None,
                   help="start the live tower from an earlier walker's hal_live_artifacts.npz")
    g.add_argument("--structure", help="starting structure (any ASE-readable file)")
    g.add_argument("--structure-index", type=int, default=0,
                   help="frame of --structure to start from; with --repeats, repeat r starts "
                        "from frame (STRUCTURE_INDEX + r) mod n_frames")
    g.add_argument("--param-dtype", default=None, help="override model dtype (float32/float64)")
    d = p.add_argument_group("dynamics")
    d.add_argument("--ensemble", choices=["nvt", "nvt_ramp", "npt", "npt_stress", "cluster"],
                   nargs="+", default=["nvt"],
                   help="periodic: nvt/nvt_ramp/npt/npt_stress; non-periodic: cluster. Several "
                        "values need --repeats (K walkers per combination)")
    d.add_argument("-T", "--temperature", type=float, nargs="+",
                   help="K; several values need --repeats (K walkers per combination)")
    d.add_argument("--n-steps", type=int)
    d.add_argument("--n-collect", type=int, default=None)
    d.add_argument("--timestep", default="auto", help="fs, or 'auto' (2 fs; 0.5 fs with H/He/Li/Be)")
    d.add_argument("--friction", type=float, default=0.01, help="Langevin friction, 1/fs")
    d.add_argument("--tdamp", type=float, default=100.0, help="MTK thermostat damping, fs")
    d.add_argument("--pdamp", type=float, default=1000.0, help="MTK barostat damping, fs")
    d.add_argument("--nvt-ramp", type=float, nargs=2, default=[1.0, 1.4], metavar=("START", "END"),
                   help="V/V0 at start and end (nvt_ramp)")
    d.add_argument("--stress-tau", type=float, default=0.02, help="uncertainty stress bias tau_s (npt_stress)")
    d.add_argument("--bulk-modulus", default="auto", help="eV/A^3, or 'auto' (5-point E-V fit)")
    d.add_argument("--volume-guard", type=float, nargs=2, default=[0.75, 1.30], metavar=("LO", "HI"),
                   help="stop when V/V0 leaves this band (npt, npt_stress)")
    d.add_argument("--cluster-wall-margin", type=float, default=2.0,
                   help="containment wall radius = initial max distance from COM + this (cluster), A")
    d.add_argument("--cluster-wall-k", type=float, default=1.0, help="wall stiffness (cluster), eV/A^2")
    d.add_argument("--remove-update-work", action=argparse.BooleanOptionalAction, default=True,
                   help="take the bias energy a live CALM update injects, "
                        "tau_r * (sum gamma_before - sum gamma_after), back out of the kinetic "
                        "energy (Langevin ensembles; MTK npt keeps its own momenta)")
    d.add_argument("--update-work-floor", type=float, default=0.5,
                   help="the removal never cools the walker below this fraction of -T")
    d.add_argument("--seed", type=int, default=42)
    b = p.add_argument_group("bias and guard")
    b.add_argument("--tau", type=float, nargs="+", default=[0.15],
                   help="force bias strength; several values run a tau scan: one walker per "
                        "tau (K with --repeats K) into OUTPUT_DIR/tau_<value>/hal_w<r>/, at most "
                        "--walkers-per-gpu at a time, table in OUTPUT_DIR/walkers.csv")
    b.add_argument("--tau-ema", type=int, default=10, help="EMA window (steps) and bias delay")
    b.add_argument("--zbl", choices=["auto", "off"], default="auto")
    b.add_argument("--zbl-switch", type=float, nargs=2, default=[0.55, 0.75], metavar=("IN", "OUT"),
                   help="ZBL fully on below IN*(rcov_i+rcov_j), off above OUT*(rcov_i+rcov_j)")
    b.add_argument("--pair-radii", default=None,
                   help='path to a JSON file {"A-B": {"zbl": [r_in, r_out], "reject": r}} with per-pair overrides (Angstrom)')
    c = p.add_argument_group("collection and CALM update")
    c.add_argument("--equil-steps", type=int, default=0,
                   help="no collection before this step (the bias still waits --tau-ema steps "
                        "for its EMAs to fill)")
    c.add_argument("--trigger-gamma", type=float, default=1.5)
    c.add_argument("--trigger-gap", type=int, default=10)
    c.add_argument("--trigger-min-atoms", type=int, default=1,
                   help="collect only when at least this many atoms have gamma > --trigger-gamma")
    c.add_argument("--reject-gamma", type=float, default=10.0)
    c.add_argument("--reject-bond-factor", type=float, default=0.65,
                   help="reject if a pair is closer than this * (rcov_i + rcov_j)")
    c.add_argument("--update", choices=["alpha_search", "none"], default="alpha_search")
    c.add_argument("--update-target", type=float, default=1.25)
    c.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="alpha grid of the additive live update C += alpha*C_frame, tried in "
                        "order (default: 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0)")
    c.add_argument("--update-inflate", action=argparse.BooleanOptionalAction, default=True,
                   help="after an alpha search that ends at alpha=1 above the target, inflate "
                        "the covariances of the clusters in the frame (bounded; paper protocol)")
    c.add_argument("--inflate-max-rounds", type=int, default=3,
                   help="at most this many inflation rounds per collection")
    c.add_argument("--inflate-max-factor", type=float, default=4.0,
                   help="covariance scale factor cap per inflation round")
    o = p.add_argument_group("output")
    o.add_argument("--output-dir", default=None,
                   help="walker directory (default hal_w0); with --repeats the parent of "
                        "OUTPUT_DIR/hal_w<i> (default hal_walkers)%(default).0s")
    o.add_argument("--walker-id", type=int, default=0)
    o.add_argument("--save-every", type=int, default=10, help="save live artifact every N collections")
    o.add_argument("--history-every", type=int, default=1000, help="flush history every N steps")
    o.add_argument("--overwrite", action="store_true", help="replace HAL files in --output-dir")
    o.add_argument("--gpus", default=None, help="value for CUDA_VISIBLE_DEVICES")
    o.add_argument("--fragment-mult", type=float, default=1.25,
                   help="n_fragments (recorded, never filtered): atoms bonded when closer than "
                        "this x (r_cov,i + r_cov,j)")
    o.add_argument("--feature-gamma-min", type=_float_or_none, default=1.0, metavar="X|none",
                   help="keep the features of the atoms with gamma_pristine >= X of every "
                        "collected frame in hal_features.npz (input of grace_uq select "
                        "--hal-dirs); none: off")
    o.add_argument("--progress-every", type=int, default=100,
                   help="print a status line every N steps (0: quiet)")
    o.add_argument("--plot-only", default=None, metavar="DIR",
                   help="only rebuild hal_diagnostics.png from DIR and exit")
    w = p.add_argument_group("several walkers")
    w.add_argument("--repeats", type=int, default=None, metavar="K",
                   help="run K independent walkers for every setting, each in its own "
                        "subprocess. A setting is one combination of --ensemble, -T and "
                        "--tau (one combination when each has one value), so the run has K x "
                        "combinations walkers. Repeat r writes OUTPUT_DIR/hal_w<r> (its output "
                        "to walker.log there), or OUTPUT_DIR/<label>/hal_w<r> when settings "
                        "vary (label: the varying ones, e.g. npt_tau_0.5), with --seed + r and "
                        "frame (--structure-index + r) mod n_frames of --structure; walker ids "
                        "count over the whole run. All other options go to every walker. Prints "
                        "a line per finished walker, then a table, also written to "
                        "OUTPUT_DIR/walkers.csv; exit code 1 if any walker failed%(default).0s")
    w.add_argument("--walkers-per-gpu", type=int, default=_WALKERS_PER_GPU, metavar="M",
                   help="with --repeats: run at most M walkers at once on each visible GPU "
                        "(--gpus / CUDA_VISIBLE_DEVICES), the rest wait for a free slot; "
                        "without a GPU, M walkers at once on the CPU. Each walker sees one GPU "
                        "and allocates its memory on demand (TF_FORCE_GPU_ALLOW_GROWTH)")
    pl = p.add_argument_group("pool collected frames (instead of running a walker)")
    pl.add_argument("--pool", nargs="+", default=None, metavar="DIR",
                    help="pool hal_collected.extxyz of these walker dirs into --pool-output and exit")
    pl.add_argument("--pool-output", default=None, metavar="FILE",
                    help="pool file: .pkl.gz (input for grace_uq select --dataset) or .extxyz")
    pl.add_argument("--gamma-pristine-min", type=float, default=None,
                    help="keep only frames with max gamma_pristine >= this")
    return p


_REQUIRED = ("structure", "temperature", "n_steps")
_MODEL = ("model_yaml", "checkpoint", "artifact_path")


def _pool_main(args, parser) -> int:
    if not args.pool_output:
        parser.error("--pool needs --pool-output FILE")
    from tensorpotential.uq.cli._common import is_extxyz_path, save_dataset_any
    from tensorpotential.uq.hal import find_walker_dirs, pool_frames

    try:
        dirs = find_walker_dirs(args.pool)
    except ValueError as e:
        parser.error(f"--pool: {e}")
    df = pool_frames(dirs, gamma_pristine_min=args.gamma_pristine_min)
    if is_extxyz_path(args.pool_output):
        warnings.warn(
            "grace_uq select / predict read extxyz only with reference energies, which HAL "
            "frames do not have; write .pkl.gz for grace_uq select --dataset",
            stacklevel=2,
        )
        cand = [c for c in df.columns if c.startswith("cand_")]
        if cand:
            warnings.warn(
                "extxyz cannot hold the candidate-atom features (hal_features.npz): "
                "dropped; write .pkl.gz to select from the pool without a prediction",
                stacklevel=2,
            )
            df = df.drop(columns=cand)
    save_dataset_any(df, args.pool_output)
    n_dirs = df["hal_dir"].nunique() if len(df) else 0
    print(f"grace_uq hal --pool: pooled {len(df)} frames from {n_dirs} dirs -> {args.pool_output}")
    return 0


# walker settings: a column only when it differs between walkers, else one line above the table
_SETTING_COLUMNS = ("repeat", "ensemble", "temperature_K", "tau", "start_frame", "device")


# in walkers.csv, not on screen
_CSV_ONLY = ("output_dir", "exit_code", "n_rejected", "max_gamma_live",
             "max_gamma_pristine_collected", "n_fragmented")


def _print_walkers_table(df) -> None:
    """The walkers table on screen: settings equal for all walkers in one line above it."""
    same = [c for c in _SETTING_COLUMNS if c in df.columns and df[c].nunique(dropna=False) == 1]
    if same:
        print("same for all walkers: " + ", ".join(f"{c} {df[c].iloc[0]:g}"
                                                  if isinstance(df[c].iloc[0], float)
                                                  else f"{c} {df[c].iloc[0]}" for c in same))
    cols = [c for c in df.columns if c not in (*_CSV_ONLY, *same)
            and not c.startswith("rejected_")]
    print(df[cols].to_string(index=False))


def _walkers_main(args, kw) -> int:
    from ase.io import read

    from tensorpotential.uq.hal.walkers import _PER_WALKER, WALKERS_CSV, run_walkers

    for k in _PER_WALKER:  # run_walkers sets these per walker
        kw.pop(k)
    # the grid axes as given: several values run K walkers per combination
    kw.update(ensemble=args.ensemble, temperature_K=args.temperature)
    frames = read(args.structure, index=":")
    df = run_walkers(frames, args.repeats,
                     walkers_per_gpu=args.walkers_per_gpu,
                     output_dir=args.output_dir, seed=args.seed, overwrite=args.overwrite,
                     structure_offset=args.structure_index, tau=args.tau, **kw)
    _print_walkers_table(df)
    failed = df.loc[df["exit_code"] != 0, "walker"].tolist()
    print(f"grace_uq hal: {len(df) - len(failed)} of {len(df)} walkers finished, "
          f"{int(df['collected'].fillna(0).sum())} frames collected"
          + (f"; FAILED: {failed}" if failed else "")
          + f" -> {os.path.join(args.output_dir, WALKERS_CSV)}")
    return 1 if failed else 0


def hal_main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.repeats is None and len(args.tau) > 1 and not (args.plot_only or args.pool):
        args.repeats = 1  # a tau scan: one walker per tau, like any other grid
    if args.repeats is None:
        if args.walkers_per_gpu != _WALKERS_PER_GPU:
            parser.error("--walkers-per-gpu applies with --repeats")
        several = [f for f, v in (("--ensemble", args.ensemble), ("-T", args.temperature))
                   if v is not None and len(v) > 1]
        if several:
            parser.error(f"several {' / '.join(several)} values need --repeats "
                         "(K walkers per combination)")
    elif args.plot_only or args.pool:
        parser.error("--repeats runs walkers; it does not combine with --plot-only or --pool")
    elif args.repeats < 1 or args.walkers_per_gpu < 1:
        parser.error("--repeats and --walkers-per-gpu must be >= 1")
    elif args.walker_id != 0:
        parser.error("--walker-id applies to a single walker (with --repeats, ids count "
                     "over the run)")
    if args.output_dir is None:
        args.output_dir = "hal_w0" if args.repeats is None else "hal_walkers"
    if args.plot_only:
        from tensorpotential.uq.hal.plot import replot_dir

        print(replot_dir(args.plot_only))
        return 0
    if args.pool:
        return _pool_main(args, parser)
    stray = [f for f, v in (("--pool-output", args.pool_output),
                            ("--gamma-pristine-min", args.gamma_pristine_min)) if v is not None]
    if stray:
        parser.error(f"{', '.join(stray)} only apply with --pool")
    missing = [n for n in _REQUIRED if getattr(args, n) is None]
    if args.model_dir is None:
        missing += [n for n in _MODEL if getattr(args, n) is None]
    if missing:
        parser.error("missing: " + ", ".join("--" + n.replace("_", "-") for n in missing)
                     + (" (or pass --model-dir)" if args.model_dir is None else ""))
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    from ase.io import read

    from tensorpotential.uq.hal import run_hal

    overrides = None
    if args.pair_radii:
        with open(args.pair_radii) as fh:
            overrides = json.load(fh)
    calc_kwargs = {"param_dtype": args.param_dtype} if args.param_dtype else {}
    kw = dict(
        model_dir=args.model_dir,
        model_yaml=args.model_yaml, checkpoint=args.checkpoint, uq_artifact=args.artifact_path,
        live_start_artifact=args.live_start_artifact_path, ensemble=args.ensemble[0],
        temperature_K=args.temperature[0], n_steps=args.n_steps, n_collect=args.n_collect,
        timestep_fs=None if args.timestep == "auto" else float(args.timestep),
        tau_ema=args.tau_ema, equil_steps=args.equil_steps,
        trigger_gamma=args.trigger_gamma, trigger_gap=args.trigger_gap,
        trigger_min_atoms=args.trigger_min_atoms, update=args.update,
        update_target=args.update_target,
        alphas=tuple(args.alphas) if args.alphas else None,
        update_inflate=args.update_inflate, inflate_max_rounds=args.inflate_max_rounds,
        inflate_max_factor=args.inflate_max_factor,
        reject_gamma=args.reject_gamma, reject_bond_factor=args.reject_bond_factor,
        zbl=args.zbl, zbl_switch=tuple(args.zbl_switch), pair_overrides=overrides,
        nvt_ramp=tuple(args.nvt_ramp), stress_tau=args.stress_tau,
        bulk_modulus=args.bulk_modulus if args.bulk_modulus == "auto" else float(args.bulk_modulus),
        volume_guard=tuple(args.volume_guard), cluster_wall_margin=args.cluster_wall_margin,
        cluster_wall_k=args.cluster_wall_k, friction=args.friction,
        tdamp_fs=args.tdamp, pdamp_fs=args.pdamp, remove_update_work=args.remove_update_work,
        update_work_floor=args.update_work_floor, seed=args.seed, walker_id=args.walker_id,
        output_dir=args.output_dir, overwrite=args.overwrite, keep_frames=False,
        save_every=args.save_every, history_every=args.history_every, calc_kwargs=calc_kwargs,
        progress_every=args.progress_every, fragment_mult=args.fragment_mult,
        feature_gamma_min=args.feature_gamma_min,
    )
    if args.repeats is not None:
        return _walkers_main(args, kw)
    atoms = read(args.structure, index=args.structure_index)
    res = run_hal(atoms, tau=args.tau[0], **kw)
    s = res.summary
    print(f"grace_uq hal: {s['n_steps']} steps, {s['n_collected']} collected, "
          f"rejected {s['rejected']}, stop: {s['stop_reason']}, {s['steps_per_s']:.2f} steps/s "
          f"-> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(hal_main())
