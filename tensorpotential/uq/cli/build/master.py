"""Master orchestration (run_master) + the build_main CLI entry point."""

from __future__ import annotations

import logging

import argparse
import glob
import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

from tensorpotential.cli.data import read_saved_dataset_stats
from tensorpotential.data.shards import discover_shards, is_sharded_dataset
from tensorpotential.instructions.base import load_instructions
from tensorpotential.metadata_utils import resolve_param_dtype
from tensorpotential.tpmodel import (
    ComputeEnergy,
    extract_cutoff_and_elements,
    extract_cutoff_dict,
)
from tensorpotential.uq import (
    GMMUQArtifactBuilder,
    GMMUQModel,
    batched_feature_iterator,
    tf_dataset_feature_iterator,
)
from tensorpotential.uq import constants as uq_constants
from tensorpotential.uq.artifact_builder import _default_eff_counts
from tensorpotential.uq.factories import (
    load_uq_model,
    make_basis_rp_spec,
)
from tensorpotential.uq.cli._common import (
    SENTINEL_CENTROID_VALUE as _SENTINEL_CENTROID_VALUE,
    alignment_perm_from_hist,
    apply_master_thread_caps,
    format_elem_label,
    resolve_threads_per_worker,
    spawn_and_monitor,
)
from tensorpotential.uq.cli.build.cli_args import (
    _BUILD_EPILOG,
    _WeightedTrainDataAction,
)
from tensorpotential.uq.cli.build.clusters import (
    _compact_step2_artifacts,
    _merge_step2_clusters,
    _pick_centroids_for_element,
)
from tensorpotential.uq.cli.build.data_resolve import (
    DEFAULT_TRAIN_DATA,
    _load_filter_fn,
    _prefetch,
    _resolve_train_data_from_input_yaml,
    _resolve_weighted_train_data,
)
from tensorpotential.uq.cli.build.paths import (
    _iter_worker_files,
    _worker_dir,
    _worker_output_paths,
)
from tensorpotential.uq.cli.build.streaming import (
    StreamingEstimate,
    stream_atoms,
)
from tensorpotential.uq.cli.build.thresholds import (
    _compute_dual_thresholds,
    _print_covariance_diagnostics,
    _save_elbow_plot,
    _save_elbow_report,
    select_optimal_clusters,
)
from tensorpotential.uq.cli.build.workers import (
    export_savedmodel,
    run_worker_step1,
    run_worker_step2,
    run_worker_step3,
)

log = logging.getLogger("grace_uq")


def _log_train_data_options(args):
    """Always log the resolved training data (files + weights), even non-verbose.

    Emitted once by the master at INFO so every build records exactly which
    pkl.gz shards (and per-file weights, when ``--train-data-weighted`` was
    used) the GMM artifact was fitted on. ``--frac`` / ``--filter-fn`` are
    noted too since they change which structures actually enter the fit.
    """
    paths = args.train_data or []
    weight_map = getattr(args, "file_weight_map", None) or {}
    weighted = bool(weight_map)
    log.info(
        "Training data: %d file(s)%s",
        len(paths),
        " (weighted)" if weighted else "",
    )
    for p in paths:
        suffix = f"  [weight {weight_map.get(p, 1.0):g}]" if weighted else ""
        log.info("  - %s%s", p, suffix)
    if getattr(args, "frac", None) is not None:
        log.info("  (subsampling --frac=%g of each shard)", args.frac)
    if getattr(args, "filter_fn", None):
        log.info("  (--filter-fn=%s applied before subsampling)", args.filter_fn)


def run_master(args):
    """Master logic: Orchestrates the parallel steps and merges results."""

    _log_train_data_options(args)

    # Resolve threads-per-worker: "auto" → physical_cores / n_workers, then
    # cap the master's own thread pools so it doesn't fight workers during
    # the merge/GMM phases (and when multiple grace_uq instances share a host).
    was_auto = args.threads_per_worker == "auto"
    args.threads_per_worker = resolve_threads_per_worker(
        args.threads_per_worker, args.n_workers
    )
    if was_auto:
        print(
            f"  [Master] threads-per-worker: auto → {args.threads_per_worker}"
            f" ({os.cpu_count() or 1} cores / {args.n_workers} workers)"
        )
    apply_master_thread_caps(args.threads_per_worker)

    # Preserve the original candidate list before any elbow mutation.
    n_clusters_candidates = list(args.n_clusters)

    # Intermediate files go next to the output file.
    work_dir = os.path.dirname(args.artifact_path) or "."

    def _wpath(name):
        return os.path.join(work_dir, name)

    # Build element index → symbol mapping (used in reports, step 2, and step 3 save).
    _, master_symbols, master_indices = extract_cutoff_and_elements(
        load_instructions(args.model_yaml)
    )
    element_names = {int(idx): sym for sym, idx in zip(master_symbols, master_indices)}

    if args.restart:
        print("  [Master] Restarting: cleaning up old checkpoint files...")
        for f in glob.glob(_wpath(".step*")):
            try:
                os.remove(f)
            except Exception:
                pass

    def cleanup_worker_files(step_idx):
        for i in range(args.n_workers):
            for path in _worker_output_paths(
                work_dir, step_idx, i, n_clusters_candidates
            ):
                if os.path.exists(path):
                    os.remove(path)

    def spawn_workers(step_idx, extra_args=None):
        workers_to_run = []
        for i in range(args.n_workers):
            expected = _worker_output_paths(
                work_dir, step_idx, i, args.n_clusters
            )
            already_done = all(os.path.exists(p) for p in expected)
            if not args.restart and already_done:
                continue
            workers_to_run.append(i)

        if not workers_to_run:
            print(f"  [Master] All workers for Step {step_idx} already finished.")
            return

        print(f"  [Master] Spawning {len(workers_to_run)} workers...")
        cmds = []
        for i in workers_to_run:
            cmd = [
                sys.executable,
                "-m",
                "tensorpotential.uq.cli.build",
                "--step",
                str(step_idx),
                "--worker-id",
                str(i),
                "--n-workers",
                str(args.n_workers),
                "--model-yaml",
                args.model_yaml,
                "--checkpoint",
                args.checkpoint,
                "--n-clusters",
                *[str(k) for k in args.n_clusters],
                "--max-neighbours-per-batch",
                str(args.max_neighbours_per_batch),
                "--seed",
                str(args.seed),
                "--gpus",
                args.gpus,
                "--artifact-path",
                args.artifact_path,
            ]
            if args.frac:
                cmd += ["--frac", str(args.frac)]
            if args.verbose:
                cmd += ["--verbose"]
            cmd += ["--rp-dim", str(args.rp_dim), "--rp-seed", str(args.rp_seed)]
            # The uqv6 normalize/density feature options are fixed (shared constants in
            # uq.constants), so the worker reproduces the master's feature without any
            # per-build flag to forward.
            if args.filter_fn:
                cmd += ["--filter-fn", args.filter_fn]
            if extra_args:
                cmd += extra_args
            if args.train_data_weighted:
                # Re-emit one --train-data-weighted block per group so the
                # worker's _WeightedTrainDataAction parses an identical
                # weight map. Avoid mixing with bare --train-data to keep the
                # mutual-exclusion check happy on the worker side.
                for weight, files in args.train_data_weighted:
                    cmd += ["--train-data-weighted", str(weight)] + list(files)
            else:
                cmd += ["--train-data"] + args.train_data
            cmds.append((i, cmd))

        # On worker failure we do NOT exit immediately: that would close pipe
        # FDs, causing surviving workers to die with BrokenPipeError before
        # they can write their checkpoint files. spawn_and_monitor waits for
        # all workers, then we raise once successful checkpoints are on disk.
        failed_workers = spawn_and_monitor(
            cmds,
            gpus=args.gpus.split(","),
            threads_per_worker=args.threads_per_worker,
            verbose=args.verbose,
            label="Worker",
        )
        if failed_workers:
            raise RuntimeError(
                f"Step {step_idx}: worker(s) {failed_workers} failed. "
                "Checkpoints from successful workers have been saved and will be "
                "reused on restart."
            )

    total_start = time.time()

    # Step 1: Clustering
    t0 = time.time()
    print("\n>>> STEP 1/3: GLOBAL CLUSTERING")
    print("=" * 40)
    step1_path = _wpath(".step1_final.npz")
    n_feats = None
    if not args.restart and os.path.exists(step1_path):
        print(f"  [Master] {step1_path} already exists. Skipping Step 1.")
        base_s1 = GMMUQArtifactBuilder.load(step1_path)
        if base_s1:
            n_feats = next(iter(base_s1.values()))[uq_constants.CENTROIDS].shape[1]
            # Infer the optimal k that was previously selected.
            args.n_clusters = [next(iter(base_s1.values()))[uq_constants.CENTROIDS].shape[0]]
    else:
        spawn_workers(1)
        print(f"  [Master] Clustering cores took {time.time() - t0:.1f}s")

        print("  [Master] Aggregating centroids and running global KMeans per k...")

        # Load per-k worker centroids and WCSS.
        all_centroids_by_k = {}
        # {elem: [wcss_k0, wcss_k1, ...]} — true data-level WCSS summed over workers.
        elem_wcss_total = {}  # {elem: {k: total_wcss}}
        elem_wcss_count = {}  # {elem: {k: total_count}}
        for k in n_clusters_candidates:
            centroids_for_k = {}
            for i in range(args.n_workers):
                path = _wpath(f".step1_w{i}_k{k}.npz")
                if not os.path.exists(path):
                    raise RuntimeError(
                        f"Worker file {path!r} missing after Step 1 completed"
                    )
                with np.load(path) as data:
                    for e in data["elements"]:
                        e_int = int(e)
                        # Meta-KMeans weight is the influence-weighted count
                        # when available, so the global centroid placement
                        # respects per-source weights. Falls back to raw
                        # counts for older worker files.
                        eff_key = f"{uq_constants.EFFECTIVE_COUNT}_{e}"
                        meta_w = (
                            data[eff_key]
                            if eff_key in data.files
                            else data[f"{uq_constants.COUNTS}_{e}"].astype(np.float64)
                        )
                        centroids_for_k.setdefault(e_int, []).append(
                            (
                                data[f"{uq_constants.CENTROIDS}_{e}"],
                                meta_w,
                            )
                        )
                        w = float(data.get(f"wcss_{e}", 0.0))
                        n = int(data.get(f"wcss_n_{e}", 0))
                        elem_wcss_total.setdefault(e_int, defaultdict(float))
                        elem_wcss_count.setdefault(e_int, defaultdict(int))
                        elem_wcss_total[e_int][k] += w
                        elem_wcss_count[e_int][k] += n
            all_centroids_by_k[k] = centroids_for_k

        # Per-element per-sample WCSS from actual data (not meta-centroids).
        per_elem_inertias = {}
        for e in sorted(elem_wcss_total.keys()):
            per_elem_inertias[e] = []
            for k in n_clusters_candidates:
                total_n = elem_wcss_count[e].get(k, 0)
                if total_n > 0:
                    per_elem_inertias[e].append(elem_wcss_total[e][k] / total_n)
                else:
                    per_elem_inertias[e].append(0.0)

        # Run global KMeans for each k (needed for step 2 centroids, not for elbow).
        global_kms_by_k = {}
        for k in n_clusters_candidates:
            elem_kms = {}
            for e, parts in all_centroids_by_k[k].items():
                c_all = np.vstack([p[0] for p in parts])
                w_all = np.concatenate([p[1] for p in parts])
                n_feats = c_all.shape[1]
                if len(c_all) < k:
                    log.warning(
                        "[Master] Element %s has only %d meta-centroids for k=%d; "
                        "skipping global KMeans for this k",
                        format_elem_label(e, element_names), len(c_all), k,
                    )
                    continue
                km = KMeans(
                    n_clusters=k, n_init=10, random_state=args.seed
                ).fit(c_all, sample_weight=w_all)
                elem_kms[e] = km
            global_kms_by_k[k] = elem_kms

        # Per-element elbow; global k = max over elements.
        per_elem_optimal_k = {}
        print("  [Master] Per-element elbow results:")
        for e in sorted(per_elem_inertias.keys()):
            opt_k = select_optimal_clusters(n_clusters_candidates, per_elem_inertias[e])
            per_elem_optimal_k[e] = opt_k
            inertia_str = "  ".join(
                f"k={k}:{v:.3e}"
                for k, v in zip(n_clusters_candidates, per_elem_inertias[e])
            )
            print(f"    {format_elem_label(e, element_names)}: optimal_k={opt_k}  [{inertia_str}]")

        if len(n_clusters_candidates) > 1:
            optimal_k = max(per_elem_optimal_k.values())
            print(
                f"  [Master] Global optimal k = {optimal_k}"
                f" (max over per-element optima: {per_elem_optimal_k})"
            )
            _save_elbow_report(
                n_clusters_candidates,
                per_elem_inertias,
                per_elem_optimal_k,
                optimal_k,
                args.artifact_path,
                element_names,
            )
            _save_elbow_plot(
                n_clusters_candidates,
                per_elem_inertias,
                per_elem_optimal_k,
                optimal_k,
                args.artifact_path,
                element_names,
            )
        else:
            optimal_k = n_clusters_candidates[0]
        args.n_clusters = [optimal_k]

        # Build step1_artifacts with per-element effective cluster count.
        # Mathematically strict floor: every real cluster must have n_atoms >= D+1
        # so that the centered sample covariance (rank <= min(D, n-1)) is
        # non-degenerate. KMeans imbalance can violate the average-based cap
        # n_atoms_e // D, so we validate post-fit using actual cluster counts
        # weighted by meta-centroid weights.
        # Elements with fewer than D+1 atoms collapse to K=1 with a warning;
        # Tikhonov regularization in step 2 stabilizes the under-determined
        # covariance for that case.
        min_atoms = (
            args.min_atoms_per_cluster
            if args.min_atoms_per_cluster > 0
            else n_feats + 1
        )
        step1_artifacts = {}
        k_min = min(n_clusters_candidates)
        print(f"  [Master] Building centroids (D={n_feats}, "
              f"min {min_atoms} atoms/cluster):")
        for e in sorted(element_names.keys()):
            lbl = format_elem_label(e, element_names)
            n_atoms_e = int(elem_wcss_count.get(e, {}).get(k_min, 0))

            if n_atoms_e == 0:
                log.warning(
                    "[Master] Step 1: element %s has no training atoms. "
                    "It will be excluded from UQ.",
                    lbl,
                )
                continue

            real_centroids, best_k = _pick_centroids_for_element(
                e=e,
                n_atoms_e=n_atoms_e,
                n_clusters_candidates=n_clusters_candidates,
                elem_optimal_k=per_elem_optimal_k.get(e, optimal_k),
                global_kms_by_k=global_kms_by_k,
                all_centroids_by_k=all_centroids_by_k,
                min_atoms=min_atoms,
                lbl=lbl,
                regularization=args.regularization,
                weighted=bool(getattr(args, "file_weight_map", None)),
            )

            n_pad = optimal_k - best_k
            if n_pad > 0:
                pad_centroids = np.full(
                    (n_pad, n_feats), _SENTINEL_CENTROID_VALUE
                )
                all_centroids = np.vstack([real_centroids, pad_centroids])
                print(
                    f"    {lbl:>10s}: {best_k}/{optimal_k} effective clusters "
                    f"({n_atoms_e} atoms, ~{n_atoms_e // max(best_k, 1)}/cluster); "
                    f"{n_pad} sentinel slots"
                )
            else:
                all_centroids = real_centroids
                print(
                    f"    {lbl:>10s}: {optimal_k}/{optimal_k} clusters "
                    f"({n_atoms_e} atoms, ~{n_atoms_e // optimal_k}/cluster)"
                )

            step1_artifacts[e] = {
                uq_constants.CENTROIDS: all_centroids,
                uq_constants.INV_COV: np.zeros((optimal_k, n_feats, n_feats)),
                uq_constants.COUNTS: np.zeros(optimal_k),
                uq_constants.SCATTER: None,
            }

        del all_centroids_by_k, global_kms_by_k

        GMMUQArtifactBuilder(n_clusters=optimal_k, feature_dim=n_feats).save(
            step1_path, step1_artifacts, store_fp32=False
        )

        cleanup_worker_files(1)

    # Step 2: Covariance
    t0 = time.time()
    print("\n>>> STEP 2/3: COVARIANCE ACCUMULATION")
    print("=" * 40)
    step2_path = _wpath(".step2_final.npz")
    if not args.restart and os.path.exists(step2_path):
        print(f"  [Master] {step2_path} already exists. Skipping Step 2.")
    else:
        spawn_workers(2, ["--step1-artifacts", step1_path])
        print(f"  [Master] Covariance accumulation took {time.time() - t0:.1f}s")

        print("  [Master] Reducing scatter matrices...")
        base = GMMUQArtifactBuilder.load(step1_path)
        for i in range(args.n_workers):
            path = _wpath(f".step2_w{i}.npz")
            if not os.path.exists(path):
                raise RuntimeError(
                    f"Worker file {path!r} missing after Step 2 completed"
                )
            w_data = GMMUQArtifactBuilder.load(path)
            for e, d in w_data.items():
                d_eff = _default_eff_counts(
                    d[uq_constants.COUNTS], d.get(uq_constants.EFFECTIVE_COUNT)
                )
                if base[e][uq_constants.SCATTER] is None:
                    base[e][uq_constants.SCATTER] = d[uq_constants.SCATTER]
                    base[e][uq_constants.COUNTS] = d[uq_constants.COUNTS].copy()
                    base[e][uq_constants.EFFECTIVE_COUNT] = d_eff.copy()
                else:
                    base[e][uq_constants.SCATTER] += d[uq_constants.SCATTER]
                    base[e][uq_constants.COUNTS] += d[uq_constants.COUNTS]
                    base[e][uq_constants.EFFECTIVE_COUNT] += d_eff

        # Step-2 merge pass (parallel-axis): tighten any real cluster that
        # now has < D+1 atoms after step-2 reassignment. Step-1's merge
        # cannot see this — its counts come from meta-centroid weights, not
        # actual step-2 atom assignments. This pass guarantees every real
        # cluster in the final artifact has count >= min_atoms (or is the
        # only real cluster, which Tikhonov handles).
        step2_min_atoms = (
            args.min_atoms_per_cluster
            if args.min_atoms_per_cluster > 0
            else n_feats + 1
        )
        for e in sorted(base.keys()):
            lbl = format_elem_label(e, element_names)
            base[e][uq_constants.EFFECTIVE_COUNT] = _default_eff_counts(
                base[e][uq_constants.COUNTS],
                base[e].get(uq_constants.EFFECTIVE_COUNT),
            )
            (
                base[e][uq_constants.CENTROIDS],
                base[e][uq_constants.SCATTER],
                base[e][uq_constants.COUNTS],
                base[e][uq_constants.EFFECTIVE_COUNT],
                _,
            ) = _merge_step2_clusters(
                base[e][uq_constants.CENTROIDS],
                base[e][uq_constants.SCATTER],
                base[e][uq_constants.COUNTS],
                min_atoms=step2_min_atoms,
                lbl=lbl,
                effective_counts=base[e][uq_constants.EFFECTIVE_COUNT],
            )

        # Compact: shrink K_max to max(K_eff) across elements so we don't
        # carry around all-sentinel rows everywhere. Updates args.n_clusters
        # so step 3 (and the threshold-calibration master loop) sees the
        # new K_max.
        K_max_old = args.n_clusters[0]
        K_max_new = _compact_step2_artifacts(base)
        if K_max_new < K_max_old:
            print(
                f"  [Master] Step 2: compacted K_max {K_max_old} → {K_max_new} "
                f"(max real-cluster count across elements after merging)."
            )
            args.n_clusters = [K_max_new]

        builder = GMMUQArtifactBuilder.from_artifacts(
            base,
            n_clusters=args.n_clusters[0],
            feature_dim=n_feats,
            regularization=args.regularization,
        )
        for e, d in base.items():
            builder.set_scatter(
                e,
                d[uq_constants.SCATTER],
                d[uq_constants.COUNTS],
                effective_counts=d.get(uq_constants.EFFECTIVE_COUNT),
            )
        step2_artifacts = builder.finalize(element_names=element_names)
        builder.save(step2_path, step2_artifacts, store_fp32=False)

        # Print covariance diagnostics summary
        _print_covariance_diagnostics(step2_artifacts, element_names)

        cleanup_worker_files(2)

    # Step 3: Thresholds
    t0 = time.time()
    print("\n>>> STEP 3/3: THRESHOLD CALIBRATION")
    print("=" * 40)
    if not args.restart and os.path.exists(args.artifact_path):
        print(f"  [Master] {args.artifact_path} already exists. Skipping Step 3.")
    else:
        spawn_workers(3, ["--step2-artifacts", step2_path])
        print(f"  [Master] Threshold calculation took {time.time() - t0:.1f}s")

        print("  [Master] Computing per-cluster p99 thresholds...")
        total_hists = {}
        total_eff_hists = {}
        for i in range(args.n_workers):
            path = _wpath(f".step3_w{i}.npz")
            if not os.path.exists(path):
                raise RuntimeError(
                    f"Worker file {path!r} missing after Step 3 completed"
                )
            with np.load(path) as d:
                for e in d["elements"]:
                    if e not in total_hists:
                        total_hists[e] = d[f"hist_{e}"]
                        total_eff_hists[e] = (
                            d[f"eff_hist_{e}"]
                            if f"eff_hist_{e}" in d.files
                            else d[f"hist_{e}"].astype(np.float64)
                        )
                    else:
                        total_hists[e] += d[f"hist_{e}"]
                        eff_inc = (
                            d[f"eff_hist_{e}"]
                            if f"eff_hist_{e}" in d.files
                            else d[f"hist_{e}"].astype(np.float64)
                        )
                        total_eff_hists[e] += eff_inc

        bins = np.linspace(0, 100, 250)
        n_clusters = args.n_clusters[0]
        # Minimum atoms required for a stable threshold estimate. A cluster with
        # fewer calibration atoms produces an unreliable threshold (its sigma
        # spread is a small-sample artifact), so it backfills from a sibling.
        # Use D as the floor — same heuristic that step 1 uses to cap K_eff.
        min_atoms_for_threshold = max(int(args.min_per_element) if hasattr(args, "min_per_element") and args.min_per_element else 50, 50)
        try:
            # Try to derive D from the step2 artifacts so the floor is
            # adaptive to the model's feature dimension.
            with np.load(step2_path) as _data:
                _elems_in_artifact = _data["elements"]
                _first_e = int(_elems_in_artifact[0])
                _D_from_artifact = _data[f"{uq_constants.CENTROIDS}_{_first_e}"].shape[1]
                min_atoms_for_threshold = max(min_atoms_for_threshold, _D_from_artifact)
        except Exception:
            pass

        # Compute both thresholds in parallel using the robust median+k*MAD
        # estimator (see thresholds._ROBUST_K). The raw histogram drives the
        # reliability gate (sample-count confidence); the effective histogram
        # drives the threshold inference uses when weights are non-trivial. Both
        # matrices share the SAME backfill positions so they stay consistent.
        thresh_matrix, eff_thresh_matrix, underpop_warnings, elementwise_fallback_warnings = (
            _compute_dual_thresholds(
                total_hists, total_eff_hists, bins, n_clusters,
                min_atoms_for_threshold, n_elements=len(master_symbols),
            )
        )

        if underpop_warnings:
            print(
                f"  [Master] {len(underpop_warnings)} cluster(s) had < "
                f"{min_atoms_for_threshold} calibration atoms — threshold replaced "
                "with element-wide max:"
            )
            for e, k, n in underpop_warnings:
                lbl = format_elem_label(e, element_names)
                print(f"    {lbl} cluster {k}: only {n} atoms")
        if elementwise_fallback_warnings:
            print(
                "  [Master] WARNING: some elements had no reliable cluster; "
                "their thresholds use the small-sample robust-threshold max as a best effort:"
            )
            for e, n_pop, fill in elementwise_fallback_warnings:
                lbl = format_elem_label(e, element_names)
                print(f"    {lbl}: {n_pop} populated cluster(s), fill={fill:.3g}")
        present_elems = sorted(total_hists.keys())
        print(
            f"Results: {len(present_elems)} species x {n_clusters} clusters "
            f"calibrated ({len(master_symbols)} rows incl. defaults)."
        )
        for e in present_elems:
            print(f"  Element {e}: thresholds(raw) = {thresh_matrix[e]}")
            if not np.allclose(thresh_matrix[e], eff_thresh_matrix[e]):
                print(f"  Element {e}: thresholds(eff) = {eff_thresh_matrix[e]}")

        # Final Save
        param_dtype = resolve_param_dtype(args.model_yaml)
        gmm_uq = GMMUQModel(step2_path, param_dtype=param_dtype)
        symbols = master_symbols

        # Defensive: realign step-3's hist/thresh rows to the post-finalize
        # cluster order. No-op when the upstream pipeline is healthy; recovers
        # the correct alignment for cached step-2 files from older builds.
        realigned_elems: list[int] = []
        unaligned_elems: list[int] = []
        for e in present_elems:
            if e not in gmm_uq._artifacts:
                continue
            counts = gmm_uq._artifacts[e][uq_constants.COUNTS]
            inv, is_identity = alignment_perm_from_hist(total_hists[e], counts)
            if inv is None:
                unaligned_elems.append(int(e))
                continue
            if is_identity:
                continue
            realigned_elems.append(int(e))
            total_hists[e] = total_hists[e][inv]
            if e in total_eff_hists:
                total_eff_hists[e] = total_eff_hists[e][inv]
            thresh_matrix[e] = thresh_matrix[e][inv]
            eff_thresh_matrix[e] = eff_thresh_matrix[e][inv]
        if realigned_elems:
            log.info(
                "Realigned histograms/thresholds for %d element(s) to match "
                "the step2 centroid/count order (upstream step-3 ordering bug; "
                "the saved artifact is consistent).",
                len(realigned_elems),
            )
        if unaligned_elems:
            preview = unaligned_elems[:10]
            ellipsis = "..." if len(unaligned_elems) > 10 else ""
            log.warning(
                "%d element(s) have histograms that cannot be matched to "
                "step2 counts within tolerance — saving as-is. Inference may "
                "use incorrect thresholds for these elements: %s%s",
                len(unaligned_elems),
                preview,
                ellipsis,
            )

        # Save per-element-per-cluster histograms (raw + weighted) for later
        # recalibration / inspection. Both threshold matrices are persisted;
        # inference prefers the weighted one when present.
        hist_kwargs = {f"hist_{e}": h for e, h in total_hists.items()}
        hist_kwargs.update(
            {f"eff_hist_{e}": h for e, h in total_eff_hists.items()}
        )

        # Stamp the self-describing basis-RP spec: regenerate the byte-identical
        # projection R (same model graph + rp-seed the workers used) and store it
        # verbatim so eval / SavedModel export reproduce the exact feature with no
        # env var. Stored float32 (the workers' model dtype).
        # feature_transform defaults to None; normalize/density are the fixed
        # production uqv6 options (the build no longer exposes per-feature toggles),
        # matching what the workers applied during feature extraction.
        rp_kwargs = make_basis_rp_spec(
            args.model_yaml,
            rp_dim=args.rp_dim,
            rp_seed=args.rp_seed,
            store_fp32=True,
            normalize=uq_constants.UQ_DEFAULT_NORMALIZE,
            add_density_channel=uq_constants.UQ_DEFAULT_DENSITY,
            density_scale=uq_constants.UQ_DEFAULT_DENSITY_SCALE,
        )

        gmm_uq.save(
            args.artifact_path,
            interp_thresholds=thresh_matrix,
            eff_interp_thresholds=eff_thresh_matrix,
            element_map=np.array(symbols),
            hist_bins=bins,
            **hist_kwargs,
            **rp_kwargs,
        )

        cleanup_worker_files(3)

    # Export SavedModel
    if not args.no_export:
        t0 = time.time()
        print("\n>>> EXPORT: SAVEDMODEL WITH UQ SIGNATURE")
        print("=" * 40)
        if not args.restart and os.path.exists(
            os.path.join(args.export_path, "saved_model.pb")
        ):
            print(f"  [Master] {args.export_path} already exists. Skipping export.")
        else:
            export_savedmodel(args)
            print(
                f"  [Master] SavedModel exported to {args.export_path}"
                f" in {time.time() - t0:.1f}s"
            )

    print(
        f"\nSUCCESS: Pipeline finished in {time.time() - total_start:.1f}s."
        f" Artifacts: {args.artifact_path}"
    )
    if not args.no_export:
        print(f"  SavedModel: {args.export_path}")


def build_main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="grace_uq build",
        description="Parallel GRACE-UQ artifact generation",
        epilog=_BUILD_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--step", type=int, choices=[1, 2, 3])
    parser.add_argument("--worker-id", type=int)
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4).",
    )
    parser.add_argument(
        "--model-yaml",
        default="model.yaml",
        help="Path to the trained model YAML (default: model.yaml in cwd).",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/checkpoint.best_test_loss.index",
        help="Path to the model checkpoint .index file "
        "(default: checkpoints/checkpoint.best_test_loss.index).",
    )
    parser.add_argument(
        "--train-data",
        nargs="+",
        default=list(DEFAULT_TRAIN_DATA),
        help="One or more paths to training data files (default: training_set.pkl.gz). "
        "If the default file is absent, the master falls back to "
        "../../input.yaml::data.filename. Required for UQ-artifact generation; "
        "not needed for export-only mode.",
    )
    parser.add_argument(
        "--n-clusters",
        "--n_clusters",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Number(s) of GMM clusters. Pass a single value to use it directly, "
        "or multiple values (e.g. --n-clusters 1 2 4 8 16) to run the elbow method "
        "and automatically select the optimal k (default: 1 2 4 8 16).",
    )
    parser.add_argument(
        "--min-atoms-per-cluster",
        type=int,
        default=-1,
        help="Minimum atoms per non-sentinel cluster. KMeans is shrunk to the "
        "largest K satisfying this. -1 (default) resolves to D+1, the strict "
        "floor for non-degenerate sample covariance in D-dim feature space.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-6,
        help="Tikhonov epsilon (eps*I) added to each cluster covariance before "
        "inversion. Default: 1e-6. Bump to 1e-5 for noisy features, lower "
        "(e.g. 1e-8) only if features are clean and you want minimal bias.",
    )
    parser.add_argument(
        "--max-neighbours-per-batch",
        type=int,
        default=15000,
        help="Target number of neighbour pairs per batch for streaming (pickle) input. "
        "Controls GPU memory usage. Ignored for sharded TF datasets.",
    )
    parser.add_argument(
        "--train-data-weighted",
        nargs="+",
        action=_WeightedTrainDataAction,
        default=None,
        metavar="WEIGHT_OR_PATH",
        help="Repeatable. First token is the per-source weight (float), "
        "remaining tokens are training shard paths. Atoms from those shards "
        "get that weight in the UQ artifact accumulation; centroid placement "
        "and per-cluster covariance are pulled toward heavy sources. "
        "Example: --train-data-weighted 20.0 /OMAT/shard_*.pkl.gz "
        "--train-data-weighted 1.0 /SMAX/shard_*.pkl.gz. "
        "Mutually exclusive with explicit --train-data.",
    )
    parser.add_argument(
        "--filter-fn",
        type=str,
        default=None,
        metavar="MODULE.PATH:FUNCTION_NAME",
        help="Optional dotted import path to a callable "
        "``filter_atoms(ase.Atoms) -> bool`` that drops structures during "
        "ingest (False = drop, True = keep). The module must be importable "
        "on the worker's PYTHONPATH. Applied per shard, before --frac.",
    )
    parser.add_argument("--frac", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--artifact-path",
        "--output-path",  # backward-compatible alias
        default="gmm_artifacts.npz",
        dest="artifact_path",
        help="Path for the .npz UQ artifact file (default: gmm_artifacts.npz). "
        "If a directory is given, gmm_artifacts.npz is appended.",
    )
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU indices")
    parser.add_argument(
        "--threads-per-worker",
        default="auto",
        help="CPU threads per worker (OMP, MKL, TF). "
        "'auto' (default) = physical_cores / n_workers. "
        "Set an integer to override, or 0 to disable limiting.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--rp-dim",
        "--rp_dim",
        dest="rp_dim",
        type=int,
        default=128,
        help="basis-RP UQ feature dimension: the random projection maps the "
        "concatenated invariant (l=0) energy-path basis to this many dims "
        "(default: 128).",
    )
    parser.add_argument(
        "--rp-seed",
        "--rp_seed",
        dest="rp_seed",
        type=int,
        default=42,
        help="Seed for the basis-RP projection matrix R (default: 42). The same "
        "(rp-dim, rp-seed) regenerates a byte-identical R across all workers.",
    )
    parser.add_argument(
        "--restart", action="store_true", help="Delete all checkpoint npz files"
    )
    parser.add_argument(
        "--export-path",
        default=None,
        help="Path for the exported SavedModel directory. "
        "Default: saved_model/ next to the artifact file.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip SavedModel export after artifact generation.",
    )
    parser.add_argument("--step1-artifacts", help="Internal")
    parser.add_argument("--step2-artifacts", help="Internal")
    args = parser.parse_args(argv)

    # Snapshot whether the user is using the default --train-data BEFORE we
    # turn paths absolute (the abspath mangling makes a literal-equality
    # check unreliable later on).
    train_data_is_default = args.train_data == DEFAULT_TRAIN_DATA

    # --train-data and --train-data-weighted are mutually exclusive once the
    # user has actually supplied a value (i.e. --train-data is not still its
    # default). Argparse's add_mutually_exclusive_group doesn't compose with
    # the custom Action, so we check manually.
    if args.train_data_weighted and not train_data_is_default:
        parser.error(
            "--train-data and --train-data-weighted are mutually exclusive. "
            "Use one or the other."
        )
    # Flatten weighted groups into a flat path list + weight map. When no
    # weighted groups were given the map is empty (back-compat: weight 1.0
    # everywhere). The flat list replaces args.train_data so downstream code
    # (worker spawn, _iter_worker_files, stream_atoms) stays uniform.
    flat_paths, file_weight_map = _resolve_weighted_train_data(args)
    if args.train_data_weighted:
        args.train_data = flat_paths
    args.file_weight_map = file_weight_map

    # Eagerly validate the filter spec so a bad import fails before workers
    # are spawned. The callable itself isn't cached on args (it's re-imported
    # per worker subprocess); we just verify it's resolvable here.
    if args.filter_fn:
        _load_filter_fn(args.filter_fn)  # raises if bad

    # Resolve all file paths to absolute so workers are CWD-independent.
    args.model_yaml = os.path.abspath(args.model_yaml)
    args.checkpoint = os.path.abspath(args.checkpoint)
    args.artifact_path = os.path.abspath(args.artifact_path)
    if os.path.isdir(args.artifact_path):
        args.artifact_path = os.path.join(args.artifact_path, "gmm_artifacts.npz")
    args.export_path = os.path.abspath(
        args.export_path
        or os.path.join(os.path.dirname(args.artifact_path) or ".", "saved_model")
    )
    if args.train_data:
        args.train_data = [os.path.abspath(p) for p in args.train_data]
    if args.step1_artifacts:
        args.step1_artifacts = os.path.abspath(args.step1_artifacts)
    if args.step2_artifacts:
        args.step2_artifacts = os.path.abspath(args.step2_artifacts)

    # If the user is relying on the default --train-data and it doesn't
    # exist on disk, look one level up for a gracemaker input.yaml and
    # resolve data.filename from it (refusing if any train/test split is
    # configured there).
    if (
        args.worker_id is None
        and train_data_is_default
        and not any(os.path.exists(p) for p in args.train_data)
    ):
        resolved = _resolve_train_data_from_input_yaml()
        if resolved is not None:
            print(
                f"  [Master] {DEFAULT_TRAIN_DATA[0]} not found; "
                f"resolved --train-data={resolved} from ../../input.yaml",
                flush=True,
            )
            args.train_data = [os.path.abspath(p) for p in resolved]

    # Early validation of --train-data.
    #
    # Master mode auto-falls-through to export-only when none of the
    # --train-data paths exist *and* an artifact is already on disk. This
    # makes `grace_uq build` runnable inside seed/{N}/ folders that have
    # already produced a gmm_artifacts.npz from a previous run, without
    # requiring the user to know about the export-only flag pattern.
    #
    # Exception: --restart explicitly asks for a full rebuild, so silently
    # falling through to export-only here would defeat the user's intent.
    train_data_exists = bool(args.train_data) and any(
        os.path.exists(p) for p in args.train_data
    )
    if args.worker_id is not None and not train_data_exists:
        parser.error(
            f"--train-data files do not exist: {args.train_data}"
        )
    if (
        args.worker_id is None
        and not train_data_exists
        and (args.restart or not os.path.exists(args.artifact_path))
    ):
        hint = (
            "--restart was given but no training data could be located. "
            if args.restart
            else ""
        )
        parser.error(
            f"{hint}--train-data files do not exist ({args.train_data}) and "
            f"--artifact-path is not usable for export-only mode "
            f"({args.artifact_path}). Provide a training set explicitly."
        )
    if args.worker_id is None and not train_data_exists:
        # Export-only fall-through: clear train_data so the master skips
        # the 3-step pipeline and goes straight to SavedModel export.
        args.train_data = None

    if args.worker_id is not None:
        # Check if this worker's output already exists — skip entirely.
        # This avoids loading the TF model when resuming a partial run.
        wdir = _worker_dir(args)
        if args.step in (1, 2, 3):
            expected = _worker_output_paths(
                wdir, args.step, args.worker_id, args.n_clusters
            )
            already_done = all(os.path.exists(p) for p in expected)
        else:
            expected, already_done = [], False
        if already_done:
            print(f"Worker {args.worker_id} Step {args.step} already completed, skipping.", flush=True)
            return
        else:
            missing = [p for p in expected if not os.path.exists(p)]
            print(f"Worker {args.worker_id} Step {args.step}: {len(missing)} file(s) missing, running."
                  f" (e.g. {missing[0] if missing else '?'})", flush=True)

        # Suppress noisy loggers in worker subprocesses
        logging.getLogger("tensorpotential.data.streaming").setLevel(logging.WARNING)
        logging.getLogger("tensorflow").setLevel(logging.ERROR)

        if args.checkpoint.endswith(".index"):
            args.checkpoint = args.checkpoint.replace(".index", "")

        # Every step only needs the side-written FEATURES key (energy-only
        # compute is sufficient — no forces are required now that the
        # force-error model has been removed).
        step1_compute_fn = ComputeEnergy(
            extra_return_keys=[uq_constants.FEATURES]
        )

        tp, instructions = load_uq_model(
            model_yaml=args.model_yaml,
            checkpoint=args.checkpoint,
            model_compute_function=step1_compute_fn,
            # The production basis-RP transform; the shared constant keeps this in
            # sync with what make_basis_rp_spec stamps into the artifact below.
            feature_spec={
                "out_dim": args.rp_dim,
                "seed": args.rp_seed,
                "transform": uq_constants.DEFAULT_FEATURE_TRANSFORM,
                "normalize": uq_constants.UQ_DEFAULT_NORMALIZE,
                "add_density_channel": uq_constants.UQ_DEFAULT_DENSITY,
                "density_scale": uq_constants.UQ_DEFAULT_DENSITY_SCALE,
            },
        )
        cutoff, symbols, indices = extract_cutoff_and_elements(instructions)
        cutoff_dict = extract_cutoff_dict(instructions)
        elem_map = {s: int(i) for s, i in zip(symbols, indices)}

        use_tf_dataset = is_sharded_dataset(args.train_data)

        if use_tf_dataset and (args.train_data_weighted or args.filter_fn):
            raise SystemExit(
                "--train-data-weighted and --filter-fn require pickle/df shards; "
                "they are not supported with sharded TF datasets."
            )

        def make_feature_iter():
            if use_tf_dataset:
                dataset_path = args.train_data[0]
                all_shards = discover_shards(dataset_path)
                my_shards = all_shards[args.worker_id :: args.n_workers]
                total_batches = None
                try:
                    stats = read_saved_dataset_stats(dataset_path)
                    if all_shards:
                        total_batches = (
                            stats.get("total_num_of_batches", 0)
                            * len(my_shards)
                            // len(all_shards)
                        )
                except (FileNotFoundError, ValueError):
                    pass
                return _prefetch(tf_dataset_feature_iterator(
                    my_shards,
                    tp.model,
                    frac=args.frac,
                    seed=args.seed + args.worker_id,
                    verbose=args.verbose,
                    total_num_batches=total_batches,
                ))
            else:
                n_my_files = sum(1 for _ in _iter_worker_files(
                    args.train_data, args.worker_id, args.n_workers
                ))
                estimate = StreamingEstimate(n_my_files)
                worker_filter_fn = _load_filter_fn(args.filter_fn)
                atoms_gen = stream_atoms(
                    args.train_data,
                    args.worker_id,
                    args.n_workers,
                    frac=args.frac,
                    seed=args.seed,
                    estimate=estimate,
                    file_weight_map=args.file_weight_map,
                    filter_fn=worker_filter_fn,
                )
                return _prefetch(batched_feature_iterator(
                    atoms_gen,
                    tp.model,
                    elem_map,
                    cutoff,
                    cutoff_dict=cutoff_dict,
                    max_num_neighbours_per_batch=args.max_neighbours_per_batch,
                    verbose=args.verbose,
                    total_num_structures=estimate,
                ))

        if args.step == 1:
            run_worker_step1(args, make_feature_iter())
        elif args.step == 2:
            run_worker_step2(args, make_feature_iter())
        elif args.step == 3:
            run_worker_step3(args, make_feature_iter(), tp)
    else:
        # Master mode
        if args.train_data is None:
            # Standalone export: artifact must already exist (validated above)
            if args.no_export:
                parser.error(
                    "Nothing to do: --train-data not provided and --no-export set"
                )
            print(f"Standalone export: {args.artifact_path} -> {args.export_path}")
            export_savedmodel(args)
            print(f"\nSUCCESS: SavedModel exported to {args.export_path}")
        else:
            run_master(args)
