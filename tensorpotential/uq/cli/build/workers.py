"""Per-worker entry points for the three build steps + SavedModel export."""

from __future__ import annotations

import logging

import itertools
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf

from tensorpotential.instructions.base import load_instructions
from tensorpotential.metadata_utils import resolve_param_dtype
from tensorpotential.tensorpot import TensorPotential
from tensorpotential.uq import GMMUQArtifactBuilder, GMMUQModel
from tensorpotential.uq import constants as uq_constants
from tensorpotential.uq.cli.build.paths import _worker_dir

log = logging.getLogger("grace_uq")

def run_worker_step1(args, feature_iter):
    """Step 1: Extract features and fit local KMeans for each candidate k."""
    n_clusters_list = args.n_clusters
    try:
        first_batch = next(feature_iter)
    except StopIteration:
        return

    feature_dim = first_batch[0].shape[1]
    builders = {
        k: GMMUQArtifactBuilder(
            n_clusters=k,
            feature_dim=feature_dim,
            random_state=args.seed + args.worker_id,
        )
        for k in n_clusters_list
    }

    # Accumulate per-element WCSS for each k (used for elbow selection).
    # Computed on actual features against current centroids after each partial_fit.
    wcss = {k: defaultdict(float) for k in n_clusters_list}
    wcss_n = {k: defaultdict(int) for k in n_clusters_list}

    # Bootstrap buffer: MiniBatchKMeans raises ValueError on the very first partial_fit
    # call when there are fewer than n_clusters samples (centers not yet initialized).
    # After the first successful call the centers are set and any batch size is accepted.
    # We buffer per-element features across batches until we have >= max_k samples, then
    # flush them all at once to bootstrap every k-builder simultaneously.
    # A separate `bootstrapped` set prevents re-entering the buffer path after init.
    bootstrapped: set[int] = set()
    init_buf: dict[int, list] = {}
    max_k = max(n_clusters_list)

    print(
        f"Worker {args.worker_id} starting feature extraction for k={n_clusters_list}...",
        flush=True,
    )
    # Per-element buffer of weights paired with init_buf entries, so the
    # bootstrap flush re-applies the same per-atom weights to step-1 counts.
    init_w_buf: dict[int, list] = {}

    # Single data pass: feed every batch to all k-builders simultaneously.
    for feats, elems, weights in itertools.chain([first_batch], feature_iter):
        feats = np.asarray(feats, dtype=np.float64)
        elems = np.asarray(elems, dtype=np.int32)
        weights = np.asarray(weights, dtype=np.float64)
        # When all atoms in the batch carry weight 1.0 we skip sklearn's
        # weighted partial_fit code path entirely. The buffer still stores
        # the weight array for consistency at flush time.
        batch_unweighted = bool(np.all(weights == 1.0))
        for e in np.unique(elems):
            mask = elems == e
            elem_feats = feats[mask]
            elem_w = None if batch_unweighted else weights[mask]
            e_int = int(e)

            if e_int in bootstrapped:
                # Centers already initialized — any batch size is safe.
                for k, builder in builders.items():
                    km = builder._get_kmeans(e_int)
                    try:
                        km.partial_fit(elem_feats, sample_weight=elem_w)
                        wcss[k][e_int] += km.inertia_
                        wcss_n[k][e_int] += len(elem_feats)
                        builder.accumulate_step1_counts(e_int, elem_feats, elem_w)
                    except ValueError:
                        log.debug(
                            "partial_fit failed for element %d, k=%d: %d samples",
                            e, k, len(elem_feats),
                        )
                continue

            # Not yet bootstrapped — buffer until we have >= max_k samples.
            buf = init_buf.setdefault(e_int, [])
            wbuf = init_w_buf.setdefault(e_int, [])
            buf.append(elem_feats)
            # Materialize per-batch weight as ones only when the buffer ever
            # mixes weighted and unweighted batches; otherwise leave it None.
            wbuf.append(elem_w if elem_w is not None else np.ones(len(elem_feats)))
            if sum(len(b) for b in buf) < max_k:
                continue  # still accumulating

            # Enough samples: flush buffer and bootstrap all k-builders at once.
            flushed = np.vstack(init_buf.pop(e_int))
            buffered_w = init_w_buf.pop(e_int)
            flushed_w = np.concatenate(buffered_w)
            # If every buffered batch was unweighted, drop the ones-array so
            # sklearn takes its fast path.
            if np.all(flushed_w == 1.0):
                flushed_w = None
            bootstrapped.add(e_int)
            for k, builder in builders.items():
                km = builder._get_kmeans(e_int)
                try:
                    km.partial_fit(flushed, sample_weight=flushed_w)
                    wcss[k][e_int] += km.inertia_
                    wcss_n[k][e_int] += len(flushed)
                    builder.accumulate_step1_counts(e_int, flushed, flushed_w)
                except ValueError:
                    log.warning(
                        "Worker %d: bootstrap partial_fit failed for element %d k=%d "
                        "with %d buffered samples",
                        args.worker_id, e_int, k, len(flushed),
                    )
                    bootstrapped.discard(e_int)  # retry on future batches

    # Last-resort flush: elements whose total samples never reached max_k.
    for e_int, bufs in init_buf.items():
        flushed = np.vstack(bufs)
        flushed_w = np.concatenate(init_w_buf.get(e_int, [np.ones(len(flushed))]))
        if np.all(flushed_w == 1.0):
            flushed_w = None
        log.warning(
            "Worker %d: element %d never accumulated >= %d samples "
            "(total=%d across all batches); attempting last-resort partial_fit",
            args.worker_id, e_int, max_k, len(flushed),
        )
        for k, builder in builders.items():
            km = builder._get_kmeans(e_int)
            try:
                km.partial_fit(flushed, sample_weight=flushed_w)
                wcss[k][e_int] += km.inertia_
                wcss_n[k][e_int] += len(flushed)
                builder.accumulate_step1_counts(e_int, flushed, flushed_w)
            except ValueError:
                log.warning(
                    "Worker %d: element %d k=%d STILL unfitted after last-resort flush "
                    "(%d samples < k=%d); will be excluded from this worker's output",
                    args.worker_id, e_int, k, len(flushed), k,
                )

    if not args.verbose:
        print(f"Worker {args.worker_id} progress: 100%", flush=True)

    for k, builder in builders.items():
        builder._fitted = True
        km_results = builder.export_kmeans_results()
        if not km_results:
            log.warning("Worker %d: no elements fitted for k=%d", args.worker_id, k)
            continue
        save_dict = {"elements": np.array(list(km_results.keys()), dtype=np.int32)}
        for elem, data in km_results.items():
            save_dict[f"{uq_constants.CENTROIDS}_{elem}"] = data[uq_constants.CENTROIDS]
            save_dict[f"{uq_constants.COUNTS}_{elem}"] = data[uq_constants.COUNTS]
            if uq_constants.EFFECTIVE_COUNT in data:
                save_dict[f"{uq_constants.EFFECTIVE_COUNT}_{elem}"] = data[
                    uq_constants.EFFECTIVE_COUNT
                ]
            save_dict[f"wcss_{elem}"] = np.float64(wcss[k].get(elem, 0.0))
            save_dict[f"wcss_n_{elem}"] = np.int64(wcss_n[k].get(elem, 0))
        np.savez(os.path.join(_worker_dir(args), f".step1_w{args.worker_id}_k{k}.npz"), **save_dict)
    print(
        f"Worker {args.worker_id} Step 1 completed (artifacts saved for k={n_clusters_list}).",
        flush=True,
    )


def run_worker_step2(args, feature_iter):
    """Step 2: accumulate scatter matrices using global centroids.

    Workers save scatter/counts in step-1 cluster order without calling
    ``finalize()``: each worker's per-subset counts can sort to a different
    permutation, and master's element-wise ``+=`` aggregation would silently
    mix scatter across clusters if workers permuted independently. Master
    applies the single sort after aggregation.
    """
    artifacts = GMMUQArtifactBuilder.load(args.step1_artifacts)
    builder = GMMUQArtifactBuilder.from_artifacts(artifacts, n_clusters=args.n_clusters[0])

    print(f"Worker {args.worker_id} starting scatter accumulation...", flush=True)
    builder.accumulate_scatter(feature_iter, verbose=False)
    if not args.verbose:
        print(f"Worker {args.worker_id} progress: 100%", flush=True)

    partial = {
        e: {
            uq_constants.CENTROIDS: builder.get_centroids(e),
            uq_constants.SCATTER: builder._scatter[e],
            uq_constants.COUNTS: builder._counts[e],
            uq_constants.EFFECTIVE_COUNT: builder._effective_counts[e],
        }
        for e in sorted(builder.elements_seen)
    }
    GMMUQArtifactBuilder.save_artifacts(
        os.path.join(_worker_dir(args), f".step2_w{args.worker_id}.npz"),
        partial,
    )
    print(f"Worker {args.worker_id} Step 2 completed (artifacts saved).", flush=True)


def run_worker_step3(args, feature_iter, tp):
    """Step 3: per-cluster σ-histogram accumulation.

    Builds raw (int64) and effective (weighted float64) σ histograms per
    element — the raw one drives ``min_atoms_for_threshold`` reliability, the
    effective one drives the robust median+k*MAD threshold under source weights.
    These histograms produce the per-element/per-cluster gamma thresholds.
    """
    param_dtype = tp.param_dtype
    gmm_uq = GMMUQModel(args.step2_artifacts, param_dtype=param_dtype)
    n_bins, low, high = 250, 0, 100
    n_clusters = args.n_clusters[0]
    scale = n_bins / (high - low)
    hists = defaultdict(lambda: np.zeros((n_clusters, n_bins), dtype=np.int64))
    eff_hists = defaultdict(lambda: np.zeros((n_clusters, n_bins), dtype=np.float64))

    print(f"Worker {args.worker_id} starting histogram evaluation...", flush=True)
    for feats, elems, weights in feature_iter:
        s, _, cluster_assign = gmm_uq._compute_compiled(
            tf.constant(feats, dtype=param_dtype),
            tf.constant(elems, dtype=tf.int32),
            1e-8,
        )
        s_np = s.numpy()
        a_np = cluster_assign.numpy()
        w_np = np.asarray(weights, dtype=np.float64)
        # Skip the weighted bincount when every atom has weight 1.0 — the
        # weighted histogram then equals the raw one and we accumulate it
        # by reusing the int64 flat result as a float view.
        batch_unweighted = bool(np.all(w_np == 1.0))
        s_bins = np.clip(((s_np - low) * scale).astype(np.int64), 0, n_bins - 1)
        for e in np.unique(elems):
            mask_e = elems == e
            combined = a_np[mask_e] * n_bins + s_bins[mask_e]
            flat = np.bincount(combined, minlength=n_clusters * n_bins)
            hists[e] += flat.reshape(n_clusters, n_bins)
            if batch_unweighted:
                eff_hists[e] += flat.reshape(n_clusters, n_bins).astype(np.float64)
            else:
                flat_w = np.bincount(
                    combined, weights=w_np[mask_e], minlength=n_clusters * n_bins
                )
                eff_hists[e] += flat_w.reshape(n_clusters, n_bins)

    if not args.verbose:
        print(f"Worker {args.worker_id} progress: 100%", flush=True)

    elem_keys = sorted(set(hists.keys()) | set(eff_hists.keys()))
    save_dict = {
        "elements": np.array(elem_keys, dtype=np.int32),
        "n_clusters": np.int32(n_clusters),
    }
    for e in elem_keys:
        save_dict[f"hist_{e}"] = hists[e]  # [n_clusters, n_bins], int64
        save_dict[f"eff_hist_{e}"] = eff_hists[e]  # [n_clusters, n_bins], float64
    np.savez(os.path.join(_worker_dir(args), f".step3_w{args.worker_id}.npz"), **save_dict)
    print(f"Worker {args.worker_id} Step 3 completed (artifacts saved).", flush=True)


def export_savedmodel(args):
    """Export a SavedModel with compute and compute_uq signatures."""
    param_dtype = resolve_param_dtype(args.model_yaml)
    gmm_uq = GMMUQModel(args.artifact_path, param_dtype=param_dtype)

    checkpoint = args.checkpoint
    if checkpoint.endswith(".index"):
        checkpoint = checkpoint[: -len(".index")]

    instructions = load_instructions(args.model_yaml)
    tp = TensorPotential(potential=instructions, param_dtype=param_dtype)
    # expect_partial=True silences the noisy "checkpoint deleted with
    # unrestored values" warnings at GC time. The unmatched values are the
    # optimizer slots/iterations, which the SavedModel export does not need.
    tp.load_checkpoint(
        checkpoint,
        assert_consumed=False,
        # GRACE_UQ_LENIENT_LOAD: see factories.load_uq_model. Strict by default.
        assert_existing_objects_matched=not os.environ.get("GRACE_UQ_LENIENT_LOAD"),
        expect_partial=True,
    )
    tp.save_model_with_aux_computes(
        exact_path=args.export_path, gmm_uq_model=gmm_uq
    )
