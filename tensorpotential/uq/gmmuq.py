"""TF-differentiable GMM-UQ forward pass for uncertainty quantification."""

from __future__ import annotations

import copy
import logging
import os
import warnings
from typing import NamedTuple

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from tensorpotential.uq.artifact_builder import (
    GMMUQArtifactBuilder,
    _default_eff_counts,
)
from tensorpotential.uq import constants as uq_constants
from tensorpotential.uq.calm_numpy import cluster_groups, nearest_cluster, sigma

log = logging.getLogger(__name__)

# inv_cov diagonal value used for elements with no training data.
# Mahalanobis distance = sqrt(delta^T @ inv_cov @ delta) ≈ sqrt(scale) * ||delta||,
# so scale=1e6 gives sigma ~1000x the feature norm — reliably flags these atoms as
# high-uncertainty regardless of the actual feature value.
_MISSING_ELEM_INV_COV_DIAG = 1e6


# Statistics that update / trial_gamma / inflate read or write. They are kept float64
# in memory whatever the stored dtype: a float32 SCATTER/INV_COV (the store_fp32
# default) would round every re-inverted inverse back to float32, and with condition
# numbers ~1e5 an additive update could then raise gamma by ~1e-4 relative.
_WORKING_STATS = (
    uq_constants.CENTROIDS,
    uq_constants.INV_COV,
    uq_constants.SCATTER,
    uq_constants.EFFECTIVE_COUNT,
)


def _float64_working_stats(artifacts: dict) -> None:
    """In place: float arrays of :data:`_WORKING_STATS` -> float64 (counts stay int)."""
    for a in artifacts.values():
        for key in _WORKING_STATS:
            v = a.get(key)
            if v is not None and np.issubdtype(np.asarray(v).dtype, np.floating):
                a[key] = np.asarray(v, dtype=np.float64)


class InflateResult(NamedTuple):
    """Outcome of :meth:`GMMUQModel.inflate`."""

    n_rounds: int
    factors: list
    gamma_max_after: float


class GMMUQModel(tf.Module):
    """Differentiable GMM-based local latent-space UQ model.

    Loads pre-computed artifacts (centroids + inverse covariance matrices per element
    per cluster) and computes per-atom Mahalanobis distances in a fully vectorized,
    GradientTape-compatible manner. Processes atoms in chunks to bound memory usage.

    Parameters
    ----------
    artifact_path : str
        Path to .npz file produced by GMMUQArtifactBuilder.save().
    param_dtype : tf.DType or None
        Dtype for artifact tensors and computation. If None (default), the dtype
        is inferred from the centroid arrays stored in the .npz file (typically
        float64, since the artifact builder always uses float64 internally).
    chunk_size : int
        Max atoms per chunk. Controls peak memory usage.
    """

    def __init__(
        self,
        artifact_path: str,
        param_dtype=None,
        chunk_size: int = 1024,
        regularization: float = 1e-6,
    ):
        super().__init__(name="GMMUQModel")
        # the artifact file this model was loaded from (None: built in memory); HAL
        # sidecars hash it so grace_uq select can check the --artifact
        self.source_path = os.path.abspath(artifact_path)
        self.chunk_size = chunk_size
        self.regularization = regularization

        # Single load: parse artifacts and extract extra data in one pass
        artifact_keys_prefixes = (
            f"{uq_constants.CENTROIDS}_",
            f"{uq_constants.INV_COV}_",
            f"{uq_constants.COUNTS}_",
            f"{uq_constants.SCATTER}_",
            "elements",
        )
        # Use object.__setattr__ to store dicts with non-string or numpy-array values
        # without TF wrapping them in DictWrapper (which requires string keys and
        # fails to checkpoint integer-keyed artifact dicts).
        object.__setattr__(self, "extra_data", {})
        with np.load(artifact_path, allow_pickle=True) as data:
            object.__setattr__(
                self,
                "_artifacts",
                GMMUQArtifactBuilder._parse_npz(data, path=artifact_path),
            )
            for k in data.files:
                if not k.startswith(artifact_keys_prefixes) or k == "element_map":
                    self.extra_data[k] = data[k]

        if param_dtype is None:
            first_elem = next(iter(self._artifacts.values()))
            param_dtype = tf.as_dtype(first_elem[uq_constants.CENTROIDS].dtype)
        self.param_dtype = param_dtype
        _float64_working_stats(self._artifacts)  # after the stored-dtype inference
        self._build_tensors(self._artifacts)

    @classmethod
    def from_artifacts(
        cls,
        artifacts,
        extra_data=None,
        param_dtype=None,
        chunk_size=1024,
        regularization=1e-6,
    ):
        """Construct a GMMUQModel directly from an artifacts dict — no disk I/O.

        The caller's dict is not modified; the model keeps float64 working statistics
        (float32 arrays are upcast into the model's own per-element dicts).

        Parameters
        ----------
        artifacts : dict
            {elem_idx: {CENTROIDS, INV_COV, COUNTS, ...}} as returned by
            ``GMMUQArtifactBuilder.finalize()``.
        extra_data : dict, optional
            Extra metadata stored on the model (e.g. ``interp_thresholds``,
            ``element_map``). Passed through to ``extra_data`` and picked up
            by ``_build_tensors`` for threshold initialisation.
        param_dtype : tf.DType or None
            Inferred from centroid dtype if not provided.
        chunk_size : int
        regularization : float
        """
        obj = cls.__new__(cls)
        tf.Module.__init__(obj, name="GMMUQModel")
        obj.source_path = None  # built in memory
        obj.chunk_size = chunk_size
        obj.regularization = regularization
        # arrays copied: the float64 upcast and the in-place updates (update_one,
        # update_many, inflate) must not change the caller's arrays
        artifacts = {e: {k: np.array(v) if isinstance(v, np.ndarray) else v
                         for k, v in a.items()} for e, a in artifacts.items()}
        object.__setattr__(obj, "_artifacts", artifacts)
        object.__setattr__(obj, "extra_data", extra_data or {})
        if param_dtype is None:
            first_elem = next(iter(artifacts.values()))
            param_dtype = tf.as_dtype(first_elem[uq_constants.CENTROIDS].dtype)
        obj.param_dtype = param_dtype
        _float64_working_stats(artifacts)  # after the stored-dtype inference
        obj._build_tensors(artifacts)
        return obj

    def __repr__(self):
        return f"GMMUQModel(n_elements={self.n_elements}, K_max={self.K_max}, D={self.D}, dtype={self.param_dtype.name})"

    @property
    def element_symbols(self) -> list[str] | None:
        """Decoded element symbols from artifact metadata."""
        if "element_map" not in self.extra_data:
            return None
        elems = self.extra_data["element_map"]
        if hasattr(elems, "dtype") and elems.dtype.kind in ("S", "U"):
            if elems.dtype.kind == "S":
                return [e.decode("utf-8") for e in elems]
            return list(elems)
        return [str(e) for e in elems]

    @property
    def thresholds_dict(self) -> dict[str, np.ndarray] | None:
        """Dictionary mapping element symbols to per-cluster interpolation thresholds.

        Returns
        -------
        dict[str, np.ndarray] or None
            {symbol: np.ndarray[K]} — per-cluster threshold array for each element.
        """
        symbols = self.element_symbols
        if symbols is None or "interp_thresholds" not in self.extra_data:
            return None
        t = self.extra_data["interp_thresholds"]
        return {s: t[i] for i, s in enumerate(symbols)}

    # XLA-GPU has no ResourceGatherNd kernel — consumers that pass these
    # variables into ``tf.gather`` / ``tf.gather_nd`` must read through the
    # ``_dense`` properties below so the read is a regular Tensor before the
    # gather op fuses.  Keep the underlying tf.Variable attributes so
    # ``_build_*_tensors`` can ``.assign(...)`` for incremental updates.
    @property
    def interp_thresholds_dense(self):
        return tf.identity(self.interp_thresholds) if self.interp_thresholds is not None else None

    def _build_tensors(self, artifacts: dict):
        """Stack per-element artifacts into padded tensors for vectorized gather.

        Tensors are sized to cover all model element type indices so that
        tf.gather(centroids, element_indices) is always in-bounds. Rows for
        elements absent from artifacts get a large isotropic inv_cov, producing
        very high Mahalanobis distances for those atoms (= maximum UQ signal).
        """
        elements = sorted(artifacts.keys())

        # Use element_map metadata for the full model element count when available.
        # This ensures rows exist for elements absent from training data (e.g. rare
        # noble gases) so that tf.gather never goes out of bounds at inference time.
        if "element_map" in self.extra_data:
            n_rows = len(self.extra_data["element_map"])
        else:
            n_rows = (max(elements) + 1) if elements else 1
        self.n_elements = n_rows
        self.element_to_idx = {e: e for e in elements}  # raw index == row index

        K_max = max(a[uq_constants.CENTROIDS].shape[0] for a in artifacts.values())
        D = artifacts[elements[0]][uq_constants.CENTROIDS].shape[1]
        self.K_max = K_max
        self.D = D

        centroids = np.zeros((n_rows, K_max, D), dtype=np.float64)
        inv_covs = np.zeros((n_rows, K_max, D, D), dtype=np.float64)
        n_clusters = np.full(n_rows, K_max, dtype=np.int32)

        for elem in elements:
            a = artifacts[elem]
            K = a[uq_constants.CENTROIDS].shape[0]
            centroids[elem, :K] = a[uq_constants.CENTROIDS]
            inv_covs[elem, :K] = a[uq_constants.INV_COV]
            n_clusters[elem] = K
            # Missing slots of a fitted element (K < K_max: never in `grace_uq build`
            # artifacts, which pad with sentinel clusters themselves, but possible via
            # from_artifacts) get the builder's sentinel centroid, not zeros, so they can
            # never be the nearest centroid (compute() takes the argmin over all K_max).
            centroids[elem, K:] = uq_constants.SENTINEL_CENTROID_VALUE

        # Fill unfitted rows with a large isotropic inv_cov so those atoms always
        # produce maximum Mahalanobis distance (= maximum UQ signal).
        faked = set(range(n_rows)) - set(elements)
        if faked:
            diag = np.arange(D)
            for e in faked:
                inv_covs[e, :, diag, diag] = _MISSING_ELEM_INV_COV_DIAG

            symbols = self.element_symbols  # handles bytes/unicode/fallback
            if symbols is not None:
                labels = [f"{symbols[e]}({e})" for e in sorted(faked)]
            else:
                labels = [str(e) for e in sorted(faked)]
            warnings.warn(
                f"GMMUQModel: {len(faked)} element(s) have no UQ training data and will use a "
                f"fictitious isotropic inv_cov (diag={_MISSING_ELEM_INV_COV_DIAG:.0e}) — "
                f"they will always produce maximum uncertainty: {', '.join(labels)}",
                UserWarning,
                stacklevel=2,
            )

        # Use tf.Variable so that update_one/update_many are visible to
        # already-traced tf.function / XLA graphs (tf.constant is baked in at
        # trace time and never re-read).
        self._assign_or_create(
            "centroids", tf.cast(centroids, self.param_dtype), "gmm_centroids"
        )
        self._assign_or_create(
            "inv_covs", tf.cast(inv_covs, self.param_dtype), "gmm_inv_covs"
        )
        self._assign_or_create(
            "n_clusters_per_elem", tf.cast(n_clusters, tf.int32), "gmm_n_clusters"
        )

        # Prefer the weighted thresholds when both are present (any non-trivial
        # source weighting renders the raw p99 quantile non-representative).
        # Falls back to ``interp_thresholds`` for older artifacts that lack the
        # weighted variant.
        thresh_key = (
            "eff_interp_thresholds"
            if "eff_interp_thresholds" in self.extra_data
            else "interp_thresholds"
        )
        if thresh_key in self.extra_data:
            self._assign_or_create(
                "interp_thresholds",
                tf.cast(self.extra_data[thresh_key], self.param_dtype),
                "gmm_interp_thresholds",
            )
        else:
            self.interp_thresholds = None

    def _assign_or_create(self, attr: str, value, name: str):
        """``self.<attr>.assign(value)`` if it's already a ``tf.Variable``,
        else create a fresh non-trainable Variable.  Keeps incremental
        artifact updates visible to already-traced tf.function / XLA graphs."""
        cur = getattr(self, attr, None)
        if isinstance(cur, tf.Variable):
            cur.assign(value)
        else:
            setattr(self, attr, tf.Variable(value, trainable=False, name=name))

    def compute(self, features, element_indices, eps=1e-8):
        """Pure TF forward pass — safe to call inside tf.function / GradientTape.

        No Python control flow, no chunking. Use this from compiled contexts (HAL).
        """
        features = tf.cast(features, self.param_dtype)
        element_indices = tf.cast(element_indices, tf.int32)
        return self._compute_core(features, element_indices, eps)

    def gamma_from_features(self, features, element_indices, eps=1e-8):
        """Compute per-atom gamma (sigma / interp_threshold) from pre-extracted features.

        Convenience wrapper around ``compute()`` that applies the stored
        interpolation thresholds. Suitable for evaluating a frozen model on
        features extracted by a different (e.g. active) model instance,
        which is the typical two-tower HAL pattern.

        Element indices must use the same mapping as this model (i.e. the same
        ``element_map`` used when building the artifacts). When Tower 2 is built
        with a consistent element map (same as Tower 1), indices pass through
        directly with no remapping needed.

        Needs the calibrated ``interp_thresholds`` that ``grace_uq build`` writes
        (step 3); an artifact saved by ``GMMUQArtifactBuilder`` alone has none, and
        then a ``ValueError`` says so (``compute()`` / sigma still work).

        Parameters
        ----------
        features : array-like [N, D]
            Hidden features per atom (numpy or tf.Tensor).
        element_indices : array-like [N]
            Element type indices per atom, consistent with this model's element map.
        eps : float
            Passed to ``compute()``.

        Returns
        -------
        gamma : np.ndarray [N]
            Per-atom normalised uncertainty.
        sigma : np.ndarray [N]
            Per-atom raw Mahalanobis distance.
        """
        if self.interp_thresholds is None:
            raise ValueError(
                "this artifact has no gamma thresholds (interp_thresholds): it gives sigma "
                "only. Build a complete artifact with grace_uq build, which calibrates the "
                "thresholds on the training data (step 3)"
            )
        feat = tf.cast(features, self.param_dtype)
        eidx = tf.cast(element_indices, tf.int32)
        sigma, _, cluster_assign = self.compute(feat, eidx, eps=eps)
        sigma_np = sigma.numpy()
        idx = tf.stack([eidx, cluster_assign], axis=1)
        thresholds = tf.gather_nd(self.interp_thresholds_dense, idx).numpy()
        gamma_np = sigma_np / thresholds
        return gamma_np, sigma_np

    def __call__(self, features, element_indices, eps=1e-8, verbose=True):
        """Chunked eager entry point for standalone evaluation.

        Parameters
        ----------
        features : tf.Tensor [N, D]
            Hidden features per atom.
        element_indices : tf.Tensor [N]
            Element type index per atom (matching artifact element ordering).
        eps : float
            Epsilon for numerical stability in sqrt.
        verbose : bool
            Show progress bar for chunked processing.

        Returns
        -------
        sigma : tf.Tensor [N]
            Per-atom uncertainty (Mahalanobis distance to assigned cluster).
        total_sigma : tf.Tensor scalar
            Sum of per-atom uncertainties.
        cluster_assign : tf.Tensor [N]
            Per-atom cluster assignment index.
        """
        features = tf.cast(features, self.param_dtype)
        element_indices = tf.cast(element_indices, tf.int32)

        N = features.shape[0] or int(tf.shape(features)[0])

        if N <= self.chunk_size:
            return self._compute_compiled(features, element_indices, eps)

        # Chunked processing to bound memory
        sigma_parts = []
        assign_parts = []
        n_chunks = (N + self.chunk_size - 1) // self.chunk_size
        chunk_iter = range(0, N, self.chunk_size)
        if verbose:
            chunk_iter = tqdm(chunk_iter, total=n_chunks, desc="GMM-UQ")
        for start in chunk_iter:
            end = min(start + self.chunk_size, N)
            sigma_chunk, _, assign_chunk = self._compute_compiled(
                features[start:end], element_indices[start:end], eps
            )
            sigma_parts.append(sigma_chunk)
            assign_parts.append(assign_chunk)

        sigma = tf.concat(sigma_parts, axis=0)
        cluster_assign = tf.concat(assign_parts, axis=0)
        total_sigma = tf.reduce_sum(sigma)
        return sigma, total_sigma, cluster_assign

    @tf.function(jit_compile=True)
    def _compute_compiled(self, features, element_indices, eps):
        """XLA-compiled entry for eager chunked calls."""
        return self._compute_core(features, element_indices, eps)

    def _compute_core(self, features, element_indices, eps):
        # Snapshot variables to plain tensors so XLA sees regular GatherNd
        # (ResourceGatherNd on tf.Variable is unsupported on XLA_CPU_JIT).
        centroids = tf.identity(self.centroids)
        inv_covs = tf.identity(self.inv_covs)

        # Step 1: Assign each atom to nearest centroid (Euclidean, non-differentiable)
        centroids_i = tf.gather(centroids, element_indices)  # [N, K, D]
        delta_all = features[:, None, :] - centroids_i  # [N, K, D]
        euclid_sq = tf.reduce_sum(delta_all**2, axis=-1)  # [N, K]
        cluster_assign = tf.stop_gradient(
            tf.argmin(euclid_sq, axis=1, output_type=tf.int32)
        )  # [N]

        # Step 2: Gather only the assigned centroid and inv_cov per atom
        elem_cluster_idx = tf.stack([element_indices, cluster_assign], axis=1)  # [N, 2]
        centroid_assigned = tf.gather_nd(centroids, elem_cluster_idx)  # [N, D]
        inv_cov_assigned = tf.gather_nd(inv_covs, elem_cluster_idx)  # [N, D, D]

        # Step 3: Mahalanobis distance to assigned centroid only
        delta = features - centroid_assigned  # [N, D]
        sigma_sq = tf.einsum("nd,nde,ne->n", delta, inv_cov_assigned, delta)  # [N]

        sigma = tf.sqrt(sigma_sq + eps)
        total_sigma = tf.reduce_sum(sigma)

        return sigma, total_sigma, cluster_assign

    def eval_numpy_chunked(self, chunk_iterator, verbose=True):
        """Evaluate sigma from numpy chunks without materializing full GPU tensor.

        Parameters
        ----------
        chunk_iterator : iterable of (np.ndarray [N,D], np.ndarray [N])
            Yields (features, element_indices) chunks (e.g. FeatureBuffer.iter_chunks()).
        verbose : bool
            Show progress bar.

        Returns
        -------
        sigma : np.ndarray [N_total]
        total_sigma : float
        cluster_assign : np.ndarray [N_total]
        """
        sigma_parts = []
        assign_parts = []
        it = tqdm(chunk_iterator, desc="GMM-UQ") if verbose else chunk_iterator
        for feats_np, elems_np in it:
            feats_tf = tf.constant(feats_np, dtype=self.param_dtype)
            elems_tf = tf.constant(elems_np, dtype=tf.int32)
            s, _, a = self._compute_compiled(feats_tf, elems_tf, 1e-8)
            sigma_parts.append(s.numpy())
            assign_parts.append(a.numpy())
        sigma = np.concatenate(sigma_parts)
        cluster_assign = np.concatenate(assign_parts)
        return sigma, float(sigma.sum()), cluster_assign

    def eval_from_atoms(
        self, calc, atoms_iterable, element_map=None, chunk_size=1024, verbose=True
    ):
        """Stream features from atoms and evaluate GMM — no full feature storage.

        Extracts features one structure at a time, batches into chunks, evaluates
        GMM on each chunk, and discards. Peak memory: O(chunk_size * D).

        Parameters
        ----------
        calc : TPCalculator
            Calculator with hidden feature extraction enabled.
        atoms_iterable : iterable of ase.Atoms
            Structures to evaluate.
        element_map : dict, optional
            {symbol: element_type_index}. If None, built from first pass
            (requires atoms_iterable to be a list/sequence).
        chunk_size : int
            Atoms per evaluation chunk.
        verbose : bool
            Show progress bar.

        Returns
        -------
        sigma : np.ndarray [N_total_atoms]
        total_sigma : float
        cluster_assign : np.ndarray [N_total_atoms]
        """
        from .feature_extraction import extract_features, batch_feature_chunks

        feature_gen = extract_features(calc, atoms_iterable, element_map)
        chunks = batch_feature_chunks(feature_gen, chunk_size=chunk_size)
        return self.eval_numpy_chunked(chunks, verbose=verbose)

    def update_one(self, features, element_indices, alpha=None, mode="convex"):
        """Incrementally update artifacts with features from one structure.

        Centroids stay fixed. Scatter matrices and counts are updated, inv_cov is
        recomputed for the clusters that received atoms, and only the rows of the
        touched elements are rewritten in the ``inv_covs`` variable.

        Parameters
        ----------
        features : np.ndarray [N_atoms, D]
            Feature vectors for atoms in the new structure.
        element_indices : np.ndarray [N_atoms]
            Element type index per atom.
        alpha : float or None
            Update factor: ``C_new = (1-alpha)*C_old + alpha*C_batch`` (``mode="convex"``)
            or ``C_new = C_old + alpha*C_batch`` (``mode="additive"``). None (default)
            uses accumulative mode.
        mode : {"convex", "additive"}
            ``"additive"`` never forgets the old data; see
            :meth:`GMMUQArtifactBuilder.update`.
        """
        GMMUQArtifactBuilder.update(
            self._artifacts,
            features,
            element_indices,
            self.regularization,
            alpha=alpha,
            mode=mode,
        )
        self._refresh_element_rows(np.unique(element_indices))

    def update_many(
        self, features_list, element_indices_list, chunk_size=4096, alpha=None,
        mode="convex",
    ):
        """Incrementally update artifacts with features from multiple structures.

        Parameters
        ----------
        features_list : list of np.ndarray [N_i, D] or single np.ndarray [N_total, D]
            Feature vectors. If a list, concatenated internally.
        element_indices_list : list of np.ndarray [N_i] or single np.ndarray [N_total]
            Element type indices. If a list, concatenated internally.
        chunk_size : int
            Process in chunks to bound memory for the assignment step.
        alpha : float or None
            Update factor: ``C_new = (1-alpha)*C_old + alpha*C_batch`` (``mode="convex"``)
            or ``C_new = C_old + alpha*C_batch`` (``mode="additive"``). None (default)
            uses accumulative mode.
        mode : {"convex", "additive"}
            See :meth:`update_one`.
        """
        if isinstance(features_list, list):
            features = np.concatenate(features_list, axis=0)
            element_indices = np.concatenate(element_indices_list, axis=0)
        else:
            features = np.asarray(features_list)
            element_indices = np.asarray(element_indices_list)

        N = len(features)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            GMMUQArtifactBuilder.update(
                self._artifacts,
                features[start:end],
                element_indices[start:end],
                self.regularization,
                alpha=alpha,
                mode=mode,
            )
        self._refresh_element_rows(np.unique(element_indices))

    def inflate(
        self, features, element_indices, target, max_rounds=3, max_factor=4.0
    ) -> InflateResult:
        """Bounded covariance inflation of the clusters one structure occupies.

        Last resort of the HAL update after the alpha search has bottomed out at
        ``alpha = 1`` (the covariance is then the frame's own scatter and re-applying
        it changes nothing). Repeats at most ``max_rounds`` times while the frame's max
        gamma is ``>= target``: every cluster (of every element in the frame) that at
        least one atom is assigned to gets ``SCATTER *= factor`` with
        ``factor = min(max_factor, (gamma_max / (0.95 * target)) ** 2)``, is
        re-inverted, and the touched element rows of ``inv_covs`` are refreshed
        (never ``_build_tensors``). Centroids, counts and thresholds are untouched.
        Mutates the model, like :meth:`update_one`.

        Returns
        -------
        InflateResult
            ``(n_rounds, factors, gamma_max_after)``; ``gamma_max_after`` is the frame's
            max gamma after the last round (before any round if none was needed).
        """
        if self.interp_thresholds is None:
            raise ValueError(
                "inflate needs calibrated thresholds (interp_thresholds) in the artifact"
            )
        features = np.asarray(features, dtype=np.float64)
        element_indices = np.asarray(element_indices, dtype=np.int32)
        touched = {}
        for e in np.unique(element_indices):
            e = int(e)
            if e not in self._artifacts:  # like update(): absent elements are skipped
                continue
            a = self._artifacts[e]
            if a[uq_constants.SCATTER] is None:
                raise ValueError(f"Scatter matrices missing for element {e}.")
            a[uq_constants.EFFECTIVE_COUNT] = _default_eff_counts(
                a[uq_constants.COUNTS], a.get(uq_constants.EFFECTIVE_COUNT)
            )
            k = nearest_cluster(features[element_indices == e], a[uq_constants.CENTROIDS])
            touched[e] = np.unique(k)
        eps_I = self.regularization * np.eye(self.D, dtype=np.float64)

        gmax = float(np.max(self.gamma_from_features(features, element_indices)[0]))
        factors = []
        while len(factors) < max_rounds and gmax >= target:
            factor = min(float(max_factor), (gmax / (0.95 * target)) ** 2)
            for e, ks in touched.items():
                a = self._artifacts[e]
                for k in ks:
                    a[uq_constants.SCATTER][k] = a[uq_constants.SCATTER][k] * factor
                    GMMUQArtifactBuilder.reinvert_cluster(a, k, eps_I)
            self._refresh_element_rows(list(touched))
            factors.append(factor)
            gmax = float(np.max(self.gamma_from_features(features, element_indices)[0]))
        return InflateResult(len(factors), factors, gmax)

    def reinvert_all(self, elements=None):
        """Re-invert every cluster from its scatter, as an update does, and refresh
        ``inv_covs``. ``elements``: only these element indices (default: all); a HAL
        walker or a selection evaluates only the elements of its structures.

        The stored ``INV_COV`` was inverted at build time and rounded to float32 on
        save; the first update of a cluster replaces it by the float64 inverse of the
        (float32-rounded) scatter, which differs by up to ~1e-4 relative in gamma.
        Calling this once up front puts every cluster on the update's footing, so
        additive updates are monotone in gamma from the start. Elements without
        scatter are left as stored. Mutates the model.
        """
        eps_I = self.regularization * np.eye(self.D, dtype=np.float64)
        wanted = None if elements is None else {int(e) for e in elements}
        done = []
        for e, a in self._artifacts.items():
            if (wanted is not None and e not in wanted) or a.get(uq_constants.SCATTER) is None:
                continue
            a[uq_constants.EFFECTIVE_COUNT] = _default_eff_counts(
                a[uq_constants.COUNTS], a.get(uq_constants.EFFECTIVE_COUNT)
            )
            for k in range(a[uq_constants.CENTROIDS].shape[0]):
                GMMUQArtifactBuilder.reinvert_cluster(a, k, eps_I)
            done.append(e)
        self._refresh_element_rows(done)

    def _refresh_element_rows(self, element_ids):
        """Write the ``inv_covs`` rows of ``element_ids`` from ``self._artifacts``.

        Cheap replacement for ``_build_tensors`` after ``GMMUQArtifactBuilder.update``:
        the update changes only ``INV_COV`` (scatter and counts are not part of the
        dense tensors), and only for the elements present in the update. Centroids,
        ``n_clusters_per_elem`` and ``interp_thresholds`` are never changed by an
        update, so those variables are left alone. Row content is exactly what
        ``_build_tensors`` produces: zeros ``[K_max, D, D]`` (float64), ``[:K]`` set
        from ``INV_COV``, then cast to ``param_dtype``.

        Elements absent from the artifacts are skipped (the builder warns and skips
        them too), so their fictitious isotropic rows are untouched. The variable is
        assigned in place, so already-traced tf.function / XLA graphs see the update.
        """
        elems = sorted({int(e) for e in element_ids} & set(self._artifacts))
        if not elems:
            return
        rows = np.zeros((len(elems), self.K_max, self.D, self.D), dtype=np.float64)
        for i, e in enumerate(elems):
            inv = self._artifacts[e][uq_constants.INV_COV]
            rows[i, : inv.shape[0]] = inv
        self.inv_covs.scatter_nd_update(
            indices=[[e] for e in elems],
            updates=tf.cast(rows, self.param_dtype),
        )

    def _trial_updates(self, features, element_indices, alphas, target, mode):
        """Yield ``(alpha, gmax, sub)`` per alpha: ``sub`` holds deep copies of the
        frame's element artifacts after a hypothetical update with ``alpha``, ``gmax``
        the frame's max gamma under it. Stops after the first ``gmax < target``
        (``target`` None: every alpha). Read-only on the model."""
        if self.interp_thresholds is None:
            raise ValueError(
                "trial_gamma needs calibrated thresholds (interp_thresholds) in the artifact"
            )
        features = np.asarray(features, dtype=np.float64)
        element_indices = np.asarray(element_indices, dtype=np.int32)
        thresholds = np.asarray(self.interp_thresholds.numpy(), dtype=np.float64)
        # elements without UQ training data are skipped, as in update() and inflate()
        present = [int(e) for e in np.unique(element_indices) if int(e) in self._artifacts]
        # the update leaves the centroids fixed: each atom's cluster is found once
        cluster = np.zeros(len(features), dtype=np.int64)
        for e in present:
            m = element_indices == e
            cluster[m] = nearest_cluster(
                features[m], np.asarray(self._artifacts[e][uq_constants.CENTROIDS], dtype=np.float64))
        rows = np.flatnonzero(np.isin(element_indices, present))  # atoms of known elements
        groups = []
        for e, k, m in cluster_groups(element_indices[rows], cluster[rows]):
            c = np.asarray(self._artifacts[e][uq_constants.CENTROIDS], dtype=np.float64)[k]
            groups.append((e, k, features[rows[m]] - c))
        for alpha in alphas:
            sub = {e: copy.deepcopy(self._artifacts[e]) for e in present}
            GMMUQArtifactBuilder.update(
                sub, features, element_indices, self.regularization, alpha=alpha,
                mode=mode,
            )
            gmax = 0.0
            for e, k, delta in groups:
                inv = np.asarray(sub[e][uq_constants.INV_COV][k], dtype=np.float64)
                gmax = max(gmax, float(np.max(sigma(delta, inv) / thresholds[e, k])))
            yield alpha, gmax, sub
            if target is not None and gmax < target:
                return

    def trial_gamma(
        self, features, element_indices, alphas, target=None, mode="convex"
    ) -> np.ndarray:
        """Max gamma of one structure after a hypothetical ``update_one`` per alpha.

        Read-only. For each alpha, the artifact entries of the elements present in
        ``features`` are deep-copied (only those, not the whole model) and updated with
        the same ``GMMUQArtifactBuilder.update`` that :meth:`update_one` uses, so trial
        and real update agree. Centroids are fixed by the update, so each atom keeps its
        cluster. Elements without UQ training data are skipped, as in the update.

        Parameters
        ----------
        target : float or None
            If given, alphas are evaluated in the order passed and evaluation stops
            after the first one whose max gamma is below ``target``; the remaining
            entries are NaN. ``None`` evaluates every alpha.
        mode : {"convex", "additive"}
            Update mode, as in :meth:`update_one`.

        Returns
        -------
        np.ndarray [len(alphas)]
            max over atoms of sigma / threshold after each trial update (NaN for
            alphas not evaluated because ``target`` was already reached).
        """
        out = np.full(len(alphas), np.nan, dtype=np.float64)
        for i, (_, gmax, _) in enumerate(
                self._trial_updates(features, element_indices, alphas, target, mode)):
            out[i] = gmax
        return out

    def alpha_search_update(
        self, features, element_indices, alphas, target, mode="additive"
    ) -> tuple[float, bool]:
        """Update with the first alpha of ``alphas`` whose trial max gamma is below
        ``target``, or with the last alpha if none is; the HAL live update and the
        calm-greedy selection tower both use it.

        The same as :meth:`trial_gamma` followed by :meth:`update_one` with the chosen
        alpha, but the chosen trial state is kept instead of being recomputed.
        Returns ``(alpha, reached)``; ``reached`` is False when even the last alpha left
        the frame at or above ``target`` (HAL then inflates the covariances).
        """
        alpha = gmax = sub = None
        for alpha, gmax, sub in self._trial_updates(
                features, element_indices, alphas, target, mode):
            pass
        if sub is None:
            raise ValueError("alpha_search_update needs at least one alpha")
        self._artifacts.update(sub)
        self._refresh_element_rows(list(sub))
        return float(alpha), bool(gmax < target)

    def save(self, path: str, **kwargs):
        """Save current artifacts (including scatter) to .npz file.
        Extra kwargs are also saved under their keyword name.
        """
        extra = {**self.extra_data, **kwargs}
        GMMUQArtifactBuilder.save_artifacts(path, self._artifacts, **extra)
