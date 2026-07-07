"""TF-differentiable GMM-UQ forward pass for uncertainty quantification."""

from __future__ import annotations

import logging
import warnings
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from tensorpotential.uq.artifact_builder import GMMUQArtifactBuilder
from tensorpotential.uq import constants as uq_constants

log = logging.getLogger(__name__)

# inv_cov diagonal value used for elements with no training data.
# Mahalanobis distance = sqrt(delta^T @ inv_cov @ delta) ≈ sqrt(scale) * ||delta||,
# so scale=1e6 gives sigma ~1000x the feature norm — reliably flags these atoms as
# high-uncertainty regardless of the actual feature value.
_MISSING_ELEM_INV_COV_DIAG = 1e6


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
        obj.chunk_size = chunk_size
        obj.regularization = regularization
        object.__setattr__(obj, "_artifacts", artifacts)
        object.__setattr__(obj, "extra_data", extra_data or {})
        if param_dtype is None:
            first_elem = next(iter(artifacts.values()))
            param_dtype = tf.as_dtype(first_elem[uq_constants.CENTROIDS].dtype)
        obj.param_dtype = param_dtype
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

        Schema-v2 artifacts always carry ``interp_thresholds``; older artifacts
        are rejected at load time, so the missing-thresholds fallback is gone.

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

    def update_one(self, features, element_indices, alpha=None):
        """Incrementally update artifacts with features from one structure.

        Centroids stay fixed. Scatter matrices and counts are updated,
        then inv_cov is recomputed and TF tensors are rebuilt.

        Parameters
        ----------
        features : np.ndarray [N_atoms, D]
            Feature vectors for atoms in the new structure.
        element_indices : np.ndarray [N_atoms]
            Element type index per atom.
        alpha : float or None
            EMA blending factor. ``C_new = (1-alpha)*C_old + alpha*C_batch``.
            None (default) uses accumulative mode.
        """
        GMMUQArtifactBuilder.update(
            self._artifacts,
            features,
            element_indices,
            self.regularization,
            alpha=alpha,
        )
        self._build_tensors(self._artifacts)

    def update_many(
        self, features_list, element_indices_list, chunk_size=4096, alpha=None
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
            EMA blending factor. ``C_new = (1-alpha)*C_old + alpha*C_batch``.
            None (default) uses accumulative mode.
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
            )
        self._build_tensors(self._artifacts)

    def save(self, path: str, **kwargs):
        """Save current artifacts (including scatter) to .npz file.
        Extra kwargs are also saved under their keyword name.
        """
        extra = {**self.extra_data, **kwargs}
        GMMUQArtifactBuilder.save_artifacts(path, self._artifacts, **extra)
