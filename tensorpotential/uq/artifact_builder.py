"""Out-of-core GMM-UQ artifact generation via two-pass streaming."""

from __future__ import annotations

import logging
import os
import warnings

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm

from tensorpotential.uq import constants as uq_constants

log = logging.getLogger(__name__)


def atomic_savez(path: str, **save_dict) -> None:
    """``np.savez`` that won't leave a corrupt half-written file on crash.

    Why: long pipeline runs that die mid-write left readers staring at a
    truncated ``.npz`` they couldn't tell from a healthy one. Writing to a
    sibling temp file and ``os.replace``-ing it makes the swap atomic on
    POSIX, so the destination is either the old file or the new file —
    never half of either.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    base = os.path.basename(path)
    # numpy auto-appends ``.npz`` to a string path that lacks it; pass an open
    # file handle so the tmp name we write to is the one we then ``os.replace``.
    tmp_path = os.path.join(directory, f".{base}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "wb") as fh:
            np.savez(fh, **save_dict)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def _default_eff_counts(counts: np.ndarray, eff: np.ndarray | None = None) -> np.ndarray:
    """Return ``eff`` cast to float64, or back-fill from ``counts`` when missing.

    Legacy artifacts predate the weighted-build feature and only carry raw
    counts; the back-fill keeps every downstream reader uniform.
    """
    if eff is None:
        return np.asarray(counts, dtype=np.float64)
    return np.asarray(eff, dtype=np.float64)


class GMMUQArtifactBuilder:
    """Two-pass streaming builder for GMM-UQ uncertainty artifacts.

    Pass 1: fit_centroids — streams features through MiniBatchKMeans.partial_fit()
    Pass 2: accumulate_scatter — accumulates per-cluster scatter matrices
    finalize() — inverts covariance matrices, returns artifacts dict

    Parameters
    ----------
    n_clusters : int
        Number of GMM clusters per element type.
    feature_dim : int
        Dimensionality of hidden features.
    regularization : float
        Tikhonov regularization (epsilon*I) added before inversion.
    random_state : int
        Random seed for MiniBatchKMeans.
    """

    def __init__(
        self,
        n_clusters: int = 32,
        feature_dim: int = 129,
        regularization: float = 1e-6,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.regularization = regularization
        self.random_state = random_state

        self._kmeans: dict[int, MiniBatchKMeans] = {}
        self._scatter: dict[int, np.ndarray] = {}  # [K, D, D]
        self._counts: dict[int, np.ndarray] = {}  # [K] int64 — raw atom count
        self._effective_counts: dict[int, np.ndarray] = {}  # [K] float64 — sum of weights
        # Per-cluster atom counts accumulated during step 1 (fit_centroids
        # / accumulate_step1_counts). MiniBatchKMeans does not expose a
        # public per-cluster atom-count attribute, so we count assignments
        # ourselves after each partial_fit. These are the weights consumed
        # by the master's post-fit cluster-merging step.
        self._step1_counts: dict[int, np.ndarray] = {}  # int64, raw
        self._step1_eff_counts: dict[int, np.ndarray] = {}  # float64, weighted
        self._elements_seen: set[int] = set()
        self._fitted = False
        self._finalized = False
        # Tracks whether any non-trivial weight has been observed. Used by
        # finalize() to switch sort key (effective_count desc when weighted,
        # raw count desc otherwise) so unweighted runs stay bit-exact.
        self._weights_seen = False

    def _get_kmeans(self, elem: int) -> MiniBatchKMeans:
        if elem not in self._kmeans:
            self._kmeans[elem] = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                batch_size=max(self.n_clusters * 10, 1024),
            )
            self._elements_seen.add(elem)
        return self._kmeans[elem]

    def accumulate_step1_counts(
        self, elem: int, features: np.ndarray, weights: np.ndarray = None
    ) -> bool:
        """Predict cluster assignments for ``features`` against the current
        KMeans centroids and accumulate per-cluster raw + effective counts.

        ``weights`` is the per-atom weight array (same length as features); if
        None, all atoms get weight 1.0.

        The KMeans centroids drift across mini-batches, so individual
        per-batch predictions use slightly stale centers; the final
        cumulative count is a close approximation of the assignment to the
        final centroids and is dramatically better than the all-ones
        fallback used when MiniBatchKMeans does not expose ``counts_``.

        Returns True if counts were accumulated; False if KMeans is not
        yet initialized for this element.
        """
        km = self._kmeans.get(elem)
        if km is None or not hasattr(km, "cluster_centers_"):
            return False
        if features is None or len(features) == 0:
            return True
        feats = np.asarray(features, dtype=np.float64)
        labels = km.predict(feats)
        bins = np.bincount(labels, minlength=self.n_clusters).astype(np.int64)
        if weights is None:
            eff_bins = bins.astype(np.float64)
        else:
            eff_bins = np.bincount(
                labels, weights=weights.astype(np.float64), minlength=self.n_clusters
            ).astype(np.float64)
        if elem not in self._step1_counts:
            self._step1_counts[elem] = np.zeros(self.n_clusters, dtype=np.int64)
            self._step1_eff_counts[elem] = np.zeros(self.n_clusters, dtype=np.float64)
        self._step1_counts[elem] += bins
        self._step1_eff_counts[elem] += eff_bins
        return True

    def fit_centroids(self, feature_iterator, verbose=True):
        """Pass 1: Stream (features, element_indices, weights) chunks through MiniBatchKMeans.

        The iterator must yield 3-tuples. Weights are per-atom and passed to
        ``MiniBatchKMeans.partial_fit(sample_weight=...)``. To run unweighted,
        supply weights of all ones (the feature iterators in
        ``feature_extraction`` already do this when no weights are attached).
        """
        it = (
            tqdm(feature_iterator, desc="fit_centroids")
            if verbose
            else feature_iterator
        )
        for features, element_indices, weights in it:
            features = np.asarray(features, dtype=np.float64)
            element_indices = np.asarray(element_indices, dtype=np.int32)
            weights = np.asarray(weights, dtype=np.float64)
            batch_unweighted = bool(np.all(weights == 1.0))
            if not self._weights_seen and not batch_unweighted:
                self._weights_seen = True
            unique_elems = np.unique(element_indices)
            for elem in unique_elems:
                elem_int = int(elem)
                mask = element_indices == elem
                elem_feats = features[mask]
                elem_w = None if batch_unweighted else weights[mask]
                km = self._get_kmeans(elem_int)
                fit_ok = False
                if len(elem_feats) >= self.n_clusters:
                    km.partial_fit(elem_feats, sample_weight=elem_w)
                    fit_ok = True
                else:
                    try:
                        km.partial_fit(elem_feats, sample_weight=elem_w)
                        fit_ok = True
                    except ValueError:
                        log.debug(
                            "Skipping partial_fit for element %d: only %d samples",
                            elem_int,
                            len(elem_feats),
                        )
                if fit_ok:
                    self.accumulate_step1_counts(elem_int, elem_feats, elem_w)
        self._fitted = True

    def accumulate_scatter(self, feature_iterator, verbose=True):
        """Pass 2: Assign to nearest centroid, accumulate weighted scatter matrices.

        Iterator must yield 3-tuples ``(features [N,D], element_indices [N],
        weights [N])``. Per-cluster scatter accumulates ``(w · δ)ᵀ δ`` and
        per-cluster effective count accumulates ``Σ w``. The raw atom count is
        also tracked for reliability diagnostics downstream.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_centroids() before accumulate_scatter()")

        D = self.feature_dim
        K = self.n_clusters

        # Initialize scatter accumulators
        for elem in self._elements_seen:
            self._scatter[elem] = np.zeros((K, D, D), dtype=np.float64)
            self._counts[elem] = np.zeros(K, dtype=np.int64)
            self._effective_counts[elem] = np.zeros(K, dtype=np.float64)

        it = (
            tqdm(feature_iterator, desc="accumulate_scatter")
            if verbose
            else feature_iterator
        )
        _warned_missing: set[int] = set()
        for features, element_indices, weights in it:
            features = np.asarray(features, dtype=np.float64)
            element_indices = np.asarray(element_indices, dtype=np.int32)
            weights = np.asarray(weights, dtype=np.float64)
            # Unweighted-batch fast path: skip the per-cluster w·δ multiply
            # (saves a [n_k, D] temporary and one matmul per cluster). Safe to
            # mix with weighted batches because each batch's contribution to
            # _scatter and _effective_counts is independent.
            batch_unweighted = bool(np.all(weights == 1.0))
            if not self._weights_seen and not batch_unweighted:
                self._weights_seen = True
            unique_elems = np.unique(element_indices)

            for elem in unique_elems:
                elem = int(elem)
                mask = element_indices == elem
                elem_feats = features[mask]

                if elem not in self._kmeans:
                    if elem not in _warned_missing:
                        log.warning(
                            "accumulate_scatter: element %d has no centroids in "
                            "step1 artifacts (not seen during Step 1); skipping",
                            elem,
                        )
                        _warned_missing.add(elem)
                    continue
                centroids = self.get_centroids(elem)
                dists = np.sum(
                    (elem_feats[:, None, :] - centroids[None, :, :]) ** 2, axis=-1
                )
                assignments = np.argmin(dists, axis=1)
                elem_w = None if batch_unweighted else weights[mask]

                for k in range(K):
                    k_mask = assignments == k
                    n_k = int(k_mask.sum())
                    if n_k == 0:
                        continue
                    delta = elem_feats[k_mask] - centroids[k]  # [n_k, D]
                    if batch_unweighted:
                        self._scatter[elem][k] += delta.T @ delta
                        self._effective_counts[elem][k] += float(n_k)
                    else:
                        w_k = elem_w[k_mask]
                        self._scatter[elem][k] += (w_k[:, None] * delta).T @ delta
                        self._effective_counts[elem][k] += float(w_k.sum())
                    self._counts[elem][k] += n_k

    def finalize(self, element_names: dict[int, str] = None) -> dict:
        """Compute inverse covariance matrices from accumulated scatter.

        Parameters
        ----------
        element_names : dict[int, str], optional
            Mapping from element index to symbol for descriptive logging.

        Returns
        -------
        dict : {elem_idx: {uq_constants.CENTROIDS: [K,D], uq_constants.INV_COV: [K,D,D], uq_constants.COUNTS: [K]}}
        """
        if not self._fitted:
            raise RuntimeError("Call fit_centroids() and accumulate_scatter() first")

        D = self.feature_dim
        K = self.n_clusters
        eps_I = self.regularization * np.eye(D, dtype=np.float64)

        artifacts = {}
        # Sentinel centroids are padded by the master at value ~1e10 to keep
        # tensor shapes uniform across elements; they intentionally never
        # attract atoms, so a zero-count there is expected and not a warning.
        _SENTINEL_THRESHOLD = 1e9
        for elem in sorted(self._elements_seen):
            centroids = self.get_centroids(elem)  # [K, D]
            counts = self._counts[elem]  # [K] int64 — raw
            eff_counts = _default_eff_counts(counts, self._effective_counts.get(elem))
            scatter = self._scatter[elem]  # [K, D, D]
            sentinel_mask = np.any(np.abs(centroids) > _SENTINEL_THRESHOLD, axis=1)

            # Reorder so cluster 0 is the dominant mode and sentinel slots sit
            # at the end. Sort key is effective_count when any weight has been
            # observed (so the dominant *influence* lands at 0); otherwise raw
            # count, which keeps unweighted artifacts bit-identical to the
            # pre-weighting build.
            if self._weights_seen:
                perm = np.lexsort((-eff_counts, sentinel_mask.astype(np.int8)))
            else:
                perm = np.lexsort((-counts, sentinel_mask.astype(np.int8)))
            centroids = centroids[perm]
            counts = counts[perm]
            eff_counts = eff_counts[perm]
            scatter = scatter[perm]
            sentinel_mask = sentinel_mask[perm]

            inv_covs = np.zeros((K, D, D), dtype=np.float64)
            cond_numbers = np.zeros(K, dtype=np.float64)
            effective_ranks = np.zeros(K, dtype=np.int32)
            n_truncated_arr = np.zeros(K, dtype=np.int32)
            elem_diags = []
            for k in tqdm(range(K), desc=f"finalize elem={elem}", leave=False):
                if eff_counts[k] > 0:
                    cov_data = scatter[k] / eff_counts[k]  # before regularization
                    cov_k = cov_data + eps_I
                else:
                    cov_data = np.zeros_like(eps_I)
                    cov_k = eps_I
                    if not sentinel_mask[k]:
                        log.warning(
                            "Element %d, cluster %d has zero effective count "
                            "(non-sentinel; KMeans assigned no atoms here)",
                            elem,
                            k,
                        )

                inv_k, d_k = self._safe_invert(cov_k)
                inv_covs[k] = inv_k
                cond_numbers[k] = d_k["cond"]
                # Report the data-supported rank (eigenvalues of the
                # unregularized scatter/counts), NOT the post-regularization
                # rank. The latter is almost always equal to D because the
                # eps_I shift floors every eigenvalue to >= eps, hiding the
                # actual span of the training data. The data rank exposes
                # cases like "Pt has 58 atoms but D=65" → rank ≤ 58 < D.
                effective_ranks[k] = self._data_rank(cov_data)
                n_truncated_arr[k] = d_k["n_truncated"]
                elem_diags.append({**d_k, "rank": int(effective_ranks[k])})

                if d_k["cond"] > 1e12:
                    warnings.warn(
                        f"Element {elem}, cluster {k}: condition number {d_k['cond']:.2e} "
                        f"is high even after regularization. Rank: {d_k['rank']}/{D}",
                        stacklevel=2,
                    )
                if d_k["error"] > 1e-5:
                    log.warning(
                        "Element %d, cluster %d: large pinv reconstruction error %.2e",
                        elem, k, d_k["error"]
                    )

            # Log summary for this element — sentinel slots are skipped so the
            # numbers describe only real, data-fitted clusters.
            e_name = f" ({element_names[elem]})" if element_names and elem in element_names else ""
            real_idx = np.flatnonzero(~sentinel_mask)
            if real_idx.size == 0:
                log.info("Element %d%s: no real clusters (all sentinels)", elem, e_name)
            else:
                real_diags = [elem_diags[k] for k in real_idx]
                avg_rank = float(np.mean([d["rank"] for d in real_diags]))
                max_trunc = int(np.max([d["n_truncated"] for d in real_diags]))
                max_cond = float(np.max(cond_numbers[real_idx]))
                k_eff = int(real_idx.size)
                log.info(
                    "Element %d%s: K_eff=%d/%d, avg_rank=%.1f/%d, "
                    "max_truncated=%d, max_cond=%.2e",
                    elem, e_name, k_eff, K, avg_rank, D, max_trunc, max_cond,
                )

            artifacts[elem] = {
                uq_constants.CENTROIDS: centroids,
                uq_constants.INV_COV: inv_covs,
                uq_constants.COUNTS: counts,
                uq_constants.EFFECTIVE_COUNT: eff_counts,
                uq_constants.SCATTER: scatter,
                uq_constants.COND_NUMBER: cond_numbers,
                uq_constants.EFFECTIVE_RANK: effective_ranks,
                uq_constants.N_TRUNCATED: n_truncated_arr,
            }

        self._finalized = True
        return artifacts

    @staticmethod
    def _data_rank(cov_data: np.ndarray, rcond: float = 1e-12) -> int:
        """Effective rank of the *unregularized* sample covariance.

        Uses ``rcond * max_eigenvalue`` as the cutoff (standard NumPy/SciPy
        convention for ``matrix_rank``). For a sample covariance built from
        ``N`` atoms in ``D``-dim feature space, the rank is bounded above by
        ``min(N, D)`` (or ``min(N - 1, D)`` if features were centered around
        their own mean). When ``N < D`` this exposes the data deficit.
        """
        if cov_data.size == 0 or not np.any(cov_data):
            return 0
        w = np.linalg.eigvalsh(cov_data)
        max_w = np.max(np.abs(w))
        if max_w == 0:
            return 0
        threshold = rcond * max_w
        return int(np.sum(w > threshold))

    @staticmethod
    def _safe_invert(cov: np.ndarray, rcond: float = 1e-15):
        """Perform symmetric pseudo-inversion and return diagnostics.

        Parameters
        ----------
        cov : np.ndarray [D, D]
            Symmetric positive semi-definite matrix.
        rcond : float
            Relative condition threshold for pseudo-inversion.

        Returns
        -------
        inv_cov : np.ndarray [D, D]
        diag : dict
            Diagnostic metrics (rank, n_truncated, cond, error).
        """
        # Symmetric eigen-decomposition
        w, v = np.linalg.eigh(cov)
        max_w = np.max(np.abs(w))
        threshold = rcond * max_w

        mask = w > threshold
        rank = np.sum(mask)
        n_truncated = len(w) - rank

        w_inv = np.zeros_like(w)
        w_inv[mask] = 1.0 / w[mask]

        inv_cov = (v * w_inv) @ v.T

        # Condition number of the non-truncated subspace
        cond = max_w / np.min(w[mask]) if rank > 0 else np.inf

        # Reconstruction error: A @ pinv(A) @ A - A
        # Using a subset for speed if D is large, but here D=129 is fine.
        reconstruction = cov @ inv_cov @ cov
        error = np.linalg.norm(reconstruction - cov) / (np.linalg.norm(cov) + 1e-10)

        return inv_cov, {
            "rank": rank,
            "n_truncated": n_truncated,
            "cond": cond,
            "error": error,
        }

    def get_centroids(self, elem: int) -> np.ndarray:
        """Retrieve centroids for a given element from the KMeans object or direct storage."""
        if elem in self._kmeans:
            km = self._kmeans[elem]
            if hasattr(km, "cluster_centers_"):
                return km.cluster_centers_
            if isinstance(km, np.ndarray):
                return km
        raise KeyError(f"No centroids found for element {elem}")

    def set_centroids(self, elem: int, centroids: np.ndarray):
        """Manually set centroids for an element, bypassing fit_centroids."""
        self._kmeans[elem] = np.asarray(centroids)
        self._elements_seen.add(elem)
        self._fitted = True

    def export_kmeans_results(self) -> dict:
        """Return per-element centroids, raw counts, and effective counts.

        Per-cluster atom counts come from ``self._step1_counts`` (raw, int64)
        and ``self._step1_eff_counts`` (weighted, float64), both populated by
        ``accumulate_step1_counts``. When the raw accumulator is empty for an
        element (e.g. partial_fit never succeeded), both fall back to
        ``np.ones(K)`` so downstream code does not crash, with a warning.

        Returns
        -------
        dict : {elem_idx: {CENTROIDS, COUNTS, EFFECTIVE_COUNT}}
        """
        results = {}
        for elem, km in self._kmeans.items():
            try:
                centroids = self.get_centroids(elem)
            except KeyError:
                log.warning(
                    "export_kmeans_results: element %d skipped — KMeans object exists "
                    "but cluster_centers_ was never set (partial_fit never succeeded "
                    "for this element)",
                    elem,
                )
                continue
            counts = self._step1_counts.get(elem)
            eff_counts = self._step1_eff_counts.get(elem)
            if counts is None or counts.sum() == 0:
                log.warning(
                    "export_kmeans_results: element %d has no accumulated "
                    "per-cluster counts (counts.sum=%s); writing all-ones "
                    "fallback. Master-side per-cluster floor checks will "
                    "be unreliable for this element.",
                    elem,
                    "0" if counts is None else int(counts.sum()),
                )
                counts = np.ones(self.n_clusters, dtype=np.int64)
                eff_counts = np.ones(self.n_clusters, dtype=np.float64)
            eff_counts = _default_eff_counts(counts, eff_counts)
            results[elem] = {
                uq_constants.CENTROIDS: centroids,
                uq_constants.COUNTS: counts,
                uq_constants.EFFECTIVE_COUNT: eff_counts,
            }
        return results

    @property
    def elements_seen(self) -> set[int]:
        """Set of element indices seen during fitting."""
        return set(self._elements_seen)

    def set_scatter(
        self,
        elem: int,
        scatter: np.ndarray,
        counts: np.ndarray,
        effective_counts: np.ndarray = None,
    ):
        """Set scatter matrix and counts for an element directly.

        ``effective_counts`` is the weighted per-cluster total (Σ weight). When
        omitted the effective counts default to ``counts.astype(float64)``,
        which is the right back-compat behavior for unweighted artifacts.
        """
        self._scatter[elem] = scatter
        self._counts[elem] = counts
        self._effective_counts[elem] = _default_eff_counts(counts, effective_counts)
        if effective_counts is not None:
            if not self._weights_seen and not np.array_equal(
                self._effective_counts[elem], self._counts[elem].astype(np.float64)
            ):
                self._weights_seen = True

    @classmethod
    def from_artifacts(
        cls,
        artifacts: dict,
        n_clusters: int,
        feature_dim: int = None,
        regularization: float = 1e-6,
    ):
        """Create a builder pre-loaded with centroids from artifacts.

        Parameters
        ----------
        artifacts : dict
            {elem_idx: {CENTROIDS, INV_COV, COUNTS, SCATTER}} from load().
        n_clusters : int
        feature_dim : int, optional
            Inferred from centroids if not given.
        regularization : float
            Tikhonov epsilon used in finalize() (eps*I added to covariance
            before inversion). Default: 1e-6.
        """
        if feature_dim is None:
            feature_dim = next(iter(artifacts.values()))[uq_constants.CENTROIDS].shape[1]
        builder = cls(
            n_clusters=n_clusters,
            feature_dim=feature_dim,
            regularization=regularization,
        )
        for elem, data in artifacts.items():
            builder.set_centroids(elem, data[uq_constants.CENTROIDS])
            # Pre-load effective counts when present so a downstream finalize()
            # can use them as the covariance divisor; otherwise default to raw.
            eff = data.get(uq_constants.EFFECTIVE_COUNT)
            if eff is not None:
                builder._effective_counts[elem] = np.asarray(eff, dtype=np.float64)
                if not np.array_equal(
                    eff, data[uq_constants.COUNTS].astype(np.float64)
                ):
                    builder._weights_seen = True
        return builder

    @staticmethod
    def save_artifacts(path: str, artifacts: dict, store_fp32: bool = True, **kwargs):
        """Save an artifacts dict to .npz file.

        Parameters
        ----------
        path : str
            Output file path.
        artifacts : dict
            {elem_idx: {CENTROIDS, INV_COV, COUNTS, SCATTER}} dict.
        store_fp32 : bool
            Cast float64 arrays to float32 before writing (default True, the
            final-artifact behaviour). Set False for intermediate build saves
            whose ``scatter`` is later inverted in float64 — casting those would
            defeat the stable inversion.
        **kwargs : dict
            Extra arrays to include (e.g. interp_thresholds, element_map).
        """
        save_dict = dict(kwargs)
        # Schema version == uqv feature-variant tag, derived from the basis-RP spec
        # flags that the caller splatted in (absent flag => off / linear). uqv6
        # (normalize+density) => 6, uqv4 (asinh) => 4, plain linear => 3.
        def _flag(key):
            v = save_dict.get(key)
            return bool(np.asarray(v).reshape(-1)[0]) if v is not None else False

        _transform = save_dict.get(uq_constants.UQ_FEATURE_TRANSFORM)
        if _transform is not None:
            _transform = str(np.asarray(_transform).reshape(-1)[0])
        save_dict[uq_constants.SCHEMA_VERSION_KEY] = np.int64(
            uq_constants.schema_version_for_spec(
                normalize=_flag(uq_constants.UQ_RP_NORMALIZE),
                add_density_channel=_flag(uq_constants.UQ_RP_DENSITY),
                transform=_transform,
            )
        )
        save_dict["elements"] = np.array(sorted(artifacts.keys()), dtype=np.int32)
        for elem, data in artifacts.items():
            save_dict[f"{uq_constants.CENTROIDS}_{elem}"] = data[uq_constants.CENTROIDS]
            if data.get(uq_constants.INV_COV) is not None:
                save_dict[f"{uq_constants.INV_COV}_{elem}"] = data[uq_constants.INV_COV]
            save_dict[f"{uq_constants.COUNTS}_{elem}"] = data[uq_constants.COUNTS]
            if (
                uq_constants.EFFECTIVE_COUNT in data
                and data[uq_constants.EFFECTIVE_COUNT] is not None
            ):
                save_dict[f"{uq_constants.EFFECTIVE_COUNT}_{elem}"] = data[
                    uq_constants.EFFECTIVE_COUNT
                ]
            if uq_constants.SCATTER in data and data[uq_constants.SCATTER] is not None:
                save_dict[f"{uq_constants.SCATTER}_{elem}"] = data[uq_constants.SCATTER]
            # Covariance diagnostics (optional)
            for diag_key in (uq_constants.COND_NUMBER, uq_constants.EFFECTIVE_RANK, uq_constants.N_TRUNCATED):
                if diag_key in data and data[diag_key] is not None:
                    save_dict[f"{diag_key}_{elem}"] = data[diag_key]
        # fp32-only storage (schema v3): GMM stats are computed in float64 for a
        # stable covariance inversion but stored float32. The model runs float32,
        # so GMMUQModel casts every stat down to float32 at load anyway — storing
        # float32 just does the same f64->f32 rounding at save time and halves the
        # artifact (validated lossless: gamma and all UQ metrics unchanged).
        # Intermediate build saves pass store_fp32=False to keep scatter float64
        # for the downstream inversion.
        #
        # Counts (raw + effective) are exempt: they are tiny per-element
        # bookkeeping for incremental updates and cluster sorting that must
        # round-trip exactly, and float32 cannot represent large counts losslessly.
        if store_fp32:
            count_prefixes = (
                f"{uq_constants.COUNTS}_",
                f"{uq_constants.EFFECTIVE_COUNT}_",
            )
            save_dict = {
                k: (
                    v.astype(np.float32)
                    if (
                        isinstance(v, np.ndarray)
                        and v.dtype == np.float64
                        and not k.startswith(count_prefixes)
                    )
                    else v
                )
                for k, v in save_dict.items()
            }
        atomic_savez(path, **save_dict)

    def save(self, path: str, artifacts: dict = None, element_names: dict[int, str] = None, **kwargs):
        """Save artifacts to .npz file.

        Parameters
        ----------
        path : str
            Output file path.
        artifacts : dict, optional
            Artifacts dict from finalize(). If None, calls finalize().
        element_names : dict[int, str], optional
            Mapping from element index to symbol for descriptive logging.
        **kwargs : dict
            Extra arrays or structures to save in the .npz file (e.g. thresholds).
        """
        if artifacts is None:
            artifacts = self.finalize(element_names=element_names)
        self.save_artifacts(path, artifacts, **kwargs)

    @staticmethod
    def _validate_schema_version(data, path: str | None = None) -> None:
        """Refuse to load artifacts that predate the current schema or sit at
        an unknown future version. Old artifacts lack the fields new loaders
        depend on (the basis-RP projection spec, effective-weight thresholds);
        silently filling with NaN/Inf hid this in the past.
        """
        where = f" in {path}" if path else ""
        if uq_constants.SCHEMA_VERSION_KEY not in data.files:
            raise ValueError(
                f"UQ artifact{where} predates schema versioning (no "
                f"'{uq_constants.SCHEMA_VERSION_KEY}' field). Rebuild with the "
                f"current `grace_uq build` to upgrade to schema v"
                f"{uq_constants.SCHEMA_VERSION}."
            )
        found = int(data[uq_constants.SCHEMA_VERSION_KEY])
        if found not in uq_constants.SUPPORTED_SCHEMA_VERSIONS:
            supported = ", ".join(
                f"v{v}" for v in sorted(uq_constants.SUPPORTED_SCHEMA_VERSIONS)
            )
            raise ValueError(
                f"UQ artifact{where} schema v{found} is not supported "
                f"(loader supports {supported}). Rebuild with the current "
                f"`grace_uq build`."
            )

    @staticmethod
    def _parse_npz(data, path: str | None = None) -> dict:
        """Parse artifact arrays from an already-opened NpzFile object.

        Parameters
        ----------
        data : np.lib.npyio.NpzFile
            Opened npz data (from np.load).
        path : str, optional
            Source path, included in the error message when schema validation
            fails (callers that loaded from disk should pass it; in-memory
            callers can omit).

        Returns
        -------
        dict : {elem_idx: {CENTROIDS, INV_COV, COUNTS, SCATTER}}
        """
        GMMUQArtifactBuilder._validate_schema_version(data, path=path)
        elements = data["elements"]
        artifacts = {}
        for elem in elements:
            elem = int(elem)
            counts = data[f"{uq_constants.COUNTS}_{elem}"]
            eff_key = f"{uq_constants.EFFECTIVE_COUNT}_{elem}"
            eff = _default_eff_counts(counts, data[eff_key] if eff_key in data else None)
            inv_cov_key = f"{uq_constants.INV_COV}_{elem}"
            entry = {
                uq_constants.CENTROIDS: data[f"{uq_constants.CENTROIDS}_{elem}"],
                uq_constants.INV_COV: data[inv_cov_key] if inv_cov_key in data else None,
                uq_constants.COUNTS: counts,
                uq_constants.EFFECTIVE_COUNT: eff,
                uq_constants.SCATTER: data[f"{uq_constants.SCATTER}_{elem}"]
                if f"{uq_constants.SCATTER}_{elem}" in data
                else None,
            }
            # Covariance diagnostics (may be absent in older artifacts)
            for diag_key in (uq_constants.COND_NUMBER, uq_constants.EFFECTIVE_RANK, uq_constants.N_TRUNCATED):
                key = f"{diag_key}_{elem}"
                entry[diag_key] = data[key] if key in data else None
            artifacts[elem] = entry
        return artifacts

    @staticmethod
    def load(path: str) -> dict:
        """Load artifacts from .npz file.

        Returns
        -------
        dict : {elem_idx: {uq_constants.CENTROIDS: [K,D], uq_constants.INV_COV: [K,D,D], uq_constants.COUNTS: [K]}}
        """
        with np.load(path) as data:
            return GMMUQArtifactBuilder._parse_npz(data, path=path)

    @staticmethod
    def update(
        artifacts,
        features,
        element_indices,
        regularization=1e-6,
        alpha=None,
        weights=None,
    ):
        """Incrementally update artifacts with new data. Centroids stay fixed.

        Parameters
        ----------
        artifacts : dict
            Artifacts dict from finalize()/load(). Must contain "scatter".
        features : np.ndarray [N, D]
            New feature vectors.
        element_indices : np.ndarray [N]
            Element type index per atom.
        regularization : float
            Tikhonov regularization for covariance inversion.
        alpha : float or None
            EMA blending factor in [0, 1]. When specified, the covariance is
            blended: ``C_new = (1 - alpha) * C_old + alpha * C_batch``.
            When None (default), uses accumulative mode (original behavior).
        weights : np.ndarray [N], optional
            Per-atom weights. Defaults to all-ones.

        Returns
        -------
        artifacts : dict
            Updated in-place and returned for convenience.
        """
        features = np.asarray(features, dtype=np.float64)
        element_indices = np.asarray(element_indices, dtype=np.int32)
        if weights is None:
            weights = np.ones(features.shape[0], dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
        unique_elems = np.unique(element_indices)
        D = features.shape[1]
        eps_I = regularization * np.eye(D, dtype=np.float64)

        for elem in unique_elems:
            elem = int(elem)
            if elem not in artifacts:
                log.warning("Element %d not in artifacts, skipping", elem)
                continue
            a = artifacts[elem]
            if a[uq_constants.SCATTER] is None:
                raise ValueError(
                    f"Scatter matrices missing for element {elem}. "
                    "Rebuild artifacts with latest code to include scatter."
                )
            a[uq_constants.EFFECTIVE_COUNT] = _default_eff_counts(
                a[uq_constants.COUNTS], a.get(uq_constants.EFFECTIVE_COUNT)
            )

            mask = element_indices == elem
            elem_feats = features[mask]
            elem_w = weights[mask]
            centroids = a[uq_constants.CENTROIDS]
            K = centroids.shape[0]

            # Assign to nearest centroid (Euclidean)
            dists = np.sum(
                (elem_feats[:, None, :] - centroids[None, :, :]) ** 2, axis=-1
            )
            assignments = np.argmin(dists, axis=1)

            # Update scatter and counts per cluster
            for k in range(K):
                k_mask = assignments == k
                if not np.any(k_mask):
                    continue
                delta = elem_feats[k_mask] - centroids[k]
                w_k = elem_w[k_mask]
                w_batch = float(w_k.sum())
                weighted_delta = w_k[:, None] * delta
                if alpha is not None:
                    # EMA blend: C_new = (1-alpha)*C_old + alpha*C_batch.
                    # Store SCATTER as (weighted_delta).T @ delta scaled by the
                    # total effective count so the downstream recompute
                    # ``cov_k = SCATTER / EFFECTIVE_COUNT`` yields cov_k = cov_new.
                    n_batch = int(k_mask.sum())
                    n_old = int(a[uq_constants.COUNTS][k])
                    w_old = float(a[uq_constants.EFFECTIVE_COUNT][k])
                    cov_old = (
                        a[uq_constants.SCATTER][k] / w_old
                        if w_old > 0
                        else np.zeros_like(a[uq_constants.SCATTER][k])
                    )
                    cov_batch = (
                        (weighted_delta.T @ delta) / w_batch
                        if w_batch > 0
                        else np.zeros_like(a[uq_constants.SCATTER][k])
                    )
                    cov_new = (1 - alpha) * cov_old + alpha * cov_batch
                    n_total = n_old + n_batch
                    w_total = w_old + w_batch
                    a[uq_constants.SCATTER][k] = cov_new * w_total
                    a[uq_constants.COUNTS][k] = n_total
                    a[uq_constants.EFFECTIVE_COUNT][k] = w_total
                else:
                    # Accumulative (original behavior)
                    a[uq_constants.SCATTER][k] += weighted_delta.T @ delta
                    a[uq_constants.COUNTS][k] += k_mask.sum()
                    a[uq_constants.EFFECTIVE_COUNT][k] += w_batch

            # Recompute inv_cov for all clusters of this element
            for k in range(K):
                if a[uq_constants.EFFECTIVE_COUNT][k] > 0:
                    cov_k = (
                        a[uq_constants.SCATTER][k] / a[uq_constants.EFFECTIVE_COUNT][k]
                        + eps_I
                    )
                else:
                    cov_k = eps_I
                # Use pseudoinverse for better stability
                inv_k, _ = GMMUQArtifactBuilder._safe_invert(cov_k)
                a[uq_constants.INV_COV][k] = inv_k

        return artifacts
