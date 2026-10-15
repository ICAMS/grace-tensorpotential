"""NumPy CALM kernel: nearest cluster, Mahalanobis sigma and its whitening.

One implementation for every NumPy path: the artifact build and update
(:class:`~tensorpotential.uq.artifact_builder.GMMUQArtifactBuilder`),
:meth:`GMMUQModel.trial_gamma` / ``alpha_search_update`` / ``inflate``, and the
selection scorers of ``grace_uq select`` (calm-greedy, fps-extrap whitening).
:meth:`GMMUQModel.compute` is the TensorFlow path (differentiable, for the HAL
forces); ``tests/test_uq_calm_numpy.py`` pins the two together.
"""

from __future__ import annotations

import numpy as np

# added to sigma^2 before the square root, as GMMUQModel.compute (eps)
SIGMA_EPS = 1e-8


def nearest_cluster(features, centroids, chunk: int = 8192) -> np.ndarray:
    """Index of the nearest centroid (Euclidean) of every row of ``features``.

    ``centroids`` are one element's real clusters [K, D]; ties go to the lowest index.
    Computed in the inputs' (promoted) dtype, chunked so the [chunk, K, D] temporary
    stays bounded.
    """
    f, c = np.asarray(features), np.asarray(centroids)
    out = np.empty(len(f), dtype=np.int64)
    for s in range(0, len(f), chunk):
        d2 = ((f[s : s + chunk, None, :] - c[None, :, :]) ** 2).sum(-1)
        out[s : s + chunk] = np.argmin(d2, axis=1)
    return out


def cluster_groups(elements, clusters):
    """``(element, cluster, atom indices)`` of every occupied (element, cluster) pair."""
    elements, clusters = np.asarray(elements), np.asarray(clusters)
    for e, k in sorted(set(zip(elements.tolist(), clusters.tolist()))):
        yield e, k, np.flatnonzero((elements == e) & (clusters == k))


def sigma(delta, inv_cov) -> np.ndarray:
    """Mahalanobis distance of the offsets ``delta`` [n, D] from one cluster with
    inverse covariance ``inv_cov`` [D, D]."""
    return np.sqrt(((delta @ inv_cov) * delta).sum(1) + SIGMA_EPS)


def cholesky_psd(a: np.ndarray) -> np.ndarray:
    """Lower factor C with a = C C^T (whitening: |C^T delta| = sigma); eigen-clipped
    fallback for a not-quite-PD matrix."""
    a = 0.5 * (a + a.T)
    try:
        return np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(a)
        return v * np.sqrt(np.clip(w, 0.0, None))
