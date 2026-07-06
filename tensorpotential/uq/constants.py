from __future__ import annotations

import os
from typing import Final, Tuple

# Model outputs
FEATURES: Final[str] = "features"
ATOMIC_SIGMA: Final[str] = "atomic_sigma"
TOTAL_SIGMA: Final[str] = "total_sigma"
DSIGMA_DR: Final[str] = "dsigma_dr"
DSIGMA_DR_PAIR: Final[str] = "dsigma_dr_pair"
VIRIAL_SIGMA: Final[str] = "virial_sigma"
ATOMIC_GAMMA: Final[str] = "gamma"
GMM_CLUSTER: Final[str] = "gmm_cluster"

# Runtime UQ modes the calculator can switch between, and the
# SavedModel signature name each mode is exported under.
UQ_MODE_FULL: Final[str] = "full"
UQ_MODE_GAMMA_ONLY: Final[str] = "gamma_only"
UQ_MODE_TO_SIGNATURE: Final[dict] = {
    UQ_MODE_FULL: "compute_uq",
    UQ_MODE_GAMMA_ONLY: "compute_uq_gamma_only",
}

# Standard UQ output keys exposed via TPCalculator.extra_properties.
# ATOMIC_GAMMA is included only when GMM thresholds are available.
UQ_EXTRA_KEYS: Final[Tuple[str, ...]] = (
    ATOMIC_SIGMA,
    TOTAL_SIGMA,
    DSIGMA_DR,
    VIRIAL_SIGMA,
    FEATURES,
    ATOMIC_GAMMA,
    GMM_CLUSTER,
)
# Subset that should be sliced down to nreal_atoms when truncating.
UQ_TRUNCATE_KEYS: Final[Tuple[str, ...]] = (
    ATOMIC_SIGMA,
    DSIGMA_DR,
    FEATURES,
    ATOMIC_GAMMA,
    GMM_CLUSTER,
)


# Artifact .npz schema version. The number doubles as the UQ feature-variant tag
# ("uqv<N>") so a loader can both gate compatibility AND know which feature to
# rebuild from the spec keys (UQ_RP_* below). Old artifacts that predate this
# gate fail to load with a clear message rather than silently falling back to
# NaN/Inf-filled diagnostics.
#
# All variants share the basis-RP layout: verbatim projection matrix + spec keys,
# GMM stats stored float32 (the model runs float32 so inference casts down anyway
# → ~half the size, lossless). The retired hidden-layer feature lived at v2 and is
# unsupported; the per-cluster force-error model (a_k + c_k·σ) and per-element τ_e
# quantile were dropped — gamma (σ / threshold) is the sole UQ signal.
#
#   v3 = linear basis-RP        proj = B @ R                      [D == rp_dim]
#   v4 = asinh basis-RP         proj = asinh(B) @ R               [D == rp_dim]
#   v6 = normalized + density   (B/‖B‖) @ R ++ log-norm density   [D == rp_dim + n_density]
#
# The version is derived at SAVE time from the feature flags (schema_version_for_spec);
# loaders accept any SUPPORTED_SCHEMA_VERSIONS and reconstruct the exact feature from
# the spec keys. The schema number is the compatibility gate; the flags are the variant.
SCHEMA_VERSION_KEY: Final[str] = "__schema_version__"
SCHEMA_VERSION_LINEAR: Final[int] = 3  # uqv3
SCHEMA_VERSION_ASINH: Final[int] = 4  # uqv4
SCHEMA_VERSION_NORM_DENSITY: Final[int] = 6  # uqv6
SUPPORTED_SCHEMA_VERSIONS: Final[frozenset] = frozenset(
    {SCHEMA_VERSION_LINEAR, SCHEMA_VERSION_ASINH, SCHEMA_VERSION_NORM_DENSITY}
)
# Latest writer version (uqv6 is production) — used only for messaging / fallbacks.
SCHEMA_VERSION: Final[int] = SCHEMA_VERSION_NORM_DENSITY


def schema_version_for_spec(
    normalize: bool, add_density_channel: bool, transform
) -> int:
    """Map a basis-RP feature spec to its artifact schema version (== uqv number).

    normalize / density (uqv6) takes precedence over an element-wise transform;
    a bare transform (asinh) is uqv4; neither set is the linear uqv3.
    """
    if normalize or add_density_channel:
        return SCHEMA_VERSION_NORM_DENSITY
    if transform not in (None, "", "linear", "none"):
        return SCHEMA_VERSION_ASINH
    return SCHEMA_VERSION_LINEAR

# basis-RP feature spec, stored verbatim in the artifact (self-describing eval /
# export — no env var). UQ_RP_MATRIX is the seeded Johnson-Lindenstrauss matrix
# R[D_basis, UQ_RP_DIM] applied to the concatenated invariant energy-path basis.
UQ_FEATURE_MODE: Final[str] = "uq_feature_mode"
UQ_RP_MATRIX: Final[str] = "uq_rp_matrix"
UQ_RP_DIM: Final[str] = "uq_rp_dim"
UQ_RP_SEED: Final[str] = "uq_rp_seed"
FEATURE_MODE_BASIS_RP: Final[str] = "basis_rp"
# Non-linear transform applied element-wise to the concatenated invariant basis
# before the random projection (``proj = T(basis) @ R``). "asinh" for current
# builds; absent == legacy linear feature (T = identity) for pre-asinh artifacts.
# Stored verbatim so eval / SavedModel reproduce the exact feature.
UQ_FEATURE_TRANSFORM: Final[str] = "uq_feature_transform"
# uqv6 feature options, stored verbatim so eval / SavedModel reproduce the feature.
# UQ_RP_NORMALIZE: L2-normalize the concatenated basis (per atom) before the
# transform + projection — the bounded unit-direction restores stretch/under-
# coordination sensitivity and keeps the covariance well-conditioned. UQ_RP_DENSITY:
# append log-norm "density" channels (full + per-reduce-block) after the projection
# so the magnitude axis (collapsed by normalize) re-enters as explicit coordinates,
# flagging both compression and stretch. UQ_RP_DENSITY_SCALE: fixed multiplier on
# those log-norm channels (balances their spread against the projected dims for
# KMeans; the per-element Mahalanobis is scale-invariant either way).
UQ_RP_NORMALIZE: Final[str] = "uq_rp_normalize"
UQ_RP_DENSITY: Final[str] = "uq_rp_add_density_channel"
UQ_RP_DENSITY_SCALE: Final[str] = "uq_rp_density_scale"
# Production basis-RP transform: the single source of truth the build's feature
# extraction (workers) and the stamped artifact must agree on. The production default
# is uqv6 — L2-normalized direction `(B/‖B‖)@R` plus log-norm density channels (always
# on; the build no longer exposes per-feature toggles) with no element-wise transform,
# so the default transform is None. Set GRACE_UQ_FEATURE_TRANSFORM=asinh for the uqv4
# feature; "none"/"linear"/"" keep the linear basis.
_FT_ENV = os.environ.get("GRACE_UQ_FEATURE_TRANSFORM", "none")
DEFAULT_FEATURE_TRANSFORM = None if _FT_ENV.lower() in ("none", "linear", "") else _FT_ENV

# Production uqv6 basis-RP feature options that `grace_uq build` stamps into every new
# artifact. The CLI no longer exposes per-build toggles, so these fix what the build
# (master stamp + workers) applies during feature extraction. The factory-level
# defaults (factories.py / instructions.py) stay False/False so the eval/read path
# still reproduces legacy (uqv3/uqv4) artifacts built before these channels existed.
UQ_DEFAULT_NORMALIZE: Final[bool] = True
UQ_DEFAULT_DENSITY: Final[bool] = True
UQ_DEFAULT_DENSITY_SCALE: Final[float] = 1.0

# Artifacts keys
CENTROIDS: Final[str] = "centroids"
INV_COV: Final[str] = "inv_cov"
COUNTS: Final[str] = "counts"
EFFECTIVE_COUNT: Final[str] = "effective_count"
SCATTER: Final[str] = "scatter"

# Covariance diagnostics (per-cluster arrays stored in artifacts)
COND_NUMBER: Final[str] = "cond_number"
EFFECTIVE_RANK: Final[str] = "effective_rank"
N_TRUNCATED: Final[str] = "n_truncated"

# ASE atoms.info key carrying the per-structure UQ weight from stream_atoms.
# Read by PerAtomWeightBuilder; broadcast to a per-atom array in each batch.
UQ_WEIGHT: Final[str] = "uq_weight"
