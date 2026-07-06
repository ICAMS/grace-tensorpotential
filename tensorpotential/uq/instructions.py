"""UQ-specific computational-graph instructions.

Currently hosts the basis random-projection (basis-RP) feature used by the
GMM-UQ pipeline:

* :func:`generate_rp_matrix` — the single source of truth for the seeded
  Johnson-Lindenstrauss projection matrix shared across build workers, the
  master artifact stamp, and eval/SavedModel.
* :class:`RandomProjectedBasisFeatures` — a :class:`TPInstruction` that
  projects the rotation-invariant (l=0) B-basis onto that fixed random
  subspace to produce the per-atom UQ feature.

These were originally defined in :mod:`tensorpotential.tpmodel`; they live
here so the UQ feature machinery is colocated with the rest of the UQ code.
:mod:`tensorpotential.tpmodel` re-exports both names for backward compatibility
with serialized artifacts that stored the old class path.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import tensorflow as tf

from tensorpotential.instructions.base import capture_init_args
from tensorpotential.instructions.compute import FunctionReduceN, TPInstruction


def generate_rp_matrix(d_basis: int, out_dim: int, seed: int) -> np.ndarray:
    """Deterministic Johnson-Lindenstrauss projection ``R`` of shape ``[d_basis, out_dim]``.

    ``R = N(0, 1) / sqrt(out_dim)`` drawn from a seeded ``np.random.default_rng``.
    This is the single source of truth for the basis-RP matrix: build workers
    (:meth:`RandomProjectedBasisFeatures.build`) and the master artifact stamp
    (:func:`tensorpotential.uq.factories.make_basis_rp_spec`) both call it with
    the same ``(d_basis, out_dim, seed)`` so the matrix is byte-identical
    everywhere and eval/SavedModel reproduce the exact feature.
    """
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((d_basis, out_dim)) / np.sqrt(out_dim)
    # Experimental row-mask hook (uqv5): if GRACE_UQ_RP_ZERO_ROWS points to a .npy
    # of integer basis-row indices, those rows of R are zeroed (== dropping those
    # basis functions from the projection). Inert when unset, so other builds are
    # unaffected. Propagates to build worker subprocesses via the inherited env and
    # is stored verbatim in the artifact, so eval/SavedModel reproduce the masked R.
    _zero_rows = os.environ.get("GRACE_UQ_RP_ZERO_ROWS")
    if _zero_rows:
        idx = np.unique(np.load(_zero_rows).astype(int))
        if idx.min() < 0 or idx.max() >= d_basis:
            raise ValueError(
                f"GRACE_UQ_RP_ZERO_ROWS indices out of range for d_basis={d_basis} "
                f"(min={idx.min()}, max={idx.max()})"
            )
        R[idx, :] = 0.0
        logging.info(
            "generate_rp_matrix: zeroed %d / %d R rows from %s",
            len(idx), d_basis, _zero_rows,
        )
    return R


@capture_init_args
class RandomProjectedBasisFeatures(TPInstruction):
    """Project the rotation-invariant (l=0) B-basis onto a fixed random subspace.

    Concatenates the invariant basis functions that enter one or more *scalar*
    ``FunctionReduceN`` reduces (``ls_max == [0, ...]``) and multiplies the
    concatenation by a fixed, seeded Johnson-Lindenstrauss matrix
    ``R \\in R^{D_basis x out_dim}`` (``R = N(0,1) / sqrt(out_dim)``). The result
    is an ``out_dim``-dimensional, rotationally-invariant, per-atom feature that
    is returned (and stored by ``TPInstruction.__call__`` under this instruction's
    ``name`` == ``uq_constants.FEATURES`` "features") for the UQ pipeline to consume
    in place of the readout hidden layer.

    The basis extraction reuses :func:`tensorpotential.tpmodel.extract_basis_functions`;
    passing the full list of scalar reduces (e.g. the 1L and 2L invariant reduces)
    concatenates both. ``R`` is a non-trainable constant — either supplied verbatim
    via ``projection_matrix`` (the byte-identical matrix shared across build workers
    and stored in the artifact) or generated deterministically from ``seed`` and
    the data-free basis dimension.

    Two optional "uqv6" knobs decouple the magnitude and direction of the basis,
    so the GMM-UQ flags BOTH compression and stretch (the linear/asinh basis-RP
    feature is one-sided — sensitive to compression but blind to under-coordination):

    * ``normalize`` — L2-normalize the concatenated basis per atom before the
      transform + projection. The projected unit-direction is bounded (no heavy
      magnitude tail → well-conditioned covariance) and swings to an extreme
      direction as bonds leave the cutoff, restoring stretch sensitivity.
    * ``add_density_channel`` — append ``density_scale * log(||·|| + eps)`` channels
      (full basis + one per reduce block) AFTER the projection, re-introducing the
      magnitude axis that ``normalize`` removes as explicit monotone coordinates
      (large under compression, → -inf under stretch). Output width becomes
      ``out_dim + 1 + n_reduce_blocks``. ``density_scale`` only balances the
      channels' spread against the projected dims for KMeans; the per-element
      Mahalanobis σ is invariant to it.
    """

    # Allowed values for ``feature_transform`` (applied element-wise to the
    # concatenated basis before the random projection). ``"asinh"`` is the default
    # for new builds: it de-collapses the GMM covariance when a minority of atoms
    # blow up the raw 2L invariant basis (a fixed linear scale folded into R cannot
    # — a linear map preserves the rank-1 domination). ``None`` is the legacy linear
    # feature, kept only so pre-asinh artifacts (which carry no transform key)
    # reproduce their exact feature at load time.
    _FEATURE_TRANSFORMS = (None, "asinh")

    def __init__(
        self,
        basis_reduce_instructions,
        out_dim: int = 128,
        seed: int = 42,
        projection_matrix=None,
        feature_transform: str | None = None,
        normalize: bool = False,
        add_density_channel: bool = False,
        density_scale: float = 1.0,
        name: str = "features",
    ):
        super().__init__(name=name)
        if isinstance(basis_reduce_instructions, FunctionReduceN):
            basis_reduce_instructions = [basis_reduce_instructions]
        self.basis_reduce_instructions = list(basis_reduce_instructions)
        self.out_dim = int(out_dim)
        self.seed = int(seed)
        if feature_transform not in self._FEATURE_TRANSFORMS:
            raise ValueError(
                f"feature_transform must be one of {self._FEATURE_TRANSFORMS}; "
                f"got {feature_transform!r}"
            )
        self.feature_transform = feature_transform
        # uqv6 options (see class docstring). Both default off → the legacy/asinh
        # basis-RP feature is unchanged.
        self.normalize = bool(normalize)
        self.add_density_channel = bool(add_density_channel)
        self.density_scale = float(density_scale)
        self._eps = 1e-12
        # Keep an explicit matrix (numpy) out of the way of TF Module tracking.
        object.__setattr__(
            self,
            "_projection_matrix",
            None if projection_matrix is None else np.asarray(projection_matrix),
        )
        # One density channel for the full basis + one per reduce block (e.g. the
        # 1L/rho and 2L/I2 invariant blocks → 3 channels for a 2L model).
        self.n_density = (
            1 + len(self.basis_reduce_instructions) if self.add_density_channel else 0
        )
        self.n_out = self.out_dim + self.n_density
        for red in self.basis_reduce_instructions:
            assert red.only_invar, (
                f"{self.__class__.__name__} requires scalar (l=0) FunctionReduceN "
                f"reduces; got '{red.name}' with ls_max={red.ls_max}"
            )

    def _basis_dim(self) -> int:
        """Total concatenated basis width (data-free, from collector metadata)."""
        d = 0
        for red in self.basis_reduce_instructions:
            for instr in red.instructions:
                c = red.collector[instr.name]
                d += int(c["n_out"]) * int(len(c["func_collect_ind"]))
        return d

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            if self._projection_matrix is not None:
                R = np.asarray(self._projection_matrix)
                assert R.shape[1] == self.out_dim, (
                    f"projection_matrix has out_dim={R.shape[1]}, expected {self.out_dim}"
                )
            else:
                R = generate_rp_matrix(self._basis_dim(), self.out_dim, self.seed)
            self.projection = tf.constant(R, dtype=float_dtype, name="rp_matrix")
            self.float_dtype = float_dtype
            self.is_built = True

    def _apply_feature_transform(self, basis):
        """Non-linear transform of the basis before the matmul: ``proj = T(basis) @ R``.

        ``basis`` is ``[N_atoms, D_basis]`` in ``self.projection.dtype``.

        * ``"asinh"`` (default for new builds) — element-wise ``asinh(basis)``;
          bounded, data-free, and the only non-linearity in the feature. It
          de-collapses the GMM covariance when a minority of atoms blow up the raw
          2L invariant basis; a fixed linear scale folded into R cannot (a linear
          map preserves the rank-1 domination).
        * ``None`` — legacy linear feature, kept so pre-asinh artifacts reproduce.
        """
        if self.feature_transform is None:
            return basis
        if self.feature_transform == "asinh":
            return tf.math.asinh(basis)
        raise ValueError(f"unknown feature_transform {self.feature_transform!r}")

    def frwrd(self, input_data, training=False, local=False):
        # Imported lazily to avoid a circular import: tpmodel re-exports this
        # class, so a module-level `from tensorpotential.tpmodel import ...`
        # here would form a cycle at load time.
        from tensorpotential.tpmodel import extract_basis_functions

        dtype = self.projection.dtype
        blocks = [
            tf.cast(extract_basis_functions(red, input_data), dtype)
            for red in self.basis_reduce_instructions
        ]
        basis = tf.concat(blocks, axis=-1)
        # Density channels: log of the RAW per-block + full L2 norms (before any
        # normalize/transform), so the magnitude axis — flattened by normalize and
        # squashed by asinh — re-enters as explicit, both-ended OOD coordinates.
        dens = None
        if self.add_density_channel:
            norms = tf.stack(
                [tf.norm(basis, axis=-1)] + [tf.norm(b, axis=-1) for b in blocks],
                axis=-1,
            )
            dens = tf.math.log(norms + self._eps) * tf.constant(
                self.density_scale, dtype=dtype
            )
        if self.normalize:
            basis = basis / (tf.norm(basis, axis=-1, keepdims=True) + self._eps)
        basis = self._apply_feature_transform(basis)
        proj = tf.matmul(basis, self.projection)
        if dens is not None:
            proj = tf.concat([proj, dens], axis=-1)
        # Returned value is stored by TPInstruction.__call__ under ``self.name``
        # (== uq_constants.FEATURES "features"), so the UQ feature consumer reads
        # it through the general mechanism — no side-write into input_data.
        return proj
