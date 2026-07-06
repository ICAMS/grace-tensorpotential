"""ComputeFunctions for GMM-UQ uncertainty quantification.

Two flavours are exported:

* :class:`ComputeStructureEnergyAndForcesAndVirialAndUncertainty` — full
  HAL compute: physical forces/virials plus uncertainty pair forces
  (``dsigma/dr``), atomic uncertainty forces, virial_sigma.  Used for
  HAL-style biased dynamics.
* :class:`ComputeStructureEnergyAndForcesAndVirialAndGammaOnly` — cheaper
  variant returning only scalar uncertainty fields (``atomic_sigma``,
  ``total_sigma``, and ``gamma`` when thresholds are available).  The GMM
  call runs *outside* the gradient tape, so no second backward pass
  through the model+GMM is needed.  Use this for active-learning
  monitoring / collection criteria where ``dsigma/dr`` is not required.
"""

from __future__ import annotations

import tensorflow as tf

from tensorpotential import constants
from tensorpotential.tpmodel import (
    ComputeFunction,
    compute_structure_virials_from_pair_forces,
    execute_instructions,
)
from tensorpotential.instructions.compute import TPInstruction

from tensorpotential.uq.gmmuq import GMMUQModel
from tensorpotential.uq import constants as uq_constants


# NOTE: constants.N_ATOMS_BATCH_REAL is consumed in __call__ but NOT declared
# in ``specs``.  The readout instruction (LinMLPOut2ScalarTarget) lists it in
# its input_tensor_spec, so it flows into compute_specs via instructions_specs.
# Declaring it here too caused XLA recompilations on every call (commit f90ae95)
# — likely a tf.function cache-key sensitivity to dict insertion order when the
# same key is updated from two sources.

_BASE_SPECS = {
    constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
    constants.BOND_IND_J: {"shape": [None], "dtype": "int"},
    constants.BOND_VECTOR: {"shape": [None, 3], "dtype": "float"},
    constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
    constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
    constants.BONDS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
    constants.N_STRUCTURES_BATCH_TOTAL: {"shape": [], "dtype": "int"},
    constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
}


class _GMMUQComputeBase(ComputeFunction):
    """Shared scaffolding for GMM-UQ compute functions: holds the GMM
    model, returns helpers for the GMM forward pass, and assembles the
    physical-prediction subset of the result dict."""

    specs = _BASE_SPECS

    def __init__(
        self,
        gmm_uq_model: GMMUQModel,
        extra_return_keys: list[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gmm_uq_model = gmm_uq_model
        self.extra_return_keys = extra_return_keys or [uq_constants.FEATURES]

    def _gmm_outputs(self, input_data):
        """GMM forward pass.  Returns ``(sigma, total_sigma, gamma_or_None,
        real_mask, cluster_assign)``.  ``gamma`` is ``None`` when the
        artifact carries no thresholds.  ``cluster_assign`` is the
        per-atom cluster index from the GMM."""
        features = input_data[uq_constants.FEATURES]
        features = tf.cast(features, dtype=self.gmm_uq_model.param_dtype)
        sigma, _, cluster_assign = self.gmm_uq_model.compute(
            features, input_data[constants.ATOMIC_MU_I]
        )
        # Sum only over real atoms (exclude padding) using a static-shape
        # mask to avoid dynamic slicing that triggers XLA recompilation.
        n_real = input_data[constants.N_ATOMS_BATCH_REAL]
        n_total = tf.shape(sigma)[0]
        real_mask = tf.cast(tf.range(n_total) < n_real, dtype=sigma.dtype)
        total_sigma = tf.reduce_sum(sigma * real_mask)

        if self.gmm_uq_model.interp_thresholds is not None:
            idx = tf.stack(
                [input_data[constants.ATOMIC_MU_I], cluster_assign], axis=1
            )
            thresholds = tf.gather_nd(
                self.gmm_uq_model.interp_thresholds_dense, idx
            )
            thresholds = tf.maximum(thresholds, tf.cast(1e-10, thresholds.dtype))
            gamma = sigma / thresholds
        else:
            gamma = None
        return sigma, total_sigma, gamma, real_mask, cluster_assign

    def _physical_outputs(self, input_data, e_atomic, pair_f):
        """Build the physical-prediction subset of the result dict."""
        e_atomic = tf.cast(e_atomic, dtype=pair_f.dtype)
        total_energy = tf.reduce_sum(e_atomic, axis=0, keepdims=True)
        nat = tf.shape(input_data[constants.ATOMIC_MU_I])[0]
        forces = tf.math.unsorted_segment_sum(
            pair_f, input_data[constants.BOND_IND_J], num_segments=nat
        ) - tf.math.unsorted_segment_sum(
            pair_f, input_data[constants.BOND_IND_I], num_segments=nat
        )
        virial = compute_structure_virials_from_pair_forces(pair_f, input_data)
        return {
            constants.PREDICT_TOTAL_ENERGY: total_energy,
            constants.PREDICT_ATOMIC_ENERGY: e_atomic,
            constants.PREDICT_FORCES: forces,
            constants.PREDICT_VIRIAL: virial,
            constants.PREDICT_PAIR_FORCES: pair_f,
        }

    def _attach_extras(self, res, input_data):
        if self.extra_return_keys:
            for k in self.extra_return_keys:
                if k in input_data:
                    res[k] = input_data[k]
        return res


class ComputeStructureEnergyAndForcesAndVirialAndUncertainty(_GMMUQComputeBase):
    """Full HAL compute: physical energy/forces/virial + uncertainty
    pair forces ``dsigma/dr`` and ``virial_sigma``.

    Uses two independent ``tf.GradientTape``s — one for the energy
    backward pass, one for the HAL objective backward pass.

    Parameters
    ----------
    gmm_uq_model : GMMUQModel
        Pre-loaded GMM-UQ model with artifacts.
    extra_return_keys : list[str], optional
        Additional keys to copy from ``input_data`` into the output dict.
        Defaults to ``[FEATURES]``.
    """

    def __call__(
        self,
        instructions: dict | list[TPInstruction],
        input_data: dict,
        training: bool = False,
    ):
        with tf.GradientTape() as tape_r, tf.GradientTape() as tape_u:
            tape_r.watch(input_data[constants.BOND_VECTOR])
            tape_u.watch(input_data[constants.BOND_VECTOR])
            execute_instructions(input_data, instructions, training)

            e_atomic = tf.reshape(input_data[constants.PREDICT_ATOMIC_ENERGY], [-1, 1])

            sigma, total_sigma, gamma, real_mask, cluster_assign = (
                self._gmm_outputs(input_data)
            )
            # HAL gradient objective: Σ(sigma_i / threshold_i) aligns the
            # gradient direction with the collection criterion max(gamma) >
            # threshold_t1.  cluster_assign is already stop_gradient'd inside
            # _compute_core, and tape_u only watches BOND_VECTOR, so
            # thresholds act as constant per-atom scaling factors on the sigma
            # gradient.  Falls back to total_sigma when thresholds are absent.
            hal_objective = (
                tf.reduce_sum(gamma * real_mask) if gamma is not None else total_sigma
            )

        pair_f = tf.negative(
            tape_r.gradient(e_atomic, input_data[constants.BOND_VECTOR])
        )
        pair_u = tape_u.gradient(hal_objective, input_data[constants.BOND_VECTOR])
        del tape_r, tape_u

        nat = tf.shape(input_data[constants.ATOMIC_MU_I])[0]
        uncertainty_forces = tf.math.unsorted_segment_sum(
            pair_u, input_data[constants.BOND_IND_J], num_segments=nat
        ) - tf.math.unsorted_segment_sum(
            pair_u, input_data[constants.BOND_IND_I], num_segments=nat
        )
        virial_sigma = compute_structure_virials_from_pair_forces(pair_u, input_data)

        res = self._physical_outputs(input_data, e_atomic, pair_f)
        res.update({
            uq_constants.TOTAL_SIGMA: total_sigma,
            uq_constants.ATOMIC_SIGMA: sigma,
            uq_constants.DSIGMA_DR: uncertainty_forces,
            uq_constants.VIRIAL_SIGMA: virial_sigma,
            uq_constants.DSIGMA_DR_PAIR: pair_u,
            uq_constants.GMM_CLUSTER: cluster_assign,
        })
        if gamma is not None:
            res[uq_constants.ATOMIC_GAMMA] = gamma

        return self._attach_extras(res, input_data)


class ComputeStructureEnergyAndForcesAndVirialAndGammaOnly(_GMMUQComputeBase):
    """Cheaper UQ compute: physical energy/forces/virial plus *scalar*
    uncertainty fields (``atomic_sigma``, ``total_sigma``, and ``gamma``
    when thresholds are available).  Does NOT compute ``dsigma/dr``,
    ``virial_sigma``, or ``dsigma_dr_pair``.

    The GMM forward runs *outside* the gradient tape, so the second
    backward pass through the model+GMM is avoided.  At the 1000-atom
    scale this is essentially free on top of a pure (no-UQ) calculator
    while the full HAL compute adds ~33 % wall-time.

    Use when only scalar per-atom uncertainty is needed (active-learning
    collection criteria, monitoring during MD).
    """

    def __call__(
        self,
        instructions: dict | list[TPInstruction],
        input_data: dict,
        training: bool = False,
    ):
        with tf.GradientTape() as tape_r:
            tape_r.watch(input_data[constants.BOND_VECTOR])
            execute_instructions(input_data, instructions, training)
            e_atomic = tf.reshape(input_data[constants.PREDICT_ATOMIC_ENERGY], [-1, 1])

        pair_f = tf.negative(
            tape_r.gradient(e_atomic, input_data[constants.BOND_VECTOR])
        )
        del tape_r

        sigma, total_sigma, gamma, _, cluster_assign = self._gmm_outputs(input_data)

        res = self._physical_outputs(input_data, e_atomic, pair_f)
        res.update({
            uq_constants.TOTAL_SIGMA: total_sigma,
            uq_constants.ATOMIC_SIGMA: sigma,
            uq_constants.GMM_CLUSTER: cluster_assign,
        })
        if gamma is not None:
            res[uq_constants.ATOMIC_GAMMA] = gamma

        return self._attach_extras(res, input_data)


def gmm_uq_compute_class(compute_dsigma_dr: bool):
    """Return the appropriate UQ compute class for the requested mode."""
    return (
        ComputeStructureEnergyAndForcesAndVirialAndUncertainty
        if compute_dsigma_dr
        else ComputeStructureEnergyAndForcesAndVirialAndGammaOnly
    )
