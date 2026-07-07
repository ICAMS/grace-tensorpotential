"""Factories for creating UQ-enabled calculators."""

from __future__ import annotations
import os
import numpy as np
import tensorflow as tf
from typing import Any

from tensorpotential import constants
from tensorpotential.instructions.base import load_instructions, TPInstruction
from tensorpotential.instructions.compute import FunctionReduceN
from tensorpotential.instructions.output import TPOutputInstruction
from tensorpotential.tensorpot import TensorPotential
from tensorpotential.tpmodel import ComputeFunction
from tensorpotential.uq.instructions import (
    RandomProjectedBasisFeatures,
    generate_rp_matrix,
)
from tensorpotential.calculator.asecalculator import TPCalculator
from tensorpotential.uq.gmmuq import GMMUQModel
from tensorpotential.uq.compute import gmm_uq_compute_class
from tensorpotential.uq import constants as uq_constants
from tensorpotential.metadata_utils import resolve_param_dtype


def _upstream_instruction_refs(ins):
    """Live upstream-instruction refs from an instruction's captured init args.

    ``_init_args`` is the authoritative graph-edge record (it is what
    ``save_instructions_dict`` rewires), so this is the canonical, class-agnostic
    way to walk dependencies. The ``target`` arg is skipped: on output
    instructions it points *downstream* to the output node, not upstream.
    """
    refs = []
    for k, v in getattr(ins, "_init_args", {}).items():
        if k == "target":
            continue
        for item in v if isinstance(v, (list, tuple)) else [v]:
            if isinstance(item, TPInstruction):
                refs.append(item)
    return refs


def trace_energy_path_invariant_reduces(instructions):
    """Reduces whose invariant (l=0) basis enters the atomic-energy path.

    Seeds from every output instruction writing ``PREDICT_ATOMIC_ENERGY`` and
    walks the dependency graph *upstream* (via :func:`_upstream_instruction_refs`),
    stopping at the first invariant ``FunctionReduceN`` (``only_invar``) on each
    branch. Robust to normalization (or any instruction) sitting between the
    readout and the reduce, and to architectures with one *or many* reduces in
    different branches — unlike a plain ``only_invar`` type-scan, which would also
    pick up l=0 reduces that only feed a later message-passing layer. Returned
    sorted by name for a deterministic basis-concatenation order.
    """
    ins_list = (
        list(instructions.values())
        if isinstance(instructions, dict)
        else list(instructions)
    )
    seeds = [
        ins
        for ins in ins_list
        if isinstance(ins, TPOutputInstruction)
        and getattr(getattr(ins, "target", None), "name", None)
        == constants.PREDICT_ATOMIC_ENERGY
    ]
    if not seeds:
        raise ValueError(
            "basis_rp feature: no output instruction targets "
            f"'{constants.PREDICT_ATOMIC_ENERGY}'"
        )
    found, visited, queue = [], {id(s) for s in seeds}, list(seeds)
    while queue:
        ins = queue.pop(0)
        for dep in _upstream_instruction_refs(ins):
            if id(dep) in visited:
                continue
            visited.add(id(dep))
            if isinstance(dep, FunctionReduceN) and getattr(dep, "only_invar", False):
                found.append(dep)  # nearest invariant reduce on this branch — stop
            else:
                queue.append(dep)
    if not found:
        raise ValueError(
            "basis_rp feature: energy-path trace found no invariant "
            "(l=0) FunctionReduceN reduce"
        )
    return sorted(found, key=lambda r: r.name)


def patch_instructions_for_basis_rp_features(
    instructions,
    out_dim: int = 128,
    seed: int = 42,
    projection_matrix=None,
    feature_transform: str | None = None,
    normalize: bool = False,
    add_density_channel: bool = False,
    density_scale: float = 1.0,
) -> RandomProjectedBasisFeatures:
    """Append a :class:`RandomProjectedBasisFeatures` instruction; return it.

    Projects the concatenated rotation-invariant (l=0) B-basis of the
    energy-path reduces (see :func:`trace_energy_path_invariant_reduces`) to
    ``out_dim`` dims under the canonical ``FEATURES`` key. ``R`` is taken verbatim
    from ``projection_matrix`` when given (eval / shared across build workers),
    else generated deterministically from ``seed``. Appended last so it runs after
    its ``ProductFunction`` dependencies; its returned value is stored under
    ``uq_constants.FEATURES`` by the general instruction mechanism. The appended
    instruction is returned so a caller patching an already-built model can
    ``.build(dtype)`` it before retracing.
    """
    reduces = trace_energy_path_invariant_reduces(instructions)
    rp = RandomProjectedBasisFeatures(
        reduces,
        out_dim=out_dim,
        seed=seed,
        projection_matrix=projection_matrix,
        feature_transform=feature_transform,
        normalize=normalize,
        add_density_channel=add_density_channel,
        density_scale=density_scale,
        name=uq_constants.FEATURES,
    )
    if isinstance(instructions, dict):
        instructions[rp.name] = rp
    else:
        instructions.append(rp)
    return rp


def load_uq_model(
    model_yaml: str,
    checkpoint: str,
    model_compute_function: ComputeFunction,
    param_dtype: tf.DType | None = None,
    feature_spec: dict | None = None,
) -> tuple[TensorPotential, dict]:
    """Load a TP model with the basis-RP UQ feature patched in.

    Shared by artifact generation (build) and runtime evaluation. UQ uses the
    basis-RP feature exclusively (the retired hidden-layer feature is no longer
    supported). ``feature_spec`` carries the projection:

    * **eval** — the self-describing artifact supplies ``matrix`` (the verbatim
      ``R``), so the exact feature is reproduced;
    * **build** — only ``out_dim`` / ``seed`` are given and ``R`` is regenerated
      deterministically (byte-identical across workers).

    Parameters
    ----------
    model_yaml, checkpoint : str
        Model graph and checkpoint to load.
    model_compute_function : ComputeFunction
        Compute function (e.g. ComputeEnergy or the UQ compute).
    param_dtype : tf.DType, optional
        Inferred from model_yaml if None.
    feature_spec : dict, optional
        ``{out_dim, seed, matrix}`` for the basis-RP projection. Defaults to the
        canonical 128-D / seed-42 projection (R regenerated) when None.

    Returns
    -------
    tuple[TensorPotential, dict]
        The initialized TensorPotential and the (patched) instructions dictionary.
    """
    param_dtype = resolve_param_dtype(model_yaml, param_dtype)
    instructions = load_instructions(model_yaml)

    feature_spec = feature_spec or {}
    patch_instructions_for_basis_rp_features(
        instructions,
        out_dim=int(feature_spec.get("out_dim", 128)),
        seed=int(feature_spec.get("seed", 42)),
        projection_matrix=feature_spec.get("matrix"),
        feature_transform=feature_spec.get("transform"),
        normalize=bool(feature_spec.get("normalize", False)),
        add_density_channel=bool(feature_spec.get("add_density_channel", False)),
        density_scale=float(feature_spec.get("density_scale", 1.0)),
    )

    tp = TensorPotential(
        potential=instructions,
        model_compute_function=model_compute_function,
        param_dtype=param_dtype,
    )
    # UQ needs only the model weights, so read the model subtree alone
    # (model_only=True): the strict object match then validates the weights
    # without tripping over training-bookkeeping trackables (step/epoch/
    # intra_epoch_save) or optimizer slots that older / foundation checkpoints
    # never wrote. expect_partial=True silences the resulting unconsumed-value
    # warnings at GC time.
    tp.load_checkpoint(
        checkpoint,
        model_only=True,
        assert_consumed=False,
        # GRACE_UQ_LENIENT_LOAD relaxes even the model-weight match, for
        # intentionally drifted graphs; the weights still load. Strict by default.
        assert_existing_objects_matched=not os.environ.get("GRACE_UQ_LENIENT_LOAD"),
        expect_partial=True,
    )
    tp.model.decorate_compute_function()
    return tp, instructions


def build_uq_compute_from_yaml(
    model_yaml: str,
    checkpoint: str,
    gmm_artifact_path: str,
    param_dtype: tf.DType | None = None,
    compute_dsigma_dr: bool = True,
):
    """Resolve dtype, load the GMM artifact, build compute_uq, and load the TP model.

    Returns ``(tp_uq, gmm_uq, compute_uq, param_dtype)``. Used by
    ``get_gmm_uq_calculator`` and ``TPCalculator.load_uq_artifacts`` to share
    the same artifact-loading flow.
    """
    if checkpoint.endswith(".index"):
        checkpoint = checkpoint.replace(".index", "")
    param_dtype = resolve_param_dtype(model_yaml, param_dtype)
    gmm_uq = GMMUQModel(gmm_artifact_path, param_dtype=param_dtype)
    compute_uq = gmm_uq_compute_class(compute_dsigma_dr)(gmm_uq_model=gmm_uq)
    # Self-describing eval: the basis-RP artifact carries the verbatim projection
    # matrix + spec, so the calculator reproduces the exact feature with no env
    # var. The retired hidden-layer feature is unsupported — an artifact without
    # the basis-RP spec is rejected with a clear rebuild message.
    feature_spec = _basis_rp_spec_from_artifact(gmm_uq)
    if feature_spec is None:
        raise ValueError(
            f"UQ artifact '{gmm_artifact_path}' was not built with the basis-RP "
            "feature (missing uq_feature_mode='basis_rp' / uq_rp_matrix). The "
            "hidden-layer UQ feature is retired — rebuild the artifact with "
            "`grace_uq build` (basis-RP is the default)."
        )
    if feature_spec.get("matrix") is None:
        raise ValueError(
            f"UQ artifact '{gmm_artifact_path}' is missing the stored projection "
            "matrix 'uq_rp_matrix'; rebuild the artifact with `grace_uq build`."
        )
    tp_uq, _ = load_uq_model(
        model_yaml=model_yaml,
        checkpoint=checkpoint,
        model_compute_function=compute_uq,
        param_dtype=param_dtype,
        feature_spec=feature_spec,
    )
    return tp_uq, gmm_uq, compute_uq, param_dtype


def make_basis_rp_spec(
    model_yaml: str,
    rp_dim: int = 128,
    rp_seed: int = 42,
    store_fp32: bool = True,
    feature_transform: str | None = uq_constants.DEFAULT_FEATURE_TRANSFORM,
    normalize: bool = False,
    add_density_channel: bool = False,
    density_scale: float = 1.0,
) -> dict:
    """Build the self-describing basis-RP spec to stamp into a GMM artifact.

    Regenerates the byte-identical projection ``R`` from ``rp_seed`` against the
    model's energy-path invariant basis (same construction the build workers and
    :meth:`RandomProjectedBasisFeatures.build` use), so eval / SavedModel export
    reproduce the exact feature. ``feature_transform`` defaults to
    ``DEFAULT_FEATURE_TRANSFORM`` (``None`` — the linear basis; override the env var
    ``GRACE_UQ_FEATURE_TRANSFORM=asinh`` for the uqv4 feature) and is stored verbatim
    so eval reproduces it. The production uqv6 feature also sets ``normalize=True`` and
    ``add_density_channel=True`` — ``grace_uq build`` always passes these (the fixed
    ``UQ_DEFAULT_*`` constants); this function keeps False defaults so the read path
    reproduces legacy artifacts. Returns a dict keyed by the artifact constants, ready
    to splat into ``GMMUQModel.save`` / ``builder.save`` as ``**spec``.
    """
    rp = RandomProjectedBasisFeatures(
        trace_energy_path_invariant_reduces(load_instructions(model_yaml)),
        out_dim=rp_dim,
        seed=rp_seed,
        feature_transform=feature_transform,
        normalize=normalize,
        add_density_channel=add_density_channel,
        density_scale=density_scale,
    )
    R = generate_rp_matrix(rp._basis_dim(), rp_dim, rp_seed)
    if store_fp32:
        R = R.astype(np.float32)
    spec = {
        uq_constants.UQ_FEATURE_MODE: np.array(uq_constants.FEATURE_MODE_BASIS_RP),
        uq_constants.UQ_RP_MATRIX: R,
        uq_constants.UQ_RP_DIM: np.int64(rp_dim),
        uq_constants.UQ_RP_SEED: np.int64(rp_seed),
    }
    # Stamp the transform only when set, so existing/identity artifacts keep their
    # exact key set and loaders treat a missing key as the linear (T=identity) feature.
    if feature_transform is not None:
        spec[uq_constants.UQ_FEATURE_TRANSFORM] = np.array(feature_transform)
    # Stamp uqv6 flags only when enabled, so pre-uqv6 artifacts keep their exact key
    # set and loaders treat the missing keys as the default (off / scale 1.0).
    if normalize:
        spec[uq_constants.UQ_RP_NORMALIZE] = np.array(True)
    if add_density_channel:
        spec[uq_constants.UQ_RP_DENSITY] = np.array(True)
        spec[uq_constants.UQ_RP_DENSITY_SCALE] = np.float64(density_scale)
    return spec


def _basis_rp_spec_from_artifact(gmm_uq_model) -> dict | None:
    """Recover the basis-RP feature spec stored in a GMM artifact, if any."""
    extra = getattr(gmm_uq_model, "extra_data", {}) or {}
    mode = extra.get(uq_constants.UQ_FEATURE_MODE)
    if mode is None or (
        str(np.asarray(mode).item() if np.ndim(mode) == 0 else mode)
        != uq_constants.FEATURE_MODE_BASIS_RP
    ):
        return None
    matrix = extra.get(uq_constants.UQ_RP_MATRIX)
    dim = extra.get(uq_constants.UQ_RP_DIM)
    if dim is not None:
        out_dim = int(np.asarray(dim).item())
    elif matrix is not None:
        out_dim = int(np.asarray(matrix).shape[1])
    else:
        out_dim = 128
    seed = extra.get(uq_constants.UQ_RP_SEED)
    transform = extra.get(uq_constants.UQ_FEATURE_TRANSFORM)
    if transform is not None:
        transform = str(np.asarray(transform).item())
    normalize = extra.get(uq_constants.UQ_RP_NORMALIZE)
    add_density = extra.get(uq_constants.UQ_RP_DENSITY)
    density_scale = extra.get(uq_constants.UQ_RP_DENSITY_SCALE)
    return {
        "mode": uq_constants.FEATURE_MODE_BASIS_RP,
        "out_dim": out_dim,
        "seed": int(np.asarray(seed).item()) if seed is not None else 42,
        "matrix": None if matrix is None else np.asarray(matrix),
        "transform": transform,
        "normalize": bool(np.asarray(normalize).item()) if normalize is not None else False,
        "add_density_channel": bool(np.asarray(add_density).item()) if add_density is not None else False,
        "density_scale": float(np.asarray(density_scale).item()) if density_scale is not None else 1.0,
    }


def default_uq_extra_properties(
    gmm_uq_model,
    compute_dsigma_dr: bool = True,
) -> tuple[list[str], list[str]]:
    """Default ``extra_properties`` and ``truncate_extras_by_natoms`` for UQ.

    Drops keys that the compute function does not produce for the current
    configuration: gamma-only mode skips uncertainty-force keys; artifacts
    without thresholds skip ``gamma``."""
    skip = set()
    if gmm_uq_model.thresholds_dict is None:
        skip.add(uq_constants.ATOMIC_GAMMA)
    if not compute_dsigma_dr:
        skip.update({uq_constants.DSIGMA_DR, uq_constants.VIRIAL_SIGMA})
    extra = [k for k in uq_constants.UQ_EXTRA_KEYS if k not in skip]
    truncate = [k for k in uq_constants.UQ_TRUNCATE_KEYS if k in extra]
    return extra, truncate


def get_gmm_uq_calculator(
    model_yaml: str,
    checkpoint: str,
    gmm_artifact_path: str,
    param_dtype: tf.DType | None = None,
    extra_properties: list[str] | None = None,
    truncate_extras_by_natoms: list[str] | None = None,
    compute_dsigma_dr: bool = True,
    **kwargs: Any,
) -> TPCalculator:
    """Create a TPCalculator with GMM-UQ uncertainty quantization (HAL).

    This factory encapsulates the boilerplate required to:
    1. Resolve the model parameter dtype.
    2. Load the GMM-UQ artifact.
    3. Patch the potential instructions to return internal features.
    4. Initialize the HAL compute function.
    5. Load the model checkpoint.
    6. Configure the ASE calculator with UQ-specific properties.

    Parameters
    ----------
    model_yaml : str
        Path to the model.yaml file.
    checkpoint : str
        Path to the model checkpoint.
    gmm_artifact_path : str
        Path to the GMM-UQ .npz artifact.
    param_dtype : tf.DType, optional
        Data type for model parameters. If None, it is inferred from model_yaml.
        Defaults to tf.float64 if not found in YAML.
    extra_properties : list[str], optional
        Additional properties to calculate. Defaults to standard UQ keys:
        [ATOMIC_SIGMA, TOTAL_SIGMA, DSIGMA_DR, VIRIAL_SIGMA, FEATURES, ATOMIC_GAMMA]
    truncate_extras_by_natoms : list[str], optional
        Properties to truncate to the number of real atoms. Defaults to:
        [ATOMIC_SIGMA, DSIGMA_DR, FEATURES, ATOMIC_GAMMA]
    **kwargs : Any
        Additional arguments passed to TPCalculator.

    Returns
    -------
    TPCalculator
        ASE-compatible calculator with UQ capabilities.
    """
    tp_uq, gmm_uq, _, _ = build_uq_compute_from_yaml(
        model_yaml=model_yaml,
        checkpoint=checkpoint,
        gmm_artifact_path=gmm_artifact_path,
        param_dtype=param_dtype,
        compute_dsigma_dr=compute_dsigma_dr,
    )
    default_extra, default_truncate = default_uq_extra_properties(
        gmm_uq, compute_dsigma_dr=compute_dsigma_dr,
    )
    if extra_properties is None:
        extra_properties = default_extra
    if truncate_extras_by_natoms is None:
        truncate_extras_by_natoms = default_truncate

    calc = TPCalculator(
        tp_uq.model,
        extra_properties=extra_properties,
        truncate_extras_by_natoms=truncate_extras_by_natoms,
        **kwargs,
    )
    calc.tensorpotential = tp_uq
    calc.gmm_uq_model = gmm_uq
    return calc
