"""
GRACE-1L, GRACE-2L and GRACE-3L weight export to .npz for the LAMMPS Kokkos
pair styles (`pair_grace_1l_kokkos`, `pair_grace_2l_kokkos`,
`pair_grace_3l_kokkos`).

The .npz key schemas are dictated by what the C++ pair styles read; do not
change key names without also updating the readers in
src/KOKKOS/pair_grace_{1l,2l,3l}_kokkos.cpp.
"""

import logging
import os

import numpy as np
import tensorflow as tf
import yaml

from tensorpotential.instructions.base import load_instructions
from tensorpotential.tpmodel import TPModel

logger = logging.getLogger(__name__)


# Compile-time hard caps from the LAMMPS Kokkos headers
# (src/KOKKOS/pair_grace_1l_kokkos.h and pair_grace_2l_kokkos.h).
# Bumping the C++ header constants requires updating these in lockstep.
CAPS_1L = {
    "MAX_MLP_DIM": 64,
    "MAX_NRADMAX": 32,
    "MAX_MLP_LAYERS": 8,
}
CAPS_2L = {
    "MAX_MLP_DIM": 210,
    "MAX_NRADMAX": 42,
    "MAX_I1_FUNCS": 16,
    "MAX_I_FUNCS": 16,
    "MAX_YI_OUT_FUNCS": 200,
    "LMAX": 4,
}


# Both 1L and 2L pair styles read `energy_mlp_activation` with codes
# {0=silu, 1=tanh}. relu/gelu are not supported by the C++ side and the pair
# style aborts at load time if it sees any other code.
_ACT_CODE = {"silu": 0, "tanh": 1, "relu": 2, "gelu": 3}
_ACT_SUPPORTED_1L = {"silu", "tanh"}
_ACT_SUPPORTED_2L = {"silu", "tanh"}

# Pure-geometry / no-parameter classes that contribute nothing to the npz but
# must not trigger the unknown-class error.
_IGNORED_CLASSES = {
    "BondLength", "ScaledBondVector", "RadialBasis",
    "BondSpecificRadialBasisFunction", "SphericalHarmonic", "CreateOutputTarget",
}
_SUPPORTED_1L = {
    "ScalarChemicalEmbedding", "MLPRadialFunction_v2",
    "SingleParticleBasisFunctionScalarInd", "FCRight2Left", "ProductFunction",
    "FunctionReduceN", "LinMLPOut2ScalarTarget",
    "TrainableShiftTarget", "ConstantScaleShiftTarget",
} | _IGNORED_CLASSES
_SUPPORTED_2L = _SUPPORTED_1L | {
    "SingleParticleBasisFunctionEquivariantInd", "InvariantLayerRMSNorm",
}


# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------

def _find_instr(instruction_dict, cls_name):
    for name, instr in instruction_dict.items():
        if type(instr).__name__ == cls_name:
            return name, instr
    return None, None


def _first_attr_numpy(obj, attrs):
    """Return the first attribute named in `attrs` that exists on `obj` and has
    a `.numpy()` method, called and returned. Else None."""
    for a in attrs:
        v = getattr(obj, a, None)
        if v is not None and hasattr(v, "numpy"):
            return v.numpy()
    return None


def _scalar_to_np(x):
    """Coerce a possibly-tensor scalar to numpy. Preserves a tensor's natural
    shape (0-d for scalar tensors); wraps raw Python numbers in a 1-element
    array. Both forms are valid in the .npz schema."""
    return x.numpy() if hasattr(x, "numpy") else np.array([float(x)])


def _walk_mlp_layers(mlp):
    """Return the ordered list of MLP layer objects, regardless of whether the
    container exposes them as `.layers` or as `.layer0`, `.layer1`, ..."""
    if hasattr(mlp, "layers") and isinstance(mlp.layers, (list, tuple)):
        return list(mlp.layers)
    layers = []
    i = 0
    while hasattr(mlp, f"layer{i}"):
        layers.append(getattr(mlp, f"layer{i}"))
        i += 1
    return layers


def _extract_layer_weights(layer, prefix, i, weights):
    """Write `{prefix}_W{i}` / `{prefix}_norm{i}` / `{prefix}_b{i}` from a Linear-
    or DenseLayer-style object. Falls back to the first 2-D trainable variable
    if the kernel is not exposed under a known attribute name."""
    w = _first_attr_numpy(layer, ("w", "kernel", "weight"))
    if w is None:
        for var in layer.trainable_variables:
            if len(var.shape) == 2:
                w = var.numpy()
                break
    if w is not None:
        weights[f"{prefix}_W{i}"] = w

    norm = getattr(layer, "norm", None)
    if norm is not None:
        weights[f"{prefix}_norm{i}"] = np.array(
            [float(norm.numpy()) if hasattr(norm, "numpy") else float(norm)]
        )

    b = _first_attr_numpy(layer, ("b", "bias"))
    if b is not None:
        weights[f"{prefix}_b{i}"] = b


# ----------------------------------------------------------------------------
# Per-instruction extractors
# ----------------------------------------------------------------------------

def _extract_chem_embedding(instruction):
    weights = {}
    emb = _first_attr_numpy(instruction, ("embedding", "chemical_embedding", "z_embedding"))
    if emb is None:
        for var in instruction.trainable_variables:
            if len(var.shape) == 2:
                emb = var.numpy()
                break
    if emb is not None:
        weights["chem_embedding"] = emb
    weights["chem_embedding_size"] = np.array([instruction.embedding_size])
    return weights


def _extract_fc_weights(instruction, prefix):
    weights = {}
    if instruction.w_left is not None:
        weights[f"{prefix}_w_left"] = instruction.w_left.numpy()
        if hasattr(instruction, "norm_left"):
            weights[f"{prefix}_norm_left"] = _scalar_to_np(instruction.norm_left)
    if instruction.w_right is not None:
        weights[f"{prefix}_w_right"] = instruction.w_right.numpy()
        if hasattr(instruction, "norm_right"):
            weights[f"{prefix}_norm_right"] = _scalar_to_np(instruction.norm_right)
    if instruction.w_tile_left is not None:
        weights[f"{prefix}_w_tile_left"] = instruction.w_tile_left.numpy().astype(np.int32)
    if instruction.w_tile_right is not None:
        weights[f"{prefix}_w_tile_right"] = instruction.w_tile_right.numpy().astype(np.int32)
    if instruction.collect_to is not None:
        weights[f"{prefix}_collect_to"] = instruction.collect_to.numpy().astype(np.int32)
    if instruction.collect_from is not None:
        weights[f"{prefix}_collect_from"] = instruction.collect_from.numpy().astype(np.int32)
    if instruction.norm_out_factor is not None:
        weights[f"{prefix}_norm_out_factor"] = instruction.norm_out_factor.numpy()
    weights[f"{prefix}_n_out"] = np.array([instruction.n_out])
    weights[f"{prefix}_left_coefs"] = np.array([1 if instruction.left_coefs else 0])
    return weights


def _extract_product_metadata(instruction, prefix):
    left_ind = instruction.left_ind.numpy()
    right_ind = instruction.right_ind.numpy()
    m_sum_ind = instruction.m_sum_ind.numpy()
    cg = instruction.cg
    cg_np = cg.numpy() if hasattr(cg, "numpy") else np.array(cg)
    return {
        f"{prefix}_left_ind": left_ind.astype(np.int32),
        f"{prefix}_right_ind": right_ind.astype(np.int32),
        f"{prefix}_m_sum_ind": m_sum_ind.astype(np.int32),
        f"{prefix}_cg_coeff": cg_np.flatten(),
        f"{prefix}_n_output_funcs": np.array([int(m_sum_ind.max()) + 1]),
        f"{prefix}_n_cg_terms": np.array([len(left_ind)]),
        f"{prefix}_lmax": np.array([instruction.lmax]),
        f"{prefix}_Lmax": np.array([getattr(instruction, "Lmax", instruction.lmax)]),
    }


def _extract_mlp_radial(instruction, prefix):
    """MLPRadialFunction_v2 — with prefix='mlp_rad' produces 1L flat layout;
    with prefix='mlp_rad_<name>' produces 2L per-name layout."""
    weights = {}
    for i, layer in enumerate(instruction.layers):
        _extract_layer_weights(layer, prefix, i, weights)

    if hasattr(instruction, "embed_transform") and instruction.embed_transform is not None:
        et = instruction.embed_transform
        weights[f"{prefix}_Wemb"] = et.w.numpy()
        norm = float(et.norm.numpy()) if hasattr(et.norm, "numpy") else float(et.norm)
        weights[f"{prefix}_norm_emb"] = np.array([norm])
        if hasattr(et, "b") and et.b is not None:
            weights[f"{prefix}_bemb"] = et.b.numpy()

    weights[f"{prefix}_n_layers"] = np.array([len(instruction.layers)])
    weights[f"{prefix}_n_rad_max"] = np.array([instruction.n_rad_max])
    weights[f"{prefix}_lmax"] = np.array([instruction.lmax])
    weights[f"{prefix}_embed_j"] = np.array(
        [1 if getattr(instruction, "embed_j", False) else 0]
    )
    weights[f"{prefix}_embed_i"] = np.array(
        [1 if getattr(instruction, "embed_i", False) else 0]
    )
    return weights


def _extract_single_particle_basis(instruction, prefix):
    weights = {}
    if instruction.inv_avg_n_neigh is not None and hasattr(instruction.inv_avg_n_neigh, "numpy"):
        weights[f"{prefix}_inv_avg_n_neigh"] = instruction.inv_avg_n_neigh.numpy()
    if instruction.lin_transform is not None:
        lt = instruction.lin_transform
        lt_w = None
        for var in lt.trainable_variables:
            if len(var.shape) == 2:
                lt_w = var.numpy()
                break
        if lt_w is None:
            lt_w = _first_attr_numpy(lt, ("w", "kernel"))
        if lt_w is not None:
            weights[f"{prefix}_lin_transform_W"] = lt_w
        norm = getattr(lt, "norm", None)
        if norm is not None:
            weights[f"{prefix}_lin_transform_norm"] = np.array(
                [float(norm.numpy()) if hasattr(norm, "numpy") else float(norm)]
            )
    weights[f"{prefix}_indicator_l_depend"] = np.array(
        [1 if instruction.indicator_l_depend else 0]
    )
    weights[f"{prefix}_sum_neighbors"] = np.array(
        [1 if instruction.sum_neighbors else 0]
    )
    return weights


def _extract_reduce(instruction, prefix, *, include_2l_keys):
    """FunctionReduceN. With include_2l_keys=True additionally writes
    `{prefix}_norm_map` / `_n_funcs` / `_only_invar` (these three keys are
    consumed by the GRACE-2L pair style only)."""
    weights = {}
    for instr_name, c in instruction.collector.items():
        w_attr = f"reducing_{instr_name}"
        if hasattr(instruction, w_attr):
            weights[f"{prefix}_reduce_{instr_name}_W"] = getattr(instruction, w_attr).numpy()
        norm_attr = f"norm_{instr_name}"
        if hasattr(instruction, norm_attr):
            norm = getattr(instruction, norm_attr)
            weights[f"{prefix}_reduce_{instr_name}_norm"] = (
                norm.numpy() if hasattr(norm, "numpy") else np.array([float(norm)])
            )
        if isinstance(c, dict):
            if "func_collect_ind" in c:
                weights[f"{prefix}_collect_ind_{instr_name}"] = np.array(
                    c["func_collect_ind"], dtype=np.int32
                )
            if "w_l_tile" in c:
                weights[f"{prefix}_w_l_tile_{instr_name}"] = np.array(
                    c["w_l_tile"], dtype=np.int32
                )
            if "total_sum_ind" in c:
                weights[f"{prefix}_total_sum_ind_{instr_name}"] = np.array(
                    c["total_sum_ind"], dtype=np.int32
                )
            if "w_shape" in c:
                weights[f"{prefix}_w_shape_{instr_name}"] = np.array([c["w_shape"]])
            if "n_out" in c:
                weights[f"{prefix}_n_in_{instr_name}"] = np.array([c["n_out"]])

    weights[f"{prefix}_n_out"] = np.array([instruction.n_out])
    weights[f"{prefix}_is_central_atom_type_dependent"] = np.array(
        [1 if instruction.is_central_atom_type_dependent else 0]
    )
    if instruction.is_central_atom_type_dependent:
        weights[f"{prefix}_number_of_atom_types"] = np.array(
            [instruction.number_of_atom_types]
        )

    if include_2l_keys:
        if hasattr(instruction, "norm_map") and instruction.norm_map is not None:
            nm = instruction.norm_map
            weights[f"{prefix}_norm_map"] = nm.numpy() if hasattr(nm, "numpy") else np.array(nm)
        if hasattr(instruction, "coupling_meta_data") and instruction.coupling_meta_data is not None:
            weights[f"{prefix}_n_funcs"] = np.array([len(instruction.coupling_meta_data)])
        weights[f"{prefix}_only_invar"] = np.array([1 if instruction.only_invar else 0])

    weights[f"{prefix}_instruction_names"] = np.array(
        list(instruction.collector.keys()), dtype="S8"
    )
    return weights


def _extract_energy_mlp(instruction, prefix, *, allowed_activations, write_activation_code,
                        arch_label):
    """LinMLPOut2ScalarTarget. write_activation_code=True writes
    `{prefix}_activation`; =False omits it. Both 1L and 2L pair styles now
    read the key, so this is True for both architectures."""
    weights = {}
    detected_act = None
    if hasattr(instruction, "mlp") and instruction.mlp is not None:
        mlp_layers = _walk_mlp_layers(instruction.mlp)
        for i, layer in enumerate(mlp_layers):
            _extract_layer_weights(layer, prefix, i, weights)
        weights[f"{prefix}_n_layers"] = np.array([len(mlp_layers)])
        detected_act = _check_activation(mlp_layers, allowed_activations, arch_label)

    if hasattr(instruction, "scale") and instruction.scale is not None:
        weights[f"{prefix}_scale"] = instruction.scale.numpy()

    if write_activation_code:
        if detected_act is None:
            raise ValueError(
                f"{arch_label}: could not detect energy MLP activation function. "
                "The Kokkos pair style requires an explicit silu or tanh activation."
            )
        weights[f"{prefix}_activation"] = np.array([_ACT_CODE[detected_act]])
    return weights


def _extract_equivariant_ind(instruction, prefix):
    lr_inds = instruction.lr_inds.numpy().astype(np.int32)
    m_sum_ind = instruction.m_sum_ind.numpy().astype(np.int32)
    cg = instruction.cg
    cg_np = cg.numpy() if hasattr(cg, "numpy") else np.array(cg)
    weights = {
        f"{prefix}_lr_inds": lr_inds,
        f"{prefix}_m_sum_ind": m_sum_ind,
        f"{prefix}_cg_coeff": cg_np.flatten(),
        f"{prefix}_n_output_funcs": np.array([int(m_sum_ind.max()) + 1]),
        f"{prefix}_n_cg_terms": np.array([len(lr_inds)]),
        f"{prefix}_n_rad_max": np.array([instruction.n_out]),
    }
    if hasattr(instruction, "inv_avg_n_neigh") and instruction.inv_avg_n_neigh is not None:
        weights[f"{prefix}_inv_avg_n_neigh"] = _scalar_to_np(instruction.inv_avg_n_neigh)
    if hasattr(instruction, "lmax"):
        weights[f"{prefix}_Lmax"] = np.array([instruction.lmax])
    return weights


def _extract_rms_norm(instruction, prefix):
    weights = {}
    if instruction.scale is not None:
        weights[f"{prefix}_scale"] = instruction.scale.numpy().flatten()
    norm_type = instruction.type
    type_map = {"full": 0, "only_nonlin": 1, "sep_lin_gate": 2}
    weights[f"{prefix}_type"] = np.array([type_map.get(norm_type, -1)])
    weights[f"{prefix}_n_out"] = np.array([instruction.n_out])
    if norm_type == "sep_lin_gate" and hasattr(instruction, "lin_scale"):
        weights[f"{prefix}_lin_scale"] = instruction.lin_scale.numpy().flatten()
    return weights


# ----------------------------------------------------------------------------
# 2L legacy-prefix aliasing
#
# pair_grace_2l_kokkos.cpp hardcodes four prefixes for the energy-MLP path:
#   rho_*, I2_*, I_nl_LN_*, I_0_LN_*
# but this exporter writes prefixes derived from YAML instruction names
# (e.g. I_out_0, I_out_1, I_out_0_LN, I_1_LN). When the YAML names drift away
# from the OMAT convention, the C++ side silently reads zeros — flat energy,
# zero forces. We fix this by walking the instruction graph, identifying which
# RMSNorm + ReduceN play each role, and emitting the legacy keys as aliases.
# ----------------------------------------------------------------------------

# RMSNorm.type → (legacy RMSNorm prefix, legacy ReduceN prefix)
_LEGACY_2L_ROLE = {
    "only_nonlin": ("I_nl_LN", "rho"),
    "full":        ("I_0_LN",  "I2"),
}

# Sentinel keys whose presence proves each legacy prefix was populated end-to-end.
_LEGACY_2L_REQUIRED = (
    ("rho_*_W",         lambda k: k.startswith("rho_reduce_") and k.endswith("_W")),
    ("I2_*_W",          lambda k: k.startswith("I2_reduce_")  and k.endswith("_W")),
    ("I_nl_LN_scale",   lambda k: k == "I_nl_LN_scale"),
    ("I_0_LN_scale",    lambda k: k == "I_0_LN_scale"),
)


def _emit_2l_legacy_aliases(instruction_dict, all_weights):
    """Mirror YAML-named keys under legacy `rho_*` / `I2_*` / `I_nl_LN_*` /
    `I_0_LN_*` prefixes that the C++ Kokkos pair style hardcodes.

    Roles are resolved from topology (RMSNorm.type + RMSNorm.input chain +
    LinMLPOut2ScalarTarget.origin), not from YAML names — so any future model
    that names its instructions differently still exports correctly. Raises
    rather than silently producing a half-populated .npz that loads as zeros.
    """
    rms_norms = [(name, instr) for name, instr in instruction_dict.items()
                 if type(instr).__name__ == "InvariantLayerRMSNorm"]
    _, mlp_instr = _find_instr(instruction_dict, "LinMLPOut2ScalarTarget")
    if not rms_norms or mlp_instr is None:
        return  # 1L or non-standard topology — nothing to alias

    # Capability check 1: exactly two RMSNorms with the two legacy types.
    types_found = sorted(getattr(r[1], "type", None) for r in rms_norms)
    if types_found != ["full", "only_nonlin"]:
        raise ValueError(
            f"GRACE-2L export_kokkos: pair_grace_2l_kokkos hardcodes one "
            f"only_nonlin (rho path) and one full (I2 path) RMSNorm feeding "
            f"the energy MLP, but the model has {len(rms_norms)} RMSNorm(s) "
            f"with types {types_found}. Other configurations are not supported."
        )

    # Capability check 2: both RMSNorms feed the energy MLP.
    origin = list(getattr(mlp_instr, "origin", []) or [])
    origin_names = {getattr(o, "name", None) for o in origin}
    rms_names_set = {n for n, _ in rms_norms}
    if not rms_names_set.issubset(origin_names):
        missing = sorted(rms_names_set - origin_names)
        raise ValueError(
            f"GRACE-2L export_kokkos: RMSNorm(s) {missing} are not in "
            f"LinMLPOut2ScalarTarget.origin {sorted(origin_names)}. The Kokkos "
            f"pair style assumes both RMSNorms feed the energy MLP directly."
        )

    # Build (yaml_prefix → legacy_prefix) renames from topology.
    renames = []
    for rms_name, rms_instr in rms_norms:
        legacy_rms, legacy_red = _LEGACY_2L_ROLE[rms_instr.type]
        if rms_name != legacy_rms:
            renames.append((rms_name, legacy_rms))

        src = getattr(rms_instr, "input", None)
        if src is None or type(src).__name__ != "FunctionReduceN":
            raise ValueError(
                f"GRACE-2L export_kokkos: RMSNorm {rms_name} input is "
                f"{type(src).__name__ if src else None}, expected "
                f"FunctionReduceN. The Kokkos pair style assumes a direct "
                f"FunctionReduceN → InvariantLayerRMSNorm chain on the energy path."
            )
        if src.name != legacy_red:
            renames.append((src.name, legacy_red))

    # Apply aliases. Both keys point at the same ndarray — np.savez will
    # write each member separately, so on-disk size grows by ~80 small
    # arrays + 2 W tensors, but in-memory state stays shared.
    aliased = 0
    for old_prefix, new_prefix in renames:
        for k in list(all_weights.keys()):
            if k.startswith(old_prefix + "_"):
                new_k = new_prefix + k[len(old_prefix):]
                if new_k not in all_weights:
                    all_weights[new_k] = all_weights[k]
                    aliased += 1
    if aliased:
        rmap = ", ".join(f"{a}→{b}" for a, b in renames)
        logger.info(
            f"Emitted {aliased} legacy alias(es) [{rmap}] so the Kokkos "
            f"pair style finds rho/I2/I_nl_LN/I_0_LN regardless of YAML names."
        )

    # Capability check 3: every legacy role ended up populated.
    missing_roles = [
        label for label, pred in _LEGACY_2L_REQUIRED
        if not any(pred(k) for k in all_weights.keys())
    ]
    if missing_roles:
        raise ValueError(
            f"GRACE-2L export_kokkos: legacy aliasing finished but the "
            f"following prefixes are still absent from the .npz: "
            f"{missing_roles}. The Kokkos pair style would load zeros for "
            f"these and produce zero forces. Likely an unexpected 2L topology."
        )


# ----------------------------------------------------------------------------
# Activation detection
# ----------------------------------------------------------------------------

def _check_activation(mlp_layers, allowed: set, arch_label: str):
    if not mlp_layers:
        return None
    layer0 = mlp_layers[0]
    act_fn = getattr(layer0, "activation", None)
    if act_fn is None:
        raise ValueError(
            f"{arch_label}: energy MLP layer0 has no `activation` attribute — "
            "cannot determine activation function for Kokkos export."
        )
    fn_name = getattr(act_fn, "__name__", str(act_fn)).lower()
    if "tanh" in fn_name:
        detected = "tanh"
    elif "silu" in fn_name or "swish" in fn_name:
        detected = "silu"
    elif "relu" in fn_name:
        detected = "relu"
    elif "gelu" in fn_name:
        detected = "gelu"
    else:
        raise ValueError(
            f"{arch_label}: unrecognized activation '{fn_name}' on energy MLP "
            f"layer0. Expected one of: silu, tanh, relu, gelu."
        )
    if detected not in allowed:
        raise ValueError(
            f"{arch_label}: energy MLP activation is '{detected}', but the "
            f"Kokkos pair style only supports {sorted(allowed)}. Retrain with "
            f"a supported activation or use the SavedModel export instead."
        )
    return detected


# ----------------------------------------------------------------------------
# Shift collapse
# ----------------------------------------------------------------------------

def _scale_value(instr):
    """Return ConstantScaleShiftTarget.scale as a float."""
    s = instr.scale.numpy() if hasattr(instr.scale, "numpy") else instr.scale
    return float(np.asarray(s).flatten()[0])


def _upstream_chain(instr):
    """Yield instr, instr.target, instr.target.target, ... while .target exists."""
    cur = instr
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = getattr(cur, "target", None)


def _collapse_shifts(instruction_dict, n_elements, chem_embedding_np):
    """Walk all shift-like instructions, sum their per-element additive
    contributions, and return (shift_values[n_elements] or None, contributions,
    output_scale).

    `output_scale` is the multiplicative factor from the (at most one)
    ConstantScaleShiftTarget with scale != 1; folded into MLP / reduce
    weights downstream by `_apply_output_scale`. Shift contributions that
    sit upstream of the scaling instruction in the .target chain are
    pre-multiplied by `output_scale` here, since the live model would
    apply the scale to them too.
    """
    scaling_instr = None
    output_scale = 1.0
    for name, instr in instruction_dict.items():
        if type(instr).__name__ != "ConstantScaleShiftTarget":
            continue
        s = _scale_value(instr)
        if np.isclose(s, 1.0):
            continue
        if scaling_instr is not None:
            raise ValueError(
                "Multiple ConstantScaleShiftTarget instructions with non-unit "
                "scale found. Only one is supported by the Kokkos exporter."
            )
        scaling_instr = instr
        output_scale = s

    inside_scale = set()
    if scaling_instr is not None:
        for u in _upstream_chain(getattr(scaling_instr, "target", None)):
            inside_scale.add(id(u))

    contributions = []
    accumulator = np.zeros(n_elements, dtype=np.float64)

    def add(values, label, *, scaled):
        nonlocal accumulator
        v = np.asarray(values).flatten().astype(np.float64)
        if v.shape[0] != n_elements:
            raise ValueError(
                f"{label}: contribution has {v.shape[0]} entries but "
                f"n_elements={n_elements}."
            )
        if scaled:
            v = v * output_scale
            label = f"{label} (× scale={output_scale:.6g})"
        accumulator += v
        contributions.append(label)

    for name, instr in instruction_dict.items():
        cls = type(instr).__name__
        if cls == "TrainableShiftTarget":
            if instr.at_shifts is None:
                continue
            scaled = id(instr) in inside_scale
            add(instr.at_shifts.numpy(), name, scaled=scaled)

        elif cls == "ConstantScaleShiftTarget":
            # The instruction's own additive shifts sit AFTER the scale
            # (`target * scale + total_shift`), so they are never scaled.
            cs = instr.constant_shift
            cs_arr = np.asarray(cs.numpy() if hasattr(cs, "numpy") else cs)
            cs_val = float(cs_arr.flatten()[0]) if cs_arr.size else 0.0
            if cs_val != 0.0:
                add(np.full(n_elements, cs_val), f"{name}.constant_shift", scaled=False)

            if instr.atomic_shift_map is not None:
                asm = instr.atomic_shift_map
                asm = asm.numpy() if hasattr(asm, "numpy") else np.asarray(asm)
                add(asm, f"{name}.atomic_shift_map", scaled=False)

            if (instr.chemical_embedding is not None
                    and hasattr(instr, "embedding_shift")):
                if chem_embedding_np is None:
                    raise ValueError(
                        f"ConstantScaleShiftTarget '{name}': uses chemical_embedding "
                        "but no ScalarChemicalEmbedding was found in the model."
                    )
                emb_shift = instr.embedding_shift.numpy()  # [embed_size, 1]
                emb_norm = (
                    instr.embedding_norm.numpy()
                    if hasattr(instr.embedding_norm, "numpy")
                    else float(instr.embedding_norm)
                )
                contribution = (chem_embedding_np @ emb_shift).flatten() * float(emb_norm)
                add(contribution, f"{name}.embedding_shift", scaled=False)

    if not contributions and np.isclose(output_scale, 1.0):
        return None, [], 1.0
    if not contributions:
        # output_scale != 1 but no additive shift; still need to fold the scale.
        return None, [], output_scale
    return accumulator, contributions, output_scale


# ----------------------------------------------------------------------------
# Caps validation
# ----------------------------------------------------------------------------

def _max_mlp_hidden_dim(all_weights, mlp_prefixes):
    """Largest hidden-buffer dim across the named MLPs. Excludes layer-0 input
    and final-layer output: those are model I/O sizes (e.g. nradmax*(lmax+1) on
    the radial MLP's output), not hidden buffers, and not what MAX_MLP_DIM caps."""
    m = 0
    for p in mlp_prefixes:
        n_key = f"{p}_n_layers"
        if n_key not in all_weights:
            continue
        n = int(all_weights[n_key][0])
        for i in range(n):
            w = all_weights.get(f"{p}_W{i}")
            if not (isinstance(w, np.ndarray) and w.ndim == 2):
                continue
            if i < n - 1:        # output of a hidden layer
                m = max(m, w.shape[1])
            if i > 0:            # input of a non-first layer
                m = max(m, w.shape[0])
    return m


def _2l_mlp_rad_prefixes(all_weights):
    if "mlp_rad_names" not in all_weights:
        return ()
    return tuple(
        f"mlp_rad_{n.decode() if isinstance(n, (bytes, np.bytes_)) else n}"
        for n in all_weights["mlp_rad_names"]
    )


def _assert_within_caps_1l(all_weights):
    caps = CAPS_1L
    violations = []

    mlp_dim = _max_mlp_hidden_dim(all_weights, ("mlp_rad", "energy_mlp"))
    if mlp_dim > caps["MAX_MLP_DIM"]:
        violations.append(
            f"max MLP hidden dim {mlp_dim} > MAX_MLP_DIM={caps['MAX_MLP_DIM']}"
        )

    if "nradmax" in all_weights:
        nrm = int(all_weights["nradmax"][0])
        if nrm > caps["MAX_NRADMAX"]:
            violations.append(f"nradmax {nrm} > MAX_NRADMAX={caps['MAX_NRADMAX']}")

    for key in ("mlp_rad_n_layers", "energy_mlp_n_layers"):
        if key in all_weights:
            n = int(all_weights[key][0])
            if n > caps["MAX_MLP_LAYERS"]:
                violations.append(f"{key}={n} > MAX_MLP_LAYERS={caps['MAX_MLP_LAYERS']}")

    if violations:
        raise ValueError(
            "GRACE-1L Kokkos compile-time caps exceeded:\n  - "
            + "\n  - ".join(violations)
            + "\nEither retrain at smaller dimensions or rebuild LAMMPS with "
            "the corresponding GRACE1L_MAX_* header constants raised."
        )


def _assert_within_caps_2l(all_weights):
    caps = CAPS_2L
    violations = []

    mlp_dim = _max_mlp_hidden_dim(
        all_weights, ("energy_mlp",) + _2l_mlp_rad_prefixes(all_weights)
    )
    if mlp_dim > caps["MAX_MLP_DIM"]:
        violations.append(
            f"max MLP hidden dim {mlp_dim} > MAX_MLP_DIM={caps['MAX_MLP_DIM']}"
        )

    for key in ("mlp_rad_R_n_rad_max", "mlp_rad_R1_n_rad_max"):
        if key in all_weights:
            n = int(all_weights[key][0])
            if n > caps["MAX_NRADMAX"]:
                violations.append(f"{key}={n} > MAX_NRADMAX={caps['MAX_NRADMAX']}")

    for key, cap_name in (("I1_n_funcs", "MAX_I1_FUNCS"),
                          ("I_n_funcs", "MAX_I_FUNCS"),
                          ("YI_n_output_funcs", "MAX_YI_OUT_FUNCS")):
        if key in all_weights:
            n = int(all_weights[key][0])
            if n > caps[cap_name]:
                violations.append(f"{key}={n} > {cap_name}={caps[cap_name]}")

    if "lmax" in all_weights:
        lm = int(all_weights["lmax"][0])
        if lm > caps["LMAX"]:
            violations.append(
                f"lmax={lm} > LMAX={caps['LMAX']} (spherical harmonic stacks "
                "require nlm <= 25, plm <= 15)"
            )

    if violations:
        raise ValueError(
            "GRACE-2L Kokkos compile-time caps exceeded:\n  - "
            + "\n  - ".join(violations)
            + "\nEither retrain at smaller dimensions or rebuild LAMMPS with "
            "the corresponding GRACE2L_MAX_* header constants raised."
        )


# ----------------------------------------------------------------------------
# Model load + checkpoint restore + globals extraction
# ----------------------------------------------------------------------------

def _load_and_restore(model_yaml_path, checkpoint_path, param_dtype_str):
    if param_dtype_str not in ("float32", "float64"):
        raise ValueError(
            f"param_dtype_str must be 'float32' or 'float64', got {param_dtype_str!r}"
        )
    param_dtype = tf.float64 if param_dtype_str == "float64" else tf.float32

    logger.info(f"Loading instructions from {model_yaml_path}...")
    instructions = load_instructions(model_yaml_path)
    instruction_dict = (
        instructions if isinstance(instructions, dict)
        else {ins.name: ins for ins in instructions}
    )

    logger.info(f"Building model with dtype={param_dtype_str}...")
    model = TPModel(instruction_dict)
    model.build(param_dtype=param_dtype)

    # Defensive: callers may have skipped preprocess_args (e.g. direct API use).
    for suffix in (".index", ".data-00000-of-00001"):
        if checkpoint_path.endswith(suffix):
            checkpoint_path = checkpoint_path[: -len(suffix)]
            logger.info(f"  Stripped suffix, using prefix: {checkpoint_path}")

    first_var = model.trainable_variables[0]
    pre_val = first_var.numpy().flat[0]

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = tf.train.Checkpoint(model=model)
    try:
        status = checkpoint.restore(checkpoint_path)
        status.expect_partial()
    except (tf.errors.NotFoundError, tf.errors.DataLossError) as e:
        raise FileNotFoundError(
            f"Checkpoint not found or corrupted: {checkpoint_path}"
        ) from e

    if first_var.numpy().flat[0] == pre_val:
        logger.warning("first variable unchanged after restore — checkpoint may be wrong")

    return instruction_dict


def _extract_globals(instruction_dict, model_yaml_path):
    out = {}

    _, Z_instr = _find_instr(instruction_dict, "ScalarChemicalEmbedding")
    _, RadBasis = _find_instr(instruction_dict, "RadialBasis")
    _, BondSpecificRBF = _find_instr(instruction_dict, "BondSpecificRadialBasisFunction")

    n_elements = (
        Z_instr.number_of_elements
        if Z_instr is not None and hasattr(Z_instr, "number_of_elements")
        else 89
    )
    out["n_elements"] = np.array([n_elements])

    element_names = _resolve_element_names(Z_instr, model_yaml_path)
    out["element_names"] = np.array(element_names, dtype="S2")
    out["embedding_size"] = np.array([Z_instr.embedding_size if Z_instr else 128])

    has_bond_specific = False
    if BondSpecificRBF is not None:
        out["nradbase"] = np.array([BondSpecificRBF.nfunc])
        out["radial_basis_p"] = np.array([int(BondSpecificRBF.cutoff_function_param)])
        bcm = np.array(BondSpecificRBF.bond_cutoff_map).reshape(n_elements, n_elements)
        out["bond_cutoff_map"] = bcm
        max_rcut = float(np.max(bcm))
        out["rcut"] = np.array([max_rcut])
        out["has_bond_specific_cutoff"] = np.array([1])
        has_bond_specific = True
        logger.info(
            f"  Bond-specific cutoffs: min={float(np.min(bcm)):.2f}, max={max_rcut:.2f}"
        )
    elif RadBasis is not None:
        out["nradbase"] = np.array([RadBasis.nfunc])
        bf = RadBasis.basis_function
        out["rcut"] = np.array([bf.rcut if bf else 6.0])
        out["radial_basis_p"] = np.array([bf.pcut if bf else 16])
        out["has_bond_specific_cutoff"] = np.array([0])
    else:
        out["nradbase"] = np.array([10])
        out["rcut"] = np.array([6.0])
        out["radial_basis_p"] = np.array([16])
        out["has_bond_specific_cutoff"] = np.array([0])

    # 0 = float64. Always 0 because _promote_to_float64 casts everything to fp64.
    out["nn_dtype"] = np.array([0])
    return out, n_elements, has_bond_specific


def _resolve_element_names(Z_instr, model_yaml_path):
    """Prefer element_map_symbols on the Z instruction; fall back to a
    metadata.yaml sitting next to the model.yaml (legacy-serialized models)."""
    if Z_instr is not None and hasattr(Z_instr, "element_map_symbols"):
        syms = Z_instr.element_map_symbols.numpy()
        idx = Z_instr.element_map_index.numpy()
        order = np.argsort(idx)
        names = [syms[i].decode() for i in order]
        logger.info(f"  Element map from Z instruction: {names}")
        return names

    model_dir = os.path.dirname(model_yaml_path)
    for candidate in (os.path.join(model_dir, "metadata.yaml"),
                      os.path.join(model_dir, "..", "metadata.yaml")):
        if os.path.exists(candidate):
            with open(candidate) as f:
                meta = yaml.safe_load(f)
            if "chemical_symbols" in meta:
                names = meta["chemical_symbols"]
                logger.info(f"  Element map from {candidate}: {names}")
                return names

    raise FileNotFoundError(
        "Could not determine element names from Z instruction or metadata.yaml"
    )


def _check_supported(instruction_dict, supported_set, arch_label):
    bad = [
        f"{name} ({type(instr).__name__})"
        for name, instr in instruction_dict.items()
        if type(instr).__name__ not in supported_set
    ]
    if bad:
        raise ValueError(
            f"{arch_label} export_kokkos: unsupported instruction class(es) "
            f"in model — the Kokkos pair style cannot evaluate them, so "
            f"writing the .npz would silently produce wrong forces:\n  - "
            + "\n  - ".join(bad)
            + f"\nSupported classes: {sorted(supported_set - _IGNORED_CLASSES)}"
        )


def _promote_to_float64(all_weights):
    """Cast non-float64 floating values to float64 in-place so model weights
    keep their full fp64 precision on disk. The Kokkos pair style's
    `npz_get_double` is word_size-aware (it upcasts float32 -> double), so a
    float32 weight would load without error — but its low mantissa bits are
    already gone, hence the promotion here. (UQ arrays are deliberately written
    float32 AFTER this call — see `_merge_uq_artifacts`.) Catches both
    np.ndarray and the np.generic scalars that `tf.Tensor.numpy()` returns
    for 0-d tensors. Integer / byte-string arrays are left alone."""
    promoted = 0
    for k, v in list(all_weights.items()):
        if isinstance(v, np.ndarray):
            if v.dtype.kind == "f" and v.dtype != np.float64:
                all_weights[k] = v.astype(np.float64)
                promoted += 1
        elif isinstance(v, np.generic):
            if np.dtype(type(v)).kind == "f" and np.dtype(type(v)) != np.float64:
                all_weights[k] = np.array(v, dtype=np.float64)
                promoted += 1
    if promoted:
        logger.info(
            f"Promoted {promoted} float32 value(s) → float64 for Kokkos pair style"
        )


def _log_summary(arch, n_elements, all_weights, dtype_str, has_bond_specific):
    total_float_params = sum(
        v.size for v in all_weights.values()
        if isinstance(v, np.ndarray) and v.dtype.kind == "f"
    )

    logger.info("=== Architecture summary ===")
    logger.info(f"  arch:            GRACE-{arch.upper()}")
    logger.info(f"  n_elements:      {n_elements}")
    logger.info(f"  embedding_size:  {int(all_weights['embedding_size'][0])}")
    if "lmax" in all_weights:
        logger.info(f"  lmax:            {int(all_weights['lmax'][0])}")
    if "nradmax" in all_weights:
        logger.info(f"  nradmax:         {int(all_weights['nradmax'][0])}")
    if "mlp_rad_R_n_rad_max" in all_weights:
        logger.info(f"  L1_nradmax:      {int(all_weights['mlp_rad_R_n_rad_max'][0])}")
    if "mlp_rad_R1_n_rad_max" in all_weights:
        logger.info(f"  L2_nradmax:      {int(all_weights['mlp_rad_R1_n_rad_max'][0])}")
    logger.info(f"  nradbase:        {int(all_weights['nradbase'][0])}")
    rcut = float(all_weights["rcut"][0])
    logger.info(f"  rcut:            {rcut}{' (bond-specific)' if has_bond_specific else ''}")
    logger.info(f"  param_dtype:     {dtype_str} (stored as float64)")
    logger.info(f"  total params:    {total_float_params:,}")
    logger.info(f"  → load with: pair_style grace/{arch}/kk")
    logger.info("=== End summary ===")


def _merge_uq_artifacts(all_weights, uq_artifacts_path, n_elements, instruction_dict):
    """Bake a GMM-UQ artifact into ``all_weights`` as dense ``uq_*`` arrays.

    Reuses :class:`GMMUQModel._build_tensors` for densification — eff-vs-raw
    threshold preference and missing-element isotropic inv_cov — then validates
    the artifact is aligned with the exported model before copying the dense
    numpy arrays out under a ``uq_`` prefix. The resulting kokkos .npz carries
    everything the C++ pair style needs to compute per-atom sigma and gamma.
    """
    from tensorpotential.uq import constants as uqc
    from tensorpotential.uq.gmmuq import GMMUQModel, _MISSING_ELEM_INV_COV_DIAG

    logger.info(f"Loading UQ artifacts from {uq_artifacts_path}...")
    # Validates the schema on load and builds the dense padded tensors.
    m = GMMUQModel(uq_artifacts_path)

    # --- alignment checks (fail hard with a clear message) ---
    if m.n_elements != n_elements:
        raise ValueError(
            f"UQ artifact element count ({m.n_elements}) != model element count "
            f"({n_elements}); the artifact was built for a different model."
        )

    raw_names = all_weights.get("element_names")
    model_symbols = (
        [e.decode() if isinstance(e, bytes) else str(e) for e in raw_names]
        if raw_names is not None
        else None
    )
    uq_symbols = m.element_symbols
    if (
        model_symbols is not None
        and uq_symbols is not None
        and uq_symbols != model_symbols
    ):
        raise ValueError(
            f"UQ artifact element order {uq_symbols} != model element "
            f"order {model_symbols}; per-element UQ rows would be misaligned."
        )

    # basis-RP feature: D must equal the stored projection out_dim (uq_rp_dim).
    # The retired readout-hidden feature (D == 1 + last energy-MLP hidden width)
    # is unsupported. Parse the spec via the single shared reader.
    from tensorpotential.uq.factories import (
        _basis_rp_spec_from_artifact,
        trace_energy_path_invariant_reduces,
    )

    spec = _basis_rp_spec_from_artifact(m)
    if spec is None or spec.get("matrix") is None:
        raise ValueError(
            "kokkos UQ export requires a basis-RP artifact (missing "
            "uq_feature_mode='basis_rp' / uq_rp_matrix); rebuild with "
            "`grace_uq build`."
        )
    rp_matrix = spec["matrix"]
    rp_dim = spec["out_dim"]
    # Feature dim D = rp_dim + n_density. uqv6 appends n_density = 1 + n_blocks
    # log-norm density channels (full basis + one per invariant reduce block);
    # uqv3/uqv4 add none (D == rp_dim). Validate against the model's actual block
    # count so an artifact built for a different feature config fails loudly.
    n_blocks = len(trace_energy_path_invariant_reduces(instruction_dict))
    expected_n_density = (1 + n_blocks) if spec["add_density_channel"] else 0
    expected_D = rp_dim + expected_n_density
    if m.D != expected_D:
        raise ValueError(
            f"UQ feature dim D={m.D} != rp_dim {rp_dim} + n_density "
            f"{expected_n_density} (={expected_D}); artifact/model feature config "
            "mismatch — rebuild with `grace_uq build`."
        )

    if m.interp_thresholds is None:
        raise ValueError(
            "UQ artifact has no interpolation thresholds; gamma cannot be computed. "
            "Rebuild the artifact with grace_uq."
        )

    # --- write dense uq_* arrays ---
    # UQ float arrays are stored float32 (the GMM stats and R carry no fp64
    # precision — the model is typically fp32 and R is a random projection), which
    # roughly halves the (large) UQ payload in the kokkos .npz. These are added
    # AFTER _promote_to_float64, so the float32 dtype survives to disk.
    # The Kokkos pair style's npz_get_double is word_size-aware (reads float32 ->
    # upcasts to double, mirroring npz_get_int), so it consumes these fp32 arrays
    # correctly. No backward-compat guard for pre-fp32 LAMMPS builds is provided
    # (none are in use).
    # Schema version == uqv feature-variant tag, derived from the artifact's own
    # spec flags (uqv6 normalize+density => 6, uqv4 asinh => 4, linear => 3), so the
    # kk loader can gate on it and reject feature combos it doesn't implement.
    uq_schema = uqc.schema_version_for_spec(
        normalize=spec["normalize"],
        add_density_channel=spec["add_density_channel"],
        transform=spec["transform"],
    )
    all_weights["uq_schema_version"] = np.array([uq_schema], dtype=np.int64)
    all_weights["uq_n_elements"] = np.array([m.n_elements], dtype=np.int64)
    all_weights["uq_max_clusters"] = np.array([m.K_max], dtype=np.int64)
    all_weights["uq_feature_dim"] = np.array([m.D], dtype=np.int64)
    all_weights["uq_missing_elem_inv_cov_diag"] = np.array(
        [_MISSING_ELEM_INV_COV_DIAG], dtype=np.float32
    )

    all_weights["uq_centroids"] = m.centroids.numpy().astype(np.float32)
    all_weights["uq_inv_cov"] = m.inv_covs.numpy().astype(np.float32)
    all_weights["uq_n_clusters"] = m.n_clusters_per_elem.numpy().astype(np.int32)
    all_weights["uq_interp_thresholds"] = m.interp_thresholds.numpy().astype(np.float32)

    # basis-RP projection R[D_basis, rp_dim], stored verbatim so the C++ pair
    # style can reproduce the feature (gather the invariant energy-path basis,
    # then matmul R). Stored float32 like the other uq_* arrays (added after
    # _promote_to_float64, so set the dtype explicitly).
    all_weights["uq_rp_dim"] = np.array([rp_dim], dtype=np.int64)
    all_weights["uq_rp_matrix"] = np.asarray(rp_matrix).astype(np.float32)
    rp_seed = m.extra_data.get(uqc.UQ_RP_SEED)
    if rp_seed is not None:
        all_weights["uq_rp_seed"] = np.array(
            [int(np.asarray(rp_seed).item())], dtype=np.int64
        )

    # uqv6 feature-construction flags. Without these the C++ pair style can only
    # reproduce the plain linear B@R feature (uqv3); uqv4/uqv6 also L2-normalize
    # the basis, append log-norm density channels, and/or apply an element-wise
    # transform. Stored as int/float scalars (the kokkos npz reader is numeric).
    # See pgs/uq/uqv6_desc.md §2.2 for the exact feature math, and §4b for this
    # key layout. n_density channels = uq_feature_dim - uq_rp_dim.
    all_weights["uq_rp_normalize"] = np.array(
        [1 if spec["normalize"] else 0], dtype=np.int64
    )
    all_weights["uq_rp_add_density_channel"] = np.array(
        [1 if spec["add_density_channel"] else 0], dtype=np.int64
    )
    all_weights["uq_rp_density_scale"] = np.array(
        [spec["density_scale"]], dtype=np.float32
    )
    # transform code: 0 = identity/none (uqv3, uqv6), 1 = asinh (uqv4).
    _transform_codes = {None: 0, "asinh": 1}
    if spec["transform"] not in _transform_codes:
        raise ValueError(
            f"kokkos UQ export: unsupported feature_transform "
            f"{spec['transform']!r} (expected one of {list(_transform_codes)})"
        )
    all_weights["uq_feature_transform"] = np.array(
        [_transform_codes[spec["transform"]]], dtype=np.int64
    )

    logger.info(
        f"Baked UQ artifacts (schema v{uq_schema}): n_elements={m.n_elements}, "
        f"K_max={m.K_max}, D={m.D}, normalize={spec['normalize']}, "
        f"density={spec['add_density_channel']} (scale={spec['density_scale']:g}), "
        f"transform={spec['transform'] or 'none'}."
    )


def _finalize_and_save(arch, instruction_dict, all_weights, n_elements, chem_embedding_np,
                       output_path, param_dtype_str, has_bond_specific, caps_validator,
                       uq_artifacts_path=None):
    shift_values, contributions, output_scale = _collapse_shifts(
        instruction_dict, n_elements, chem_embedding_np
    )
    if shift_values is not None:
        all_weights["shift_values"] = shift_values
        logger.info(
            f"Collapsed shifts from: {contributions} → shift_values[{n_elements}]"
        )
    if not np.isclose(output_scale, 1.0):
        all_weights["output_scale"] = np.array([output_scale])
        logger.info(
            f"Wrote output_scale={output_scale:.6g} (applied by the C++ pair "
            f"style as `E_atom = (mlp(features) + lin_origin) * output_scale + "
            f"shift_values[type]`)."
        )

    _promote_to_float64(all_weights)
    if arch == "2l":
        _emit_2l_legacy_aliases(instruction_dict, all_weights)
    caps_validator(all_weights)
    _log_summary(arch, n_elements, all_weights, param_dtype_str, has_bond_specific)

    # Bake UQ artifacts in last, after caps validation, so the uq_* keys are never
    # inspected by the architecture caps validators above.
    if uq_artifacts_path is not None:
        _merge_uq_artifacts(all_weights, uq_artifacts_path, n_elements, instruction_dict)

    logger.info(f"Saving {len(all_weights)} arrays to {output_path}...")
    np.savez(output_path, **all_weights)
    logger.info(f"Output saved to: {output_path}")


# ----------------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------------

def export_1l_npz(model_yaml_path, checkpoint_path, output_path,
                  param_dtype_str="float64", uq_artifacts_path=None):
    """Export GRACE-1L model to flat-prefix .npz layout."""
    instruction_dict = _load_and_restore(model_yaml_path, checkpoint_path, param_dtype_str)
    _check_supported(instruction_dict, _SUPPORTED_1L, "GRACE-1L")

    all_weights, n_elements, has_bond_specific = _extract_globals(
        instruction_dict, model_yaml_path
    )

    _, R_instr = _find_instr(instruction_dict, "MLPRadialFunction_v2")
    all_weights["lmax"] = np.array([R_instr.lmax if R_instr else 4])
    all_weights["nradmax"] = np.array([R_instr.n_rad_max if R_instr else 32])

    has_chem_emb_in_R = (
        R_instr is not None
        and hasattr(R_instr, "chem_embedding")
        and R_instr.chem_embedding is not None
    )
    all_weights["mlp_rad_has_chem_emb"] = np.array([1 if has_chem_emb_in_R else 0])

    chem_embedding_np = None
    for name, instr in instruction_dict.items():
        cls = type(instr).__name__
        logger.info(f"Processing {name} ({cls})...")

        if cls == "ScalarChemicalEmbedding":
            d = _extract_chem_embedding(instr)
            all_weights.update(d)
            chem_embedding_np = d.get("chem_embedding")
        elif cls == "MLPRadialFunction_v2":
            all_weights.update(_extract_mlp_radial(instr, prefix="mlp_rad"))
        elif cls == "SingleParticleBasisFunctionScalarInd":
            all_weights.update(_extract_single_particle_basis(instr, prefix="A"))
        elif cls == "FCRight2Left":
            all_weights.update(_extract_fc_weights(instr, prefix=f"fc_{name}"))
        elif cls == "ProductFunction":
            all_weights.update(_extract_product_metadata(instr, prefix=f"prod_{name}"))
        elif cls == "FunctionReduceN":
            all_weights.update(_extract_reduce(instr, prefix="rho", include_2l_keys=False))
        elif cls == "LinMLPOut2ScalarTarget":
            all_weights.update(_extract_energy_mlp(
                instr, prefix="energy_mlp",
                allowed_activations=_ACT_SUPPORTED_1L,
                write_activation_code=True,
                arch_label="GRACE-1L",
            ))

    _finalize_and_save(
        "1l", instruction_dict, all_weights, n_elements, chem_embedding_np,
        output_path, param_dtype_str, has_bond_specific, _assert_within_caps_1l,
        uq_artifacts_path=uq_artifacts_path,
    )
    return all_weights


def export_2l_npz(model_yaml_path, checkpoint_path, output_path,
                  param_dtype_str="float64", uq_artifacts_path=None):
    """Export GRACE-2L model to named-prefix .npz layout with *_names index lists."""
    instruction_dict = _load_and_restore(model_yaml_path, checkpoint_path, param_dtype_str)
    _check_supported(instruction_dict, _SUPPORTED_2L, "GRACE-2L")

    all_weights, n_elements, has_bond_specific = _extract_globals(
        instruction_dict, model_yaml_path
    )

    _, Y_instr = _find_instr(instruction_dict, "SphericalHarmonic")
    all_weights["lmax"] = np.array([Y_instr.lmax if Y_instr else 4])

    mlp_rad_names, spb_names, fc_names = [], [], []
    prod_names, reduce_names, rms_names = [], [], []
    chem_embedding_np = None

    for name, instr in instruction_dict.items():
        cls = type(instr).__name__
        logger.info(f"Processing {name} ({cls})...")

        if cls == "ScalarChemicalEmbedding":
            d = _extract_chem_embedding(instr)
            all_weights.update(d)
            chem_embedding_np = d.get("chem_embedding")
        elif cls == "MLPRadialFunction_v2":
            prefix = f"mlp_rad_{name}"
            all_weights.update(_extract_mlp_radial(instr, prefix=prefix))
            mlp_rad_names.append(name)
            has_chem_emb = (
                hasattr(instr, "chem_embedding") and instr.chem_embedding is not None
            )
            all_weights[f"{prefix}_has_chem_emb"] = np.array([1 if has_chem_emb else 0])
        elif cls == "SingleParticleBasisFunctionScalarInd":
            all_weights.update(_extract_single_particle_basis(instr, prefix=name))
            spb_names.append(name)
        elif cls == "SingleParticleBasisFunctionEquivariantInd":
            all_weights.update(_extract_equivariant_ind(instr, prefix=name))
        elif cls == "FCRight2Left":
            all_weights.update(_extract_fc_weights(instr, prefix=f"fc_{name}"))
            fc_names.append(name)
        elif cls == "ProductFunction":
            all_weights.update(_extract_product_metadata(instr, prefix=f"prod_{name}"))
            prod_names.append(name)
        elif cls == "FunctionReduceN":
            all_weights.update(_extract_reduce(instr, prefix=name, include_2l_keys=True))
            reduce_names.append(name)
        elif cls == "InvariantLayerRMSNorm":
            all_weights.update(_extract_rms_norm(instr, prefix=name))
            rms_names.append(name)
        elif cls == "LinMLPOut2ScalarTarget":
            all_weights.update(_extract_energy_mlp(
                instr, prefix="energy_mlp",
                allowed_activations=_ACT_SUPPORTED_2L,
                write_activation_code=True,
                arch_label="GRACE-2L",
            ))

    all_weights["mlp_rad_names"] = np.array(mlp_rad_names, dtype="S8")
    all_weights["spb_names"] = np.array(spb_names, dtype="S8")
    all_weights["fc_names"] = np.array(fc_names, dtype="S8")
    all_weights["prod_names"] = np.array(prod_names, dtype="S8")
    all_weights["reduce_names"] = np.array(reduce_names, dtype="S8")
    all_weights["rms_names"] = np.array(rms_names, dtype="S8")

    _finalize_and_save(
        "2l", instruction_dict, all_weights, n_elements, chem_embedding_np,
        output_path, param_dtype_str, has_bond_specific, _assert_within_caps_2l,
        uq_artifacts_path=uq_artifacts_path,
    )
    return all_weights


# ----------------------------------------------------------------------------
# GRACE-3L export
#
# The .npz key schema is dictated by what the C++ pair style reads; do not
# change key names without also updating the reader in
# src/KOKKOS/pair_grace_3l_kokkos.cpp.
# ----------------------------------------------------------------------------

# Energy-MLP activations the grace/3l/kk pair style supports. Mirrors the
# 1L/2L C++ codes {0=silu, 1=tanh}; anything else aborts at load. The reference
# model uses silu (swish) on the readout MLP.
_ACT_SUPPORTED_3L = {"silu", "tanh"}


# Compile-time caps — AUTHORITATIVE for the C++ header (mirrored into
# GRACE3L_MAX_* constants). Each value is the dimension measured off
# GRACE-3L-OMAT-large-ft-AM plus ~25% slack, rounded sensibly. `MAX_NRADMAX` and
# `LMAX` are architectural ceilings (harmonic stacks require nlm<=25 for lmax=4)
# and carry no slack. `_assert_within_caps_3l` validates every instruction
# against these on export; the C++ loader re-checks on load.
#
#   cap                  raw max (this model)   source
#   MAX_NRADMAX          64                     SPBF.n_rad_max (A1/A2/A3)
#   LMAX                 4                      SPBF._lmax / Lmax
#   MAX_NRADBASIS        10                     SPBF.n_rad_basis (Chebyshev)
#   MAX_MLP_DIM          256                    radial-MLP hidden (also >= 320 radial out = 64*(4+1))
#   MAX_MLP_LAYERS       3                      radial MLP (138,256)->(256,128)->(128,320)
#   MAX_SPBF_NFUNC       750                    A3 CG-couple nfunc
#   MAX_SPBF_NLMIND      50                     A3 n_lm_indicator
#   MAX_PROD_RANK        64                     A1_2/A2_2/A3_2 cp_l rank
#   MAX_PROD_NFUNC       1500                   A2_3 product nfunc
#   MAX_PROD_NIN         128                    max(n_left,n_right) across products
#   MAX_PROD_NGROUPS     10                     n_groups_left/right (L2/L3 products)
#   MAX_PROD_NCG         8208                   A2_3 cp_l coupling terms
#   MAX_REDUCE_NOUT      128                    A*_2_red / A*_red n_out
#   MAX_REDUCE_NIN       128                    largest reduce collector input width
#   MAX_REDUCE_NFUNCS    50                     largest reduce coupling_meta_data (A2/A3 equiv)
#   MAX_REDUCE_WSHAPE    260                    A2_3_red reducing-weight w_shape
#   MAX_FC_NOUT          128                    FC n_out / feature width
#   MAX_FC_TILE          10                     FC w_right tile dim (A2/A3 FCs)
#   MAX_RHO_NOUT         32                     rho*_norm / rho* density width
#   MAX_ENERGY_MLP_DIM   64                     readout MLP hidden (31->64->1)
#   MAX_ATOM_TYPES       89                     n_elements / elem-dependent reduce leading dim
CAPS_3L = {
    "MAX_NRADMAX": 64,        # architectural ceiling (no slack)
    "LMAX": 4,                # architectural ceiling (no slack)
    "MAX_NRADBASIS": 12,      # 10 -> 12
    "MAX_MLP_DIM": 320,       # 256 -> 320 (also covers radial output 64*(lmax+1)=320)
    "MAX_MLP_LAYERS": 4,      # 3 -> 4
    "MAX_SPBF_NFUNC": 960,    # 750 -> 960
    "MAX_SPBF_NLMIND": 64,    # 50 -> 64
    "MAX_PROD_RANK": 80,      # 64 -> 80
    "MAX_PROD_NFUNC": 1920,   # 1500 -> 1920
    "MAX_PROD_NIN": 160,      # 128 -> 160
    "MAX_PROD_NGROUPS": 16,   # 10 -> 16
    "MAX_PROD_NCG": 10400,    # 8208 -> 10400
    "MAX_REDUCE_NOUT": 160,   # 128 -> 160
    "MAX_REDUCE_NIN": 160,    # 128 -> 160
    "MAX_REDUCE_NFUNCS": 64,  # 50 -> 64
    "MAX_REDUCE_WSHAPE": 336, # 260 -> 336
    "MAX_FC_NOUT": 160,       # 128 -> 160
    "MAX_FC_TILE": 16,        # 10 -> 16
    "MAX_RHO_NOUT": 40,       # 32 -> 40
    "MAX_ENERGY_MLP_DIM": 80, # 64 -> 80
    "MAX_ATOM_TYPES": 112,    # 89 -> 112
}


# Supported-class set for GRACE-3L. `_IGNORED_CLASSES` (geometry instructions such as
# BondLength/ScaledBondVector/SphericalHarmonic/CreateOutputTarget) contribute nothing to the
# npz but must not trigger the unknown-class error.
_SUPPORTED_3L = {
    "SPBF",
    "GeneralProductFunction",
    "FunctionReduceN",
    "FCRight2Left",
    "InvariantLayerRMSNorm",
    "EquivariantRMSNorm",
    "LinMLPOut2ScalarTarget",
    "ScalarChemicalEmbedding",
    "ConstantScaleShiftTarget",
} | _IGNORED_CLASSES


def _check_supported_3l(instruction_dict):
    # Delegate to the generic checker (used by the 1L/2L exporters) so this
    # stays consistent with the shared error semantics.
    _check_supported(instruction_dict, _SUPPORTED_3L, "GRACE-3L")


def _decode_names(arr):
    """Decode an S-dtype names index array to a list of python str."""
    return [
        n.decode() if isinstance(n, (bytes, np.bytes_)) else str(n)
        for n in arr
    ]


def _mlp_hidden_max_3l(all_weights, prefix, n_layers):
    """Largest hidden-buffer dim of an MLP stored as `{prefix}_W{i}`. Excludes the
    layer-0 input and the final-layer output (those are model I/O sizes, not the
    fixed hidden buffers `MAX_MLP_DIM` caps) — mirrors `_max_mlp_hidden_dim` in
    the shared exporter."""
    m = 0
    for i in range(n_layers):
        w = all_weights.get(f"{prefix}_W{i}")
        if not (isinstance(w, np.ndarray) and w.ndim == 2):
            continue
        if i < n_layers - 1:   # output of a hidden layer
            m = max(m, w.shape[1])
        if i > 0:              # input of a non-first layer
            m = max(m, w.shape[0])
    return m


def _assert_within_caps_3l(all_weights):
    """Validate every extracted GRACE-3L instruction dimension against CAPS_3L.
    Raises with a full violation list (like `_assert_within_caps_2l`) so a model
    that exceeds the compiled C++ buffers fails loudly at export rather than
    loading into a too-small kernel and silently producing wrong forces. Caps are
    logged in the export summary. Iterates via the `*_names` index lists."""
    caps = CAPS_3L
    v = []

    def _get(key):
        a = all_weights.get(key)
        return int(a[0]) if a is not None else None

    def _chk(val, cap_name, label):
        if val is not None and val > caps[cap_name]:
            v.append(f"{label}={val} > {cap_name}={caps[cap_name]}")

    # --- SPBF (radial MLP + basis) ---
    for name in _decode_names(all_weights.get("spbf_names", np.array([], dtype="S8"))):
        _chk(_get(f"{name}_n_rad_max"), "MAX_NRADMAX", f"{name}.n_rad_max")
        _chk(_get(f"{name}_n_rad_basis"), "MAX_NRADBASIS", f"{name}.n_rad_basis")
        _chk(_get(f"{name}_lmax"), "LMAX", f"{name}._lmax")
        _chk(_get(f"{name}_Lmax"), "LMAX", f"{name}.Lmax")
        nl = _get(f"{name}_mlp_n_layers")
        if nl is not None:
            _chk(nl, "MAX_MLP_LAYERS", f"{name}.mlp_n_layers")
            _chk(_mlp_hidden_max_3l(all_weights, f"{name}_mlp", nl),
                 "MAX_MLP_DIM", f"{name}.mlp_hidden")
        if f"{name}_nfunc" in all_weights:   # equivariant mode only
            _chk(_get(f"{name}_nfunc"), "MAX_SPBF_NFUNC", f"{name}.nfunc")
            _chk(_get(f"{name}_n_lm_ind"), "MAX_SPBF_NLMIND", f"{name}.n_lm_ind")

    # --- cp_l products ---
    for name in _decode_names(all_weights.get("prod_names", np.array([], dtype="S8"))):
        _chk(_get(f"{name}_rank"), "MAX_PROD_RANK", f"{name}.rank")
        _chk(_get(f"{name}_nfunc"), "MAX_PROD_NFUNC", f"{name}.nfunc")
        _chk(_get(f"{name}_n_left"), "MAX_PROD_NIN", f"{name}.n_left")
        _chk(_get(f"{name}_n_right"), "MAX_PROD_NIN", f"{name}.n_right")
        _chk(_get(f"{name}_n_groups_left"), "MAX_PROD_NGROUPS", f"{name}.n_groups_left")
        _chk(_get(f"{name}_n_groups_right"), "MAX_PROD_NGROUPS", f"{name}.n_groups_right")
        cg = all_weights.get(f"{name}_cg")
        if cg is not None:
            _chk(int(cg.shape[0]), "MAX_PROD_NCG", f"{name}.n_cg")

    # --- reduces (product reduces + elem-dep eq*/rho*) ---
    for name in _decode_names(all_weights.get("reduce_names", np.array([], dtype="S8"))):
        _chk(_get(f"{name}_n_out"), "MAX_REDUCE_NOUT", f"{name}.n_out")
        _chk(_get(f"{name}_n_funcs"), "MAX_REDUCE_NFUNCS", f"{name}.n_funcs")
        if _get(f"{name}_is_central_atom_type_dependent"):
            _chk(_get(f"{name}_number_of_atom_types"), "MAX_ATOM_TYPES",
                 f"{name}.number_of_atom_types")
        for cname in _decode_names(
            all_weights.get(f"{name}_instruction_names", np.array([], dtype="S8"))
        ):
            _chk(_get(f"{name}_w_shape_{cname}"), "MAX_REDUCE_WSHAPE",
                 f"{name}.w_shape[{cname}]")
            _chk(_get(f"{name}_n_in_{cname}"), "MAX_REDUCE_NIN",
                 f"{name}.n_in[{cname}]")

    # --- FCs ---
    for name in _decode_names(all_weights.get("fc_names", np.array([], dtype="S8"))):
        _chk(_get(f"fc_{name}_n_out"), "MAX_FC_NOUT", f"fc_{name}.n_out")
        wr = all_weights.get(f"fc_{name}_w_right")
        if wr is not None and wr.ndim == 3:
            _chk(int(wr.shape[1]), "MAX_FC_NOUT", f"fc_{name}.w_right.feat")
            _chk(int(wr.shape[2]), "MAX_FC_TILE", f"fc_{name}.w_right.tile")

    # --- invariant RMS-norms (readout densities) ---
    for name in _decode_names(all_weights.get("rmsnorm_names", np.array([], dtype="S8"))):
        _chk(_get(f"{name}_n_out"), "MAX_RHO_NOUT", f"{name}.n_out")

    # --- readout energy MLP ---
    nl = _get("energy_mlp_n_layers")
    if nl is not None:
        _chk(nl, "MAX_MLP_LAYERS", "energy_mlp.n_layers")
        _chk(_mlp_hidden_max_3l(all_weights, "energy_mlp", nl),
             "MAX_ENERGY_MLP_DIM", "energy_mlp.hidden")

    # --- global atom-type count ---
    _chk(_get("n_elements"), "MAX_ATOM_TYPES", "n_elements")

    logger.info(
        "CAPS_3L check: %s",
        ", ".join(f"{k}={val}" for k, val in caps.items()),
    )
    if v:
        raise ValueError(
            "GRACE-3L Kokkos compile-time caps exceeded:\n  - "
            + "\n  - ".join(v)
            + "\nEither retrain at smaller dimensions or rebuild LAMMPS with the "
            "corresponding GRACE3L_MAX_* header constants raised (and bump CAPS_3L)."
        )


def _norm_to_np(norm):
    """1-element-array form of a possibly-tensor scalar norm, matching how the shared
    exporter stores `lin_transform`/`layer` norms (`_extract_single_particle_basis` /
    `_extract_layer_weights`). The norm is stored *separately* — NOT folded into the
    weight — because the consumer applies `w*norm` (and `b*norm`) once at runtime,
    exactly as `tensorpotential.functions.nn.Linear.__call__` does. Pre-folding here
    would make the C++/oracle double-apply it."""
    return np.array([float(norm.numpy()) if hasattr(norm, "numpy") else float(norm)])


def _extract_spbf(instruction, prefix):
    """Extract an `SPBF` (Single Particle Basis Function) instruction — the fused
    Chebyshev-radial + MLP + angular-couple + neighbor-sum basis builder.

    Two modes (auto-detected from `instruction.equivariant_mode`):
      * **scalar** (`A1`; indicator is a `ScalarChemicalEmbedding`): produces the
        L1 scalar-indicator basis; carries a `lin_transform` species modulation.
      * **equivariant** (`A2`/`A3`; indicator is a `TPEquivariantInstruction`):
        produces the equivariant-indicator basis with an L=0 chemistry injection
        (`chem_linear` + `chem_l0_mask`) and a dense Clebsch-Gordan couple matrix.

    Norm-fold convention: the radial-MLP `Linear` layers, `lin_transform` and
    `chem_linear` all store the *raw* weight/bias plus the separate scalar
    `norm = 1/sqrt(n_in)`; the consumer applies the norm once (see `_norm_to_np`).
    This mirrors `_extract_layer_weights` / `_extract_single_particle_basis` exactly.
    """
    p = prefix
    weights = {}

    # --- Radial MLP layers: raw W{i} / scalar norm{i} / raw b{i} (hidden only) ---
    # `instruction.mlp_layers` are `Linear` objects (hidden layers use bias, output
    # layer has use_bias=False so no b key). Reuse the shared per-layer extractor so
    # the storage (dtype, norm-separate) convention is byte-for-byte the same as 1L/2L.
    for i, layer in enumerate(instruction.mlp_layers):
        _extract_layer_weights(layer, f"{p}_mlp", i, weights)
    weights[f"{p}_mlp_n_layers"] = np.array([len(instruction.mlp_layers)])

    # --- Scalars / metadata ---
    weights[f"{p}_n_rad_max"] = np.array([instruction.n_rad_max])
    weights[f"{p}_n_rad_basis"] = np.array([instruction.n_rad_basis])
    # `_lmax` = angular coupling lmax (R*Y and l_tile span (lmax+1)^2 channels);
    # `lmax` (set via TPEquivariantInstruction super().__init__) = output Lmax.
    weights[f"{p}_lmax"] = np.array([instruction._lmax])
    weights[f"{p}_Lmax"] = np.array([instruction.lmax])
    weights[f"{p}_p"] = np.array([instruction.p])
    weights[f"{p}_rcut"] = np.array([float(instruction.get_cutoff())])
    # inv_avg_n_neigh = 1/avg_n_neigh (float scalar here; per-species is an array).
    # Taken straight off the instruction attribute (built from the model's
    # avg_n_neigh), NOT hardcoded.
    weights[f"{p}_inv_avg_n_neigh"] = _scalar_to_np(instruction.inv_avg_n_neigh)
    # l_tile: maps (lmax+1) radial channels -> (lmax+1)^2 by l index.
    weights[f"{p}_l_tile"] = instruction.l_tile.numpy().astype(np.int32)
    weights[f"{p}_equivariant"] = np.array(
        [1 if instruction.equivariant_mode else 0]
    )

    if not instruction.equivariant_mode:
        # --- Scalar mode: species modulation z_tr = lin_transform(Z) ---
        lt = instruction.lin_transform
        if lt is not None:
            weights[f"{p}_lin_transform_W"] = lt.w.numpy()
            weights[f"{p}_lin_transform_norm"] = _norm_to_np(lt.norm)
    else:
        # --- Equivariant mode: L=0 chemistry injection + dense CG couple ---
        cl = instruction.chem_linear
        weights[f"{p}_chem_linear_W"] = cl.w.numpy()
        weights[f"{p}_chem_linear_norm"] = _norm_to_np(cl.norm)
        weights[f"{p}_chem_l0_mask"] = instruction.chem_l0_mask.numpy()

        # Dense CG couple matrix W[lm_y*lm_ind, nfunc], reproducing EXACTLY the
        # `_USE_GEMM_COUPLE` path of `_equiv_cg_couple` (compute.py:99-114):
        #   lr_flat = lr_inds[:,0]*lm_ind + lr_inds[:,1]   (row-major into (lm_y,lm_ind))
        #   W = scatter_nd([lr_flat, m_sum_ind], cg, [lm_y*lm_ind, nfunc])
        # so the oracle/C++ couple is a plain matmul prod_flat @ W.
        # lm_y = a_nl angular dim = (lmax+1)^2 ; lm_ind = indicator lm dim.
        lr = instruction.lr_inds.numpy().astype(np.int64)  # [n_cg, 2]
        m_sum = instruction.m_sum_ind.numpy().astype(np.int64)  # [n_cg]
        cg = (
            instruction.cg.numpy()
            if hasattr(instruction.cg, "numpy")
            else np.asarray(instruction.cg)
        )
        cg = np.reshape(cg, [-1])  # cg is stored [n_cg,1,1]; flatten as compute.py does
        lm_y = (int(instruction._lmax) + 1) ** 2
        lm_ind = int(instruction.n_lm_indicator)
        nfunc = int(
            instruction.nfunc.numpy()
            if hasattr(instruction.nfunc, "numpy")
            else instruction.nfunc
        )
        P = lm_y * lm_ind
        lr_flat = lr[:, 0] * lm_ind + lr[:, 1]
        cg_W = np.zeros((P, nfunc), dtype=cg.dtype)
        # np.add.at (accumulate) matches tf.scatter_nd's duplicate-index summation.
        np.add.at(cg_W, (lr_flat, m_sum), cg)
        weights[f"{p}_cg_W"] = cg_W
        weights[f"{p}_nfunc"] = np.array([nfunc])
        weights[f"{p}_n_lm_ind"] = np.array([lm_ind])

    return weights


def _extract_general_product(instruction, prefix):
    """Extract a `GeneralProductFunction` in `cp_l` mode — the low-rank CP
    decomposition product that couples a layer's basis with (a re-channeled copy
    of) itself (A1_2, A1_3, ... A3_4).

    Forward math it must feed (compute.py `_frwrd_cp_l`, 2775-2791; spec §4.6):
        U_tiled[w] = U[group_left[w]]              # gather per lm channel
        xL~[a,r,w] = norm_u * Σ_n U_tiled[w,r,n] · xL[a,n,w]     # :2784
        xR~[a,r,w] = norm_v * Σ_n V_tiled[w,r,n] · xR[a,n,w]     # :2785
        p[c,a,r]   = xL~[a,r,left_ind[c]] · xR~[a,r,right_ind[c]] · cg[c]
        y[f,a,r]   = Σ_{c: m_sum_ind[c]=f} p[c,a,r]      # unsorted_segment_sum
    with `use_S=False` ⇒ rank == n_out (no S output projection), so the coupled
    `y` IS the output (`_apply_S` is a no-op, compute.py 2745-2746).

    Norm convention (PINNED, matches `_norm_to_np`): `norm_u=1/√n_left`
    and `norm_v=1/√n_right` are stored as SEPARATE scalars, NOT folded into U/V.
    `__call__` multiplies each into the *projected* features exactly once
    (compute.py 2784-2785: `... * self.norm_u` / `... * self.norm_v`); the oracle
    and C++ apply them once likewise. Folding here would double-apply.

    Everything else (U, V, group_left/right, left_ind/right_ind/m_sum_ind/cg) is
    stored raw straight off the built instance — the cp_l coupling table is NOT
    interchangeable with ProductFunction's (different optimize_ms_comb), so it must
    come from THIS instruction (spec §4.6).
    """
    p = prefix
    weights = {}

    # --- Guard: only cp_l with use_S=False (rank==n_out) is supported. ---
    assert instruction.mode == "cp_l", (
        f"{p}: _extract_general_product supports mode='cp_l' only, got "
        f"'{instruction.mode}'"
    )
    assert instruction.use_S is False, (
        f"{p}: use_S=True (S output-projection path) is unsupported by the 3L "
        "Kokkos exporter"
    )
    rank = int(instruction.rank)
    n_out = int(instruction.n_out)
    assert rank == n_out, (
        f"{p}: with use_S=False, rank must equal n_out (source asserts this), got "
        f"rank={rank}, n_out={n_out}"
    )

    n_left = int(instruction.n_left)
    n_right = int(instruction.n_right)

    # --- CP factors (raw; orientation must match the einsum "wrn,anw->arw"). ---
    U = instruction.U.numpy()  # [n_groups_left, rank, n_left]
    V = instruction.V.numpy()  # [n_groups_right, rank, n_right]
    n_groups_left = int(instruction.n_groups_left)
    n_groups_right = int(instruction.n_groups_right)
    assert U.shape == (n_groups_left, rank, n_left), (
        f"{p}_U shape {U.shape} != (n_groups_left={n_groups_left}, rank={rank}, "
        f"n_left={n_left}) — factor orientation mismatch"
    )
    assert V.shape == (n_groups_right, rank, n_right), (
        f"{p}_V shape {V.shape} != (n_groups_right={n_groups_right}, rank={rank}, "
        f"n_right={n_right}) — factor orientation mismatch"
    )
    weights[f"{p}_U"] = U
    weights[f"{p}_V"] = V

    # --- Per-lm-channel -> (l,hist,parity) group maps (compute.py 2545-2556). ---
    weights[f"{p}_group_left"] = instruction.group_left.numpy().astype(np.int32)
    weights[f"{p}_group_right"] = instruction.group_right.numpy().astype(np.int32)

    # --- cp_l coupling table (straight off this instance; init_coupling 2492-2513). ---
    weights[f"{p}_left_ind"] = instruction.left_ind.numpy().astype(np.int32)
    weights[f"{p}_right_ind"] = instruction.right_ind.numpy().astype(np.int32)
    weights[f"{p}_m_sum_ind"] = instruction.m_sum_ind.numpy().astype(np.int32)
    # cg is stored [1,1,n_cg] (standard, non-lm_first) then cast in build(); the
    # kernel/oracle index it as cg[c], so flatten to [n_cg] (compute.py 2726).
    cg = instruction.cg.numpy() if hasattr(instruction.cg, "numpy") else np.asarray(
        instruction.cg
    )
    weights[f"{p}_cg"] = np.reshape(cg, [-1])

    nfunc = int(
        instruction.nfunc.numpy()
        if hasattr(instruction.nfunc, "numpy")
        else instruction.nfunc
    )

    # --- Scalars / metadata ---
    weights[f"{p}_nfunc"] = np.array([nfunc])
    weights[f"{p}_rank"] = np.array([rank])
    weights[f"{p}_n_left"] = np.array([n_left])
    weights[f"{p}_n_right"] = np.array([n_right])
    weights[f"{p}_n_groups_left"] = np.array([n_groups_left])
    weights[f"{p}_n_groups_right"] = np.array([n_groups_right])
    # Runtime norms kept SEPARATE (applied once in __call__ 2784-2785).
    weights[f"{p}_norm_u"] = _norm_to_np(instruction.norm_u)
    weights[f"{p}_norm_v"] = _norm_to_np(instruction.norm_v)

    # --- Per-output-function angular character (coupling_meta_data rows are 1:1
    # with output functions f; nfunc == len(coupling_meta_data)). ---
    cmd = instruction.coupling_meta_data
    out_l = cmd["l"].values.astype(np.int32)
    out_parity = cmd["parity"].values.astype(np.int32)
    assert len(out_l) == nfunc, (
        f"{p}: out_l len {len(out_l)} != nfunc {nfunc}"
    )
    weights[f"{p}_out_l"] = out_l
    weights[f"{p}_out_parity"] = out_parity

    return weights


def _extract_equivariant_rms_norm(instruction, prefix):
    """Extract an `EquivariantRMSNorm` instruction (eq1_norm/eq2_norm) — degree-
    balanced per-atom RMS norm with per-(l,parity,hist) affine scale.

    Confirmed on the built GRACE-3L-OMAT-large-ft-AM instances (both eq1_norm and
    eq2_norm): `center_l0=True`, `balance_degrees=True`, `init="ones"`, and
    `center_l0_bias=split_norm=normalize_l0_only=False` — i.e. exactly the plain
    path this extractor implements (spec §4.6). The asserts below guard against
    a future checkpoint silently taking one of the other (unimplemented) paths.

    Forward math it must feed (compute.py `EquivariantRMSNorm.frwrd`, 4275-4365;
    spec §4.6), with `x` shape `[atoms, n_features, n_angular]` (non-lm_first):
        # 1. center_l0 (:4296-4298): subtract the cross-feature mean, L=0 columns only
        l0_mean[a,m] = mean_f(x[a,f,m]) * l0_mask[m]
        x[a,f,m]    -= l0_mean[a,m]
        # 2. degree-balanced RMS, ONE scalar per atom (:4323-4328)
        norm[a] = mean_f( Σ_m degree_weights[m] * x[a,f,m]^2 )
        rms[a]  = rsqrt(norm[a] + eps)
        # 3. per-(l,parity,hist) affine scale (:4342-4343, 4350-4357)
        scale[m,f] = affine_weight[expand_index[m], f]
        out[a,f,m]  = x[a,f,m] * rms[a] * scale[m,f]
    `center_l0_bias`/`split_norm`/`normalize_l0_only` all being False means no
    `l0_bias`, `split_l0_weights`/`split_lgt0_weights`, or `l0_norm_weights`
    buffers exist on the built instance — nothing to export for those paths.

    Buffers pulled straight off the built instance (raw, no folding):
      - `affine_weight` (:4226-4231, a `tf.Variable`) `[n_lph_groups, n_out]`.
      - `_degree_weights_np` (:4158, precomputed in `__init__`, NOT the reshaped
        `degree_weights` tensor from `build()`) `[M]`, `M` = number of angular
        (lm) channels; Σ over distinct l of `degree_weights[mask_l].sum()` == 1.
      - `_expand_index_np` (:4168) `[M]` int, each entry the (l,parity,hist)
        group id in `[0, n_lph_groups)` that channel `m`'s affine scale is
        gathered from.
      - `_l0_mask_np` (:4198) `[M]` bool → stored as float64 (1.0/0.0) for
        dtype consistency with `degree_weights` and this file's other mask-like
        buffers (e.g. `chem_l0_mask` in `_extract_spbf`).
      - `epsilon` (:4221, a `tf.constant`) — `1e-8` for float32 build dtype.
      - `center_l0` (:4136, a python bool) → 0/1.
    """
    p = prefix
    assert instruction.balance_degrees, f"{p}: balance_degrees=False is unsupported"
    assert instruction.init == "ones", f"{p}: init={instruction.init!r} != 'ones'"
    assert not instruction.center_l0_bias, (
        f"{p}: center_l0_bias=True needs an extra {p}_center_l0_bias export "
        "(l0_bias variable) — not implemented by this extractor"
    )
    assert not instruction.split_norm, (
        f"{p}: split_norm=True uses split_l0/lgt0_weights buffers instead of "
        "degree_weights — not implemented by this extractor"
    )
    assert not instruction.normalize_l0_only, (
        f"{p}: normalize_l0_only=True uses l0_norm_weights instead of "
        "degree_weights — not implemented by this extractor"
    )

    weights = {
        f"{p}_affine_weight": instruction.affine_weight.numpy(),
        f"{p}_degree_weights": instruction._degree_weights_np,
        f"{p}_expand_index": instruction._expand_index_np.astype(np.int32),
        f"{p}_l0_mask": instruction._l0_mask_np.astype(np.float64),
        f"{p}_eps": _scalar_to_np(instruction.epsilon),
        f"{p}_center_l0": np.array([1 if instruction.center_l0 else 0]),
    }
    return weights


def _to_np(x):
    """Coerce a tf.Tensor / tf.Variable / numpy array to numpy, tolerating either.
    The 3L instructions mix the two (e.g. FCRight2Left.w_tile_left is a plain
    numpy array while w_tile_right is a tf.Tensor)."""
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _extract_fc_weights_3l(instruction, prefix):
    """3L-specific reimplementation of the shared `_extract_fc_weights`.

    DEVIATION (documented): the shared helper is not usable by pure call for
    this model's `FCRight2Left` (tensorpotential 0.5.10). Two 2L-era assumptions
    break on it:
      1. It reads `instruction.w_left` *directly*; the 3L FC is a pure right->left
         contraction (`left_coefs=False`) and does NOT define `w_left`/`norm_left`
         at all → AttributeError.
      2. It calls `.numpy()` unconditionally on `w_tile_left`; on the 3L FC that
         attribute is already a plain numpy array (only w_tile_right/collect_*/
         norm_out_factor are tf tensors here) → AttributeError.

    This reimplementation produces the EXACT same key layout / dtypes the shared
    helper would (so the oracle/C++ read the same keys): `{p}_w_left`/`_norm_left`
    only if a left weight exists (skipped here), `{p}_w_right`/`_norm_right`,
    `{p}_w_tile_left`/`_w_tile_right`, `{p}_collect_to`/`_collect_from` (int32),
    `{p}_norm_out_factor`, `{p}_n_out`, `{p}_left_coefs`. Robust to
    tf-tensor-or-numpy via `_to_np`. Norm scalars are kept SEPARATE (pinned
    convention — the consumer applies the norm once)."""
    weights = {}
    w_left = getattr(instruction, "w_left", None)
    if w_left is not None:
        weights[f"{prefix}_w_left"] = _to_np(w_left)
        if getattr(instruction, "norm_left", None) is not None:
            weights[f"{prefix}_norm_left"] = _scalar_to_np(instruction.norm_left)
    w_right = getattr(instruction, "w_right", None)
    if w_right is not None:
        weights[f"{prefix}_w_right"] = _to_np(w_right)
        if getattr(instruction, "norm_right", None) is not None:
            weights[f"{prefix}_norm_right"] = _scalar_to_np(instruction.norm_right)
    if getattr(instruction, "w_tile_left", None) is not None:
        weights[f"{prefix}_w_tile_left"] = _to_np(instruction.w_tile_left).astype(np.int32)
    if getattr(instruction, "w_tile_right", None) is not None:
        weights[f"{prefix}_w_tile_right"] = _to_np(instruction.w_tile_right).astype(np.int32)
    if getattr(instruction, "collect_to", None) is not None:
        weights[f"{prefix}_collect_to"] = _to_np(instruction.collect_to).astype(np.int32)
    if getattr(instruction, "collect_from", None) is not None:
        weights[f"{prefix}_collect_from"] = _to_np(instruction.collect_from).astype(np.int32)
    if getattr(instruction, "norm_out_factor", None) is not None:
        weights[f"{prefix}_norm_out_factor"] = _to_np(instruction.norm_out_factor)
    weights[f"{prefix}_n_out"] = np.array([instruction.n_out])
    weights[f"{prefix}_left_coefs"] = np.array([1 if instruction.left_coefs else 0])
    return weights


def export_3l_npz(model_yaml_path, checkpoint_path, output_path,
                  param_dtype_str="float64", uq_artifacts_path=None):
    """Export GRACE-3L model to named-prefix .npz layout with `*_names` index lists.

    Runs load/restore, the supported-class check, global-metadata extraction, the
    full per-instruction weight-extraction loop (SPBF / cp_l products / reduces /
    FCs / equivariant + invariant RMS-norms / 3-origin readout MLP), then collapses
    shifts and saves. Reuses the shared `_extract_*` helpers; the 3L-specific
    `_extract_spbf` / `_extract_general_product` / `_extract_equivariant_rms_norm` /
    `_extract_fc_weights_3l` live in this module.
    """
    instruction_dict = _load_and_restore(model_yaml_path, checkpoint_path, param_dtype_str)
    _check_supported_3l(instruction_dict)

    all_weights, n_elements, has_bond_specific = _extract_globals(
        instruction_dict, model_yaml_path
    )

    chem_embedding_np = None
    spbf_names = []
    prod_names = []
    reduce_names = []
    fc_names = []
    eqnorm_names = []
    rmsnorm_names = []
    out1_origins = []
    energy_mlp_origin_n_out = None
    for name, instr in instruction_dict.items():
        cls = type(instr).__name__
        logger.info(f"Processing {name} ({cls})...")

        if cls == "ScalarChemicalEmbedding":
            # The [n_elements, embedding_size] table the SPBF radial MLP consumes
            # as its [Z_i, Z_j] input channels (A*_mlp_W0 is 138 = 10 cheb + 2*64).
            # Also feeds `_collapse_shifts` if a shift uses embedding_shift.
            d = _extract_chem_embedding(instr)
            all_weights.update(d)
            chem_embedding_np = d.get("chem_embedding")
        elif cls == "SPBF":
            all_weights.update(_extract_spbf(instr, name))
            spbf_names.append(name)
        elif cls == "GeneralProductFunction":
            all_weights.update(_extract_general_product(instr, name))
            prod_names.append(name)
        elif cls == "FunctionReduceN":
            # include_2l_keys=True also emits `{name}_only_invar` / `_n_funcs` /
            # `_norm_map`. `only_invar` distinguishes scalar reduces (rho*, A3_3_red,
            # A3_4_red; single l=0 output) from equivariant ones (eq*, A*_red;
            # l<=4). The `is_central_atom_type_dependent` flag distinguishes the
            # elem-dependent eq*/rho* (per-central-type weight [n_types,...]) from
            # the plain product reduces. The helper is fully generic over both
            # kinds via its `collector` iteration — no scalar-vs-equivariant path
            # is baked in — so a single call per reduce is correct.
            all_weights.update(_extract_reduce(instr, prefix=name, include_2l_keys=True))
            reduce_names.append(name)
        elif cls == "FCRight2Left":
            # Mirror the 2L convention: keys are prefixed `fc_<name>` while the
            # index list stores the bare instruction name. (These 3L FCs have
            # w_left=None / left_coefs=False — the helper handles that.)
            all_weights.update(_extract_fc_weights_3l(instr, prefix=f"fc_{name}"))
            fc_names.append(name)
        elif cls == "EquivariantRMSNorm":
            all_weights.update(_extract_equivariant_rms_norm(instr, name))
            eqnorm_names.append(name)
        elif cls == "InvariantLayerRMSNorm":
            # Pure per-instruction extractor (scale / type / n_out). The 2L
            # global type-set assertion (`types_found == [full, only_nonlin]`)
            # lives in `_emit_2l_legacy_aliases`, which `_finalize_and_save`
            # runs ONLY for arch=="2l" — so calling this once per rms-norm is
            # safe for the 3L inventory (rho1_norm=only_nonlin, rho2/3_norm=full).
            all_weights.update(_extract_rms_norm(instr, prefix=name))
            rmsnorm_names.append(name)
        elif cls == "LinMLPOut2ScalarTarget":
            all_weights.update(_extract_energy_mlp(
                instr, prefix="energy_mlp",
                allowed_activations=_ACT_SUPPORTED_3L,
                write_activation_code=True,
                arch_label="GRACE-3L",
            ))
            # 3-origin readout (spec 4.5): rho1_norm/rho2_norm/rho3_norm are
            # SUMMED element-wise (LinMLPOut2ScalarTarget.frwrd does `+=` over
            # origins, NOT concat), and only the l=0 slice is used. So the
            # energy-MLP input width is `origin.n_out - 1` (=31, the l=0 density
            # width minus the channel-0 linear skip), NOT 3x. Record the origin
            # names + density width so the oracle/C++ sum-then-read l=0.
            origins = list(instr.origin)
            out1_origins = [o.name for o in origins]
            out_shapes = {int(o.n_out) for o in origins}
            assert len(out_shapes) == 1, (
                f"out1 origins have mismatched n_out {sorted(out_shapes)}; the "
                "3-origin element-wise SUM requires equal widths."
            )
            energy_mlp_origin_n_out = out_shapes.pop()
            assert getattr(instr, "normalize", None) is None, (
                f"out1.normalize={instr.normalize!r}; the exporter/oracle assume "
                "the plain readout path (no scalar_rms_ln on the summed origins)."
            )

    # Names index lists (S16: `rho{1,2,3}_norm` are 9 chars — S8 would truncate).
    all_weights["spbf_names"] = np.array(spbf_names, dtype="S16")
    all_weights["prod_names"] = np.array(prod_names, dtype="S16")
    all_weights["reduce_names"] = np.array(reduce_names, dtype="S16")
    all_weights["fc_names"] = np.array(fc_names, dtype="S16")
    all_weights["eqnorm_names"] = np.array(eqnorm_names, dtype="S16")
    all_weights["rmsnorm_names"] = np.array(rmsnorm_names, dtype="S16")
    all_weights["out1_origins"] = np.array(out1_origins, dtype="S16")
    if energy_mlp_origin_n_out is not None:
        all_weights["energy_mlp_origin_n_out"] = np.array([energy_mlp_origin_n_out])
        all_weights["energy_mlp_n_origins"] = np.array([len(out1_origins)])

    _finalize_and_save(
        "3l",
        instruction_dict,
        all_weights,
        n_elements,
        chem_embedding_np,
        output_path,
        param_dtype_str,
        has_bond_specific,
        _assert_within_caps_3l,
        uq_artifacts_path=uq_artifacts_path,
    )
    return all_weights
