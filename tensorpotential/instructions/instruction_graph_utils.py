from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Tuple, Set, Optional
import tensorflow as tf

if TYPE_CHECKING:
    from tensorpotential.instructions.base import TPInstruction, InstructionManager


def get_dependencies(instruction: TPInstruction) -> List[str]:
    """
    Extract instruction names that a given instruction depends on by inspecting _init_args.
    """
    dependencies = []
    if hasattr(instruction, "_init_args"):
        from tensorpotential.instructions.base import (
            TPInstruction,
            recursive_walk_and_modify,
        )

        def find_instr_dependencies(k, v, container):
            if isinstance(v, TPInstruction):
                dependencies.append(v.name)
            elif isinstance(v, dict) and "_instruction_" in v:
                dependencies.append(v["name"])

        # We don't want to modify, just walk. recursive_walk_and_modify is perfect for walking.
        # However, _init_args might contain TPInstruction objects or serialized dicts.
        import copy

        # Work on a copy to be safe, although we aren't modifying
        args_copy = copy.copy(instruction._init_args)
        recursive_walk_and_modify(args_copy, find_instr_dependencies)

    return sorted(list(set(dependencies)))


def build_dependency_graph(
    instructions: Dict[str, TPInstruction],
) -> Dict[str, List[str]]:
    """
    Build a complete dependency graph mapping each instruction name to its dependencies.
    """
    return {name: get_dependencies(instr) for name, instr in instructions.items()}


def print_dependency_tree(
    instructions: Dict[str, TPInstruction],
    root: Optional[str] = None,
    max_depth: int = 10,
) -> str:
    """
    Generate a text-based tree visualization of instruction dependencies.
    """
    graph = build_dependency_graph(instructions)

    if root is None:
        # Try to find root(s) - nodes that are not dependencies of any other node
        all_deps = set()
        for deps in graph.values():
            all_deps.update(deps)
        roots = [name for name in graph.keys() if name not in all_deps]
        if not roots:
            # Fallback to all nodes if we can't find a clear root (maybe a cycle, though unlikely)
            roots = list(graph.keys())
    else:
        roots = [root]

    lines = []

    def _walk(node_name, prefix="", is_last=True, depth=0):
        if depth > max_depth:
            lines.append(f"{prefix}└── ... (max depth reached)")
            return

        instr = instructions.get(node_name)
        instr_type = instr.__class__.__name__ if instr else "Unknown"

        marker = "└── " if is_last else "├── "
        lines.append(f"{prefix}{marker}{node_name} ({instr_type})")

        new_prefix = prefix + ("    " if is_last else "│   ")
        deps = graph.get(node_name, [])

        for i, dep_name in enumerate(deps):
            _walk(dep_name, new_prefix, i == len(deps) - 1, depth + 1)

    for i, r in enumerate(roots):
        _walk(r, is_last=(i == len(roots) - 1))

    return "\n".join(lines)


def get_communication_keys(
    layer1_instr: Dict[str, TPInstruction],
    layer2_instr: Dict[str, TPInstruction],
    recompute_instructions: Optional[List[str]] = None,
) -> List[str]:
    """
    Identify which outputs from Layer 1 are used as inputs in Layer 2.
    Instructions that are present in both layers are considered recomputed and not communicated.
    """
    l1_names = set(layer1_instr.keys())
    l2_names = set(layer2_instr.keys())
    recompute_set = set(recompute_instructions or [])
    comm_keys = set()

    for instr in layer2_instr.values():
        deps = get_dependencies(instr)
        for dep in deps:
            # If it's in L1, but not in L2, and not explicitly recomputed, it must be communicated
            if dep in l1_names and dep not in l2_names and dep not in recompute_set:
                comm_keys.add(dep)

    return sorted(list(comm_keys))


def split_2layer_instructions(
    instructions: Dict[str, TPInstruction], communicated_keys: List[str]
) -> Tuple[Dict[str, TPInstruction], Dict[str, TPInstruction]]:
    """
    Split instruction dict into layer 1 and layer 2 based on provided communicated keys.
    Returns:
        - Layer 1 instructions dict (needed to compute communicated_keys)
        - Layer 2 instructions dict (everything else needed for final results, recomputing shared deps)
    """
    graph = build_dependency_graph(instructions)
    l1_set = set()

    # Layer 1: All recursive dependencies of communicated_keys + the keys themselves
    def _collect_l1(node_name):
        if node_name in l1_set:
            return
        l1_set.add(node_name)
        for dep in graph.get(node_name, []):
            _collect_l1(dep)

    for comm_key in communicated_keys:
        if comm_key in instructions:
            _collect_l1(comm_key)

    l1_instr = {name: instr for name, instr in instructions.items() if name in l1_set}

    # Layer 2: Instructions needed for final outputs starting from communicated_keys
    # Final outputs are targets (e.g., CreateOutputTarget) or anything not dependent on by anything else
    all_deps = set()
    for deps in graph.values():
        all_deps.update(deps)
    possible_roots = [name for name in instructions.keys() if name not in all_deps]

    # We want everything that is NOT strictly needed ONLY for L1 and is needed for final output.
    # Actually, L2 should contain all instructions that use communicated_keys as input,
    # plus their dependencies that are NOT the communicated_keys themselves (recomputation).

    l2_set = set()

    def _collect_l2(node_name):
        if node_name in l2_set or node_name in communicated_keys:
            return
        l2_set.add(node_name)
        for dep in graph.get(node_name, []):
            _collect_l2(dep)

    for root in possible_roots:
        _collect_l2(root)

    l2_instr = {name: instr for name, instr in instructions.items() if name in l2_set}

    return l1_instr, l2_instr


def find_non_local_keys(
    instructions: Dict[str, TPInstruction],
    communicated_keys: List[str],
    non_local_instruction_marker: str = "Y",
):
    dep_graph = build_dependency_graph(instructions)

    # Identify local vs non-local keys
    # Reverse graph: consumer map
    consumers = defaultdict(list)
    for name, deps in dep_graph.items():
        for dep in deps:
            consumers[dep].append(name)

    non_local_keys = []

    for key in communicated_keys:
        is_non_local = False
        # Find consumers of this key
        key_consumers = consumers[key]
        for consumer in key_consumers:
            # Check if this consumer also depends on 'Y'
            consumer_deps = dep_graph.get(consumer, [])
            if non_local_instruction_marker in consumer_deps:
                non_local_keys.append(key)
                break

    return non_local_keys


def infer_communicated_keys(instructions: Dict[str, TPInstruction]) -> List[str]:
    """Auto-detect communicated_keys for 2L models from the instruction graph."""
    from tensorpotential.instructions.compute import (
        SingleParticleBasisFunctionEquivariantInd,
    )
    from tensorpotential.instructions.output import TPOutputInstruction

    # Step 1: Find primary indicator names (these are the cross-layer boundary instructions)
    indicator_names = []
    for instr in instructions.values():
        if isinstance(instr, SingleParticleBasisFunctionEquivariantInd):
            # SPBFEI.indicator points to the actual instruction object
            indicator_names.append(instr.indicator.name)

    if not indicator_names:
        raise ValueError(
            "No SingleParticleBasisFunctionEquivariantInd found in instructions"
        )

    # Step 1.5: Find extra communication keys
    # These are instructions that feed into output targets but do not depend on indicator_names
    # This happens when some L1 features are passed directly to the output block (L2)
    graph = build_dependency_graph(instructions)

    # Memoization for dependency check
    memo_depends = {}

    def depends_on_any(node_name, ref_nodes):
        ref_nodes = tuple(sorted(ref_nodes))
        if (node_name, ref_nodes) in memo_depends:
            return memo_depends[(node_name, ref_nodes)]

        deps = graph.get(node_name, [])
        for d in deps:
            if d in ref_nodes:
                memo_depends[(node_name, ref_nodes)] = True
                return True
            if depends_on_any(d, ref_nodes):
                memo_depends[(node_name, ref_nodes)] = True
                return True

        memo_depends[(node_name, ref_nodes)] = False
        return False

    extra_keys = set()
    for instr in instructions.values():
        if isinstance(instr, TPOutputInstruction):
            # Check 'origin' or 'instructions' depending on the output instruction type
            origins = getattr(instr, "origin", [])
            if not isinstance(origins, list):
                origins = [origins]

            # Some instructions use 'instructions' instead of 'origin'
            if not origins:
                origins = getattr(instr, "instructions", [])
                if not isinstance(origins, list):
                    origins = [origins]

            for origin in origins:
                origin_name = (
                    origin.name if hasattr(origin, "name") else origin
                )
                if origin_name not in instructions:
                    continue

                # If this origin does not depend on primary indicators, it's computed in L1
                # and since it's used in an output target (L2), it must be communicated.
                if origin_name not in indicator_names and not depends_on_any(
                    origin_name, indicator_names
                ):
                    extra_keys.add(origin_name)

    all_initial_keys = sorted(list(set(indicator_names) | extra_keys))

    # Step 2: Use detected names as a baseline to split the graph
    # and find all outputs that cross from L1 to L2
    l1_instr, l2_instr = split_2layer_instructions(instructions, all_initial_keys)
    all_comm_keys = get_communication_keys(l1_instr, l2_instr)

    return all_comm_keys


def build_split_tpmodel(
    instructions: Dict[str, TPInstruction],
    communicated_keys: List[str],
    float_dtype=tf.float64,
    input_dtype=None,
    jit_compile=True,
    extra_aux_computes=None,
):
    """
    Automate the creation and building of a split TPModel.

    Args:
        instructions: Original instructions dictionary.
        communicated_keys: Keys to be passed between layers.
        float_dtype: DType for variables/computation.
        input_dtype: DType for inputs.
        jit_compile: Whether to use XLA.

    Returns:
        Built TPModel with aux_computes for forward L1, backward L2, and backward L1.
    """
    import tensorflow as tf
    from tensorpotential.tpmodel import (
        TPModel,
        ComputeBlock,
        ComputeBlockInputGradient,
        ComputeBlockOutputGradient,
    )
    from tensorpotential import constants

    # 1. Split instructions
    l1_instr, l2_instr = split_2layer_instructions(instructions, communicated_keys)

    # 2. Infer specs for communicated keys
    # Shapes are [None, n_out, n_features]
    comm_specs = {}
    for key in communicated_keys:
        instr = instructions[key]
        n_out = getattr(instr, "n_out", None)

        # Infer number of equivariant/invariant features
        n_features = 1
        curr_instr = instr
        # Some instructions like RMSNorm wrap others
        if hasattr(curr_instr, "input"):
            curr_instr = curr_instr.input

        if (
            hasattr(curr_instr, "coupling_meta_data")
            and curr_instr.coupling_meta_data is not None
        ):
            n_features = len(curr_instr.coupling_meta_data)

        comm_specs[key] = {
            "shape": [None, n_out, n_features],
            "dtype": "float64" if float_dtype == tf.float64 else "float32",
        }

    l2_grad_specs = {f"grad_{k}": v for k, v in comm_specs.items()}

    # 3. Define Aux Computes
    aux_computes = {
        # Step 1: Forward Layer 1 (Outputs features for L2)
        "forward_layer_1": ComputeBlock(
            instructions=l1_instr, output_keys=communicated_keys
        ),
        # Step 2: Backward Layer 2 (Target -> communicated grads)
        "backward_layer_2": ComputeBlockInputGradient(
            instructions=l2_instr,
            wrt_keys=communicated_keys + [constants.BOND_VECTOR],
            target_key=constants.PREDICT_ATOMIC_ENERGY,
            output_keys=[constants.PREDICT_ATOMIC_ENERGY],
            specs=comm_specs,
        ),
        # Step 3: Backward Layer 1 (communicated grads -> coord grads)
        "backward_layer_1": ComputeBlockOutputGradient(
            instructions=l1_instr,
            wrt_keys=[constants.BOND_VECTOR],
            output_keys=communicated_keys,
            specs=l2_grad_specs,
        ),
    }
    if extra_aux_computes:
        aux_computes.update(extra_aux_computes)

    # 4. Build Model
    m = TPModel(instructions=instructions, aux_compute=aux_computes)
    m.build(float_dtype, jit_compile=jit_compile, input_dtype=input_dtype)
    m.decorate_compute_function(
        float_dtype, jit_compile=jit_compile, input_dtype=input_dtype
    )

    # this is not ideal, but it's the best we can do for now
    non_local_comm_keys = find_non_local_keys(instructions, communicated_keys)
    m.non_local_communication_keys = non_local_comm_keys
    return m
