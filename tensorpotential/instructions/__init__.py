from tensorpotential.instructions.base import (
    load_instructions,
    capture_init_args,
    InstructionManager,
    NoInstructionManager,
    TPInstruction,
    TPEquivariantInstruction,
    save_instructions_dict,
    load_instructions,
)

from tensorpotential.instructions.compute import (
    BondLength,
    ScaledBondVector,
    SphericalHarmonic,
    ScalarChemicalEmbedding,
    RadialBasis,
    LinearRadialFunction,
    MLPRadialFunction,
    SingleParticleBasisFunctionScalarInd,
    SingleParticleBasisFunctionEquivariantInd,
    ProductFunction,
    FunctionReduce,
    FunctionReduceParticular,
    FunctionReduceN,
    FCRight2Left,
    ZBLPotential,
    CropProductFunction,
    BondSpecificRadialBasisFunction,
    MLPRadialFunction_v2,
    InvariantLayerRMSNorm,
    InvariantPade,
)

from tensorpotential.instructions.instruction_graph_utils import (
    get_dependencies,
    build_dependency_graph,
    print_dependency_tree,
    get_communication_keys,
    split_2layer_instructions,
    build_split_tpmodel,
)

from tensorpotential.instructions.output import (
    CreateOutputTarget,
    LinearOut2Target,
    FSOut2ScalarTarget,
    MLPOut2ScalarTarget,
    ConstantScaleShiftTarget,
    LinMLPOut2ScalarTarget,
    LinearOut2EquivarTarget,
    TrainableShiftTarget,
)

__all__ = [
    "capture_init_args",
    "TPEquivariantInstruction",
    "TPInstruction",
    "load_instructions",
    "InstructionManager",
    "NoInstructionManager",
    "BondLength",
    "ScaledBondVector",
    "SphericalHarmonic",
    "RadialBasis",
    "LinearRadialFunction",
    "MLPRadialFunction",
    "SingleParticleBasisFunctionScalarInd",
    "SingleParticleBasisFunctionEquivariantInd",
    "ScalarChemicalEmbedding",
    "ProductFunction",
    "FunctionReduce",
    "FunctionReduceParticular",
    "CreateOutputTarget",
    "LinearOut2Target",
    "FSOut2ScalarTarget",
    "MLPOut2ScalarTarget",
    "LinMLPOut2ScalarTarget",
    "ConstantScaleShiftTarget",
    "FunctionReduceN",
    "FCRight2Left",
    "CropProductFunction",
    "ZBLPotential",
    "LinearOut2EquivarTarget",
    "BondSpecificRadialBasisFunction",
    "MLPRadialFunction_v2",
    "TrainableShiftTarget",
    "InvariantLayerRMSNorm",
    "save_instructions_dict",
    "load_instructions",
    'InvariantPade',
    "get_dependencies",
    "build_dependency_graph",
    "print_dependency_tree",
    "get_communication_keys",
    "split_2layer_instructions",
    "build_split_tpmodel",
]

try:
    from tensorpotential.experimental.instructions.aux_compute import (
        InvariantLayerDTNorm,
        EquivariantRMSNorm,
        MLPRadialFunction_v3,
        MLPRadialFunction_v4,
        MLPRadialFunction_v5,
    )

    __all__.extend(
        [
            "EquivariantRMSNorm",
            "InvariantLayerDTNorm",
            "MLPRadialFunction_v3",
            "MLPRadialFunction_v4",
            "MLPRadialFunction_v5",
        ]
    )
except ImportError:
    pass
