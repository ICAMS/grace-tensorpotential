from tensorpotential.instructions.base import (
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
    GeneralProductFunction,
    FunctionReduce,
    FunctionReduceParticular,
    FunctionReduceN,
    FCRight2Left,
    ZBLPotential,
    CropProductFunction,
    BondSpecificRadialBasisFunction,
    MLPRadialFunction_v2,
    SPBF,
    InvariantLayerRMSNorm,
    EquivariantRMSNorm,
    EquivariantGate,
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
    LinMLPScalarReadOut,
    LinearOut2EquivarTarget,
    TrainableShiftTarget,
    TrainableShiftTarget_v2,
)


def __getattr__(name: str):
    # Old notebooks did ``from tensorpotential.instructions import
    # read_model_metadata``; the helper moved to tensorpotential.metadata_utils
    # but the legacy import path is kept alive (with a DeprecationWarning) so
    # those notebooks keep working.
    if name == "read_model_metadata":
        import warnings
        from tensorpotential.metadata_utils import read_model_metadata

        warnings.warn(
            "Importing read_model_metadata from tensorpotential.instructions "
            "is deprecated; use tensorpotential.metadata_utils.read_model_metadata.",
            DeprecationWarning,
            stacklevel=2,
        )
        return read_model_metadata
    raise AttributeError(f"module 'tensorpotential.instructions' has no attribute {name!r}")

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
    "GeneralProductFunction",
    "FunctionReduce",
    "FunctionReduceParticular",
    "CreateOutputTarget",
    "LinearOut2Target",
    "FSOut2ScalarTarget",
    "MLPOut2ScalarTarget",
    "LinMLPOut2ScalarTarget",
    "LinMLPScalarReadOut",
    "ConstantScaleShiftTarget",
    "FunctionReduceN",
    "FCRight2Left",
    "CropProductFunction",
    "ZBLPotential",
    "LinearOut2EquivarTarget",
    "BondSpecificRadialBasisFunction",
    "MLPRadialFunction_v2",
    "TrainableShiftTarget",
    "TrainableShiftTarget_v2",
    "InvariantLayerRMSNorm",
    "EquivariantRMSNorm",
    "EquivariantGate",
    "save_instructions_dict",
    "load_instructions",
    "SPBF",
    "get_dependencies",
    "build_dependency_graph",
    "print_dependency_tree",
    "get_communication_keys",
    "split_2layer_instructions",
    "build_split_tpmodel",
]

try:
    from tensorpotential.experimental.instructions.aux_compute import (  # noqa: F401
        StructuredGridProductFunction,
        StructuredGridMessagePassing,
    )

    __all__.extend(
        [
            "StructuredGridProductFunction",
            "StructuredGridMessagePassing",
        ]
    )
except ImportError:
    pass
