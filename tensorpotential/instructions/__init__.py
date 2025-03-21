from tensorpotential.instructions.base import (
    load_instructions,
    capture_init_args,
    InstructionManager,
    NoInstructionManager,
    TPInstruction,
    TPEquivariantInstruction,
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
)

from tensorpotential.instructions.output import (
    CreateOutputTarget,
    LinearOut2Target,
    FSOut2ScalarTarget,
    MLPOut2ScalarTarget,
    ConstantScaleShiftTarget,
    LinMLPOut2ScalarTarget,
    LinearOut2EquivarTarget,
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
]

try:
    from tensorpotential.experimental.instructions.aux_compute import (
        InvariantLayerRMSNorm,
        InvariantGate,
        EquivariantRMSNorm,
    )

    __all__.extend(
        [
            "InvariantLayerRMSNorm",
            "InvariantGate",
            "EquivariantRMSNorm",
        ]
    )
except ImportError:
    pass
