from tensorpotential.instructions.output import (
    CreateOutputTarget,
    LinearOut2Target,
    MLPOut2ScalarTarget,
    ConstantScaleShiftTarget,
)
from tensorpotential.instructions.base import *
from tensorpotential.instructions.compute import *
from tensorpotential.potentials.presets import Parity


def custom_model(
    element_map: dict,
    rcut: float = 6.0,
    avg_n_neigh: float = 1.0,
    constant_out_shift: float = 0.0,
    constant_out_scale: float = 1.0,
    atomic_shift_map=None,
    lmax=1,
    n_rad_max=2,
    embedding_size=3,
    n_mlp_dens=2,
    n_rad_func=2,
    **kwargs
):
    num_elements = len(element_map)
    with InstructionManager() as instructor:
        d_ij = BondLength()
        rhat = ScaledBondVector(bond_length=d_ij)

        z = ScalarChemicalEmbedding(
            element_map=element_map, embedding_size=embedding_size, name="Z_I"
        )
        z_j = ScalarChemicalEmbedding(
            element_map=element_map, embedding_size=embedding_size, name="Z_J"
        )

        g_k = RadialBasis(bonds=d_ij, basis_type="SBessel", nfunc=n_rad_func, rcut=rcut)
        R_nl = MLPRadialFunction(
            n_rad_max=n_rad_max,
            lmax=lmax,
            basis=g_k,
            name="R",
            chemical_embedding_i=z,
            chemical_embedding_j=z_j,
        )

        Y = SphericalHarmonic(vhat=rhat, lmax=lmax, name="Y")

        A = SingleParticleBasisFunctionScalarInd(
            radial=R_nl,
            angular=Y,
            indicator=z,
            name="A",
            avg_n_neigh=avg_n_neigh,
        )

        AA = ProductFunction(
            left=A,
            right=A,
            name="AA",
            lmax=lmax,
            Lmax=lmax,
            keep_parity=Parity.FULL_PARITY,
        )
        AAA = ProductFunction(
            left=AA,
            right=A,
            name="AAA",
            lmax=lmax,
            Lmax=0,
            keep_parity=Parity.FULL_PARITY,
        )

        AAAA = ProductFunction(
            left=AA,
            right=AA,
            name="AAAA",
            lmax=lmax,
            Lmax=0,
            keep_parity=Parity.FULL_PARITY,
        )

        I_l = FunctionReduce(
            instructions=[A, AA, AAA, AAAA],
            name="E",
            ls_max=0,
            n_in=n_rad_max,
            n_out=1,
            is_central_atom_type_dependent=False,
            number_of_atom_types=num_elements,
            allowed_l_p=Parity.SCALAR,
        )

        out_instr = CreateOutputTarget(name=constants.PREDICT_ATOMIC_ENERGY)
        LinearOut2Target(origin=[I_l], target=out_instr)

    return instructor.instruction_list
