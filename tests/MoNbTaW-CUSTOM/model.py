from tensorpotential.instructions.output import (
    CreateOutputTarget,
    LinearOut2Target,
    MLPOut2ScalarTarget,
    ConstantScaleShiftTarget,
)
from tensorpotential.instructions.base import *
from tensorpotential.instructions.compute import *
from tensorpotential.potentials.presets import Parity


@capture_init_args
class SingleParticleBasisFunctionScalarInd2(TPEquivariantInstruction):
    """
    Compute ACE single particle basis function with scalar indicator

    """

    input_tensor_spec = {
        constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
        constants.BOND_MU_I: {"shape": [None], "dtype": "int"},
        constants.BOND_MU_J: {"shape": [None], "dtype": "int"},
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
    }

    def __init__(
        self,
        radial: TPInstruction,
        angular: SphericalHarmonic,
        indicator_i: ScalarChemicalEmbedding,
        indicator_j: ScalarChemicalEmbedding,
        avg_n_neigh: float,
        name: str,
    ):
        super().__init__(name=name, lmax=angular.lmax)
        self.radial = radial
        self.angular = angular
        self.indicator_i = indicator_i
        self.indicator_j = indicator_j
        self.avg_n_neigh = avg_n_neigh
        self.lin_transform_in = DenseLayer(
            n_in=self.indicator_i.embedding_size,
            n_out=self.radial.n_rad_max,
            activation=None,
            name="ChemIndTransf_I",
        )
        self.lin_transform_jn = DenseLayer(
            n_in=self.indicator_j.embedding_size,
            n_out=self.radial.n_rad_max,
            activation=None,
            name="ChemIndTransf_J",
        )
        self.coupling_meta_data = self.angular.coupling_meta_data
        self.coupling_origin = self.angular.coupling_origin
        self.n_out = self.radial.n_rad_max

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        self.avg_n_neigh = tf.reshape(
            tf.convert_to_tensor(self.avg_n_neigh, dtype=float_dtype), [1, 1]
        )

    def frwrd(self, input_data: dict, training=False):
        z_i = input_data[self.indicator_i.name]
        z_j = input_data[self.indicator_j.name]
        r_ij = input_data[self.radial.name]
        y_ij = input_data[self.angular.name]

        mu_i = input_data[constants.BOND_MU_I]
        mu_j = input_data[constants.BOND_MU_J]
        ind_i = input_data[constants.BOND_IND_I]
        batch_tot_nat = input_data[constants.N_ATOMS_BATCH_TOTAL]

        z_tr_in = self.lin_transform_in(z_i)
        z_tr_jn = self.lin_transform_jn(z_j)
        bond_z_tr_in = tf.gather(z_tr_in, mu_i, axis=0)
        bond_z_tr_jn = tf.gather(z_tr_jn, mu_j, axis=0)

        bond_a_nl = tf.einsum(
            "bnl,bl,bn,bn->bnl", r_ij, y_ij, bond_z_tr_in, bond_z_tr_jn
        )

        atom_a_nl = tf.math.unsorted_segment_sum(
            bond_a_nl, segment_ids=ind_i, num_segments=batch_tot_nat
        )
        # atom_a_nl /= self.avg_n_neigh

        return atom_a_nl


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

        g_k = RadialBasis(bonds=d_ij, basis_type="SBessel", nfunc=n_rad_func, rcut=rcut)
        R_nl = MLPRadialFunction(n_rad_max=n_rad_max, lmax=lmax, basis=g_k, name="R")

        Y = SphericalHarmonic(vhat=rhat, lmax=lmax, name="Y")

        z_j = ScalarChemicalEmbedding(
            element_map=element_map, embedding_size=embedding_size, name="Z_J"
        )

        z_i = ScalarChemicalEmbedding(
            element_map=element_map, embedding_size=embedding_size, name="Z_I"
        )

        A = SingleParticleBasisFunctionScalarInd2(
            radial=R_nl,
            angular=Y,
            indicator_i=z_i,
            indicator_j=z_j,
            name="A",
            avg_n_neigh=avg_n_neigh,
        )
        ConstantScaleShiftTarget(
            target=A, scale=1 / avg_n_neigh, shift=0.0, l=A.lmax, name="norm_A"
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
            ls_max=[0, 0, 0, 0],
            n_in=n_rad_max,
            n_out=1,
            is_central_atom_type_dependent=False,
            number_of_atom_types=num_elements,
            allowed_l_p=Parity.SCALAR,
        )

        I_nl = FunctionReduce(
            instructions=[A, AA, AAA, AAAA],
            name="rho",
            ls_max=[0, 0, 0, 0],
            n_in=n_rad_max,
            n_out=n_mlp_dens,
            is_central_atom_type_dependent=False,
            number_of_atom_types=num_elements,
            allowed_l_p=Parity.SCALAR,
        )

        out_instr = CreateOutputTarget(name=constants.PREDICT_ATOMIC_ENERGY)
        LinearOut2Target(origin=[I_l], target=out_instr)
        MLPOut2ScalarTarget(origin=[I_nl], target=out_instr)
        ConstantScaleShiftTarget(
            target=out_instr,
            scale=constant_out_scale,
            shift=constant_out_shift,
            atomic_shift_map=atomic_shift_map,
        )

    return instructor.instruction_list
