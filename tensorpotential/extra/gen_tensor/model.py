from tensorpotential.instructions import *
from tensorpotential.potentials.registry import register_preset
from tensorpotential.utils import Parity
from tensorpotential.extra.gen_tensor import constants as tensor_constants
from tensorpotential import constants as tc
from tensorpotential.tpmodel import (
    TrainFunction,
    compute_batch_virials_from_pair_forces,
    execute_instructions,
)

import tensorflow as tf


@register_preset("TENSOR_1L")
def grace_1(
    element_map: dict,
    rcut: float = 6.0,
    avg_n_neigh: float = 1.0,
    constant_out_shift: float = 0.0,
    constant_out_scale: float = 1.0,
    atomic_shift_map: dict = None,
    lmax=4,
    basis_type="Cheb",  # SBessel
    n_rad_base=8,
    n_rad_max=32,
    embedding_size=32,
    n_mlp_dens=16,
    max_order=4,
    compute_energy: bool = False,
    tensor_components: list = None,
    **kwargs,
):
    chem_init = "random"
    num_elements = len(element_map)
    if tensor_components is None:
        tensor_components = [0, 1, 2]
    if len(tensor_components) == 1:
        assert (
            tensor_components[0] != 0
        ), "Tensor output with single component can not be a scalar"
    assert (
        max(tensor_components) < 3
    ), f"Maximum output tensor rank is 2, but {max(tensor_components)} is specified"
    with InstructionManager() as instructor:
        d_ij = BondLength()
        rhat = ScaledBondVector(bond_length=d_ij)

        g_k = RadialBasis(
            bonds=d_ij,
            basis_type=basis_type,
            nfunc=n_rad_base,
            p=16,
            normalized=False,
            rcut=rcut,
        )

        R_nl = MLPRadialFunction_v2(
            n_rad_max=n_rad_max,
            lmax=lmax,
            basis=g_k,
            name="R",
            hidden_layers=[64, 64],
            activation=["silu", "silu"],
        )

        Y = SphericalHarmonic(vhat=rhat, lmax=lmax, name="Y")
        z = ScalarChemicalEmbedding(
            element_map=element_map,
            embedding_size=embedding_size,
            name="Z",
            init=chem_init,
        )
        A = SingleParticleBasisFunctionScalarInd(
            radial=R_nl, angular=Y, indicator=z, name="A", avg_n_neigh=avg_n_neigh
        )

        instructions = [A]

        if max_order > 1:
            A1 = FCRight2Left(
                left=A,
                right=A,
                name="A1",
                n_out=n_rad_max,
                is_central_atom_type_dependent=False,
                norm_out=True,
            )
            AA = ProductFunction(
                left=A1,
                right=A1,
                name="AA",
                lmax=lmax,
                Lmax=lmax,
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
                # n_out=n_rad_max[1]  # reduce
            )
            instructions.append(AA)

        if max_order > 2:
            AA1 = FCRight2Left(
                left=AA,
                right=A,
                name="AA1",
                n_out=n_rad_max,
                is_central_atom_type_dependent=False,
                norm_out=True,
            )
            AAA = ProductFunction(
                left=AA1,
                right=A,
                name="AAA",
                lmax=lmax,
                Lmax=int(max(tensor_components)),
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions.append(AAA)
        if max_order > 3:
            AA2 = FCRight2Left(
                left=AA,
                right=A,
                name="AA2",
                n_out=n_rad_max,
                is_central_atom_type_dependent=False,
                norm_out=True,
            )
            AAAA = ProductFunction(
                left=AA2,
                right=AA2,
                name="AAAA",
                lmax=lmax,
                Lmax=int(max(tensor_components)),
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions.append(AAAA)
        if max_order > 4:
            raise NotImplementedError(
                "Product function of order > 4 is not implemented in this model"
            )
        if compute_energy:
            out_instr = CreateOutputTarget(name=tc.PREDICT_ATOMIC_ENERGY)
            I_e = FunctionReduceN(
                instructions=instructions,
                name="rho",
                ls_max=[0] * len(instructions),
                n_out=n_mlp_dens + 1,
                is_central_atom_type_dependent=True,
                number_of_atom_types=num_elements,
                allowed_l_p=Parity.SCALAR,
            )
            LinMLPOut2ScalarTarget(
                origin=[I_e],
                target=out_instr,
                hidden_layers=[64],
                activation="tanh",
            )
            ConstantScaleShiftTarget(
                target=out_instr,
                scale=constant_out_scale,
                shift=constant_out_shift,
                atomic_shift_map=atomic_shift_map,
            )
        for component in tensor_components:
            if component == 0:
                out_0 = CreateOutputTarget(name=tensor_constants.PREDICT_L0_term, l=0)
                I0 = FunctionReduceParticular(
                    instructions=instructions,
                    name="I0",
                    selected_l=0,
                    selected_p=1,
                    n_out=n_mlp_dens + 1,
                    is_central_atom_type_dependent=True,
                    number_of_atom_types=num_elements,
                )
                LinMLPOut2ScalarTarget(
                    origin=[I0],
                    name="a0",
                    target=out_0,
                    hidden_layers=[64],
                    activation="tanh",
                )
            if component == 1:
                out_1 = CreateOutputTarget(name=tensor_constants.PREDICT_L1_term, l=1)
                I1 = FunctionReduceParticular(
                    instructions=instructions,
                    name="I1",
                    selected_l=1,
                    selected_p=-1,
                    n_out=1,
                    is_central_atom_type_dependent=True,
                    number_of_atom_types=num_elements,
                )
                LinearOut2EquivarTarget(origin=[I1], name="a1", target=out_1, l=1)
            elif component == 2:
                out_2 = CreateOutputTarget(name=tensor_constants.PREDICT_L2_term, l=2)
                I2 = FunctionReduceParticular(
                    instructions=instructions,
                    name="I2",
                    selected_l=2,
                    selected_p=1,
                    n_out=1,
                    is_central_atom_type_dependent=True,
                    number_of_atom_types=num_elements,
                )
                LinearOut2EquivarTarget(
                    origin=[I2], name="a2", target=out_2, l=2, full_r2_form=False
                )

    return instructor.get_instructions()


@register_preset("TENSOR_2L")
def grace_2(
    element_map: dict,
    rcut: float = 5.0,
    avg_n_neigh: float = 1.0,
    constant_out_shift: float = 0.0,
    constant_out_scale: float = 1.0,
    atomic_shift_map: dict = None,
    lmax=4,
    lmax_indicator=1,
    basis_type="Cheb",  # SBessel
    n_rad_base=8,
    n_rad_max=[32, 42],
    embedding_size=128,
    n_mlp_dens=16,
    max_order=4,
    compute_energy: bool = False,
    tensor_components: list = None,
    **kwargs,
):
    chem_init = "random"
    num_elements = len(element_map)
    if tensor_components is None:
        tensor_components = [0, 1, 2]
    if len(tensor_components) == 1:
        assert (
            tensor_components[0] != 0
        ), "Tensor output with single component can not be a scalar"
    assert (
        max(tensor_components) < 3
    ), f"Maximum output tensor rank is 2, but {max(tensor_components)} is specified"
    with InstructionManager() as instructor:
        d_ij = BondLength()
        rhat = ScaledBondVector(bond_length=d_ij)

        g_k = RadialBasis(
            bonds=d_ij,
            basis_type=basis_type,
            nfunc=n_rad_base,
            p=16,
            normalized=False,
            rcut=rcut,
        )

        R_nl = MLPRadialFunction_v2(
            n_rad_max=n_rad_max[0],
            lmax=lmax,
            basis=g_k,
            name="R",
            hidden_layers=[64, 64],
            activation=["silu", "silu"],
        )

        Y = SphericalHarmonic(vhat=rhat, lmax=lmax, name="Y")
        z = ScalarChemicalEmbedding(
            element_map=element_map,
            embedding_size=embedding_size,
            name="Z",
            init=chem_init,
        )
        A = SingleParticleBasisFunctionScalarInd(
            radial=R_nl, angular=Y, indicator=z, name="A", avg_n_neigh=avg_n_neigh
        )

        instructions = [A]

        if max_order > 1:
            A1 = FCRight2Left(
                left=A,
                right=A,
                name="A1",
                n_out=n_rad_max[0],
                is_central_atom_type_dependent=False,
                norm_out=True,
            )
            AA = ProductFunction(
                left=A1,
                right=A1,
                name="AA",
                lmax=lmax,
                Lmax=lmax,
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
                # n_out=n_rad_max[1]  # reduce
            )
            instructions.append(AA)

        if max_order > 2:
            AA1 = FCRight2Left(
                left=AA,
                right=A,
                name="AA1",
                n_out=n_rad_max[0],
                is_central_atom_type_dependent=False,
                norm_out=True,
            )
            AAA = ProductFunction(
                left=AA1,
                right=A,
                name="AAA",
                lmax=lmax,
                Lmax=int(max(max(tensor_components), lmax_indicator)),
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions.append(AAA)
        if max_order > 3:
            AA2 = FCRight2Left(
                left=AA,
                right=A,
                name="AA2",
                n_out=n_rad_max[0],
                is_central_atom_type_dependent=False,
                norm_out=True,
            )
            AAAA = ProductFunction(
                left=AA2,
                right=AA2,
                name="AAAA",
                lmax=lmax,
                Lmax=int(max(tensor_components)),
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions.append(AAAA)
        if max_order > 4:
            raise NotImplementedError(
                "Product function of order > 4 is not implemented in this model"
            )
        assert (
            lmax_indicator < 4
        ), f"lmax_indicator is limited to 3, but {lmax_indicator} is set"
        parity1 = [
            [0, 1],
            [1, -1],
            [2, 1],
            [3, -1],
        ]
        IM = FunctionReduceN(
            name="IM",
            instructions=instructions,
            ls_max=[lmax_indicator, lmax_indicator, lmax_indicator, 1][
                : len(instructions)
            ],
            n_out=12,
            is_central_atom_type_dependent=True,
            number_of_atom_types=num_elements,
            allowed_l_p=parity1,
        )
        I = FunctionReduceN(
            name="I",
            instructions=[IM],
            ls_max=[lmax_indicator],
            n_out=32,
            is_central_atom_type_dependent=False,
            allowed_l_p=parity1,
        )

        R1_nl = MLPRadialFunction_v2(
            n_rad_max=32,
            lmax=lmax,
            basis=g_k,
            name="R1",
            hidden_layers=[64, 64],
            activation=["silu", "silu"],
        )
        B0 = SingleParticleBasisFunctionScalarInd(
            radial=R1_nl, angular=Y, indicator=z, name="B0", avg_n_neigh=avg_n_neigh
        )
        YI = SingleParticleBasisFunctionEquivariantInd(
            radial=R1_nl,
            angular=Y,
            indicator=I,
            name="YI",
            lmax=lmax,
            Lmax=3,
            avg_n_neigh=avg_n_neigh,
            keep_parity=Parity.FULL_PARITY,
            normalize=True,
        )
        B = FunctionReduceN(
            instructions=[YI, B0],
            name="B",
            ls_max=[3, 3],
            out_norm=False,
            n_out=n_rad_max[1],
            is_central_atom_type_dependent=False,
            allowed_l_p=Parity.FULL_PARITY,
        )
        instructions2 = [B]

        if max_order > 1:
            B1 = FCRight2Left(
                left=B,
                right=B,
                name="B1",
                n_out=n_rad_max[1],
                norm_out=True,
            )
            BB = ProductFunction(
                left=B1,
                right=B1,
                name="BB",
                lmax=3,
                Lmax=3,
                keep_parity=Parity.FULL_PARITY + [[0, -1]],
                is_left_right_equal=True,
                normalize=True,
            )
            instructions2.append(BB)
        if max_order > 2:
            BB1 = FCRight2Left(
                left=BB,
                right=B,
                name="BB1",
                n_out=n_rad_max[1],
                norm_out=True,
            )
            BBB = ProductFunction(
                left=BB1,
                right=B,
                name="BBB",
                lmax=3,
                Lmax=int(max(tensor_components)),
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions2.append(BBB)
        if max_order > 3:
            BB2 = FCRight2Left(
                left=BB,
                right=B,
                name="BB2",
                n_out=n_rad_max[1],
                norm_out=True,
            )
            BBBB = ProductFunction(
                left=BB2,
                right=BB2,
                name="BBBB",
                lmax=3,
                Lmax=int(max(tensor_components)),
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions2.append(BBBB)
        if compute_energy:
            I_e = FunctionReduceN(
                instructions=instructions,
                name="rho_1",
                ls_max=[0] * len(instructions),
                n_out=n_mlp_dens + 1,
                is_central_atom_type_dependent=True,
                number_of_atom_types=num_elements,
                allowed_l_p=Parity.SCALAR,
            )
            I_e_LN = InvariantLayerRMSNorm(inpt=I_e, name="I_e_LN", type="only_nonlin")
            I2_e = FunctionReduceN(
                instructions=instructions2,
                name="rho_2",
                ls_max=0,
                n_out=n_mlp_dens + 1,
                is_central_atom_type_dependent=False,
                allowed_l_p=Parity.SCALAR,
            )
            I2_e_LN = InvariantLayerRMSNorm(inpt=I2_e, name="I2_e_LN", type="full")
            out_instr = CreateOutputTarget(name=tc.PREDICT_ATOMIC_ENERGY)
            LinMLPOut2ScalarTarget(
                origin=[I_e_LN, I2_e_LN],
                target=out_instr,
                hidden_layers=[64],
                activation="tanh",
            )
            ConstantScaleShiftTarget(
                target=out_instr,
                scale=constant_out_scale,
                shift=constant_out_shift,
                atomic_shift_map=atomic_shift_map,
            )

        for component in tensor_components:
            if component == 0:
                out_0 = CreateOutputTarget(name=tensor_constants.PREDICT_L0_term, l=0)
                I0 = FunctionReduceParticular(
                    instructions=[*instructions, *instructions2],
                    name="I0",
                    selected_l=0,
                    selected_p=1,
                    n_out=n_mlp_dens + 1,
                    is_central_atom_type_dependent=True,
                    number_of_atom_types=num_elements,
                )
                LinMLPOut2ScalarTarget(
                    origin=[I0],
                    name="a0",
                    target=out_0,
                    hidden_layers=[64],
                    activation="tanh",
                )
            if component == 1:
                out_1 = CreateOutputTarget(name=tensor_constants.PREDICT_L1_term, l=1)
                I1 = FunctionReduceParticular(
                    instructions=[*instructions, *instructions2],
                    name="I1",
                    selected_l=1,
                    selected_p=-1,
                    n_out=1,
                    is_central_atom_type_dependent=True,
                    number_of_atom_types=num_elements,
                )
                LinearOut2EquivarTarget(origin=[I1], name="a1", target=out_1, l=1)
            elif component == 2:
                out_2 = CreateOutputTarget(name=tensor_constants.PREDICT_L2_term, l=2)
                I2 = FunctionReduceParticular(
                    instructions=[*instructions, *instructions2],
                    name="I2",
                    selected_l=2,
                    selected_p=1,
                    n_out=1,
                    is_central_atom_type_dependent=True,
                    number_of_atom_types=num_elements,
                )
                LinearOut2EquivarTarget(
                    origin=[I2], name="a2", target=out_2, l=2, full_r2_form=False
                )

    return instructor.get_instructions()


class ComputeBatchEFTensor(TrainFunction):
    specs = {
        tc.BOND_IND_I: {"shape": [None], "dtype": "int"},
        tc.BOND_IND_J: {"shape": [None], "dtype": "int"},
        tc.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
        tc.BONDS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
        tc.N_STRUCTURES_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        tc.BOND_VECTOR: {"shape": [None, 3], "dtype": "float"},
        tc.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
    }

    def __init__(self, compute_function_config: dict):
        super().__init__()
        self.tensor_components: list = compute_function_config.get(
            "tensor_components", [0, 1, 2]
        )
        self.is_per_structure_tensor_output: bool = compute_function_config.get(
            "per_structure", False
        )
        self.compute_forces: bool = compute_function_config.get("compute_forces", True)
        self.compute_energy: bool = compute_function_config.get("compute_energy", True)

        assert len(set(self.tensor_components)) == len(
            self.tensor_components
        ), "Only unique tensor components from (0, 1, 2) are supported"

        if set(self.tensor_components) == {0, 1, 2}:
            self.evaluate = compute_012_tensor
        elif set(self.tensor_components) == {1, 2}:
            self.evaluate = compute_12_tensor
        elif set(self.tensor_components) == {0, 2}:
            self.evaluate = compute_02_tensor
        elif set(self.tensor_components) == {1}:
            self.evaluate = compute_1_tensor
        elif set(self.tensor_components) == {2}:
            self.evaluate = compute_2_tensor
        else:
            raise ValueError(f"Unsupported tensor components {self.tensor_components}")

    def __call__(
        self,
        instructions: dict,
        input_data: dict,
        training: bool = False,
    ):
        result = {}
        if self.compute_energy:
            if self.compute_forces:
                with tf.GradientTape() as tape:
                    tape.watch(input_data[tc.BOND_VECTOR])
                    execute_instructions(input_data, instructions, training=training)
                    e_atomic = tf.reshape(input_data[tc.PREDICT_ATOMIC_ENERGY], [-1, 1])
                nat = tf.reshape(input_data[tc.N_ATOMS_BATCH_TOTAL], [])
                pair_f = tf.negative(
                    tape.gradient(e_atomic, input_data[tc.BOND_VECTOR])
                )
                result["z_" + tc.PREDICT_PAIR_FORCES] = pair_f
                total_f = tf.math.unsorted_segment_sum(
                    pair_f, input_data[tc.BOND_IND_J], num_segments=nat
                ) - tf.math.unsorted_segment_sum(
                    pair_f, input_data[tc.BOND_IND_I], num_segments=nat
                )
                result[tc.PREDICT_FORCES] = total_f
                virial = compute_batch_virials_from_pair_forces(pair_f, input_data)
                result[tc.PREDICT_VIRIAL] = virial
            else:
                execute_instructions(input_data, instructions, training=training)
                e_atomic = tf.reshape(input_data[tc.PREDICT_ATOMIC_ENERGY], [-1, 1])
            total_energy = tf.math.unsorted_segment_sum(
                e_atomic,
                input_data[tc.ATOMS_TO_STRUCTURE_MAP],
                num_segments=input_data[tc.N_STRUCTURES_BATCH_TOTAL],
            )
            result[tc.PREDICT_TOTAL_ENERGY] = total_energy
        else:
            execute_instructions(input_data, instructions, training=training)
        tensor_output = self.evaluate(input_data, self.is_per_structure_tensor_output)
        result[tensor_constants.PREDICT_TENSOR] = tensor_output

        return result




def process_l0(a0):
    I = tf.eye(3, batch_shape=[1], dtype=a0.dtype)
    A0 = I * tf.reshape(a0, [-1, 1, 1])
    return tf.reshape(A0, [-1, 9])


def process_l1(a1):
    e_ijk = tf.constant(
        [
            [[0, 0, 0], [0, 0, 1], [0, -1, 0]],
            [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
            [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
        ],
        dtype=a1.dtype,
    )
    A1 = tf.einsum("...k,kbc->...bc", a1, e_ijk)
    return tf.reshape(A1, [-1, 9])


def process_l2(a2):
    A2 = tf.gather(a2, [0, 3, 4, 3, 1, 5, 4, 5, 2], axis=-1)
    return tf.reshape(A2, [-1, 9])


def reduce_atoms_to_structure(x, input_data):
    return tf.math.unsorted_segment_sum(
        x,
        input_data[tc.ATOMS_TO_STRUCTURE_MAP],
        num_segments=input_data[tc.N_STRUCTURES_BATCH_TOTAL],
    )


def compute_012_tensor(
    input_data,
    is_per_structure_tensor_output: bool,
):
    a0 = input_data[tensor_constants.PREDICT_L0_term]
    A0 = process_l0(a0)

    a1 = input_data[tensor_constants.PREDICT_L1_term][:, 0, :]
    A1 = process_l1(a1)

    a2 = input_data[tensor_constants.PREDICT_L2_term][:, 0, :]
    A2 = process_l2(a2)

    full_tensor = A0 + A1 + A2
    if is_per_structure_tensor_output:
        full_tensor = reduce_atoms_to_structure(full_tensor, input_data)

    return full_tensor


def compute_12_tensor(input_data, is_per_structure_tensor_output: bool):
    a1 = input_data[tensor_constants.PREDICT_L1_term][:, 0, :]
    A1 = process_l1(a1)

    a2 = input_data[tensor_constants.PREDICT_L2_term][:, 0, :]
    A2 = process_l2(a2)

    full_tensor = A1 + A2
    if is_per_structure_tensor_output:
        full_tensor = reduce_atoms_to_structure(full_tensor, input_data)

    return full_tensor


def compute_02_tensor(input_data, is_per_structure_tensor_output: bool):
    a0 = input_data[tensor_constants.PREDICT_L0_term]
    A0 = process_l0(a0)

    a2 = input_data[tensor_constants.PREDICT_L2_term][:, 0, :]
    A2 = process_l2(a2)

    full_tensor = A0 + A2
    if is_per_structure_tensor_output:
        full_tensor = reduce_atoms_to_structure(full_tensor, input_data)

    return full_tensor


def compute_1_tensor(input_data, is_per_structure_tensor_output: bool):
    a1 = input_data[tensor_constants.PREDICT_L1_term][:, 0, :]

    full_tensor = a1
    if is_per_structure_tensor_output:
        full_tensor = reduce_atoms_to_structure(full_tensor, input_data)

    return full_tensor


def compute_2_tensor(input_data, is_per_structure_tensor_output: bool):
    a2 = input_data[tensor_constants.PREDICT_L2_term][:, 0, :]
    A2 = process_l2(a2)

    full_tensor = A2
    if is_per_structure_tensor_output:
        full_tensor = reduce_atoms_to_structure(full_tensor, input_data)

    return full_tensor


ComputeStructureEFTensor = ComputeBatchEFTensor
