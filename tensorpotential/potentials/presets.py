from __future__ import annotations

from tensorpotential.instructions.base import *
from tensorpotential.instructions.compute import *
from tensorpotential.instructions.output import (
    CreateOutputTarget,
    LinearOut2Target,
    LinearOut2EquivarTarget,
    FSOut2ScalarTarget,
    MLPOut2ScalarTarget,
    ConstantScaleShiftTarget,
    LinMLPOut2ScalarTarget,
)


def LINEAR(
    element_map: dict,
    rcut: float = 6.0,
    avg_n_neigh: float = 1.0,
    constant_out_shift: float = 0.0,
    constant_out_scale: float = 1.0,
    atomic_shift_map: dict = None,
    lmax=4,
    basis_type="SBessel",  # SBessel
    rad_base_normalized=False,
    n_rad_base=8,
    n_rad_max=24,
    embedding_size=32,
    max_order=3,
    crad_init="random",
    func_init="random",
    chem_init="random",
    normalize_prod=False,
    **kwargs,
):
    num_elements = len(element_map)
    with InstructionManager() as instructor:
        d_ij = BondLength()
        rhat = ScaledBondVector(bond_length=d_ij)
        g_k = RadialBasis(
            bonds=d_ij,
            basis_type=basis_type,
            rcut=rcut,
            nfunc=n_rad_base,
            p=5,
            normalized=rad_base_normalized,
        )
        R_nl = LinearRadialFunction(
            n_rad_max=n_rad_max, lmax=lmax, basis=g_k, name="R", init=crad_init
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
            AA = ProductFunction(
                left=A,
                right=A,
                name="AA",
                lmax=lmax,
                Lmax=lmax,
                keep_parity=Parity.FULL_PARITY,
                normalize=normalize_prod,
            )
            instructions.append(AA)

        if max_order > 2:
            AAA = ProductFunction(
                left=AA,
                right=A,
                name="AAA",
                lmax=lmax,
                Lmax=0,
                keep_parity=Parity.FULL_PARITY,
                normalize=normalize_prod,
            )
            instructions.append(AAA)

        if max_order > 3:
            AAAA = ProductFunction(
                left=AA,
                right=AA,
                name="AAAA",
                lmax=lmax,
                Lmax=0,
                keep_parity=Parity.FULL_PARITY,
                normalize=normalize_prod,
            )
            instructions.append(AAAA)

        ls_max = [0] * len(instructions)

        I_l = FunctionReduce(
            instructions=instructions,
            name="E",
            ls_max=ls_max,
            n_in=n_rad_max,
            n_out=1,
            is_central_atom_type_dependent=True,
            number_of_atom_types=num_elements,
            allowed_l_p=Parity.SCALAR,
            init_vars=func_init,
        )

        out_instr = CreateOutputTarget(name=constants.PREDICT_ATOMIC_ENERGY)
        LinearOut2Target(origin=[I_l], target=out_instr)
        ConstantScaleShiftTarget(
            target=out_instr,
            scale=constant_out_scale,
            shift=constant_out_shift,
            atomic_shift_map=atomic_shift_map,
        )

    return instructor.instruction_list


def FS(
    element_map: dict,
    rcut: float = 6.0,
    avg_n_neigh: float = 1.0,
    constant_out_shift: float = 0.0,
    constant_out_scale: float = 1.0,
    atomic_shift_map: dict = None,
    lmax=(5, 5, 4, 3),
    Lmax=(None, 3, 0, 0),
    max_sum_l=(None, None, 6, 4),
    lmax_hist=(None, None, None, 3),
    basis_type="SBessel",  # SBessel
    rad_base_normalized=False,
    n_rad_base=20,
    n_rad_max=(20, 15, 10, 5),
    embedding_size=64,
    max_order=4,
    crad_init: Literal["random", "delta"] = "random",
    func_init: Literal["random", "zeros"] = "random",
    chem_init: Literal["random", "zeros", "delta"] = "random",
    fs_parameters=((1.0, 1.0), (1.0, 0.5), (1.0, 2), (1.0, 0.75)),
    normalize_prod=True,
    parity="real",
    simplify_prod=True,
    **kwargs,
    # mlp_embedding=None,  # {ndens: 8, hidden_layers: [32] } # not yet supported for export
):
    if isinstance(n_rad_max, int):
        n_rad_max = [n_rad_max] * max_order
    if isinstance(lmax, int):
        lmax = [lmax] * max_order
    num_elements = len(element_map)
    with InstructionManager() as instructor:
        d_ij = BondLength()
        rhat = ScaledBondVector(bond_length=d_ij)

        g_k = RadialBasis(
            bonds=d_ij,
            basis_type=basis_type,
            nfunc=n_rad_base,
            p=5,
            normalized=rad_base_normalized,
            rcut=rcut,
        )
        R_nl = LinearRadialFunction(
            n_rad_max=n_rad_max[0], lmax=max(lmax), basis=g_k, name="R", init=crad_init
        )

        Y = SphericalHarmonic(vhat=rhat, lmax=max(lmax), name="Y")

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
        keep_parity = {"full": Parity.FULL_PARITY, "real": Parity.REAL_PARITY}[parity]
        drop_list = "DEFAULT" if simplify_prod else None
        if max_order > 1:
            AA = CropProductFunction(
                left=A,
                right=A,
                name="AA",
                lmax=lmax[1],
                Lmax=Lmax[1],
                max_sum_l=max_sum_l[1],
                lmax_hist=lmax_hist[1],
                n_crop=n_rad_max[1],
                keep_parity=keep_parity,
                normalize=normalize_prod,
                history_drop_list=drop_list,
            )
            instructions.append(AA)

        if max_order > 2:
            AAA = CropProductFunction(
                left=AA,
                right=A,
                name="AAA",
                lmax=lmax[2],
                Lmax=Lmax[2],
                lmax_hist=lmax_hist[2],
                max_sum_l=max_sum_l[2],
                n_crop=n_rad_max[2],
                keep_parity=keep_parity,
                normalize=normalize_prod,
                history_drop_list=drop_list,
            )
            instructions.append(AAA)

        if max_order > 3:
            AAAA = CropProductFunction(
                left=AA,
                right=AA,
                name="AAAA",
                lmax=lmax[3],
                Lmax=Lmax[3],
                lmax_hist=lmax_hist[3],
                max_sum_l=max_sum_l[3],
                n_crop=n_rad_max[3],
                keep_parity=keep_parity,
                normalize=normalize_prod,
                history_drop_list=drop_list,
            )
            instructions.append(AAAA)

        out_instr = CreateOutputTarget(name=constants.PREDICT_ATOMIC_ENERGY)

        n_out = len(fs_parameters)
        I = FunctionReduceN(
            instructions=instructions,
            name="E",
            ls_max=0,
            n_in=n_rad_max,  # TODO: why do we need it?
            n_out=n_out,
            is_central_atom_type_dependent=True,
            number_of_atom_types=num_elements,
            allowed_l_p=Parity.SCALAR,
            init_vars=func_init,
            simplify=simplify_prod,
        )
        FSOut2ScalarTarget(origin=[I], target=out_instr, fs_parameters=fs_parameters)

        # if not isinstance(mlp_embedding, dict):
        #     mlp_embedding = {}
        # n_out = mlp_embedding.get("ndens", 8)
        # hidden_layers = mlp_embedding.get("hidden_layers", [32])
        #
        # I_l = FunctionReduce(
        #     instructions=instructions,
        #     name="E",
        #     ls_max=0,
        #     n_in=n_rad_max,
        #     n_out=1,
        #     is_central_atom_type_dependent=True,
        #     number_of_atom_types=num_elements,
        #     allowed_l_p=Parity.SCALAR,
        #     init_vars=func_init,
        # )
        # I_nl = FunctionReduce(
        #     instructions=instructions,
        #     name="rho",
        #     ls_max=0,
        #     n_in=n_rad_max,
        #     n_out=n_out,
        #     is_central_atom_type_dependent=True,
        #     number_of_atom_types=num_elements,
        #     allowed_l_p=Parity.SCALAR,
        #     init_vars=func_init,
        # )
        # LinearOut2Target(origin=[I_l], target=out_instr)
        # MLPOut2ScalarTarget(
        #     origin=[I_nl],
        #     target=out_instr,
        #     hidden_layers=hidden_layers,
        #     normalize=True,
        # )
        ConstantScaleShiftTarget(
            target=out_instr,
            scale=constant_out_scale,
            shift=constant_out_shift,
            atomic_shift_map=atomic_shift_map,
        )
    return instructor.instruction_list


def GRACE_1LAYER(
    element_map: dict,
    rcut: float = 6.0,
    avg_n_neigh: float = 1.0,
    constant_out_shift: float = 0.0,
    constant_out_scale: float = 1.0,
    n_rad_base=8,
    basis_type="Cheb",
    cutoff_function_order=5,
    rad_base_normalized=False,
    embedding_size=128,
    lmax=4,
    n_rad_max=42,
    n_mlp_dens=16,
    max_order=4,
    mlp_radial=True,
    norm_out=False,
    atomic_shift_map: dict = None,
    func_init: Literal["random", "zeros"] = "random",
    chem_init: Literal["random", "zeros"] = "random",
    cutoff_dict: dict = None,
    zbl_cutoff: dict = None,
    **kwargs,
):
    num_elements = len(element_map)
    with InstructionManager() as instructor:
        d_ij = BondLength()
        rhat = ScaledBondVector(bond_length=d_ij)

        if cutoff_dict is None:
            g_k = RadialBasis(
                bonds=d_ij,
                basis_type=basis_type,
                nfunc=n_rad_base,
                p=cutoff_function_order,
                normalized=rad_base_normalized,
                rcut=rcut,
            )
        else:
            g_k = BondSpecificRadialBasisFunction(
                bonds=d_ij,
                element_map=element_map,
                cutoff_dict=cutoff_dict,
                cutoff=rcut,
                cutoff_type="symmetric_bond",
                cutoff_function_param=cutoff_function_order,
                basis_type=basis_type,
                nfunc=n_rad_base,
            )

        if mlp_radial:
            R_nl = MLPRadialFunction(
                n_rad_max=n_rad_max,
                lmax=lmax,
                basis=g_k,
                name="R",
                hidden_layers=[64, 64],
                activation="tanh",
            )
        else:
            R_nl = LinearRadialFunction(
                n_rad_max=n_rad_max, lmax=lmax, basis=g_k, name="R"
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
                Lmax=0,
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
                Lmax=0,
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions.append(AAAA)

        I_nl = FunctionReduceN(
            instructions=instructions,
            name="rho",
            ls_max=0,
            n_out=n_mlp_dens + 1,  # +1 for linear input
            is_central_atom_type_dependent=True,
            number_of_atom_types=num_elements,
            allowed_l_p=Parity.SCALAR,
            init_vars=func_init,
        )

        out_instr = CreateOutputTarget(name=constants.PREDICT_ATOMIC_ENERGY)
        LinMLPOut2ScalarTarget(
            origin=[I_nl], target=out_instr, hidden_layers=[64], normalize=norm_out
        )
        ConstantScaleShiftTarget(
            target=out_instr,
            scale=constant_out_scale,
            shift=constant_out_shift,
            atomic_shift_map=atomic_shift_map,
        )
        if zbl_cutoff is not None:
            zbl = ZBLPotential(bonds=d_ij, cutoff=zbl_cutoff, element_map=element_map)
            LinearOut2Target(origin=[zbl], target=out_instr, name="zbl_output")

    return instructor.instruction_list


GRACE_1LAYER_v2 = GRACE_1LAYER  # for backward compatibility


def GRACE_2LAYER(
    element_map: dict,
    rcut: float = 5.0,
    avg_n_neigh: float = 1.0,
    constant_out_shift: float = 0.0,
    constant_out_scale: float = 1.0,
    n_rad_base: int = 8,
    basis_type: str = "Cheb",
    cutoff_function_order: int = 5,
    rad_base_normalized: bool = False,
    embedding_size: int = 128,
    lmax=(3, 3),  # first and second layer
    n_rad_max=(32, 48),  # first and second layer
    n_mlp_dens: int = 10,  # readout MLP input dimensions
    max_order: int = 4,  # max product order
    mlp_radial: bool = True,
    indicator_lmax: int = 1,  # maximum l of indicator
    atomic_shift_map: dict = None,
    func_init: Literal["random", "zeros"] = "random",
    chem_init: Literal["random", "zeros"] = "random",
    cutoff_dict: dict = None,
    zbl_cutoff: dict = None,
    **kwargs,
):
    if isinstance(lmax, int):
        # expand parameter for both layers
        lmax = [lmax, lmax]
    if isinstance(n_rad_max, int):
        # expand parameter for both layers
        n_rad_max = [n_rad_max, n_rad_max]
    Il = indicator_lmax
    num_elements = len(element_map)
    with InstructionManager() as instructor:
        d_ij = BondLength()
        rhat = ScaledBondVector(bond_length=d_ij)

        if cutoff_dict is None:
            g_k = RadialBasis(
                bonds=d_ij,
                basis_type=basis_type,
                nfunc=n_rad_base,
                p=cutoff_function_order,
                normalized=rad_base_normalized,
                rcut=rcut,
            )
        else:
            g_k = BondSpecificRadialBasisFunction(
                bonds=d_ij,
                element_map=element_map,
                cutoff_dict=cutoff_dict,
                cutoff=rcut,
                cutoff_type="symmetric_bond",
                cutoff_function_param=cutoff_function_order,
                basis_type=basis_type,
                nfunc=n_rad_base,
            )

        if mlp_radial:
            R_nl = MLPRadialFunction(
                n_rad_max=n_rad_max[0],
                lmax=lmax[0],
                basis=g_k,
                name="R",
                hidden_layers=[64, 64],
                activation="tanh",
            )
        else:
            R_nl = LinearRadialFunction(
                n_rad_max=n_rad_max[0], lmax=lmax[0], basis=g_k, name="R"
            )

        Y = SphericalHarmonic(vhat=rhat, lmax=lmax[0], name="Y")
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
                norm_out=True,
            )
            AA = ProductFunction(
                left=A1,
                right=A1,
                name="AA",
                lmax=lmax[0],
                Lmax=lmax[0],
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions.append(AA)

        if max_order > 2:
            AA1 = FCRight2Left(
                left=AA,
                right=A,
                name="AA1",
                n_out=n_rad_max[0],
                norm_out=True,
            )
            AAA = ProductFunction(
                left=AA1,
                right=A,
                name="AAA",
                lmax=lmax[0],
                Lmax=Il,
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
                norm_out=True,
            )
            AAAA = ProductFunction(
                left=AA2,
                right=AA2,
                name="AAAA",
                lmax=lmax[0],
                Lmax=0,
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions.append(AAAA)

        indicator_l_parity = Parity.REAL_PARITY
        I0 = FunctionReduceN(
            name="I0",
            instructions=instructions,
            ls_max=[Il, Il, Il, 0][: len(instructions)],
            n_out=16,
            is_central_atom_type_dependent=True,
            number_of_atom_types=num_elements,
            allowed_l_p=indicator_l_parity[: Il + 1],
        )
        I = FunctionReduceN(
            name="I",
            instructions=[I0],
            ls_max=[Il],
            n_out=n_rad_max[1],
            is_central_atom_type_dependent=False,
            allowed_l_p=indicator_l_parity[: Il + 1],
        )

        R1_nl = MLPRadialFunction(
            n_rad_max=n_rad_max[1],
            lmax=lmax[0],
            basis=g_k,
            name="R1",
            hidden_layers=[64, 64],
            activation="tanh",
        )
        B0 = SingleParticleBasisFunctionScalarInd(
            radial=R1_nl, angular=Y, indicator=z, name="B0", avg_n_neigh=avg_n_neigh
        )
        YI = SingleParticleBasisFunctionEquivariantInd(
            radial=R1_nl,
            angular=Y,
            indicator=I,
            name="YI",
            lmax=lmax[0],
            Lmax=lmax[1],
            avg_n_neigh=avg_n_neigh,
            keep_parity=Parity.REAL_PARITY,
            normalize=True,
        )
        B = FunctionReduceN(
            instructions=[YI, B0],
            name="B",
            ls_max=lmax[1],
            out_norm=False,
            n_out=n_rad_max[1],
            is_central_atom_type_dependent=False,
            allowed_l_p=Parity.REAL_PARITY,
        )

        instructions2 = [B]

        if max_order > 1:
            B1 = FCRight2Left(
                left=B,
                right=B,
                name="B1",
                n_out=n_rad_max[1],
                is_central_atom_type_dependent=False,
                norm_out=True,
            )
            BB = ProductFunction(
                left=B1,
                right=B1,
                name="BB",
                lmax=lmax[1],
                Lmax=lmax[1],
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions2.append(BB)
        if max_order > 2:
            BB1 = FCRight2Left(
                left=BB,
                right=B,
                name="BB1",
                n_out=n_rad_max[1],
                is_central_atom_type_dependent=False,
                norm_out=True,
            )
            BBB = ProductFunction(
                left=BB1,
                right=B,
                name="BBB",
                lmax=lmax[1],
                Lmax=0,
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
                is_central_atom_type_dependent=False,
                norm_out=True,
            )
            BBBB = ProductFunction(
                left=BB2,
                right=BB2,
                name="BBBB",
                lmax=lmax[1],
                Lmax=0,
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions2.append(BBBB)

        full_ins = instructions + instructions2
        I_out = FunctionReduceN(
            name="I_out",
            instructions=full_ins,
            ls_max=0,
            n_out=n_mlp_dens + 1,  # +1 for linear output
            is_central_atom_type_dependent=True,
            number_of_atom_types=num_elements,
            allowed_l_p=[[0, 1]],
            init_vars=func_init,
        )

        out_instr = CreateOutputTarget(name=constants.PREDICT_ATOMIC_ENERGY)
        LinMLPOut2ScalarTarget(origin=[I_out], target=out_instr, hidden_layers=[64])
        ConstantScaleShiftTarget(
            target=out_instr,
            scale=constant_out_scale,
            shift=constant_out_shift,
            atomic_shift_map=atomic_shift_map,
        )
        if zbl_cutoff is not None:
            zbl = ZBLPotential(bonds=d_ij, cutoff=zbl_cutoff, element_map=element_map)
            LinearOut2Target(origin=[zbl], target=out_instr, name="zbl_output")

    return instructor.instruction_list
