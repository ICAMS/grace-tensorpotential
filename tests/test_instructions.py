import os

import pytest
from ase.neighborlist import neighbor_list
from tensorpotential.functions.lora import (
    initialize_lora_tensors,
    lora_reconstruction,
    apply_lora_update,
)

from tensorpotential.functions.nn import DenseLayer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

from ase.build import bulk
from ase import Atoms
from matplotlib import pylab as plt
from scipy.spatial.transform import Rotation
from tensorflow import float64

from tensorpotential.instructions import (
    BondLength,
    BondSpecificRadialBasisFunction,
    CreateOutputTarget,
    EquivariantGate,
    EquivariantRMSNorm,
    FCRight2Left,
    FunctionReduce,
    FunctionReduceN,
    GeneralProductFunction,
    InstructionManager,
    LinMLPOut2ScalarTarget,
    LinMLPScalarReadOut,
    LinearOut2Target,
    LinearRadialFunction,
    MLPRadialFunction,
    MLPRadialFunction_v2,
    ProductFunction,
    RadialBasis,
    ScalarChemicalEmbedding,
    ScaledBondVector,
    SingleParticleBasisFunctionEquivariantInd,
    SingleParticleBasisFunctionScalarInd,
    SphericalHarmonic,
    ZBLPotential,
)
from tensorpotential import constants
from tensorpotential.utils import Parity
from .utils import print_full

np.random.seed(322)


def assert_tp_var(tensor, meta_data_df, check=True):
    tensor = tensor.reshape(tensor.shape[0], -1)
    h_g = meta_data_df.groupby(["l", "hist", "parity"]).indices
    for h, ii in h_g.items():
        print(h, ii)
        x = tensor[:, ii]
        x2 = np.mean(x**2)
        # var = np.var(x)
        print("<x**2>=", x2, np.var(x))
        if check:
            assert np.abs(x2 - 1.0) < 1e-5


def test_radial_basis():
    rcut = 4
    nfunc = 8
    fake_bonds = np.random.uniform(0, rcut, size=(10, 1))
    basis = RadialBasis(
        bonds="fake_bonds", basis_type="SBessel", nfunc=nfunc, rcut=rcut
    )
    basis.build(float64)
    baza = basis.frwrd({"fake_bonds": fake_bonds})
    assert baza.shape == (10, nfunc)

    nfunc = 10
    # fake_bonds = np.random.normal(0, rcut, size=(100, 1))
    # print(baza, np.var(baza, axis=1))
    fake_bonds = np.linspace(0, rcut, 100).reshape(-1, 1)
    basis = RadialBasis(
        bonds="fake_bonds",
        basis_type="Gaussian",
        # basis_type="Cheb",
        nfunc=nfunc,
        rcut=rcut,
        p=16,
        normalized=False,
        init_gamma=1,
    )
    basis.build(float64)
    baza = basis.frwrd({"fake_bonds": fake_bonds})
    # plt.plot(fake_bonds, baza)
    # plt.show()
    print(np.var(baza, axis=0))
    assert baza.shape == (100, nfunc)
    print("=================================")

    nfunc = 10
    fake_bonds = np.linspace(0, rcut, 100).reshape(-1, 1)
    # fake_bonds = np.random.uniform(0, rcut, size=(1000, 1))
    basis = RadialBasis(
        bonds="fake_bonds",
        basis_type="Cheb",
        nfunc=nfunc,
        rcut=rcut,
        normalized=True,
    )
    basis.build(float64)
    baza = basis.frwrd({"fake_bonds": fake_bonds})
    # plt.plot(fake_bonds, baza)
    # plt.show()
    assert baza.shape == (100, nfunc)


def test_linear_radial_function():
    nfunc = 8
    n_rad_max = 7
    lmax = 2

    fake_basis = np.random.uniform(0, 4, size=(10, nfunc))
    rfunc = LinearRadialFunction(
        n_rad_max=n_rad_max, input_shape=nfunc, lmax=lmax, basis="fake_basis", name="R"
    )
    rfunc.build(float64)

    rf = rfunc.frwrd({"fake_basis": fake_basis})
    assert rf.shape == (10, n_rad_max, int((lmax + 1) ** 2))


def n_act(x):
    return 1.59278 * tf.nn.tanh(x)


def test_mlp_function():
    nfunc = 8
    rcut = 6.2
    n_rad_max = 22
    lmax = 2

    # fake_bonds = np.linspace(0, rcut, 100).reshape(-1, 1)
    fake_bonds = np.random.uniform(0, rcut, size=(100_000, 1))
    data = {"fake_bonds": fake_bonds}
    basis = RadialBasis(
        name="base",
        bonds="fake_bonds",
        # basis_type="Cheb",
        basis_type="RadSinBessel",
        p=5,
        nfunc=nfunc,
        rcut=rcut,
        normalized=True,
    )
    basis.build(float64)
    data = basis(data)

    rfunc = MLPRadialFunction(
        n_rad_max=n_rad_max,
        input_shape=nfunc,
        lmax=lmax,
        basis=basis,
        name="R",
        activation="tanh",
    )
    rfunc.build(float64)

    data = rfunc(data)
    rf = data[rfunc.name].numpy()[:, :, [0, 1, 4]]
    # print(rf)
    print(np.var(rf, axis=0))
    # assert rf.shape == (10, n_rad_max, int((lmax + 1) ** 2))


def test_bond_cut():
    from ase.build import molecule
    from tensorpotential.data.databuilder import GeometricalDataBuilder

    el_map = {"C": 0, "Cl": 1, "H": 2, "O": 3}

    cutoff_dict = {
        "CC": 4.1,
        "CH": 2.1,
        "CO": 3,
        # ("C", "Cl"): 2,
        "ClCl": 2.9,
        "ClH": 2.2,
        "ClO": 2,
        "HH": 3,
        "HO": 2.11,
        "OO": 2,
    }
    default_cut = 5
    db = GeometricalDataBuilder(
        elements_map=el_map,
        cutoff=default_cut,
    )
    data = db.extract_from_ase_atoms(molecule("CH3COCl"))
    data = tf.data.Dataset.from_tensors(data).get_single_element()
    bonds = BondLength()
    bonds.build(float64)
    data = bonds(data)

    bs = BondSpecificRadialBasisFunction(
        bonds,
        element_map=el_map,
        cutoff_dict=cutoff_dict,
        cutoff=default_cut,
        nfunc=2,
    )
    bs.build(float64)
    data = bs(data)
    b = data[bs.name].numpy()[:, 0]
    print(b)
    dist = data[bonds.name].numpy()
    inv_map = {v: k for k, v in el_map.items()}
    non0_d = []
    cut_d = bs.cutoff_dict
    for bi, mui, muj, dd in zip(b, data["mu_i"].numpy(), data["mu_j"].numpy(), dist):
        mini = np.min((mui, muj))
        maxi = np.max((mui, muj))
        k = (inv_map[mini], inv_map[maxi])
        curr_cut = cut_d.get(k, default_cut)
        print(f"({inv_map[mui]}, {inv_map[muj]}), [{curr_cut}] : {dd}, {bi}")
        if dd >= curr_cut:
            assert bi == 0
        else:
            assert bi != 0
            non0_d.append(dd)

    print("#" * 100)
    print("#" * 100)
    print("#" * 100)
    db = GeometricalDataBuilder(
        elements_map=el_map,
        cutoff=default_cut,
        cutoff_dict=cutoff_dict,
    )
    data = db.extract_from_ase_atoms(molecule("CH3COCl"))
    data = tf.data.Dataset.from_tensors(data).get_single_element()
    bonds = BondLength()
    bonds.build(float64)
    data = bonds(data)

    bs = BondSpecificRadialBasisFunction(
        bonds,
        element_map=el_map,
        cutoff_dict=cutoff_dict,
        cutoff=default_cut,
        nfunc=2,
    )
    bs.build(float64)
    data = bs(data)
    b = data[bs.name].numpy()[:, 0]
    print(b)
    dist = data[bonds.name].numpy()
    inv_map = {v: k for k, v in el_map.items()}
    non0_d2 = []
    cut_d = bs.cutoff_dict
    for bi, mui, muj, dd in zip(b, data["mu_i"].numpy(), data["mu_j"].numpy(), dist):
        mini = np.min((mui, muj))
        maxi = np.max((mui, muj))
        k = (inv_map[mini], inv_map[maxi])
        # curr_cut = cutoff_dict.get(k, default_cut)
        # k = f'{inv_map[mini]}{inv_map[maxi]}'
        curr_cut = cut_d.get(k, default_cut)
        print(f"({inv_map[mui]}, {inv_map[muj]}), [{curr_cut}] : {dd}, {bi}")
        assert bi != 0
        non0_d2.append(dd)
    print("!" * 100)
    print(non0_d)
    print(non0_d2)
    assert len(non0_d) == len(non0_d2)
    assert np.allclose(np.array(non0_d) - np.array(non0_d2), 0)

    # =======================================================================================
    # =======================================================================================

    from tensorpotential.tpmodel import TPModel, ComputePlaceholder
    from tensorpotential.calculator import TPCalculator

    d_ij = BondLength()
    bs = BondSpecificRadialBasisFunction(
        d_ij,
        element_map=el_map,
        cutoff_dict=cutoff_dict,
        cutoff=default_cut,
        nfunc=1,
        name=constants.PLACEHOLDER,
    )
    z = ScalarChemicalEmbedding("z", element_map=el_map, embedding_size=2)
    model = TPModel(instructions=[d_ij, bs, z], compute_function=ComputePlaceholder())
    model.build(float64)
    calc = TPCalculator(
        model=model,
        cutoff=6,
        extra_properties=[bs.name],
        pad_atoms_number=None,
        pad_neighbors_fraction=None,
    )
    at = molecule("CH3COCl")

    at.calc = calc
    at.get_potential_energy()
    b_val = at.calc.results[constants.PLACEHOLDER]
    print(b_val)
    assert np.allclose(b_val.flatten(), b.flatten())
    print(at.calc.cutoff_dict)


def test_zbl(do_plot=False):
    rcut = 4.0
    r_in = 3.0
    size = 10
    np.random.seed(322)
    fake_bonds = np.random.uniform(0, rcut + 1, size=(size, 1)).astype(np.float64)
    elm = {
        s: i
        for i, s in enumerate(["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"])
    }
    zbl = ZBLPotential(
        bonds="fake_bonds", cutoff=rcut, delta_cutoff=r_in, element_map=elm
    )
    zbl.build(float64)
    dat = {"fake_bonds": fake_bonds}
    dat[constants.BOND_IND_I] = [0] * size
    dat[constants.N_ATOMS_BATCH_TOTAL] = 1
    dat[constants.BOND_MU_I] = [0] * size
    dat[constants.BOND_MU_J] = np.arange(size)
    dat = zbl(dat)
    print(dat[zbl.name])

    assert np.allclose(dat[zbl.name].numpy(), 288.62542471)

    from tensorpotential.tpmodel import TPModel
    from tensorpotential.calculator import TPCalculator

    rcut = 1.0
    r_in = 0.5
    d_ij = BondLength()
    zbl = ZBLPotential(
        bonds=d_ij,
        # cutoff=rcut,
        cutoff={
            "Al": 1.0,
            "AlH": 3.5,
            # ("H", "Al"): 5.5,
            "H": 1.0,
        },
        element_map={"Al": 0, "H": 1},
        delta_cutoff=r_in,
    )
    out_instr = CreateOutputTarget(name=constants.PREDICT_ATOMIC_ENERGY)
    lo = LinearOut2Target(origin=[zbl], target=out_instr)
    model = TPModel(instructions=[d_ij, zbl, out_instr, lo])
    model.build(float64)
    calc = TPCalculator(model=model, cutoff=6)
    at = bulk("Al", cubic=True) * (2, 2, 2)
    # at.rattle(stdev=0.01)
    at.calc = calc
    print(at.get_potential_energy())

    dd = np.linspace(0.8, 8, 100)
    es = []
    for d in dd:
        at = Atoms("AlH", positions=[[0, 0, 0], [0, 0, d]])
        at.calc = calc
        es.append(at.get_potential_energy())
    if do_plot:
        plt.plot(dd, es, label="AlH")

    es = []
    for d in dd:
        at = Atoms("AlAl", positions=[[0, 0, 0], [0, 0, d]])
        at.calc = calc
        es.append(at.get_potential_energy())
    if do_plot:
        plt.plot(dd, es, label="AlAl")

    es = []
    for d in dd:
        at = Atoms("HH", positions=[[0, 0, 0], [0, 0, d]])
        at.calc = calc
        es.append(at.get_potential_energy())

    if do_plot:
        plt.plot(dd, es, label="HH")

        plt.axhline(y=0, ls="--", c="r")
        plt.yscale("symlog")
        plt.legend()
        plt.show()


def test_bond_spherical_harmonics():
    size = 322
    lmax = 2
    fake_projections = np.random.normal(0, 1, size=(size, 3))
    fake_projections = fake_projections / np.linalg.norm(
        fake_projections, axis=1, keepdims=True
    )

    Y = SphericalHarmonic(vhat="fake_projections", lmax=lmax, name="Y")
    Y.build(float64)
    sg = Y.frwrd({"fake_projections": fake_projections})
    assert sg.shape == (size, int((lmax + 1) ** 2))
    assert np.allclose(np.var(sg), 1.0, atol=1e-1)


def test_product_function():
    # size = 2
    size = 22
    lmax = 2
    fake_projections = np.random.normal(0, 1, size=(size, 3))
    fake_projections = fake_projections / np.linalg.norm(
        fake_projections, axis=1, keepdims=True
    )

    Y = SphericalHarmonic(vhat="fake_projections", lmax=lmax, name="Y")
    Y.build(float64)
    sg = Y.frwrd({"fake_projections": fake_projections}).numpy()
    inpt_data = {"Y": sg.reshape(size, 1, int((lmax + 1) ** 2))}
    print_full(Y.coupling_meta_data)

    p = ProductFunction(
        left=Y,
        right=Y,
        name="AA",
        lmax=lmax,
        Lmax=1,
        is_left_right_equal=True,
        keep_parity=[[0, 1], [1, -1], [1, 1], [2, -1], [2, 1]],
        # keep_parity=[[2, -1], [2, 1]],
    )
    print_full(p.coupling_meta_data)
    print("-" * 200)
    # p.build(float64)
    # aa = p.frwrd(inpt_data)
    # print(aa, aa.shape)
    pp = ProductFunction(
        left=p,
        right=Y,
        name="AAA",
        lmax=lmax,
        Lmax=1,
        is_left_right_equal=True,
        keep_parity=[[0, 1], [1, -1], [1, 1], [2, -1], [2, 1]],
    )
    print_full(pp.coupling_meta_data)
    print("-" * 200)
    ddf = p.coupling_meta_data
    lll = ddf.groupby(["l", "parity", "hist"]).indices
    print_full(lll)
    print(len(lll))
    u_i = np.concatenate([[i] * len(v) for i, (k, v) in enumerate(lll.items())])
    print(f"{u_i=}", len(u_i))
    count_i = ddf.loc[np.concatenate([v for k, v in lll.items()]), "left_inds"].values
    map_u_i = np.concatenate([[i] * len(count) for (i, count) in zip(u_i, count_i)])
    print(f"{count_i=}", len(count_i))
    print(f"{map_u_i=}", len(map_u_i))
    print(p.m_sum_ind.numpy(), len(p.m_sum_ind.numpy()))
    l1 = np.array(p.coupling_meta_data["l1"]).reshape(1, -1)
    l2 = np.array(p.coupling_meta_data["l2"]).reshape(1, -1)
    l_arr = np.array(p.coupling_meta_data["l"]).reshape(1, -1)
    # print(np.concatenate([l_arr, l1, l2], axis=0).reshape(1, -1), len(l_arr))
    lll = np.ravel_multi_index(
        np.concatenate([l_arr, l1, l2], axis=0),
        [np.max(l_arr) + 1, np.max(l1) + 1, np.max(l2) + 1],
    )
    print(lll, lll.shape)
    p.build(float64)
    aa = p.frwrd(inpt_data)
    assert aa.shape == (size, 1, 9)
    for i, row in p.coupling_meta_data.iterrows():
        assert max(set(row["left_inds"])) < (row["l1"] + 1) ** 2
        assert max(set(row["right_inds"])) < (row["l2"] + 1) ** 2


def test_product_function_norm():
    size = 1000
    lmax = 3
    np.random.seed(322)
    fake_projections = np.random.normal(0, 1, size=(size, 3))
    fake_projections = fake_projections / np.linalg.norm(
        fake_projections, axis=1, keepdims=True
    )

    Y = SphericalHarmonic(vhat="fake_projections", lmax=lmax, name="Y")
    Y.build(float64)
    sh = Y.frwrd({"fake_projections": fake_projections}).numpy()
    tensor_dict = {"Y": sh.reshape(size, 1, int((lmax + 1) ** 2))}
    sh_var = sh.var()
    print(f"{sh_var=}")
    assert_tp_var(sh, Y.coupling_meta_data, check=False)

    print("AA")
    AA = ProductFunction(
        left=Y,
        right=Y,
        name="AA",
        lmax=lmax,
        Lmax=lmax,
        is_left_right_equal=True,
        normalize=True,
    )
    AA.build(float64)
    tensor_dict = AA(tensor_dict)
    aa = tensor_dict[AA.name].numpy()
    aa_var = aa.var()
    print(f"{aa_var=}")
    assert_tp_var(aa, AA.coupling_meta_data, check=True)

    print("AAA")
    AAA = ProductFunction(
        left=AA,
        right=Y,
        name="AAA",
        lmax=lmax,
        Lmax=lmax,
        normalize=True,
    )
    AAA.build(float64)
    tensor_dict = AAA(tensor_dict)
    aaa = tensor_dict[AAA.name].numpy()
    aaa_var = aaa.var()
    print(f"{aaa_var=}")
    assert_tp_var(aaa, AAA.coupling_meta_data, check=False)


def test_avg_sg():
    from ase.build import bulk
    from ase.neighborlist import neighbor_list

    cutoff = 5
    c = bulk("C", "diamond", cubic=True) * (3, 3, 3)
    c.rattle(stdev=0.1)
    ind_i, ind_j, bond_vector = neighbor_list("ijD", c, cutoff=cutoff)
    _, nn = np.unique(ind_i, return_counts=True)
    # size = 10
    lmax = 4
    # np.random.seed(322)
    # fake_neighbors = np.random.normal(0, 3, size=(size, 3))
    # fake_center = np.array([0, 0, 0]).reshape(1, -1)
    # fake_bonds = fake_neighbors - fake_center
    tensor_dict = {
        constants.BOND_VECTOR: bond_vector,
        constants.BOND_IND_I: ind_i,
        constants.N_ATOMS_BATCH_TOTAL: len(c),
    }
    d_ij = BondLength()
    d_ij.build(float64)
    tensor_dict = d_ij(tensor_dict)

    rhat = ScaledBondVector(bond_length=d_ij)
    rhat.build(float64)
    tensor_dict = rhat(tensor_dict)

    sg = SphericalHarmonic(vhat=rhat, lmax=lmax, name="Y")
    sg.build(float64)
    tensor_dict = sg(tensor_dict)
    assert_tp_var(tensor_dict[sg.name].numpy(), sg.coupling_meta_data, check=False)

    from tensorpotential.instructions.compute import BondAvgSphericalHarmonic

    avg_sg = BondAvgSphericalHarmonic(
        spherical_harmonics=sg,
        bonds=d_ij,
        rcut=cutoff,
        name="avg_Y",
        avg_n_neigh=np.mean(nn),
    )
    avg_sg.build(float64)
    tensor_dict = avg_sg(tensor_dict)

    print(tensor_dict[avg_sg.name])
    print(avg_sg.coupling_meta_data)
    assert_tp_var(
        tensor_dict[avg_sg.name].numpy(), avg_sg.coupling_meta_data, check=False
    )


def test_fc_func():

    cutoff = 4
    c = bulk("C", "diamond", cubic=True)
    c.rattle(stdev=0.1)
    ind_i, ind_j, bond_vector = neighbor_list("ijD", c, cutoff=cutoff)
    _, nn = np.unique(ind_i, return_counts=True)
    # size = 10
    lmax = 4
    # np.random.seed(322)
    # fake_neighbors = np.random.normal(0, 3, size=(size, 3))
    # fake_center = np.array([0, 0, 0]).reshape(1, -1)
    # fake_bonds = fake_neighbors - fake_center
    tensor_dict = {
        constants.BOND_VECTOR: bond_vector,
        constants.BOND_IND_I: ind_i,
        constants.N_ATOMS_BATCH_TOTAL: len(c),
        constants.ATOMIC_MU_I: np.zeros(len(c)).astype(np.int32),
    }
    d_ij = BondLength()
    d_ij.build(float64)
    tensor_dict = d_ij(tensor_dict)

    rhat = ScaledBondVector(bond_length=d_ij)
    rhat.build(float64)
    tensor_dict = rhat(tensor_dict)

    Y = SphericalHarmonic(vhat=rhat, lmax=lmax, name="Y")
    Y.build(float64)
    Y.n_out = 1
    tensor_dict = Y(tensor_dict)
    tensor_dict[Y.name] = tensor_dict[Y.name][:, tf.newaxis, :]

    p = ProductFunction(
        left=Y,
        right=Y,
        name="AA",
        lmax=lmax,
        Lmax=1,
        # keep_parity=[[0, 1], [1, -1], [1, 1], [2, 1], [2, -1], [3, 1], [3, -1]],
        keep_parity=[[0, 1], [1, -1], [2, 1], [3, -1]],
    )
    p.build(float64)
    # print_full(p.coupling_meta_data)
    tensor_dict = p(tensor_dict)
    print(tensor_dict[p.name].numpy().shape, "P_SHAPE!!!!!!!!!!!")
    p_p = FCRight2Left(left=Y, right=p, n_out=1, name="p_p", norm_out=True)
    p_p.build(float64)
    tensor_dict = p_p(tensor_dict)
    print(tensor_dict[p_p.name].numpy().shape)
    assert ~np.isnan(tensor_dict[p_p.name].numpy()).any()

    p_p2 = FCRight2Left(
        left=Y, right=p, n_out=1, name="p_p2", norm_out=False, left_coefs=False
    )
    p_p2.build(float64)
    tensor_dict = p_p2(tensor_dict)
    assert ~np.isnan(tensor_dict[p_p2.name].numpy()).any()


def test_function_collector():
    size = 22
    lmax = 4
    np.random.seed(322)
    tf.random.set_seed(322)
    fake_projections = np.random.normal(0, 1, size=(size, 3))
    fake_projections = fake_projections / np.linalg.norm(
        fake_projections, axis=1, keepdims=True
    )

    Y = SphericalHarmonic(vhat="fake_projections", lmax=lmax, name="Y")
    Y.build(float64)
    sg = Y.frwrd({"fake_projections": fake_projections}).numpy()
    inpt_data = {"Y": sg.reshape(size, 1, int((lmax + 1) ** 2))}
    Y.n_out = 1
    p = ProductFunction(
        left=Y,
        right=Y,
        name="AA",
        lmax=lmax,
        Lmax=3,
        keep_parity=[[0, 1], [1, -1], [1, 1], [2, 1], [2, -1], [3, 1], [3, -1]],
    )
    aa = p.frwrd(inpt_data)
    print("-----------------------------" * 50)
    for i, row in p.coupling_meta_data.iterrows():
        print(
            f'({row["l"]},{row["m"]}), {row["hist"]}; {row["parity"]}: {aa[0, 0, i].numpy()}'
        )
        print("&" * 25)
    print("-----------------------------" * 50)
    inpt_data["AA"] = aa
    print(inpt_data["AA"][0])
    print("***************************" * 50)
    inpt_data[constants.N_ATOMS_BATCH_TOTAL] = size
    inpt_data[constants.ATOMIC_MU_I] = np.zeros((size)).astype(np.int32)

    instr_reduce = FunctionReduce(
        instructions=[p],
        name="I",
        ls_max=[0],
        n_in=1,
        n_out=3,
        is_central_atom_type_dependent=True,
        number_of_atom_types=1,
        allowed_l_p=[[0, 1]],
    )
    instr_reduce.build(float64)
    inpt_data = instr_reduce(inpt_data)
    assert inpt_data[instr_reduce.name].numpy().shape == (size, 3, 1)

    II = FunctionReduceN(
        instructions=[p, instr_reduce],
        name="II",
        ls_max=[1, 1],
        n_in=1,
        n_out=3,
        is_central_atom_type_dependent=True,
        number_of_atom_types=1,
        allowed_l_p=[[0, 1], [1, -1], [1, 1]],
    )
    II.build(float64)
    inpt_data = II(inpt_data)


def test_invar_collect():
    from scipy.spatial.transform import Rotation

    # stage 0: rotation:
    axis = np.array([1, 2, 3])
    theta = np.pi / 2.17
    axis = axis / np.linalg.norm(axis)  # normalize the rotation vector first
    rot = Rotation.from_rotvec(theta * axis)

    # stage 1: input vectors:
    lmax = 3
    np.random.seed(322)
    tf.random.set_seed(322)
    input_vectors = [[3, 2, 1]]
    input_vectors.append(rot.apply(input_vectors[-1]))
    input_vectors.append(rot.apply(input_vectors[-1]))
    input_vectors.append(rot.apply(input_vectors[-1]))

    input_vectors = np.array(input_vectors)

    size = len(input_vectors)
    input_vectors = input_vectors / np.linalg.norm(input_vectors, axis=1, keepdims=True)
    assert np.allclose(rot.apply(input_vectors[0]), input_vectors[1])
    print("input_vectors=", input_vectors)
    # input_vectors = rot.apply(input_vectors)

    Y = SphericalHarmonic(vhat="input_vectors", lmax=lmax, name="Y")
    Y.build(float64)
    inpt_dict = {"input_vectors": input_vectors}
    inpt_dict = Y(inpt_dict)
    inpt_dict[Y.name] = inpt_dict[Y.name].numpy().reshape(size, 1, int((lmax + 1) ** 2))
    print("Y=", inpt_dict[Y.name])
    Y.n_out = 1
    p = ProductFunction(
        left=Y,
        right=Y,
        name="AA",
        lmax=lmax,
        Lmax=3,
        keep_parity=[[0, 1], [1, -1], [1, 1], [2, 1], [2, -1], [3, 1], [3, -1]],
    )
    inpt_dict = p(inpt_dict)

    p2 = ProductFunction(
        left=p,
        right=Y,
        name="AAA",
        lmax=3,
        Lmax=0,
        keep_parity=[[0, 1], [1, -1], [1, 1], [2, 1], [2, -1]],
    )
    inpt_dict = p2(inpt_dict)

    print("p2=", inpt_dict[p2.name])

    inpt_dict[constants.N_ATOMS_BATCH_TOTAL] = size
    inpt_dict[constants.ATOMIC_MU_I] = np.zeros((size)).astype(np.int32)

    n_out = 1
    instr_reduce_2 = FunctionReduce(
        instructions=[p, p2],
        name="I",
        ls_max=[0, 0],
        n_in=1,
        n_out=n_out,
        is_central_atom_type_dependent=True,
        number_of_atom_types=1,
        allowed_l_p=[[0, 1]],
    )
    instr_reduce_2.build(float64)
    print("I.Coupling: ")
    print(instr_reduce_2.coupling_meta_data)
    inpt_dict = instr_reduce_2(inpt_dict)
    I_out = inpt_dict[instr_reduce_2.name].numpy()[:, :, 0]
    print("I(xyz)=", I_out)

    v0 = I_out[0]
    for i in range(3):
        vv = I_out[i + 1]
        assert np.allclose(v0, vv)


@pytest.mark.parametrize(
    "norm",
    [
        True,
        False,
    ],
)
def test_rot_invar(norm):
    lmax_bond = 5
    n_rad_base = 8
    n_rad_max_bond = 5
    # n_rad_max_out = 6
    cutoff_rad = 6.0

    np.random.seed(322)
    tf.random.set_seed(322)

    axis = np.array([1, 1, 3])
    theta = np.pi / 2.2
    axis = axis / np.linalg.norm(axis)  # normalize the rotation vector first
    rot = Rotation.from_rotvec(theta * axis)

    coord = np.array([[0.2, 0.3, 2.1], [0.1, 3.2, 0.5], [0.3, 1, 2]])
    coord_r = rot.apply(coord)
    coord_2 = np.vstack([coord, coord_r]).astype(np.float64)
    print(coord_2)
    # stage 1: input vectors:

    indj = [1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4]
    indi = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    atomic_mu_i = [0, 1, 0, 0, 1, 0]
    # muj = [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    muj = np.take(atomic_mu_i, indj, axis=0)
    rj = np.take(coord_2, indj, axis=0)
    ri = np.take(coord_2, indi, axis=0)
    rij = rj - ri
    inpt_dict = {
        constants.BOND_VECTOR: rij,
        constants.BOND_IND_J: indj,
        constants.BOND_MU_J: muj,
        constants.BOND_IND_I: indi,
        constants.N_ATOMS_BATCH_TOTAL: len(coord_2),
        constants.ATOMIC_MU_I: atomic_mu_i,
    }
    ###########################################################################################
    d_ij = BondLength()
    d_ij.build(float64)
    inpt_dict = d_ij(inpt_dict)

    rhat = ScaledBondVector(bond_length=d_ij)
    rhat.build(float64)
    inpt_dict = rhat(inpt_dict)

    g_k = RadialBasis(
        bonds=d_ij,
        basis_type="SBessel",
        nfunc=n_rad_base,
        rcut=cutoff_rad,
        p=5,
    )
    g_k.build(float64)
    inpt_dict = g_k(inpt_dict)

    R_nl = MLPRadialFunction(
        n_rad_max=n_rad_max_bond, lmax=lmax_bond, basis=g_k, name="R"
    )
    R_nl.build(float64)
    inpt_dict = R_nl(inpt_dict)

    Y = SphericalHarmonic(vhat=rhat, lmax=lmax_bond, name="Y")
    Y.build(float64)
    inpt_dict = Y(inpt_dict)

    z = ScalarChemicalEmbedding(
        element_map={"H": 0, "C": 1},
        embedding_size=54,
        name="Z",
    )
    z.build(float64)
    inpt_dict = z(inpt_dict)

    A = SingleParticleBasisFunctionScalarInd(
        radial=R_nl,
        angular=Y,
        indicator=z,
        name="A",
    )
    A.build(float64)
    A.lin_transform.build(float64)
    inpt_dict = A(inpt_dict)

    AA = ProductFunction(
        left=A,
        right=A,
        name="AA",
        lmax=lmax_bond,
        Lmax=lmax_bond,
        normalize=norm,
    )
    AA.build(float64)
    inpt_dict = AA(inpt_dict)

    AAA = ProductFunction(
        left=AA,
        right=A,
        name="AAA",
        lmax=lmax_bond,
        Lmax=0,
        normalize=norm,
    )
    AAA.build(float64)
    inpt_dict = AAA(inpt_dict)
    I_l = FunctionReduce(
        instructions=[A, AA, AAA],
        name="E",
        ls_max=[0, 0, 0],
        n_in=n_rad_max_bond,
        n_out=1,
        is_central_atom_type_dependent=True,
        number_of_atom_types=2,
        allowed_l_p=[[0, 1]],
    )
    I_l.build(float64)
    inpt_dict = I_l(inpt_dict)
    out = inpt_dict[I_l.name]
    print(out)
    print(out[:3] - out[3:])
    assert np.allclose(out[:3], out[3:])


def test_Dense_layer():

    float_dtype = tf.float64
    n_in = 10
    n_out = 20
    dense = DenseLayer(n_in=n_in, n_out=n_out, name="TestDenseLayer")
    dense.build(float_dtype)

    batch_size = 15
    x = tf.random.uniform(
        shape=(
            batch_size,
            n_in,
        ),
        dtype=float_dtype,
    )
    assert x.shape == (batch_size, n_in)
    y = dense(x)
    assert y.shape == (batch_size, n_out)
    assert not np.allclose(y.numpy(), 0.0)


def test_LORA_Dense_layer_activate_reduce():

    float_dtype = tf.float64
    n_in = 10
    n_out = 20
    lora_rank = 2
    lora_config = {"rank": lora_rank, "alpha": 1.0}
    dense = DenseLayer(
        n_in=n_in,
        n_out=n_out,
        name="TestDenseLayer",
    )
    dense.build(float_dtype)

    batch_size = 15
    x = tf.random.uniform(
        shape=(
            batch_size,
            n_in,
        ),
        dtype=float_dtype,
    )
    assert x.shape == (batch_size, n_in)
    y = dense(x)
    assert y.shape == (batch_size, n_out)
    assert dense.w._trainable
    assert not hasattr(dense, "lora_tensors")
    assert not dense.lora

    # activate lora
    dense.enable_lora_adaptation(lora_config)
    y_lora = dense(x)

    assert hasattr(dense, "lora_tensors")
    assert dense.lora
    assert not dense.w._trainable
    assert np.allclose(y.numpy(), y_lora.numpy())

    # reduce_lora
    dense.finalize_lora_update()
    y_reduce_lora = dense(x)

    assert not hasattr(dense, "lora_tensors")
    assert not dense.lora
    assert dense.w._trainable
    assert np.allclose(y.numpy(), y_reduce_lora.numpy())


def test_LORA_Dense_layer_init():

    float_dtype = tf.float64
    n_in = 10
    n_out = 20
    lora_rank = 2
    lora_config = {"rank": lora_rank, "alpha": 1.0}
    dense = DenseLayer(
        n_in=n_in, n_out=n_out, name="TestDenseLayer", lora_config=lora_config
    )
    dense.build(float_dtype)

    batch_size = 15
    x = tf.random.uniform(
        shape=(
            batch_size,
            n_in,
        ),
        dtype=float_dtype,
    )
    assert x.shape == (batch_size, n_in)
    y = dense(x).numpy()
    assert y.shape == (batch_size, n_out)
    assert not dense.w._trainable
    assert hasattr(dense, "lora_tensors")
    assert dense.lora

    # modify lora tensors
    for t in dense.lora_tensors:
        t.assign(1e-2 * tf.ones_like(t))

    yb = dense(x).numpy()
    assert not np.allclose(y, yb)

    # reduce_lora
    dense.finalize_lora_update()
    y_reduce_lora = dense(x).numpy()
    assert not hasattr(dense, "lora_tensors")
    assert not dense.lora
    assert dense.w._trainable
    assert np.allclose(yb, y_reduce_lora)


def test_LORA_ScalarChemicalEmbedding():
    float_dtype = tf.float64
    Z = ScalarChemicalEmbedding(
        name="Z", element_map={"H": 0, "C": 1}, embedding_size=7
    )
    Z.build(float_dtype)

    w = Z.frwrd({}).numpy()
    assert w.shape == (2, 7)
    assert not Z.lora

    Z.enable_lora_adaptation({"rank": 5, "alpha": 1})

    w_new = Z.frwrd({}).numpy()
    assert Z.lora
    assert w_new.shape == (2, 7)
    assert np.allclose(w_new, w)
    assert hasattr(Z, "lora_tensors")

    # modify lora tensors
    for t in Z.lora_tensors:
        t.assign(tf.ones_like(t))

    w_newb = Z.frwrd({}).numpy()
    assert not np.allclose(w_new, w_newb)

    Z.finalize_lora_update()

    w_new_2 = Z.frwrd({}).numpy()
    assert not Z.lora
    assert w_new_2.shape == (2, 7)
    assert np.allclose(w_new_2, w_newb)
    assert not hasattr(Z, "lora_tensors")


def test_LORA_FCRight2Left():
    cutoff = 4
    c = bulk("C", "diamond", cubic=True)
    c.rattle(stdev=0.1)
    ind_i, ind_j, bond_vector = neighbor_list("ijD", c, cutoff=cutoff)
    # size = 10
    lmax = 4
    n_out = 23
    np.random.seed(322)
    # fake_neighbors = np.random.normal(0, 3, size=(size, 3))
    # fake_center = np.array([0, 0, 0]).reshape(1, -1)
    tensor_dict = {
        constants.BOND_VECTOR: bond_vector,
        constants.BOND_IND_I: ind_i,
        constants.N_ATOMS_BATCH_TOTAL: len(c),
        constants.ATOMIC_MU_I: np.zeros(len(c)).astype(np.int32),
    }
    d_ij = BondLength()
    d_ij.build(float64)
    tensor_dict = d_ij(tensor_dict)

    rhat = ScaledBondVector(bond_length=d_ij)
    rhat.build(float64)
    tensor_dict = rhat(tensor_dict)

    Y = SphericalHarmonic(vhat=rhat, lmax=lmax, name="Y")
    Y.build(float64)
    Y.n_out = n_out
    tensor_dict = Y(tensor_dict)
    #
    # tensor_dict[Y.name] =  tensor_dict[Y.name][:, tf.newaxis, :]
    tensor_dict[Y.name] = tf.repeat(
        tensor_dict[Y.name][:, tf.newaxis, :], repeats=n_out, axis=1
    )

    p = ProductFunction(
        left=Y,
        right=Y,
        name="AA",
        lmax=lmax,
        Lmax=1,
        keep_parity=[[0, 1], [1, -1], [2, 1], [3, -1]],
    )
    p.build(float64)
    tensor_dict = p(tensor_dict)

    fc = FCRight2Left(left=Y, right=p, n_out=7, name="fc", norm_out=True)
    fc.build(float64)
    fc_res = fc.frwrd(tensor_dict).numpy()

    print(fc_res.shape)
    print(f"{fc.w_left.shape=}")
    print(f"{fc.w_right.shape=}")

    # activate LORA
    lora_config = {"rank": 2, "alpha": 1}
    fc.enable_lora_adaptation(lora_config)
    assert fc.lora
    assert len(fc.w_left_lora_tensors) == 3
    assert len(fc.w_right_lora_tensors) == 3
    assert not fc.w_left.trainable
    assert not fc.w_right.trainable

    fc_res_2 = fc.frwrd(tensor_dict).numpy()
    assert np.allclose(fc_res, fc_res_2)

    # upd LORA tensors
    for t in fc.w_left_lora_tensors:
        t.assign(1e-2 * tf.ones_like(t))

    for t in fc.w_right_lora_tensors:
        t.assign(1e-2 * tf.ones_like(t))

    fc_res_2b = fc.frwrd(tensor_dict).numpy()
    assert not np.allclose(fc_res, fc_res_2b)

    # reduce LORA
    fc.finalize_lora_update()

    assert not fc.lora
    assert not hasattr(fc, "w_left_lora_tensors")
    assert not hasattr(fc, "w_right_lora_tensors")
    assert fc.w_left.trainable
    assert fc.w_right.trainable

    fc_res_3 = fc.frwrd(tensor_dict).numpy()

    assert np.allclose(fc_res_2b, fc_res_3)


def test_LORA_MLPRadialFunction():
    nfunc = 8
    rcut = 6.2
    n_rad_max = 22
    lmax = 2

    fake_bonds = np.random.uniform(0, rcut, size=(100_000, 1))
    data = {"fake_bonds": fake_bonds}
    g_k = RadialBasis(
        name="base",
        bonds="fake_bonds",
        basis_type="RadSinBessel",
        p=5,
        nfunc=nfunc,
        rcut=rcut,
        normalized=True,
    )
    g_k.build(float64)
    data = g_k(data)

    R_nl = MLPRadialFunction(
        n_rad_max=n_rad_max,
        input_shape=nfunc,
        lmax=lmax,
        basis=g_k,
        name="R",
        activation="tanh",
    )
    R_nl.build(float64)

    r1 = R_nl.frwrd(data).numpy()
    print("Before LORA")
    train_var_names_1 = sorted([var.name for var in R_nl.trainable_variables])
    non_train_var_names_1 = sorted([var.name for var in R_nl.non_trainable_variables])
    print(f"{train_var_names_1=}")
    print(f"{non_train_var_names_1=}")

    # activate LORA
    lora_config = {"rank": 2, "alpha": 1}
    R_nl.enable_lora_adaptation(lora_config)

    r2 = R_nl.frwrd(data).numpy()

    print("AFTER LORA")
    train_var_names_2 = sorted([var.name for var in R_nl.trainable_variables])
    non_train_var_names_2 = sorted([var.name for var in R_nl.non_trainable_variables])
    print(f"{train_var_names_2=}")
    print(f"{non_train_var_names_2=}")

    assert np.allclose(r1, r2)
    assert train_var_names_1 != train_var_names_2
    assert non_train_var_names_1 != non_train_var_names_2

    for i in range(R_nl.mlp.nlayers):
        layer = getattr(R_nl.mlp, f"layer{i}")
        for t in layer.lora_tensors:
            t.assign(tf.ones_like(t))

    r2b = R_nl.frwrd(data).numpy()
    assert not np.allclose(r1, r2b)

    R_nl.finalize_lora_update()

    print("AFTER REDUCE LORA")
    r3 = R_nl.frwrd(data).numpy()
    train_var_names_3 = sorted([var.name for var in R_nl.trainable_variables])
    non_train_var_names_3 = sorted([var.name for var in R_nl.non_trainable_variables])
    print(f"{train_var_names_3=}")
    print(f"{non_train_var_names_3=}")

    assert train_var_names_1 == train_var_names_3
    assert non_train_var_names_1 == non_train_var_names_3

    assert np.allclose(r2b, r3)

    R_nl_lora = MLPRadialFunction(
        n_rad_max=n_rad_max,
        input_shape=nfunc,
        lmax=lmax,
        basis=g_k,
        name="R",
        activation="tanh",
        lora_config=lora_config,
    )
    R_nl_lora.build(float64)

    train_var_names_4 = sorted([var.name for var in R_nl_lora.trainable_variables])
    non_train_var_names_4 = sorted(
        [var.name for var in R_nl_lora.non_trainable_variables]
    )
    print(f"{train_var_names_4=}")
    print(f"{non_train_var_names_4=}")

    assert train_var_names_2 == train_var_names_4
    assert non_train_var_names_2 == non_train_var_names_4


def test_LORA_MLPRadialFunction_v2():
    nfunc = 8
    rcut = 6.2
    n_rad_max = 22
    lmax = 2

    fake_bonds = np.random.uniform(0, rcut, size=(100_000, 1))
    data = {"fake_bonds": fake_bonds}
    g_k = RadialBasis(
        name="base",
        bonds="fake_bonds",
        basis_type="RadSinBessel",
        p=5,
        nfunc=nfunc,
        rcut=rcut,
        normalized=True,
    )
    g_k.build(float64)
    data = g_k(data)

    R_nl = MLPRadialFunction_v2(
        n_rad_max=n_rad_max,
        input_shape=nfunc,
        lmax=lmax,
        basis=g_k,
        name="R",
        activation="tanh",
    )
    R_nl.build(float64)

    r1 = R_nl.frwrd(data).numpy()
    print("Before LORA")
    train_var_names_1 = sorted([var.name for var in R_nl.trainable_variables])
    non_train_var_names_1 = sorted([var.name for var in R_nl.non_trainable_variables])
    print(f"{train_var_names_1=}")
    print(f"{non_train_var_names_1=}")

    # activate LORA
    lora_config = {"rank": 2, "alpha": 1}
    R_nl.enable_lora_adaptation(lora_config)

    r2 = R_nl.frwrd(data).numpy()

    print("AFTER LORA")
    train_var_names_2 = sorted([var.name for var in R_nl.trainable_variables])
    non_train_var_names_2 = sorted([var.name for var in R_nl.non_trainable_variables])
    print(f"{train_var_names_2=}")
    print(f"{non_train_var_names_2=}")

    assert np.allclose(r1, r2)
    assert train_var_names_1 != train_var_names_2
    assert non_train_var_names_1 != non_train_var_names_2

    for layer in R_nl.layers:
        for t in layer.lora_tensors:
            t.assign(tf.ones_like(t))

    r2b = R_nl.frwrd(data).numpy()
    assert not np.allclose(r1, r2b)

    R_nl.finalize_lora_update()

    print("AFTER REDUCE LORA")
    r3 = R_nl.frwrd(data).numpy()
    train_var_names_3 = sorted([var.name for var in R_nl.trainable_variables])
    non_train_var_names_3 = sorted([var.name for var in R_nl.non_trainable_variables])
    print(f"{train_var_names_3=}")
    print(f"{non_train_var_names_3=}")

    assert train_var_names_1 == train_var_names_3
    assert non_train_var_names_1 == non_train_var_names_3

    assert np.allclose(r2b, r3)

    R_nl_lora = MLPRadialFunction_v2(
        n_rad_max=n_rad_max,
        input_shape=nfunc,
        lmax=lmax,
        basis=g_k,
        name="R",
        activation="tanh",
        lora_config=lora_config,
    )
    R_nl_lora.build(float64)

    train_var_names_4 = sorted([var.name for var in R_nl_lora.trainable_variables])
    non_train_var_names_4 = sorted(
        [var.name for var in R_nl_lora.non_trainable_variables]
    )
    print(f"{train_var_names_4=}")
    print(f"{non_train_var_names_4=}")

    assert train_var_names_2 == train_var_names_4
    assert non_train_var_names_2 == non_train_var_names_4


def test_initialize_lora_tensors_keep_dims():
    w_shape = [89, 13, 32, 5]
    w = tf.Variable(tf.zeros(w_shape, dtype=tf.float64))
    lora_config = dict(rank=3, alpha=1, keep_dims=1)

    lora_tensors = initialize_lora_tensors(w, lora_config=lora_config, name="w")
    for t in lora_tensors:
        print(t.name, t.shape)
    assert len(lora_tensors) == 3
    assert lora_tensors[0].shape.as_list() == [89, 13, 3]
    assert lora_tensors[1].shape.as_list() == [89, 32, 3]
    assert lora_tensors[2].shape.as_list() == [89, 5, 3]

    delta_w = lora_reconstruction(*lora_tensors, lora_config=lora_config)

    print(f"{delta_w.shape.as_list()=}")
    assert delta_w.shape.as_list() == w_shape

    apply_lora_update(w, *lora_tensors, lora_config=lora_config)
    assert w._trainable


def test_initialize_lora_tensors_additive():
    w_shape = [89, 13, 32, 5]
    w = tf.Variable(tf.zeros(w_shape, dtype=tf.float64))
    lora_config = dict(mode="full_additive")

    lora_tensors = initialize_lora_tensors(w, lora_config=lora_config, name="w")
    assert not w._trainable
    for t in lora_tensors:
        print(t.name, t.shape)
        assert t._trainable
    assert len(lora_tensors) == 1
    assert lora_tensors[0].shape.as_list() == w_shape

    delta_w = lora_reconstruction(*lora_tensors, lora_config=lora_config)

    print(f"{delta_w.shape.as_list()=}")
    assert delta_w.shape.as_list() == w_shape

    apply_lora_update(w, *lora_tensors, lora_config=lora_config)
    assert w._trainable


@pytest.mark.parametrize(
    "full_par",
    [
        True,
        False,
    ],
)
def test_rot_invar_2l(full_par):
    from tensorpotential.utils import Parity

    lmax_bond = 5
    n_rad_base = 8
    n_rad_max_bond = 5
    # n_rad_max_out = 6
    cutoff_rad = 6.0

    np.random.seed(322)
    tf.random.set_seed(322)

    axis = np.array([1, 1, 3])
    theta = np.pi / 2.2
    axis = axis / np.linalg.norm(axis)  # normalize the rotation vector first
    rot = Rotation.from_rotvec(theta * axis)

    coord = np.array([[0.2, 0.3, 2.1], [0.1, 3.2, 0.5], [0.3, 1, 2]])
    coord_r = rot.apply(coord)
    coord_2 = np.vstack([coord, coord_r]).astype(np.float64)
    print(coord_2)
    # stage 1: input vectors:

    indj = [1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4]
    indi = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    atomic_mu_i = [0, 1, 0, 0, 1, 0]
    # muj = [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    muj = np.take(atomic_mu_i, indj, axis=0)
    rj = np.take(coord_2, indj, axis=0)
    ri = np.take(coord_2, indi, axis=0)
    rij = rj - ri
    inpt_dict = {
        constants.BOND_VECTOR: rij,
        constants.BOND_IND_J: indj,
        constants.BOND_MU_J: muj,
        constants.BOND_IND_I: indi,
        constants.N_ATOMS_BATCH_TOTAL: len(coord_2),
        constants.ATOMIC_MU_I: atomic_mu_i,
    }
    ###########################################################################################
    with InstructionManager() as instructor:
        d_ij = BondLength()
        # d_ij.build(float64)
        # inpt_dict = d_ij(inpt_dict)
        rhat = ScaledBondVector(bond_length=d_ij)
        # rhat.build(float64)
        # inpt_dict = rhat(inpt_dict)
        g_k = RadialBasis(
            bonds=d_ij,
            basis_type="SBessel",
            nfunc=n_rad_base,
            rcut=cutoff_rad,
            p=5,
        )
        # g_k.build(float64)
        # inpt_dict = g_k(inpt_dict)

        R_nl = MLPRadialFunction(
            n_rad_max=n_rad_max_bond, lmax=lmax_bond, basis=g_k, name="R"
        )
        # R_nl.build(float64)
        # inpt_dict = R_nl(inpt_dict)

        Y = SphericalHarmonic(vhat=rhat, lmax=lmax_bond, name="Y")
        # Y.build(float64)
        # inpt_dict = Y(inpt_dict)

        z = ScalarChemicalEmbedding(
            element_map={"H": 0, "C": 1},
            embedding_size=54,
            name="Z",
        )
        # z.build(float64)
        # inpt_dict = z(inpt_dict)

        A = SingleParticleBasisFunctionScalarInd(
            radial=R_nl,
            angular=Y,
            indicator=z,
            name="A",
        )
        # A.build(float64)
        # A.lin_transform.build(float64)
        # inpt_dict = A(inpt_dict)

        AA = ProductFunction(
            left=A,
            right=A,
            name="AA",
            lmax=lmax_bond,
            Lmax=lmax_bond,
            normalize=True,
        )
        # AA.build(float64)
        # inpt_dict = AA(inpt_dict)

        AAA = ProductFunction(
            left=AA,
            right=A,
            name="AAA",
            lmax=lmax_bond,
            Lmax=lmax_bond,
            normalize=True,
        )
        # AAA.build(float64)
        # inpt_dict = AAA(inpt_dict)
        instr_reduce_n = FunctionReduceN(
            instructions=[A, AA, AAA],
            name="E",
            ls_max=[lmax_bond, lmax_bond, lmax_bond],
            n_out=n_rad_max_bond,
            is_central_atom_type_dependent=True,
            number_of_atom_types=2,
            allowed_l_p=Parity.REAL_PARITY,
        )
        # I_l.build(float64)
        # inpt_dict = I_l(inpt_dict)
        R1_nl = MLPRadialFunction(
            n_rad_max=n_rad_max_bond,
            lmax=lmax_bond,
            basis=g_k,
            name="R1",
            hidden_layers=[64, 64],
            activation="tanh",
        )
        if full_par:
            p_sec_lay = Parity.FULL_PARITY
        else:
            p_sec_lay = Parity.REAL_PARITY
        YI = SingleParticleBasisFunctionEquivariantInd(
            radial=R1_nl,
            angular=Y,
            indicator=instr_reduce_n,
            name="YI",
            lmax=lmax_bond,
            Lmax=lmax_bond,
            avg_n_neigh=1.0,
            keep_parity=p_sec_lay,
            normalize=True,
        )

        B = FunctionReduceN(
            instructions=[YI, instr_reduce_n],
            name="B",
            ls_max=lmax_bond,
            out_norm=False,
            n_out=n_rad_max_bond,
            is_central_atom_type_dependent=False,
            allowed_l_p=p_sec_lay,
        )
        B1 = FCRight2Left(
            left=B,
            right=B,
            name="B1",
            n_out=n_rad_max_bond,
            is_central_atom_type_dependent=False,
            norm_out=True,
        )
        BB = ProductFunction(
            left=B1,
            right=B1,
            name="BB",
            lmax=lmax_bond,
            Lmax=lmax_bond,
            keep_parity=p_sec_lay,
            normalize=True,
        )

        BB1 = FCRight2Left(
            left=BB,
            right=B,
            name="BB1",
            n_out=n_rad_max_bond,
            is_central_atom_type_dependent=False,
            norm_out=True,
        )
        BBB = ProductFunction(
            left=BB1,
            right=B,
            name="BBB",
            lmax=0,
            Lmax=0,
            keep_parity=[[0, 1]],
            normalize=True,
        )
        I2 = FunctionReduceN(
            instructions=[B, BB, BBB],
            name="E2",
            ls_max=[0, 0, 0],
            # n_in=n_rad_max_bond,
            n_out=n_rad_max_bond,
            is_central_atom_type_dependent=True,
            number_of_atom_types=2,
            allowed_l_p=Parity.REAL_PARITY,
        )
    ins = instructor.get_instructions()
    for i_n, i in ins.items():
        i.build(tf.float64)
        inpt_dict = i(inpt_dict)

    out = inpt_dict[I2.name]
    print(out)
    print(out[:3] - out[3:])
    assert np.allclose(out[:3], out[3:])


def test_linmlp_normalize_layer():
    """Test LinMLPOut2ScalarTarget with normalize='layer' vs normalize=None."""
    n_atoms = 10
    n_out = 8  # total features per atom (1 linear + 7 MLP input)

    # Mock origin object with required attributes
    class MockOrigin:
        def __init__(self, name, n_out):
            self.name = name
            self.n_out = n_out
            self.lmax = 0

    origin = MockOrigin("mock_reduce", n_out)
    target = CreateOutputTarget(name="energy", initial_value=0.0, l=0)
    target.build(float64)

    # Build LinMLPOut2ScalarTarget with normalize="layer"
    linmlp_norm = LinMLPOut2ScalarTarget(
        origin=[origin],
        target=target,
        hidden_layers=[32],
        normalize="layer",
        name="linmlp_norm",
    )
    linmlp_norm.build(float64)

    # Build LinMLPOut2ScalarTarget without normalize
    linmlp_no_norm = LinMLPOut2ScalarTarget(
        origin=[origin],
        target=target,
        hidden_layers=[32],
        normalize=None,
        name="linmlp_no_norm",
    )
    linmlp_no_norm.build(float64)

    # Create mock input data
    np.random.seed(42)
    origin_data = np.random.randn(n_atoms, n_out, 1).astype(np.float64)
    input_data = {
        "energy": target.frwrd({}),
        "mock_reduce": tf.constant(origin_data),
        constants.N_ATOMS_BATCH_REAL: tf.constant(n_atoms, dtype=tf.int32),
        constants.N_ATOMS_BATCH_TOTAL: tf.constant(n_atoms, dtype=tf.int32),
    }

    # Verify scale variable exists and is trainable
    assert hasattr(
        linmlp_norm, "scale"
    ), "scale variable should exist when normalize='layer'"
    assert linmlp_norm.scale.trainable, "scale should be trainable"
    assert linmlp_norm.scale.shape == [1, n_out - 1]

    # Forward pass with normalization
    out_norm = linmlp_norm.frwrd(input_data)
    # Forward pass without normalization
    out_no_norm = linmlp_no_norm.frwrd(input_data)

    # Outputs should differ (normalization changes the MLP input)
    assert not np.allclose(
        out_norm.numpy(), out_no_norm.numpy()
    ), "Normalized and non-normalized outputs should differ"

    # Verify gradients exist for scale
    with tf.GradientTape() as tape:
        out = linmlp_norm.frwrd(input_data)
        loss = tf.reduce_sum(out)
    grad = tape.gradient(loss, linmlp_norm.scale)
    assert grad is not None, "Gradient for scale should exist"
    assert not np.allclose(grad.numpy(), 0.0), "Gradient for scale should be non-zero"


# ============================================================================
# LinMLPScalarReadOut tests
# ============================================================================


class _MockOrigin:
    def __init__(self, name, n_out):
        self.name = name
        self.n_out = n_out
        self.lmax = 0


def _make_readout_input_data(origins, n_atoms=10, seed=42):
    """Build mock input_data dict for LinMLPScalarReadOut tests."""
    np.random.seed(seed)
    target = CreateOutputTarget(name="energy", initial_value=0.0, l=0)
    target.build(float64)
    data = {
        "energy": target.frwrd({}),
        constants.N_ATOMS_BATCH_REAL: tf.constant(n_atoms, dtype=tf.int32),
        constants.N_ATOMS_BATCH_TOTAL: tf.constant(n_atoms, dtype=tf.int32),
    }
    for o in origins:
        data[o.name] = tf.constant(
            np.random.randn(n_atoms, o.n_out, 1).astype(np.float64)
        )
    return target, data


def test_linmlp_scalar_readout_init_state():
    """At init: alpha=[1, 0, ...], scales=0 ⇒ out == target + lin_0."""
    n_out = 8
    origins = [_MockOrigin("r0", n_out), _MockOrigin("r1", n_out)]
    target, data = _make_readout_input_data(origins)

    readout = LinMLPScalarReadOut(
        origin=origins, target=target, hidden_layers=[16], name="ro_init",
    )
    readout.build(float64)

    assert len(readout.alpha) == 2
    np.testing.assert_allclose(readout.alpha[0].numpy(), 1.0)
    np.testing.assert_allclose(readout.alpha[1].numpy(), 0.0)
    assert len(readout.scales) == 2
    for s in readout.scales:
        np.testing.assert_allclose(s.numpy(), 0.0)

    out = readout.frwrd(data).numpy()
    expected = data[origins[0].name].numpy()[:, 0, 0].reshape(-1, 1)
    np.testing.assert_allclose(out, expected, atol=1e-12)

    # Gradients flow through alpha, scales, and MLP layer weights.
    with tf.GradientTape(persistent=True) as tape:
        loss = tf.reduce_sum(readout.frwrd(data))
    assert tape.gradient(loss, readout.alpha[0]) is not None
    assert tape.gradient(loss, readout.scales[0]) is not None
    assert tape.gradient(loss, readout.mlp_layers[0][0].w) is not None


def test_linmlp_scalar_readout_per_input_different_shapes():
    """mlp_mode='per_input' supports origins with different n_out."""
    o1 = _MockOrigin("r0", 6)
    o2 = _MockOrigin("r1", 10)
    target, data = _make_readout_input_data([o1, o2])

    readout = LinMLPScalarReadOut(
        origin=[o1, o2], target=target, hidden_layers=[16], name="ro_per",
    )
    readout.build(float64)

    assert len(readout.mlp_layers) == 2
    assert readout.mlp_layers[0][0].n_in == 5  # 6 - 1
    assert readout.mlp_layers[1][0].n_in == 9  # 10 - 1
    assert readout.mlp_layers[0][-1].n_out == 1
    assert readout.mlp_layers[1][-1].n_out == 1

    out = readout.frwrd(data)
    assert out.shape == (10, 1)


def test_linmlp_scalar_readout_shared():
    """mlp_mode='shared': single MLP after summing non-linear parts."""
    n_out = 8
    origins = [_MockOrigin("r0", n_out), _MockOrigin("r1", n_out)]
    target, data = _make_readout_input_data(origins)

    readout = LinMLPScalarReadOut(
        origin=origins, target=target, hidden_layers=[16],
        mlp_mode="shared", name="ro_shared",
    )
    readout.build(float64)

    assert len(readout.mlp_layers) == 1
    assert readout.mlp_layers[0][0].n_in == n_out - 1

    out = readout.frwrd(data).numpy()
    expected = data[origins[0].name].numpy()[:, 0, 0].reshape(-1, 1)
    np.testing.assert_allclose(out, expected, atol=1e-12)
    assert out.shape == (10, 1)

    # Mismatched shapes must raise.
    with pytest.raises(AssertionError):
        LinMLPScalarReadOut(
            origin=[_MockOrigin("a", 6), _MockOrigin("b", 8)],
            target=target, hidden_layers=[16], mlp_mode="shared",
            name="ro_shared_bad",
        )


def test_linmlp_scalar_readout_activation_str_and_list():
    """activation accepts str or per-hidden-layer list."""
    n_out = 8
    origins = [_MockOrigin("r0", n_out)]
    target, data = _make_readout_input_data(origins)

    readout_str = LinMLPScalarReadOut(
        origin=origins, target=target, hidden_layers=[16, 8],
        activation="silu", name="ro_act_str",
    )
    readout_str.build(float64)
    assert readout_str.activation == ["silu", "silu"]
    assert readout_str.frwrd(data).shape == (10, 1)

    readout_list = LinMLPScalarReadOut(
        origin=origins, target=target, hidden_layers=[16, 8],
        activation=["tanh", "sigmoid"], name="ro_act_list",
    )
    readout_list.build(float64)
    assert readout_list.activation == ["tanh", "sigmoid"]
    assert readout_list.frwrd(data).shape == (10, 1)

    # Mismatched list length raises.
    with pytest.raises(AssertionError):
        LinMLPScalarReadOut(
            origin=origins, target=target, hidden_layers=[16, 8],
            activation=["tanh"], name="ro_act_bad",
        )


def test_linmlp_scalar_readout_element_dependent():
    """element_dependent=True uses ElementDependentLinear with per-element weights."""
    from tensorpotential.functions.nn import ElementDependentLinear

    n_atoms = 10
    n_types = 3
    n_out = 8
    origins = [_MockOrigin("r0", n_out), _MockOrigin("r1", n_out)]
    target, data = _make_readout_input_data(origins, n_atoms=n_atoms)
    mu_i = tf.constant(
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=tf.int32
    )
    data[constants.ATOMIC_MU_I] = mu_i
    data[constants.ATOMIC_MU_I_LOCAL] = mu_i

    readout = LinMLPScalarReadOut(
        origin=origins, target=target, hidden_layers=[16],
        element_dependent=True, number_of_atom_types=n_types,
        name="ro_ed",
    )
    readout.build(float64)

    for layer_list in readout.mlp_layers:
        for layer in layer_list:
            assert isinstance(layer, ElementDependentLinear)
            assert layer.w.shape == (n_types, layer.n_in, layer.n_out)

    expected = data[origins[0].name].numpy()[:, 0, 0].reshape(-1, 1)
    out_global = readout.frwrd(data, local=False).numpy()
    out_local = readout.frwrd(data, local=True).numpy()
    assert out_global.shape == (n_atoms, 1)
    assert out_local.shape == (n_atoms, 1)
    np.testing.assert_allclose(out_global, expected, atol=1e-12)
    np.testing.assert_allclose(out_local, expected, atol=1e-12)

    # Missing number_of_atom_types must raise.
    with pytest.raises(AssertionError):
        LinMLPScalarReadOut(
            origin=origins, target=target, hidden_layers=[16],
            element_dependent=True, name="ro_ed_bad",
        )


def test_linmlp_scalar_readout_lm_first():
    """Origins with lm_first=True ([lm, atoms, n_out]) give identical output
    to the lm-last layout ([atoms, n_out, lm])."""
    n_atoms = 10
    n_out = 8
    origins = [_MockOrigin("r0", n_out), _MockOrigin("r1", n_out)]
    target, data = _make_readout_input_data(origins, n_atoms=n_atoms)

    readout = LinMLPScalarReadOut(
        origin=origins, target=target, hidden_layers=[16], name="ro_lmf",
    )
    readout.build(float64)
    # Non-trivial alpha/scales so both the linear and MLP paths are exercised.
    for i in range(len(origins)):
        readout.alpha[i].assign(tf.constant(0.5 + i, dtype=float64))
        readout.scales[i].assign(
            tf.constant(
                np.random.randn(1, n_out - 1), dtype=float64
            )
        )

    out_last = readout.frwrd(data).numpy()

    # Same values in lm-first layout: [atoms, n_out, lm] -> [lm, atoms, n_out]
    for o in origins:
        data[o.name] = tf.transpose(data[o.name], [2, 0, 1])
        o.lm_first = True

    out_first = readout.frwrd(data).numpy()
    assert out_first.shape == (n_atoms, 1)
    np.testing.assert_allclose(out_first, out_last, atol=1e-12)


# ============================================================================
# GeneralProductFunction tests
# ============================================================================


def _build_general_product_test_data(dtype=float64):
    """Helper: build A-basis from a rotated pair of 3-atom clusters."""
    from scipy.spatial.transform import Rotation

    lmax_bond = 4
    n_rad_base = 8
    n_rad_max = 8
    cutoff_rad = 6.0

    np.random.seed(322)
    tf.random.set_seed(322)

    axis = np.array([1, 1, 3])
    theta = np.pi / 2.2
    axis = axis / np.linalg.norm(axis)
    rot = Rotation.from_rotvec(theta * axis)

    coord = np.array([[0.2, 0.3, 2.1], [0.1, 3.2, 0.5], [0.3, 1, 2]])
    coord_r = rot.apply(coord)
    coord_2 = np.vstack([coord, coord_r]).astype(np.float64)

    indj = [1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4]
    indi = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    atomic_mu_i = [0, 1, 0, 0, 1, 0]
    muj = np.take(atomic_mu_i, indj, axis=0)
    rj = np.take(coord_2, indj, axis=0)
    ri = np.take(coord_2, indi, axis=0)
    rij = rj - ri

    inpt_dict = {
        constants.BOND_VECTOR: rij,
        constants.BOND_IND_J: indj,
        constants.BOND_MU_J: muj,
        constants.BOND_IND_I: indi,
        constants.N_ATOMS_BATCH_TOTAL: len(coord_2),
        constants.ATOMIC_MU_I: atomic_mu_i,
    }

    d_ij = BondLength()
    d_ij.build(dtype)
    inpt_dict = d_ij(inpt_dict)

    rhat = ScaledBondVector(bond_length=d_ij)
    rhat.build(dtype)
    inpt_dict = rhat(inpt_dict)

    g_k = RadialBasis(
        bonds=d_ij, basis_type="SBessel", nfunc=n_rad_base, rcut=cutoff_rad, p=5
    )
    g_k.build(dtype)
    inpt_dict = g_k(inpt_dict)

    R_nl = MLPRadialFunction(n_rad_max=n_rad_max, lmax=lmax_bond, basis=g_k, name="R")
    R_nl.build(dtype)
    inpt_dict = R_nl(inpt_dict)

    Y = SphericalHarmonic(vhat=rhat, lmax=lmax_bond, name="Y")
    Y.build(dtype)
    inpt_dict = Y(inpt_dict)

    z = ScalarChemicalEmbedding(
        element_map={"H": 0, "C": 1}, embedding_size=32, name="Z"
    )
    z.build(dtype)
    inpt_dict = z(inpt_dict)

    A = SingleParticleBasisFunctionScalarInd(
        radial=R_nl, angular=Y, indicator=z, name="A", avg_n_neigh=10.0
    )
    A.build(dtype)
    A.lin_transform.build(dtype)
    inpt_dict = A(inpt_dict)

    return inpt_dict, A, lmax_bond, n_rad_max


@pytest.mark.parametrize("mode", ["elementwise", "cp", "cp_l", "cp_lL"])
@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_general_product_function_rot_invar(mode, param_dtype):
    """Test that GeneralProductFunction preserves rotational invariance
    when contracted to scalars via FunctionReduce, for all 4 modes."""
    from tensorpotential.utils import Parity

    inpt_dict, A, lmax_bond, n_rad_max = _build_general_product_test_data(
        dtype=param_dtype
    )

    if mode == "elementwise":
        AA = GeneralProductFunction(
            left=A,
            right=A,
            name="AA",
            lmax=lmax_bond,
            Lmax=lmax_bond,
            mode="elementwise",
            normalize=True,
        )
    else:
        AA = GeneralProductFunction(
            left=A,
            right=A,
            name="AA",
            lmax=lmax_bond,
            Lmax=lmax_bond,
            mode=mode,
            rank=6,
            n_out=n_rad_max,
            use_S=True,
            normalize=True,
        )
    AA.build(param_dtype)
    inpt_dict = AA(inpt_dict)

    AAA = GeneralProductFunction(
        left=AA,
        right=A,
        name="AAA",
        lmax=lmax_bond,
        Lmax=0,
        mode="elementwise",
        normalize=True,
    )
    AAA.build(param_dtype)
    inpt_dict = AAA(inpt_dict)

    I_l = FunctionReduce(
        instructions=[A, AA, AAA],
        name="E",
        ls_max=[0, 0, 0],
        n_in=n_rad_max,
        n_out=1,
        is_central_atom_type_dependent=True,
        number_of_atom_types=2,
        allowed_l_p=[[0, 1]],
    )
    I_l.build(param_dtype)
    inpt_dict = I_l(inpt_dict)

    out = inpt_dict[I_l.name]
    # First 3 atoms and last 3 atoms are related by rotation
    atol = 1e-10
    assert np.allclose(out[:3], out[3:], atol=atol), (
        f"mode={mode}: rotational invariance broken, "
        f"max diff={np.max(np.abs(out[:3] - out[3:]))}"
    )


@pytest.mark.parametrize("mode", ["cp", "cp_l", "cp_lL"])
@pytest.mark.parametrize("use_S", [True, False])
def test_general_product_function_shapes(mode, use_S):
    """Test output shapes and trainable variable counts for all CP modes."""
    from tensorpotential.utils import Parity

    size = 10
    lmax = 2
    n_rad = 5
    rank = 4
    n_out = 7 if use_S else rank

    np.random.seed(42)
    tf.random.set_seed(42)

    fake_projections = np.random.normal(0, 1, size=(size, 3))
    fake_projections = fake_projections / np.linalg.norm(
        fake_projections, axis=1, keepdims=True
    )

    Y = SphericalHarmonic(vhat="fake_projections", lmax=lmax, name="Y")
    Y.build(float64)
    sg = Y.frwrd({"fake_projections": fake_projections}).numpy()
    Y.n_out = n_rad
    inpt_data = {"Y": np.repeat(sg[:, np.newaxis, :], n_rad, axis=1)}

    AA = GeneralProductFunction(
        left=Y,
        right=Y,
        name="AA",
        lmax=lmax,
        Lmax=lmax,
        mode=mode,
        rank=rank,
        n_out=n_out,
        use_S=use_S,
        is_left_right_equal=True,
        keep_parity=Parity.REAL_PARITY,
        normalize=True,
    )
    AA.build(float64)
    result = AA.frwrd(inpt_data)

    n_LM = AA.coupling_meta_data.shape[0]
    assert result.shape == (
        size,
        n_out,
        n_LM,
    ), f"Expected shape ({size}, {n_out}, {n_LM}), got {result.shape}"

    # Check trainable variables exist
    trainable_names = [v.name for v in AA.trainable_variables]
    assert any(
        "U_" in n for n in trainable_names
    ), f"U variable missing: {trainable_names}"
    assert any(
        "V_" in n for n in trainable_names
    ), f"V variable missing: {trainable_names}"
    if use_S:
        assert any(
            "S_" in n for n in trainable_names
        ), f"S variable missing: {trainable_names}"
    else:
        assert not any(
            "S_" in n for n in trainable_names
        ), f"S variable should not exist: {trainable_names}"


def test_general_product_function_elementwise_matches_product_function():
    """Verify that GeneralProductFunction(mode='elementwise') produces
    the same output as ProductFunction."""
    from tensorpotential.utils import Parity

    inpt_dict, A, lmax_bond, n_rad_max = _build_general_product_test_data()

    # Original ProductFunction
    AA_orig = ProductFunction(
        left=A,
        right=A,
        name="AA_orig",
        lmax=lmax_bond,
        Lmax=lmax_bond,
        normalize=True,
    )
    AA_orig.build(float64)
    inpt_dict = AA_orig(inpt_dict)

    # GeneralProductFunction elementwise
    AA_gen = GeneralProductFunction(
        left=A,
        right=A,
        name="AA_gen",
        lmax=lmax_bond,
        Lmax=lmax_bond,
        mode="elementwise",
        normalize=True,
    )
    AA_gen.build(float64)
    inpt_dict = AA_gen(inpt_dict)

    assert np.allclose(
        inpt_dict["AA_orig"].numpy(), inpt_dict["AA_gen"].numpy()
    ), "elementwise mode should match ProductFunction exactly"


@pytest.mark.parametrize("mode", ["cp", "cp_l", "cp_lL"])
def test_general_product_function_gradients(mode):
    """Test that gradients flow through all trainable variables."""
    size = 5
    lmax = 2
    n_rad = 4
    rank = 3
    n_out = 4

    np.random.seed(42)
    tf.random.set_seed(42)

    fake_projections = np.random.normal(0, 1, size=(size, 3))
    fake_projections = fake_projections / np.linalg.norm(
        fake_projections, axis=1, keepdims=True
    )

    Y = SphericalHarmonic(vhat="fake_projections", lmax=lmax, name="Y")
    Y.build(float64)
    sg = Y.frwrd({"fake_projections": fake_projections}).numpy()
    Y.n_out = n_rad
    left_data = tf.constant(np.repeat(sg[:, np.newaxis, :], n_rad, axis=1))

    AA = GeneralProductFunction(
        left=Y,
        right=Y,
        name="AA",
        lmax=lmax,
        Lmax=0,
        mode=mode,
        rank=rank,
        n_out=n_out,
        use_S=True,
        is_left_right_equal=True,
        normalize=True,
    )
    AA.build(float64)

    with tf.GradientTape() as tape:
        result = AA.frwrd({"Y": left_data})
        loss = tf.reduce_sum(result)

    grads = tape.gradient(loss, AA.trainable_variables)
    for var, grad in zip(AA.trainable_variables, grads):
        assert grad is not None, f"No gradient for {var.name}"
        # tf.gather can produce IndexedSlices instead of dense tensors
        if isinstance(grad, tf.IndexedSlices):
            grad_dense = tf.convert_to_tensor(grad)
        else:
            grad_dense = grad
        assert not np.allclose(grad_dense.numpy(), 0.0), f"Zero gradient for {var.name}"


def test_general_product_function_compatible_with_reduce_n():
    """Test full pipeline: GeneralProductFunction -> FunctionReduceN."""
    from tensorpotential.utils import Parity

    inpt_dict, A, lmax_bond, n_rad_max = _build_general_product_test_data()
    rank = 6

    AA = GeneralProductFunction(
        left=A,
        right=A,
        name="AA",
        lmax=lmax_bond,
        Lmax=lmax_bond,
        mode="cp",
        rank=rank,
        n_out=n_rad_max,
        normalize=True,
    )
    AA.build(float64)
    inpt_dict = AA(inpt_dict)

    rho = FunctionReduceN(
        instructions=[A, AA],
        name="rho",
        ls_max=[0, 0],
        n_out=5,
        is_central_atom_type_dependent=True,
        number_of_atom_types=2,
        allowed_l_p=Parity.SCALAR,
    )
    rho.build(float64)
    inpt_dict = rho(inpt_dict)

    out = inpt_dict[rho.name]
    assert out.shape == (6, 5, 1)
    assert not np.any(np.isnan(out.numpy())), "NaN in output"


@pytest.mark.parametrize("mode", ["cp", "cp_l", "cp_lL"])
def test_general_product_function_group_counts(mode):
    """Verify the group assignment dimensions are sensible."""
    from tensorpotential.utils import Parity

    size = 5
    lmax = 3
    n_rad = 4

    np.random.seed(42)
    fake_projections = np.random.normal(0, 1, size=(size, 3))
    fake_projections = fake_projections / np.linalg.norm(
        fake_projections, axis=1, keepdims=True
    )

    Y = SphericalHarmonic(vhat="fake_projections", lmax=lmax, name="Y")
    Y.build(float64)
    Y.n_out = n_rad

    AA = GeneralProductFunction(
        left=Y,
        right=Y,
        name="AA",
        lmax=lmax,
        Lmax=lmax,
        mode=mode,
        rank=4,
        n_out=4,
        is_left_right_equal=True,
        keep_parity=Parity.REAL_PARITY,
        normalize=True,
    )

    if mode == "cp":
        # cp mode has no group assignments
        assert not hasattr(AA, "group_left")
        assert not hasattr(AA, "cg_u_group")
    elif mode == "cp_l":
        # Number of groups should be <= number of (l, hist, parity) groups in input
        n_groups_left = AA.n_groups_left
        n_groups_right = AA.n_groups_right
        # For uncoupled Y with lmax=3, there are 4 groups (l=0..3)
        assert n_groups_left == lmax + 1
        assert n_groups_right == lmax + 1
        # group_left should have length = number of angular channels in Y
        assert len(AA.group_left.numpy()) == len(Y.coupling_meta_data)
    elif mode == "cp_lL":
        # Groups should incorporate output L, so more groups than cp_l
        assert AA.n_groups_u > 0
        assert AA.n_groups_v > 0
        # cg_u_group length should equal total CG pairs
        n_cg = len(AA.left_ind.numpy())
        assert len(AA.cg_u_group.numpy()) == n_cg
        assert len(AA.cg_v_group.numpy()) == n_cg


def _compute_wigner_d_real(rot, lmax):
    """Compute real Wigner D-matrices D^L(R) for L=0..lmax from spherical harmonics.

    Uses Y_L(Rv) = D^L(R) @ Y_L(v) with a set of random unit vectors.
    """
    rng = np.random.RandomState(12345)
    n_vecs = max(2 * lmax + 1, 5)
    vecs = rng.normal(size=(n_vecs, 3))
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    rvecs = rot.apply(vecs)

    Y = SphericalHarmonic(vhat="v", lmax=lmax, name="_wigner_Y")
    Y.build(float64)
    y_orig = Y.frwrd({"v": vecs}).numpy()  # [n_vecs, (lmax+1)^2]
    y_rot = Y.frwrd({"v": rvecs}).numpy()

    D = {}
    for L in range(lmax + 1):
        start = L * L
        end = (L + 1) * (L + 1)
        Y_orig = y_orig[:, start:end].T  # [2L+1, n_vecs]
        Y_rot = y_rot[:, start:end].T
        # Y_rot = D @ Y_orig => D = Y_rot @ pinv(Y_orig)
        D[L] = Y_rot @ np.linalg.pinv(Y_orig)
    return D


@pytest.mark.parametrize("mode", ["elementwise", "cp", "cp_l", "cp_lL"])
@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_general_product_function_rot_equivar(mode, param_dtype):
    """Test that GeneralProductFunction with Lmax>0 preserves rotational equivariance.

    For each (L, hist, parity) output group, the (2L+1) m-components must
    transform under the Wigner D-matrix: out(Rx) = D^L(R) @ out(x).
    """
    inpt_dict, A, lmax_bond, n_rad_max = _build_general_product_test_data(
        dtype=param_dtype
    )

    # Recreate the same rotation used in _build_general_product_test_data
    axis = np.array([1, 1, 3])
    theta = np.pi / 2.2
    axis = axis / np.linalg.norm(axis)
    rot = Rotation.from_rotvec(theta * axis)

    Lmax_out = lmax_bond

    if mode == "elementwise":
        AA = GeneralProductFunction(
            left=A,
            right=A,
            name="AA",
            lmax=lmax_bond,
            Lmax=Lmax_out,
            mode="elementwise",
            normalize=True,
        )
    else:
        AA = GeneralProductFunction(
            left=A,
            right=A,
            name="AA",
            lmax=lmax_bond,
            Lmax=Lmax_out,
            mode=mode,
            rank=6,
            n_out=n_rad_max,
            use_S=True,
            normalize=True,
        )
    AA.build(param_dtype)
    inpt_dict = AA(inpt_dict)

    out = inpt_dict[AA.name].numpy()  # [6, n_out, n_LM]
    out_orig = out[:3]  # atoms 0,1,2 = original
    out_rot = out[3:]  # atoms 3,4,5 = rotated by R

    # Compute Wigner D-matrices for the rotation
    D = _compute_wigner_d_real(rot, Lmax_out)

    # Check equivariance for each (L, hist, parity) group
    cmd = AA.coupling_meta_data
    groups = cmd.groupby(["l", "hist", "parity"]).indices
    for (L, hist, parity), indices in groups.items():
        # indices into the n_LM dimension of the output
        # Ensure they are sorted by m for correct D-matrix application
        sub = cmd.iloc[indices].sort_values("m")
        m_indices = sub.index.values
        ms = sub["m"].values
        assert list(ms) == list(
            range(-L, L + 1)
        ), f"Expected m = {list(range(-L, L + 1))}, got {list(ms)}"

        atol = 1e-12 if param_dtype == tf.float64 else 1e-9
        D_L = D[L]  # [2L+1, 2L+1]
        for atom in range(3):
            for n in range(out_orig.shape[1]):
                v_orig = out_orig[atom, n, m_indices]  # [2L+1]
                v_rot = out_rot[atom, n, m_indices]  # [2L+1]
                v_expected = D_L @ v_orig
                assert np.allclose(v_rot, v_expected, atol=atol), (
                    f"mode={mode}, L={L}, hist={hist}, p={parity}, "
                    f"atom={atom}, n={n}: equivariance broken, "
                    f"max diff={np.max(np.abs(v_rot - v_expected))}"
                )


# ============================================================================
# EquivariantRMSNorm tests
# ============================================================================


def _build_equivariant_rms_norm_test_data(param_dtype):
    """Helper: build AA equivariant data for EquivariantRMSNorm tests."""
    inpt_dict, A, lmax_bond, n_rad_max = _build_general_product_test_data(
        dtype=param_dtype
    )
    # Add N_ATOMS_BATCH_REAL (all 6 atoms are real, no padding)
    inpt_dict[constants.N_ATOMS_BATCH_REAL] = tf.constant(6, dtype=tf.int32)

    AA = GeneralProductFunction(
        left=A,
        right=A,
        name="AA",
        lmax=lmax_bond,
        Lmax=lmax_bond,
        mode="cp_l",
        rank=n_rad_max,
        use_S=False,
        normalize=True,
    )
    AA.build(param_dtype)
    inpt_dict = AA(inpt_dict)
    return inpt_dict, AA, lmax_bond, n_rad_max


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("init", ["zeros", "ones"])
def test_equivariant_rms_norm_shape(param_dtype, init):
    """Output shape matches input shape."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_rms_norm_test_data(
        param_dtype
    )

    norm = EquivariantRMSNorm(input=AA, name="AA_norm", init=init)
    norm.build(param_dtype)
    inpt_dict = norm(inpt_dict)

    out = inpt_dict[norm.name]
    aa_out = inpt_dict[AA.name]
    assert out.shape == aa_out.shape, f"Shape mismatch: {out.shape} vs {aa_out.shape}"


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_equivariant_rms_norm_variance(param_dtype):
    """With init='ones', output should have bounded variance."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_rms_norm_test_data(
        param_dtype
    )

    norm = EquivariantRMSNorm(input=AA, name="AA_norm", init="ones")
    norm.build(param_dtype)
    inpt_dict = norm(inpt_dict)

    out = inpt_dict[norm.name].numpy()
    # Degree-balanced RMS of normalized output should be approximately 1
    # (since affine scale is all ones and rms normalizes to unit variance)
    out_var = np.mean(out**2)
    assert out_var < 10.0, f"Output variance {out_var} too large (expected ~1)"
    assert out_var > 1e-6, f"Output variance {out_var} too small"


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_equivariant_rms_norm_gradient(param_dtype):
    """Gradients flow to both input and affine_weight."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_rms_norm_test_data(
        param_dtype
    )

    norm = EquivariantRMSNorm(input=AA, name="AA_norm", init="ones")
    norm.build(param_dtype)

    with tf.GradientTape() as tape:
        tape.watch(norm.affine_weight)
        out = norm.frwrd(inpt_dict)
        loss = tf.reduce_sum(out**2)

    grad = tape.gradient(loss, norm.affine_weight)
    assert grad is not None, "No gradient for affine_weight"
    assert tf.reduce_any(grad != 0), "Gradient is all zeros"


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_equivariant_rms_norm_rot_equivar(param_dtype):
    """EquivariantRMSNorm preserves rotational equivariance.

    For each (L, hist, parity) output group, the (2L+1) m-components must
    transform under the Wigner D-matrix: out(Rx) = D^L(R) @ out(x).
    """
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_rms_norm_test_data(
        param_dtype
    )

    # Recreate the same rotation used in _build_general_product_test_data
    axis = np.array([1, 1, 3])
    theta = np.pi / 2.2
    axis = axis / np.linalg.norm(axis)
    rot = Rotation.from_rotvec(theta * axis)

    Lmax_out = lmax_bond

    norm = EquivariantRMSNorm(input=AA, name="AA_norm", init="ones")
    norm.build(param_dtype)
    inpt_dict = norm(inpt_dict)

    out = inpt_dict[norm.name].numpy()  # [6, n_out, n_LM]
    out_orig = out[:3]  # atoms 0,1,2 = original
    out_rot = out[3:]  # atoms 3,4,5 = rotated by R

    # Compute Wigner D-matrices for the rotation
    D = _compute_wigner_d_real(rot, Lmax_out)

    # Check equivariance for each (L, hist, parity) group
    cmd = norm.coupling_meta_data
    groups = cmd.groupby(["l", "hist", "parity"]).indices
    for (L, hist, parity), indices in groups.items():
        sub = cmd.iloc[indices].sort_values("m")
        m_indices = sub.index.values
        ms = sub["m"].values
        assert list(ms) == list(
            range(-L, L + 1)
        ), f"Expected m = {list(range(-L, L + 1))}, got {list(ms)}"

        atol = 1e-12 if param_dtype == tf.float64 else 1e-6
        D_L = D[L]  # [2L+1, 2L+1]
        for atom in range(3):
            for n in range(out_orig.shape[1]):
                v_orig = out_orig[atom, n, m_indices]  # [2L+1]
                v_rot = out_rot[atom, n, m_indices]  # [2L+1]
                v_expected = D_L @ v_orig
                assert np.allclose(v_rot, v_expected, atol=atol), (
                    f"L={L}, hist={hist}, p={parity}, "
                    f"atom={atom}, n={n}: equivariance broken after "
                    f"EquivariantRMSNorm, "
                    f"max diff={np.max(np.abs(v_rot - v_expected))}"
                )


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_equivariant_rms_norm_split_norm_rot_equivar(param_dtype):
    """EquivariantRMSNorm with split_norm=True preserves equivariance."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_rms_norm_test_data(
        param_dtype
    )

    axis = np.array([1, 1, 3])
    theta = np.pi / 2.2
    axis = axis / np.linalg.norm(axis)
    rot = Rotation.from_rotvec(theta * axis)

    Lmax_out = lmax_bond

    norm = EquivariantRMSNorm(input=AA, name="AA_norm", init="ones", split_norm=True)
    norm.build(param_dtype)
    inpt_dict = norm(inpt_dict)

    out = inpt_dict[norm.name].numpy()
    out_orig = out[:3]
    out_rot = out[3:]

    D = _compute_wigner_d_real(rot, Lmax_out)

    cmd = norm.coupling_meta_data
    groups = cmd.groupby(["l", "hist", "parity"]).indices
    for (L, hist, parity), indices in groups.items():
        sub = cmd.iloc[indices].sort_values("m")
        m_indices = sub.index.values
        ms = sub["m"].values
        assert list(ms) == list(range(-L, L + 1))

        atol = 1e-12 if param_dtype == tf.float64 else 1e-6
        D_L = D[L]
        for atom in range(3):
            for n in range(out_orig.shape[1]):
                v_orig = out_orig[atom, n, m_indices]
                v_rot = out_rot[atom, n, m_indices]
                v_expected = D_L @ v_orig
                assert np.allclose(v_rot, v_expected, atol=atol), (
                    f"L={L}, hist={hist}, p={parity}, "
                    f"atom={atom}, n={n}: equivariance broken after "
                    f"EquivariantRMSNorm with split_norm, "
                    f"max diff={np.max(np.abs(v_rot - v_expected))}"
                )


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_equivariant_rms_norm_split_norm_independent(param_dtype):
    """With split_norm, l=0 and l>0 should be normalized independently."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_rms_norm_test_data(
        param_dtype
    )

    norm_global = EquivariantRMSNorm(
        input=AA, name="AA_norm_global", init="ones"
    )
    norm_global.build(param_dtype)
    inpt_dict = norm_global(inpt_dict)

    norm_split = EquivariantRMSNorm(
        input=AA, name="AA_norm_split", init="ones", split_norm=True
    )
    norm_split.build(param_dtype)
    inpt_dict = norm_split(inpt_dict)

    out_global = inpt_dict[norm_global.name].numpy()
    out_split = inpt_dict[norm_split.name].numpy()

    # They should generally differ (different normalization)
    assert not np.allclose(out_global, out_split, atol=1e-6), (
        "split_norm output should differ from global norm"
    )

    # Check that l=0 and l>0 RMS are independently ~1
    cmd = AA.coupling_meta_data
    l_values = cmd["l"].values
    l0_mask = l_values == 0
    lgt0_mask = ~l0_mask

    # For split norm, each group should have its own scale
    real_atoms = out_split[:3]  # only real atoms
    if l0_mask.any():
        l0_rms = np.sqrt(np.mean(real_atoms[:, :, l0_mask] ** 2))
        assert l0_rms > 0, "l=0 RMS should be non-zero"
    if lgt0_mask.any():
        lgt0_rms = np.sqrt(np.mean(real_atoms[:, :, lgt0_mask] ** 2))
        assert lgt0_rms > 0, "l>0 RMS should be non-zero"


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_equivariant_rms_norm_rot_invar_l0(param_dtype):
    """L=0 (scalar) features must be rotationally invariant after EquivariantRMSNorm.

    This is a tighter check than full equivariance — for L=0 the Wigner D-matrix
    is identity, so orig and rotated values must match directly without any
    matrix multiplication error from D.
    """
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_rms_norm_test_data(
        param_dtype
    )

    norm = EquivariantRMSNorm(input=AA, name="AA_norm", init="ones")
    norm.build(param_dtype)
    inpt_dict = norm(inpt_dict)

    out = inpt_dict[norm.name].numpy()  # [6, n_out, n_LM]
    out_orig = out[:3]
    out_rot = out[3:]

    # Extract L=0 channels
    cmd = norm.coupling_meta_data
    l0_mask = cmd["l"].values == 0
    assert l0_mask.any(), "No L=0 channels found"

    l0_orig = out_orig[:, :, l0_mask]
    l0_rot = out_rot[:, :, l0_mask]

    atol = 1e-6 if param_dtype == tf.float32 else 1e-10
    print(np.max(np.abs(l0_rot - l0_orig)), param_dtype)
    assert np.allclose(l0_orig, l0_rot, atol=atol), (
        f"L=0 rotational invariance broken after EquivariantRMSNorm, "
        f"max diff={np.max(np.abs(l0_orig - l0_rot)):.3e}"
    )


def _build_equivariant_rms_norm_test_data_with_padding(param_dtype):
    """Build test data with 3 real atoms + 3 rotated + 2 padding atoms.

    Padding atoms have bonds far beyond cutoff, so their features are zero.
    """
    lmax_bond = 4
    n_rad_base = 8
    n_rad_max = 8
    cutoff_rad = 6.0

    np.random.seed(322)
    tf.random.set_seed(322)

    axis = np.array([1, 1, 3])
    theta = np.pi / 2.2
    axis = axis / np.linalg.norm(axis)
    rot = Rotation.from_rotvec(theta * axis)

    coord = np.array([[0.2, 0.3, 2.1], [0.1, 3.2, 0.5], [0.3, 1, 2]])
    coord_r = rot.apply(coord)
    # 2 padding atoms placed far away (>cutoff from everything)
    coord_pad = np.array([[100.0, 100.0, 100.0], [200.0, 200.0, 200.0]])
    coord_all = np.vstack([coord, coord_r, coord_pad]).astype(np.float64)
    n_real = 6  # 3 orig + 3 rotated
    n_total = 8  # + 2 padding

    # Bonds: real atoms connect to each other; padding atoms connect to
    # each other (far beyond cutoff, so radial basis will be zero)
    indi = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7]
    indj = [1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4, 7, 6]
    atomic_mu_i = [0, 1, 0, 0, 1, 0, 0, 1]
    muj = np.take(atomic_mu_i, indj, axis=0)
    rj = np.take(coord_all, indj, axis=0)
    ri = np.take(coord_all, indi, axis=0)
    rij = rj - ri

    inpt_dict = {
        constants.BOND_VECTOR: rij,
        constants.BOND_IND_J: indj,
        constants.BOND_MU_J: muj,
        constants.BOND_IND_I: indi,
        constants.N_ATOMS_BATCH_TOTAL: n_total,
        constants.N_ATOMS_BATCH_REAL: tf.constant(n_real, dtype=tf.int32),
        constants.ATOMIC_MU_I: atomic_mu_i,
    }

    d_ij = BondLength()
    d_ij.build(param_dtype)
    inpt_dict = d_ij(inpt_dict)

    rhat = ScaledBondVector(bond_length=d_ij)
    rhat.build(param_dtype)
    inpt_dict = rhat(inpt_dict)

    g_k = RadialBasis(
        bonds=d_ij, basis_type="SBessel", nfunc=n_rad_base, rcut=cutoff_rad, p=5
    )
    g_k.build(param_dtype)
    inpt_dict = g_k(inpt_dict)

    R_nl = MLPRadialFunction(n_rad_max=n_rad_max, lmax=lmax_bond, basis=g_k, name="R")
    R_nl.build(param_dtype)
    inpt_dict = R_nl(inpt_dict)

    Y = SphericalHarmonic(vhat=rhat, lmax=lmax_bond, name="Y")
    Y.build(param_dtype)
    inpt_dict = Y(inpt_dict)

    z = ScalarChemicalEmbedding(
        element_map={"H": 0, "C": 1}, embedding_size=32, name="Z"
    )
    z.build(param_dtype)
    inpt_dict = z(inpt_dict)

    A = SingleParticleBasisFunctionScalarInd(
        radial=R_nl, angular=Y, indicator=z, name="A", avg_n_neigh=10.0
    )
    A.build(param_dtype)
    A.lin_transform.build(param_dtype)
    inpt_dict = A(inpt_dict)

    AA = GeneralProductFunction(
        left=A,
        right=A,
        name="AA",
        lmax=lmax_bond,
        Lmax=lmax_bond,
        mode="cp_l",
        rank=n_rad_max,
        use_S=False,
        normalize=True,
    )
    AA.build(param_dtype)
    inpt_dict = AA(inpt_dict)

    return inpt_dict, AA, n_real, n_total


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("center_l0", [False, True])
def test_equivariant_rms_norm_padding_zeros(param_dtype, center_l0):
    """Padding atoms (index >= n_at_b_real) must remain exactly zero."""
    inpt_dict, AA, n_real, n_total = _build_equivariant_rms_norm_test_data_with_padding(
        param_dtype
    )

    norm = EquivariantRMSNorm(
        input=AA, name="AA_norm", init="ones", center_l0=center_l0
    )
    norm.build(param_dtype)
    inpt_dict = norm(inpt_dict)

    out = inpt_dict[norm.name].numpy()  # [n_total, n_out, n_angular]
    padding_out = out[n_real:]  # [2, n_out, n_angular]

    assert np.all(padding_out == 0.0), (
        f"Padding atoms should be exactly zero, "
        f"max abs value={np.max(np.abs(padding_out)):.3e}, "
        f"center_l0={center_l0}"
    )


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_equivariant_rms_norm_l0_only_shape(param_dtype):
    """normalize_l0_only=True: output shape matches input shape."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_rms_norm_test_data(
        param_dtype
    )

    norm = EquivariantRMSNorm(
        input=AA, name="AA_norm_l0only", init="ones", normalize_l0_only=True
    )
    norm.build(param_dtype)
    inpt_dict = norm(inpt_dict)

    out = inpt_dict[norm.name]
    aa_out = inpt_dict[AA.name]
    assert out.shape == aa_out.shape, f"Shape mismatch: {out.shape} vs {aa_out.shape}"


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_equivariant_rms_norm_l0_only_higher_l_unchanged(param_dtype):
    """normalize_l0_only=True: l>0 channels must be identical to the input."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_rms_norm_test_data(
        param_dtype
    )

    norm = EquivariantRMSNorm(
        input=AA, name="AA_norm_l0only", init="ones", normalize_l0_only=True
    )
    norm.build(param_dtype)
    inpt_dict = norm(inpt_dict)

    out = inpt_dict[norm.name].numpy()
    inp = inpt_dict[AA.name].numpy()

    cmd = AA.coupling_meta_data
    high_l_mask = cmd["l"].values > 0
    assert high_l_mask.any(), "No l>0 channels to test"

    atol = 1e-6 if param_dtype == tf.float32 else 1e-12
    assert np.allclose(out[:, :, high_l_mask], inp[:, :, high_l_mask], atol=atol), (
        f"l>0 channels should be unchanged, "
        f"max diff={np.max(np.abs(out[:, :, high_l_mask] - inp[:, :, high_l_mask])):.3e}"
    )


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_equivariant_rms_norm_l0_only_l0_normalized(param_dtype):
    """normalize_l0_only=True: l=0 channels must differ from raw input (are normalized)."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_rms_norm_test_data(
        param_dtype
    )

    norm = EquivariantRMSNorm(
        input=AA, name="AA_norm_l0only", init="ones", normalize_l0_only=True
    )
    norm.build(param_dtype)
    inpt_dict = norm(inpt_dict)

    out = inpt_dict[norm.name].numpy()
    inp = inpt_dict[AA.name].numpy()

    cmd = AA.coupling_meta_data
    l0_mask = cmd["l"].values == 0
    assert l0_mask.any(), "No l=0 channels found"

    # l=0 output should not equal the raw input (normalization applied)
    assert not np.allclose(
        out[:, :, l0_mask], inp[:, :, l0_mask]
    ), "l=0 channels should be modified by normalization"


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_equivariant_rms_norm_l0_only_rot_equivar(param_dtype):
    """normalize_l0_only=True: rotational equivariance is preserved for all channels."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_rms_norm_test_data(
        param_dtype
    )

    axis = np.array([1, 1, 3])
    theta = np.pi / 2.2
    axis = axis / np.linalg.norm(axis)
    rot = Rotation.from_rotvec(theta * axis)

    norm = EquivariantRMSNorm(
        input=AA, name="AA_norm_l0only", init="ones", normalize_l0_only=True
    )
    norm.build(param_dtype)
    inpt_dict = norm(inpt_dict)

    out = inpt_dict[norm.name].numpy()  # [6, n_out, n_LM]
    out_orig = out[:3]
    out_rot = out[3:]

    D = _compute_wigner_d_real(rot, lmax_bond)

    cmd = norm.coupling_meta_data
    groups = cmd.groupby(["l", "hist", "parity"]).indices
    atol = 1e-12 if param_dtype == tf.float64 else 1e-5
    for (L, hist, parity), indices in groups.items():
        sub = cmd.iloc[indices].sort_values("m")
        m_indices = sub.index.values
        D_L = D[L]
        for atom in range(3):
            for n in range(out_orig.shape[1]):
                v_orig = out_orig[atom, n, m_indices]
                v_rot = out_rot[atom, n, m_indices]
                v_expected = D_L @ v_orig
                assert np.allclose(v_rot, v_expected, atol=atol), (
                    f"L={L}, hist={hist}, p={parity}, atom={atom}, n={n}: "
                    f"equivariance broken with normalize_l0_only, "
                    f"max diff={np.max(np.abs(v_rot - v_expected)):.3e}"
                )


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_equivariant_rms_norm_l0_only_padding_zeros(param_dtype):
    """normalize_l0_only=True: padding atoms remain exactly zero."""
    inpt_dict, AA, n_real, n_total = _build_equivariant_rms_norm_test_data_with_padding(
        param_dtype
    )

    norm = EquivariantRMSNorm(
        input=AA, name="AA_norm_l0only", init="ones", normalize_l0_only=True
    )
    norm.build(param_dtype)
    inpt_dict = norm(inpt_dict)

    out = inpt_dict[norm.name].numpy()
    padding_out = out[n_real:]

    assert np.all(padding_out == 0.0), (
        f"Padding atoms should be exactly zero with normalize_l0_only, "
        f"max abs value={np.max(np.abs(padding_out)):.3e}"
    )


# ===========================================================================
# EquivariantGate tests
# ===========================================================================


def _build_equivariant_gate_test_data(param_dtype):
    """Helper: build AA equivariant data for EquivariantGate tests."""
    inpt_dict, A, lmax_bond, n_rad_max = _build_general_product_test_data(
        dtype=param_dtype
    )
    inpt_dict[constants.N_ATOMS_BATCH_REAL] = tf.constant(6, dtype=tf.int32)

    AA = GeneralProductFunction(
        left=A,
        right=A,
        name="AA",
        lmax=lmax_bond,
        Lmax=lmax_bond,
        mode="cp_l",
        rank=n_rad_max,
        use_S=False,
        normalize=True,
    )
    AA.build(param_dtype)
    inpt_dict = AA(inpt_dict)
    return inpt_dict, AA, lmax_bond, n_rad_max


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("hidden_dim", [None, 8])
def test_equivariant_gate_shape(param_dtype, hidden_dim):
    """Output shape matches input shape."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_gate_test_data(param_dtype)

    gate = EquivariantGate(input=AA, name="AA_gate", hidden_dim=hidden_dim)
    gate.build(param_dtype)
    inpt_dict = gate(inpt_dict)

    out = inpt_dict[gate.name]
    aa_out = inpt_dict[AA.name]
    assert out.shape == aa_out.shape, f"Shape mismatch: {out.shape} vs {aa_out.shape}"


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("hidden_dim", [None, 8])
def test_equivariant_gate_rot_equivar(param_dtype, hidden_dim):
    """EquivariantGate preserves rotational equivariance.

    For each (L, hist, parity) output group, the (2L+1) m-components must
    transform under the Wigner D-matrix: out(Rx) = D^L(R) @ out(x).
    """
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_gate_test_data(param_dtype)

    # Recreate the same rotation used in _build_general_product_test_data
    axis = np.array([1, 1, 3])
    theta = np.pi / 2.2
    axis = axis / np.linalg.norm(axis)
    rot = Rotation.from_rotvec(theta * axis)

    Lmax_out = lmax_bond

    gate = EquivariantGate(input=AA, name="AA_gate", hidden_dim=hidden_dim)
    gate.build(param_dtype)
    inpt_dict = gate(inpt_dict)

    out = inpt_dict[gate.name].numpy()  # [6, n_out, n_LM]
    out_orig = out[:3]  # atoms 0,1,2 = original
    out_rot = out[3:]  # atoms 3,4,5 = rotated by R
    print(out)
    # Compute Wigner D-matrices for the rotation
    D = _compute_wigner_d_real(rot, Lmax_out)

    # Check equivariance for each (L, hist, parity) group
    cmd = gate.coupling_meta_data
    groups = cmd.groupby(["l", "hist", "parity"]).indices
    for (L, hist, parity), indices in groups.items():
        sub = cmd.iloc[indices].sort_values("m")
        m_indices = sub.index.values
        ms = sub["m"].values
        assert list(ms) == list(
            range(-L, L + 1)
        ), f"Expected m = {list(range(-L, L + 1))}, got {list(ms)}"

        atol = 1e-14 if param_dtype == tf.float64 else 1e-8
        D_L = D[L]  # [2L+1, 2L+1]
        for atom in range(3):
            for n in range(out_orig.shape[1]):
                v_orig = out_orig[atom, n, m_indices]  # [2L+1]
                v_rot = out_rot[atom, n, m_indices]  # [2L+1]
                v_expected = D_L @ v_orig
                assert np.allclose(v_rot, v_expected, atol=atol), (
                    f"hidden_dim={hidden_dim}, L={L}, hist={hist}, p={parity}, "
                    f"atom={atom}, n={n}: equivariance broken after "
                    f"EquivariantGate, "
                    f"max diff={np.max(np.abs(v_rot - v_expected))}"
                )


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("hidden_dim", [None, 8])
def test_equivariant_gate_rot_invar(param_dtype, hidden_dim):
    """EquivariantGate preserves rotational invariance when contracted to scalars."""
    inpt_dict, A, lmax_bond, n_rad_max = _build_general_product_test_data(
        dtype=param_dtype
    )

    AA = GeneralProductFunction(
        left=A,
        right=A,
        name="AA",
        lmax=lmax_bond,
        Lmax=lmax_bond,
        mode="cp_l",
        rank=n_rad_max,
        use_S=False,
        normalize=True,
    )
    AA.build(param_dtype)
    inpt_dict = AA(inpt_dict)

    gate = EquivariantGate(input=AA, name="AA_gate", hidden_dim=hidden_dim)
    gate.build(param_dtype)
    inpt_dict = gate(inpt_dict)

    # Contract to L=0 scalars
    AAA = GeneralProductFunction(
        left=gate,
        right=A,
        name="AAA",
        lmax=lmax_bond,
        Lmax=0,
        mode="elementwise",
        normalize=True,
    )
    AAA.build(param_dtype)
    inpt_dict = AAA(inpt_dict)

    I_l = FunctionReduce(
        instructions=[A, AA, AAA],
        name="E",
        ls_max=[0, 0, 0],
        n_in=n_rad_max,
        n_out=1,
        is_central_atom_type_dependent=True,
        number_of_atom_types=2,
        allowed_l_p=[[0, 1]],
    )
    I_l.build(param_dtype)
    inpt_dict = I_l(inpt_dict)

    out = inpt_dict[I_l.name]
    atol = 1e-14 if param_dtype == tf.float64 else 1e-8
    assert np.allclose(out[:3], out[3:], atol=atol), (
        f"hidden_dim={hidden_dim}: rotational invariance broken, "
        f"max diff={np.max(np.abs(out[:3] - out[3:]))}"
    )


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("hidden_dim", [None, 8])
def test_equivariant_gate_gradient(param_dtype, hidden_dim):
    """Gradients flow through the gate to both input and gate weights."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_gate_test_data(param_dtype)

    gate = EquivariantGate(input=AA, name="AA_gate", hidden_dim=hidden_dim)
    gate.build(param_dtype)

    if hidden_dim is None:
        watch_vars = [gate.gate_weight]
    else:
        watch_vars = [gate.gate_w1, gate.gate_w2]

    with tf.GradientTape() as tape:
        for v in watch_vars:
            tape.watch(v)
        out = gate.frwrd(inpt_dict)
        loss = tf.reduce_sum(out**2)

    grads = tape.gradient(loss, watch_vars)
    for g, v in zip(grads, watch_vars):
        assert g is not None, f"No gradient for {v.name}"
        assert tf.reduce_any(g != 0), f"Gradient is all zeros for {v.name}"


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_equivariant_gate_output_bounded(param_dtype):
    """Gate values are in (0, 1) so output magnitude <= input magnitude."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_gate_test_data(param_dtype)

    gate = EquivariantGate(input=AA, name="AA_gate")
    gate.build(param_dtype)
    inpt_dict = gate(inpt_dict)

    out = np.abs(inpt_dict[gate.name].numpy())
    inp = np.abs(inpt_dict[AA.name].numpy())
    assert np.all(out <= inp + 1e-7), (
        f"Gated output should not exceed input magnitude, "
        f"max excess={np.max(out - inp):.3e}"
    )


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_equivariant_gate_zero_input(param_dtype):
    """Zero input produces zero output regardless of gate weights."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_gate_test_data(param_dtype)

    gate = EquivariantGate(input=AA, name="AA_gate", use_bias=True)
    gate.build(param_dtype)

    # Zero out the input
    inpt_dict[AA.name] = tf.zeros_like(inpt_dict[AA.name])
    inpt_dict = gate(inpt_dict)

    out = inpt_dict[gate.name].numpy()
    assert np.all(out == 0.0), (
        f"Zero input should produce zero output, "
        f"max abs value={np.max(np.abs(out)):.3e}"
    )


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_equivariant_gate_coupling_metadata_preserved(param_dtype):
    """Gate preserves coupling_meta_data from input instruction."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_gate_test_data(param_dtype)

    gate = EquivariantGate(input=AA, name="AA_gate")
    assert gate.coupling_meta_data.equals(
        AA.coupling_meta_data
    ), "coupling_meta_data should be identical to input"
    assert gate.n_out == AA.n_out, "n_out should match input"
    assert gate.lmax == AA.lmax, "lmax should match input"


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("hidden_dim", [None, 8])
def test_equivariant_gate_mix_channels_rot_equivar(param_dtype, hidden_dim):
    """EquivariantGate with mix_channels preserves rotational equivariance."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_gate_test_data(
        param_dtype
    )

    axis = np.array([1, 1, 3])
    theta = np.pi / 2.2
    axis = axis / np.linalg.norm(axis)
    rot = Rotation.from_rotvec(theta * axis)

    Lmax_out = lmax_bond

    gate = EquivariantGate(
        input=AA, name="AA_gate_mix", hidden_dim=hidden_dim, mix_channels=4
    )
    gate.build(param_dtype)
    inpt_dict = gate(inpt_dict)

    out = inpt_dict[gate.name].numpy()
    out_orig = out[:3]
    out_rot = out[3:]

    D = _compute_wigner_d_real(rot, Lmax_out)

    cmd = gate.coupling_meta_data
    groups = cmd.groupby(["l", "hist", "parity"]).indices
    for (L, hist, parity), indices in groups.items():
        sub = cmd.iloc[indices].sort_values("m")
        m_indices = sub.index.values
        ms = sub["m"].values
        assert list(ms) == list(range(-L, L + 1))

        atol = 1e-12 if param_dtype == tf.float64 else 1e-6
        D_L = D[L]
        for atom in range(3):
            for n in range(out_orig.shape[1]):
                v_orig = out_orig[atom, n, m_indices]
                v_rot = out_rot[atom, n, m_indices]
                v_expected = D_L @ v_orig
                assert np.allclose(v_rot, v_expected, atol=atol), (
                    f"mix_channels, hidden_dim={hidden_dim}, L={L}: "
                    f"equivariance broken, "
                    f"max diff={np.max(np.abs(v_rot - v_expected))}"
                )


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_equivariant_gate_mix_channels_shape(param_dtype):
    """mix_channels does not change output shape."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_gate_test_data(
        param_dtype
    )

    gate = EquivariantGate(input=AA, name="AA_gate_mix", mix_channels=4)
    gate.build(param_dtype)
    inpt_dict = gate(inpt_dict)

    out = inpt_dict[gate.name]
    aa_out = inpt_dict[AA.name]
    assert out.shape == aa_out.shape, f"Shape mismatch: {out.shape} vs {aa_out.shape}"


@pytest.mark.parametrize("param_dtype", [tf.float32, tf.float64])
def test_equivariant_gate_mix_channels_gradient(param_dtype):
    """Gradients flow through mix_weight and mix_weight_back."""
    inpt_dict, AA, lmax_bond, n_rad_max = _build_equivariant_gate_test_data(
        param_dtype
    )

    gate = EquivariantGate(input=AA, name="AA_gate_mix", mix_channels=4)
    gate.build(param_dtype)

    watch_vars = [gate.gate_weight, gate.mix_weight, gate.mix_weight_back]

    with tf.GradientTape() as tape:
        for v in watch_vars:
            tape.watch(v)
        out = gate.frwrd(inpt_dict)
        loss = tf.reduce_sum(out**2)

    grads = tape.gradient(loss, watch_vars)
    for g, v in zip(grads, watch_vars):
        assert g is not None, f"No gradient for {v.name}"
        assert tf.reduce_any(g != 0), f"Gradient is all zeros for {v.name}"


