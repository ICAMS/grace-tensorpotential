import os

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

from ase.build import bulk
from ase import Atoms
from matplotlib import pylab as plt
from scipy.spatial.transform import Rotation
from tensorflow import float64

from tensorpotential.instructions import *
from tensorpotential import constants
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
        p=10,
        normalized=False,
        init_gamma=100,
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
        name="baza",
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
    model = TPModel(instructions=[d_ij, bs, z], compute_function=ComputePlaceholder)
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
    r_in = 3
    size = 10
    np.random.seed(322)
    fake_bonds = np.random.uniform(0, rcut + 1, size=(size, 1))
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
    l = np.array(p.coupling_meta_data["l"]).reshape(1, -1)
    # print(np.concatenate([l, l1, l2], axis=0).reshape(1, -1), len(l))
    lll = np.ravel_multi_index(
        np.concatenate([l, l1, l2], axis=0),
        [np.max(l) + 1, np.max(l1) + 1, np.max(l2) + 1],
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
    size = 10
    lmax = 4
    np.random.seed(322)
    fake_neighbors = np.random.normal(0, 3, size=(size, 3))
    fake_center = np.array([0, 0, 0]).reshape(1, -1)
    fake_bonds = fake_neighbors - fake_center
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
    from ase.build import bulk
    from ase.neighborlist import neighbor_list
    from tensorpotential.instructions.compute import FCRight2Left

    cutoff = 4
    c = bulk("C", "diamond", cubic=True)
    c.rattle(stdev=0.1)
    ind_i, ind_j, bond_vector = neighbor_list("ijD", c, cutoff=cutoff)
    _, nn = np.unique(ind_i, return_counts=True)
    size = 10
    lmax = 4
    np.random.seed(322)
    fake_neighbors = np.random.normal(0, 3, size=(size, 3))
    fake_center = np.array([0, 0, 0]).reshape(1, -1)
    fake_bonds = fake_neighbors - fake_center
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
    inpt_data["AA"] = aa
    inpt_data[constants.N_ATOMS_BATCH_TOTAL] = size
    inpt_data[constants.ATOMIC_MU_I] = np.zeros((size)).astype(np.int32)

    I = FunctionReduce(
        instructions=[p],
        name="I",
        ls_max=[0],
        n_in=1,
        n_out=3,
        is_central_atom_type_dependent=True,
        number_of_atom_types=1,
        allowed_l_p=[[0, 1]],
    )
    I.build(float64)
    inpt_data = I(inpt_data)
    assert inpt_data[I.name].numpy().shape == (size, 3, 1)

    II = FunctionReduceN(
        instructions=[p, I],
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
    I = FunctionReduce(
        instructions=[p, p2],
        name="I",
        ls_max=[0, 0],
        n_in=1,
        n_out=n_out,
        is_central_atom_type_dependent=True,
        number_of_atom_types=1,
        allowed_l_p=[[0, 1]],
    )
    I.build(float64)
    print("I.Coupling: ")
    print(I.coupling_meta_data)
    inpt_dict = I(inpt_dict)
    I_out = inpt_dict[I.name].numpy()[:, :, 0]
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
    n_rad_max_out = 6
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
