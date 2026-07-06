import os
import shutil

import numpy as np
import pytest
import tensorflow as tf
from ase import Atoms
from ase.build import bulk
from tensorflow import float64
from tensorflow.data import Dataset

from tensorpotential import TPModel
from tensorpotential.calculator import TPCalculator
from tensorpotential.calculator.asecalculator import (
    AdaptivePaddingConfig,
    PaddingManager,
)
from tensorpotential.data.databuilder import GeometricalDataBuilder
from tensorpotential.instructions import load_instructions

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



def test_savedmodel_dual_dense_signature(tmp_path):
    """A dense_nbr model exports ONE SavedModel with two interchangeable signatures
    sharing weights: `compute` (segment_sum) and `compute_dense` (dense reshape). They
    have the SAME input signature (reshape adds no input) and differ only in the bond
    LAYOUT the caller supplies; the calculator's `mode` picks the engine
    and get_data prepares the matching layout. They must agree to round-off."""
    from tensorpotential.tensorpot import TensorPotential
    from tensorpotential.potentials.presets import GRACE_2LAYER_v2_25

    tf.random.set_seed(7)
    np.random.seed(7)
    ins = GRACE_2LAYER_v2_25(
        element_map={"Cu": 0}, rcut=6.0, dense_nbr=True
    ).get_instructions()
    model = TPModel(ins)
    model.build(tf.float64)

    path = str(tmp_path / "saved_dual")
    model.save_model(path, input_signature_float_dtype=tf.float64)

    m = TensorPotential.load_model(path)
    assert {"compute", "compute_dense"} <= set(m.signatures)
    # identical input signature -- the mode lives in the graph + bond layout, not inputs
    assert set(m.signatures["compute"]._arg_keywords) == set(
        m.signatures["compute_dense"]._arg_keywords
    )

    s = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    s.rattle(stdev=0.1, seed=7)

    c_seg = TPCalculator(model=path, mode="diverse")
    c_den = TPCalculator(model=path, mode="uniform")
    assert c_seg.dense_reshape is False  # diverse -> compute: flat bonds
    assert c_den.dense_reshape is True   # uniform -> compute_dense: reordered per-atom-uniform bonds

    a_seg = s.copy()
    a_seg.calc = c_seg
    a_den = s.copy()
    a_den.calc = c_den
    e_seg, f_seg = a_seg.get_potential_energy(), a_seg.get_forces()
    e_den, f_den = a_den.get_potential_energy(), a_den.get_forces()
    assert np.allclose(e_seg, e_den, atol=1e-7, rtol=0)
    assert np.allclose(f_seg, f_den, atol=1e-6, rtol=0)


def test_calculator_dense_reshape_parity():
    """Dense RESHAPE mode (dense_nbr='reshape', per-atom-uniform bond layout) through
    the calculator matches the segment_sum baseline E/F. The calculator auto-detects
    reshape from the model's instructions and reorders bonds in get_data."""
    from tensorpotential import constants
    from tensorpotential.potentials.presets import GRACE_2LAYER_v2_25

    def mk(dn):
        tf.random.set_seed(7)
        np.random.seed(7)
        ins = GRACE_2LAYER_v2_25(element_map={"Cu": 0}, rcut=6.0, dense_nbr=dn).get_instructions()
        m = TPModel(ins)
        m.build(tf.float64)
        m.decorate_compute_function(input_signature_float_dtype=tf.float64)
        return TPCalculator(model=m, mode="uniform" if dn else "diverse")

    c_base, c_reshape = mk(False), mk("reshape")
    assert c_base.dense_reshape is False
    assert c_reshape.dense_reshape is True

    s = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    s.rattle(stdev=0.1, seed=3)
    a_b = s.copy()
    a_b.calc = c_base
    a_r = s.copy()
    a_r.calc = c_reshape
    eb, fb = a_b.get_potential_energy(), a_b.get_forces()
    er, fr = a_r.get_potential_energy(), a_r.get_forces()
    assert np.allclose(eb, er, atol=1e-7, rtol=0)
    assert np.allclose(fb, fr, atol=1e-6, rtol=0)

    # reshape produced the per-atom-uniform [n_atoms*max_neigh] bond layout
    c_reshape.get_data(s)
    nb = int(np.asarray(c_reshape.data[constants.BOND_IND_I]).shape[0])
    nat = int(np.asarray(c_reshape.data[constants.ATOMIC_MU_I]).shape[0])
    assert nb % nat == 0, f"{nb} bonds not a multiple of {nat} atoms"


def test_calculator_mode_resolution_and_tight_padding():
    """`mode` selects the engine: 'uniform'->dense (tight initial padding), 'diverse'->segment_sum.
    When the model can't run the requested engine it falls back to the available one; an unknown
    mode raises."""
    import pytest
    from tensorpotential.potentials.presets import GRACE_2LAYER_v2_25

    def mk(dense):
        tf.random.set_seed(7); np.random.seed(7)
        ins = GRACE_2LAYER_v2_25(element_map={"Cu": 0}, rcut=6.0, dense_nbr=dense).get_instructions()
        m = TPModel(ins); m.build(tf.float64)
        m.decorate_compute_function(input_signature_float_dtype=tf.float64)
        return m

    dense_model, seg_model = mk(True), mk(False)

    # uniform -> dense, with TIGHT initial padding (small step, no floor64 / geometric ladder)
    c = TPCalculator(model=dense_model, mode="uniform")
    assert c.dense_reshape is True and c.mode == "uniform"
    pm = c.dense_padding_manager
    assert pm.width_floor == 8 and pm.width_step == 8 and pm.atom_step == 8
    assert pm.width_geom_factor is None and pm.atom_geom_factor is None  # linear-only (no ladder)

    # diverse on a single-engine (dense-only) in-memory model -> falls back to dense
    assert TPCalculator(model=dense_model, mode="diverse").dense_reshape is True

    # segment_sum model: diverse uses seg; uniform falls back to seg (no dense engine)
    assert TPCalculator(model=seg_model, mode="diverse").dense_reshape is False
    assert TPCalculator(model=seg_model, mode="uniform").dense_reshape is False

    # default mode is 'diverse'
    assert TPCalculator(model=seg_model).mode == "diverse"

    # unknown mode -> ValueError
    with pytest.raises(ValueError):
        TPCalculator(model=seg_model, mode="fast")


def test_dense_padding_manager_shape_reuse():
    """DensePaddingManager collapses many distinct (nat, max_neigh) structures onto few reused
    (atoms, width) shapes (== XLA compiles), the lever for sequential dense evaluation."""
    from tensorpotential.calculator.asecalculator import (
        DensePaddingManager, _snap_up_hybrid,
    )

    # hybrid ladder: fine (step 16) up to 128, geometric (x1.5) above
    assert _snap_up_hybrid(41, step=16, linear_max=128) == 48
    assert _snap_up_hybrid(128, step=16, linear_max=128) == 128
    assert _snap_up_hybrid(129, step=16, linear_max=128, geom_factor=1.5) == 192
    assert _snap_up_hybrid(129, step=16, linear_max=128, geom_factor=None) == 144  # linear-only

    dm = DensePaddingManager(data_builders=[], reuse_tolerance=0.5, atom_step=4, width_step=16)
    # identical structures -> one compile, the rest reuse
    s0 = dm.select_shape(10, 40)
    for _ in range(5):
        assert dm.select_shape(10, 40) == s0
    assert dm.n_compiles == 1 and dm.n_reuse == 5

    # a slightly larger structure that fits s0 within tolerance reuses it (no new compile)
    assert dm.select_shape(10, 41) == s0
    assert dm.n_compiles == 1

    # a clearly larger structure mints a new shape (fits nothing) and covers itself
    big = dm.select_shape(120, 300)
    assert big[0] >= 120 and big[1] >= 300 and dm.n_compiles == 2

    # a diverse random sequence -> compiles far below the number of distinct inputs
    rng = np.random.RandomState(0)
    dm2 = DensePaddingManager(data_builders=[], reuse_tolerance=0.5, atom_step=4, width_step=16)
    pairs = [(int(rng.randint(2, 120)), int(rng.randint(8, 200))) for _ in range(2000)]
    for nat, mx in pairs:
        dm2.select_shape(nat, mx)
    assert dm2.n_compiles < len(set(pairs)) / 2  # heavy collapse vs distinct shapes


def _drive(dm, seq, compile_s=0.10, steady_s=0.05):
    """Run ``(nat, true_max)`` pairs through ``dm`` like the calculator would: select, then feed
    the eval wall-time back (first eval of a shape == its compile, later == steady)."""
    for nat, tm in seq:
        dm.select_shape(nat, tm)
        first_eval = dm._eval_seen.get(dm._last_shape, 0) == 0
        dm.note_eval(compile_s if first_eval else steady_s)


def test_dense_padding_manager_hot_shape_promotion():
    """Measured ski-rental promotion: a hot, STABLE region (MD/relaxation: fixed nat, narrow
    max_neigh drift) is promoted to a tight shape (atoms exact, fewer slots) once the padding it
    wastes has paid back a measured compile -- but only then, so a too-short run never promotes,
    and a diverse stream (wide envelope -> no gain) never promotes."""
    from tensorpotential.calculator.asecalculator import DensePaddingManager

    # MD-like Cu: 256 atoms (fixed), high coordination max_neigh ~90 drifting tightly. The coarse
    # ladder pads atoms 256 -> 288; promotion should recover that by setting atoms back to 256.
    # compile_s=0.10, steady_s=0.05, frac saved ~0.11 -> pays back a compile after ~18 reuses.
    dm = DensePaddingManager(data_builders=[], promote_safety=1.0)
    dm.select_shape(256, 90); dm.note_eval(0.10)  # coarse mint + its compile
    first = dm.visited[0]
    assert first[0] > 256  # coarse ladder padded atoms up
    _drive(dm, [(256, 88 + (i % 5)) for i in range(60)])  # long, stable -> eventually pays back
    assert dm.n_promote == 1
    promoted = dm.select_shape(256, 90)  # cheapest fit is now the tight shape
    assert promoted[0] == 256  # atoms exact (no padding for fixed-N)
    assert promoted[0] * promoted[1] < first[0] * first[1]  # strictly fewer padded slots

    # a run too SHORT to pay back a compile must NOT promote (the key property vs a fixed count)
    dm_short = DensePaddingManager(data_builders=[], promote_safety=1.0)
    dm_short.select_shape(256, 90); dm_short.note_eval(0.10)
    _drive(dm_short, [(256, 88 + (i % 5)) for i in range(5)])  # only 5 reuses
    assert dm_short.n_promote == 0

    # promotion can be disabled outright
    dm_off = DensePaddingManager(data_builders=[], promote=False)
    dm_off.select_shape(256, 90); dm_off.note_eval(0.10)
    _drive(dm_off, [(256, 88 + (i % 5)) for i in range(60)])
    assert dm_off.n_promote == 0

    # an already-tight hot region gains nothing -> no promotion, no wasted compile
    dm_tight = DensePaddingManager(data_builders=[], promote_safety=1.0,
                                   atom_floor=1, width_floor=1, atom_step=1, width_step=1)
    _drive(dm_tight, [(64, 64) for _ in range(40)])
    assert dm_tight.n_promote == 0 and dm_tight.n_compiles == 1

    # diverse stream: wide envelopes -> no/negligible promotion -> compile count not inflated
    rng = np.random.RandomState(0)
    pairs = [(int(rng.randint(2, 120)), int(rng.randint(8, 200))) for _ in range(2000)]
    off = DensePaddingManager(data_builders=[], promote=False)
    on = DensePaddingManager(data_builders=[])  # default promotion (safety=2.0)
    _drive(off, pairs); _drive(on, pairs)
    assert on.n_compiles <= off.n_compiles * 1.3


def test_calculator_model():
    ins = load_instructions("model_grace.yaml")
    for instr in ins:
        if hasattr(instr, "inv_avg_n_neigh"):
            setattr(instr, "inv_avg_n_neigh", 1 / 30.0)
    model = TPModel(ins)
    model.build(tf.float64)
    calc = TPCalculator(model=model)

    np.random.seed(322)
    s = bulk("W", cubic=True) * (2, 2, 2)
    s.positions += np.random.uniform(-0.2, 0.2, size=(len(s), 3))
    s.calc = calc
    e0 = s.get_potential_energy()
    print(e0)
    f0 = s.get_forces()
    print(f0)
    st0 = s.get_stress()
    print(st0)
    st0_num = calc.calculate_numerical_stress(s, d=5e-5)
    print(st0_num - st0)
    assert np.allclose(st0, st0_num, atol=1e-4)

    shutil.rmtree("./test_calculator_model_tmp", ignore_errors=True)
    model.save_model("./test_calculator_model_tmp")

    calc1 = TPCalculator(model="./test_calculator_model_tmp")

    np.random.seed(322)
    s = bulk("W", cubic=True) * (2, 2, 2)
    s.positions += np.random.uniform(-0.2, 0.2, size=(len(s), 3))
    s.calc = calc1
    e1 = s.get_potential_energy()
    print(e1)
    f1 = s.get_forces()
    print(f1)
    st1 = s.get_stress()
    print(st1)
    st1_num = calc.calculate_numerical_stress(s, d=5e-5)
    assert np.allclose(st1, st1_num, atol=1e-4)

    assert np.allclose(e0, e1)
    print(f0 - f1)
    assert np.allclose(f0, f1, atol=1e-7)
    assert np.allclose(st0, st1)
    shutil.rmtree("./test_calculator_model_tmp", ignore_errors=True)


def test_calculator_with_fake_neighbors():
    calc = TPCalculator(
        model="./test_calculator_model", pad_neighbors_fraction=0.25, pad_atoms_number=1
    )
    calc1 = TPCalculator(
        model="./test_calculator_model",
        pad_neighbors_fraction=None,
        pad_atoms_number=None,
    )
    # print(f"{calc.padding_manager.current_max_neighbors=} at the start")
    np.random.seed(322)
    s = bulk("W", cubic=True) * (2, 2, 2)
    s.positions += np.random.uniform(-0.2, 0.2, size=(len(s), 3))
    s.calc = calc
    e = s.get_potential_energy()
    print(e)
    f = s.get_forces()
    print(f)
    assert np.shape(f) == (len(s), 3)
    st = s.get_stress()
    print(st)
    # print(f"{calc.padding_manager.current_max_neighbors=} at the end")
    st_num = calc.calculate_numerical_stress(s)
    assert np.allclose(st, st_num, rtol=1e-5)

    s.calc = calc1
    e1 = s.get_potential_energy()
    f1 = s.get_forces()
    st1 = s.get_stress()
    assert np.shape(f1) == (len(s), 3)

    assert np.allclose(e, e1, rtol=1e-5)
    assert np.allclose(f, f1, rtol=1e-5)
    assert np.allclose(st, st1, rtol=1e-5)


def test_dynamic_padding():
    calc = TPCalculator(
        model="./test_calculator_model",
        pad_neighbors_fraction=0.25,
        pad_atoms_number=1,
        debug_padding_verbose=3,
    )

    assert len(calc.padding_manager.padding_bounds) == 0
    s1 = bulk("W", a=3.165, cubic=True)
    s1.calc = calc
    e1 = s1.get_potential_energy()
    print(f"{e1=}")
    assert len(calc.padding_manager.padding_bounds) == 1
    # print(
    #     calc.padding_manager.current_max_atoms,
    #     calc.padding_manager.current_max_neighbors,
    # )

    s2 = bulk("W", a=3.165, cubic=True) * (2, 2, 2)
    s2.calc = calc
    e2 = s2.get_potential_energy()
    print(f"{e2=}")
    assert e1 != e2
    assert len(calc.padding_manager.padding_bounds) == 2
    # print(calc.current_max_atoms, calc.current_max_neighbors)

    s3 = bulk("W", a=2.5, cubic=True)
    s3.calc = calc
    e3 = s3.get_potential_energy()
    print(f"{e3=}")
    assert e1 != e3
    assert len(calc.padding_manager.padding_bounds) == 2
    # print(calc.current_max_atoms, calc.current_max_neighbors)

    s4 = bulk("W", a=2.75, cubic=True)
    s4.calc = calc
    e4 = s4.get_potential_energy()
    print(f"{e4=}")
    assert e4 != e3
    assert len(calc.padding_manager.padding_bounds) == 2
    # print(calc.current_max_atoms, calc.current_max_neighbors)


def test_dynamic_padding_reducing():
    calc = TPCalculator(
        model="./test_calculator_model", pad_neighbors_fraction=0.25, pad_atoms_number=5
    )

    assert len(calc.padding_manager.padding_bounds) == 0

    s2 = bulk("W", a=3.165, cubic=True) * (2, 2, 2)
    s2.calc = calc
    e2 = s2.get_potential_energy()
    print(f"{e2=}")
    assert len(calc.padding_manager.padding_bounds) == 1
    # print(calc.current_max_atoms, calc.current_max_neighbors)
    print("padding history=", calc.padding_manager.padding_bounds)

    s1 = bulk("W", a=3.165, cubic=True)
    s1.calc = calc
    e1 = s1.get_potential_energy()
    print(f"{e1=}")
    assert e1 != e2
    assert len(calc.padding_manager.padding_bounds) == 2
    # print(calc.current_max_atoms, calc.current_max_neighbors)
    print("padding history=", calc.padding_manager.padding_bounds)


def test_dynamic_padding_reducing_limit():
    calc = TPCalculator(
        model="./test_calculator_model",
        pad_neighbors_fraction=0.25,
        pad_atoms_number=5,
        max_number_reduction_recompilation=0,
    )

    assert len(calc.padding_manager.padding_bounds) == 0

    s2 = bulk("W", a=3.165, cubic=True) * (2, 2, 2)
    s2.calc = calc
    e2 = s2.get_potential_energy()
    print(f"{e2=}")
    assert len(calc.padding_manager.padding_bounds) == 1
    # print(calc.current_max_atoms, calc.current_max_neighbors)
    print("padding history=", calc.padding_manager.padding_bounds)

    s1 = bulk("W", a=3.165, cubic=True)
    s1.calc = calc
    e1 = s1.get_potential_energy()
    print(f"{e1=}")
    assert e1 != e2
    assert len(calc.padding_manager.padding_bounds) == 1
    # print(calc.current_max_atoms, calc.current_max_neighbors)
    print("padding history=", calc.padding_manager.padding_bounds)


def test_calculator_with_fake_atoms():
    calc = TPCalculator(
        model="./test_calculator_model",
        pad_neighbors_fraction=0.25,
        pad_atoms_number=10,
    )
    calc1 = TPCalculator(
        model="./test_calculator_model",
    )
    # print(f"{calc.current_max_neighbors=} at the start")
    # print(f"{calc.current_max_atoms=} at the start")
    np.random.seed(322)
    s = bulk("W", cubic=True) * (2, 2, 2)
    s.positions += np.random.uniform(-0.2, 0.2, size=(len(s), 3))
    s.calc = calc
    e = s.get_potential_energy()
    print(e)
    f = s.get_forces()
    assert np.shape(f) == (len(s), 3)
    print(f)
    st = s.get_stress()
    print(st)
    # print(f"{calc.padding_manager.current_max_neighbors=} at the end")
    # print(f"{calc.padding_manager.current_max_atoms=} at the end")
    st_num = calc.calculate_numerical_stress(s)
    assert np.allclose(st, st_num, rtol=1e-5)

    s.calc = calc1
    e1 = s.get_potential_energy()
    f1 = s.get_forces()
    assert np.shape(f1) == (len(s), 3)

    st1 = s.get_stress()

    assert np.allclose(e, e1, rtol=1e-5)
    assert np.allclose(f, f1, rtol=1e-5)
    assert np.allclose(st, st1, rtol=1e-5)


def test_ensemble_calculator_model():
    ins = load_instructions("model_grace.yaml")
    for instr in ins:
        if hasattr(instr, "inv_avg_n_neigh"):
            setattr(instr, "inv_avg_n_neigh", 1 / 30.0)
    model = TPModel(ins)
    model.build(float64)
    calc = TPCalculator(model=[model, model])

    np.random.seed(322)
    s = bulk("W", cubic=True) * (2, 2, 2)
    s.rattle(0.2)
    s.calc = calc
    e0 = s.get_potential_energy()
    print(e0)
    f0 = s.get_forces()
    print(f0)
    st0 = s.get_stress()
    print(st0)
    st0_num = calc.calculate_numerical_stress(s, d=5e-5)
    assert np.allclose(st0, st0_num, atol=1e-5)

    assert "energy_std" in calc.results
    assert "forces_std" in calc.results
    assert "stress_std" in calc.results

    assert np.allclose(calc.results["energy_std"], 0)
    assert np.allclose(calc.results["forces_std"], 0)
    assert np.allclose(calc.results["stress_std"], 0)


def test_calculator_min_dist():
    calc = TPCalculator(
        model="./test_calculator_model",
        pad_neighbors_fraction=0.25,
        min_dist=1.0,
    )

    at = Atoms("W2", positions=[[0, 0, 0], [0, 0, 2]], cell=(10, 10, 10), pbc=True)
    at.calc = calc

    at.get_potential_energy()

    at2 = Atoms("W2", positions=[[0, 0, 0], [0, 0, 0.5]], cell=(10, 10, 10), pbc=True)
    at2.calc = calc
    with pytest.raises(RuntimeError):
        at2.get_potential_energy()


def test_calculator_model_with_custom_cutoff():
    calc = TPCalculator(
        model="./test_model_custom_cutoff",
        pad_neighbors_fraction=0.25,
        min_dist=1.0,
    )

    def compute_e(symbs, z):
        at = Atoms(
            symbs, positions=[[0, 0, 0], [0, 0, z]], cell=(100, 100, 100), pbc=True
        )
        desc = f"{symbs} (z={z})"
        at.calc = calc
        e = at.get_potential_energy()
        print(f"{desc} e=", e)
        return e

    # Ta-*: 7
    assert compute_e("Ta2", 7.5) == 0
    assert compute_e("Ta2", 5.5) != 0

    # Ta-Mo: 7
    assert compute_e("TaMo", 7.5) == 0
    assert compute_e("MoTa", 5.5) != 0

    # Mo: 4
    assert compute_e("Mo2", 5.5) == 0
    assert compute_e("Mo2", 3.5) != 0

    # W: 5
    assert compute_e("W2", 5.5) == 0
    assert compute_e("W2", 3.5) != 0

    # Mo-Nb: 3
    assert compute_e("MoNb", 5.5) == 0
    assert compute_e("MoNb", 2.5) != 0


def test_padding_manager():
    print()
    element_map = {"W": 0}
    cutoff = 6
    geom_data_builder = GeometricalDataBuilder(
        elements_map=element_map,
        cutoff=cutoff,
        cutoff_dict=None,
    )
    data_builders = [geom_data_builder]

    pm = PaddingManager(
        data_builders=data_builders,
        pad_neighbors_fraction=0.25,
        pad_atoms_number=5,
        max_number_reduction_recompilation=0,
        debug_padding_verbose=3,
    )

    assert len(pm.padding_bounds) == 0

    atoms = bulk("W", a=3.165, cubic=True) * (2, 2, 2)
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 1

    del atoms[0]
    del atoms[1]
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 1

    atoms = bulk("W", a=3.165, cubic=True)
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 1

    print("set max_number_reduction_recompilation=1")
    pm.max_number_reduction_recompilation = 1
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 2

    atoms = bulk("W", a=3.165, cubic=True) * (1, 2, 2)
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 2

    # print("data=", data)
    data = {k: v for k, v in data.items()}

    ins = load_instructions("model_grace.yaml")
    model = TPModel(ins)
    model.build(float64)

    data_keys = [k for k, v in model.compute_specs.items()]
    data = {k: v for k, v in data.items() if k in data_keys}

    tf_data = Dataset.from_tensors(data).get_single_element()
    model.compute(tf_data)

    # print("output=", output)


def test_padding_manager_no_padding():
    print()
    element_map = {"W": 0}
    cutoff = 6
    geom_data_builder = GeometricalDataBuilder(
        elements_map=element_map,
        cutoff=cutoff,
        cutoff_dict=None,
    )
    data_builders = [geom_data_builder]

    pm = PaddingManager(
        data_builders=data_builders,
        pad_neighbors_fraction=None,
        pad_atoms_number=None,
        debug_padding_verbose=3,
    )

    assert len(pm.padding_bounds) == 0

    atoms = bulk("W", a=3.165, cubic=True) * (2, 2, 2)
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 0

    del atoms[0]
    del atoms[1]
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 0

    atoms = bulk("W", a=3.165, cubic=True)
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 0

    print("set max_number_reduction_recompilation=1")
    pm.max_number_reduction_recompilation = 1
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 0

    atoms = bulk("W", a=3.165, cubic=True) * (1, 2, 2)
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 0

    # print("data=", data)
    data = {k: v for k, v in data.items()}

    ins = load_instructions("model_grace.yaml")
    model = TPModel(ins)
    model.build(float64)

    data_keys = [k for k, v in model.compute_specs.items()]
    data = {k: v for k, v in data.items() if k in data_keys}

    tf_data = Dataset.from_tensors(data).get_single_element()
    model.compute(tf_data)


def _make_padding_manager(**kwargs):
    """Helper to create a PaddingManager with W element map and cutoff=6."""
    element_map = {"W": 0}
    geom_data_builder = GeometricalDataBuilder(
        elements_map=element_map, cutoff=6, cutoff_dict=None
    )
    defaults = dict(
        data_builders=[geom_data_builder],
        pad_neighbors_fraction=0.25,
        pad_atoms_number=5,
        max_number_reduction_recompilation=0,
        debug_padding_verbose=3,
    )
    defaults.update(kwargs)
    return PaddingManager(**defaults)


def test_adaptive_padding_grows_on_frequent_misses():
    """Feed strictly increasing structures so every call is a cache miss.
    After the window fills, margins should have grown."""
    pm = _make_padding_manager(
        adaptive_padding=True,
        adaptive_padding_config=AdaptivePaddingConfig(
            miss_rate_window=5,
            miss_rate_threshold=0.5,
            atoms_growth_factor=2.0,
            neighbors_growth_factor=1.5,
        ),
    )
    initial_atoms = pm._current_pad_atoms_number
    initial_neigh = pm._current_pad_neighbors_fraction

    # Generate structures of increasing size to force cache misses
    for n in range(1, 8):
        atoms = bulk("W", a=3.165, cubic=True) * (n, 1, 1)
        pm.get_data(atoms)

    assert pm._n_adaptations >= 1
    assert pm._current_pad_atoms_number > initial_atoms
    assert pm._current_pad_neighbors_fraction > initial_neigh


def test_adaptive_padding_no_growth_on_hits():
    """Feed the same structure repeatedly — after the first miss, all are hits."""
    pm = _make_padding_manager(
        adaptive_padding=True,
        adaptive_padding_config=AdaptivePaddingConfig(
            miss_rate_window=5,
            miss_rate_threshold=0.5,
        ),
    )
    atoms = bulk("W", a=3.165, cubic=True) * (2, 2, 2)
    for _ in range(20):
        pm.get_data(atoms)

    assert pm._n_adaptations == 0


def test_adaptive_padding_respects_caps():
    """Set low caps and force many misses — values must not exceed caps."""
    pm = _make_padding_manager(
        adaptive_padding=True,
        adaptive_padding_config=AdaptivePaddingConfig(
            miss_rate_window=3,
            miss_rate_threshold=0.3,
            atoms_growth_factor=3.0,
            neighbors_growth_factor=3.0,
            max_pad_atoms_number=12,
            max_pad_neighbors_fraction=0.40,
        ),
    )
    for n in range(1, 20):
        atoms = bulk("W", a=3.165, cubic=True) * (n, 1, 1)
        pm.get_data(atoms)

    assert pm._current_pad_atoms_number <= 12
    assert pm._current_pad_neighbors_fraction <= 0.40


def test_adaptive_padding_atoms_grow_faster():
    """After adaptation, the relative growth of atoms should exceed neighbors."""
    pm = _make_padding_manager(
        adaptive_padding=True,
        adaptive_padding_config=AdaptivePaddingConfig(
            miss_rate_window=5,
            miss_rate_threshold=0.5,
            atoms_growth_factor=2.0,
            neighbors_growth_factor=1.5,
        ),
    )
    initial_atoms = pm._current_pad_atoms_number
    initial_neigh = pm._current_pad_neighbors_fraction

    for n in range(1, 15):
        atoms = bulk("W", a=3.165, cubic=True) * (n, 1, 1)
        pm.get_data(atoms)

    assert pm._n_adaptations >= 1
    atoms_ratio = pm._current_pad_atoms_number / initial_atoms
    neigh_ratio = pm._current_pad_neighbors_fraction / initial_neigh
    assert atoms_ratio > neigh_ratio


def test_adaptive_padding_disabled():
    """With adaptive_padding=False, margins never change regardless of misses."""
    pm = _make_padding_manager(
        adaptive_padding=False,
        adaptive_padding_config=AdaptivePaddingConfig(
            miss_rate_window=3,
            miss_rate_threshold=0.3,
        ),
    )
    initial_atoms = pm._current_pad_atoms_number
    initial_neigh = pm._current_pad_neighbors_fraction

    for n in range(1, 15):
        atoms = bulk("W", a=3.165, cubic=True) * (n, 1, 1)
        pm.get_data(atoms)

    assert pm._n_adaptations == 0
    assert pm._current_pad_atoms_number == initial_atoms
    assert pm._current_pad_neighbors_fraction == initial_neigh


def test_padding_bounds_remain_sorted_after_insertions():
    """``bisect.insort`` keeps ``padding_bounds`` lexicographically sorted
    even when entries arrive in arbitrary order. The atoms dimension is
    monotone after sort; the neighbors dimension is only monotone *within*
    a fixed atoms value (the lookup relies on this hybrid structure)."""
    pm = _make_padding_manager()
    # Feed structures in deliberately non-monotone order.
    sizes = [(2, 1, 1), (1, 1, 1), (3, 1, 1), (2, 2, 1), (1, 1, 1)]
    for nx, ny, nz in sizes:
        atoms = bulk("W", a=3.165, cubic=True) * (nx, ny, nz)
        pm.get_data(atoms)

    bounds = pm.padding_bounds
    assert bounds == sorted(bounds), "bisect.insort must keep bounds sorted"
    # Lexicographic order ⇒ atoms dimension is monotone non-decreasing.
    atoms_dim = [b[0] for b in bounds]
    assert atoms_dim == sorted(atoms_dim)


def test_find_upper_padding_bound_returns_valid_superset():
    """Lookup result must satisfy *both* atoms >= max_nat AND neighbors >=
    max_nneigh — the linear tail scan exists precisely because the
    neighbors dimension isn't globally monotone."""
    pm = _make_padding_manager()
    # Manually seed padding_bounds with a mixed configuration that exposes
    # the non-monotone neighbors-by-atoms shape: small structures with many
    # neighbors, large structures with fewer.
    import bisect

    candidates = [(10, 200), (20, 100), (30, 150), (40, 250)]
    for c in candidates:
        bisect.insort(pm.padding_bounds, c)

    # Query (atoms=15, neighbors=120): need atoms>=15, neighbors>=120.
    # Candidates that satisfy: (20,100) NO (n=100<120), (30,150) YES,
    # (40,250) YES. Smallest atoms among valid = 30.
    found = pm.find_upper_padding_bound(15, 120)
    assert found is not None
    assert found[0] >= 15 and found[1] >= 120
    assert found == (30, 150)

    # Query that no bucket satisfies → None.
    assert pm.find_upper_padding_bound(50, 50) is None

    # Query exactly matching a bucket.
    assert pm.find_upper_padding_bound(10, 200) == (10, 200)
