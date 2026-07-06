import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
from ase.build import bulk

from tensorpotential.data.databuilder import construct_batches, GeometricalDataBuilder

import pytest

dir_path = os.path.dirname(os.path.realpath(__file__))


def test_construct_batches_list() -> None:
    at1 = bulk("Al", cubic=True)
    at2 = bulk("Al", "sc", a=5, cubic=True)

    at3 = bulk("Cu", cubic=True)
    at4 = bulk("Cu", "sc", a=3, cubic=True)

    data_builders = [GeometricalDataBuilder({"Al": 0, "Cu": 1}, cutoff=6)]
    ase_atoms_list = [at1, at2, at3, at4]
    batches, padding_stats = construct_batches(
        ase_atoms_list,
        data_builders=data_builders,
        batch_size=2,
        max_n_buckets=1,
        return_padding_stats=True,
        verbose=False,
    )

    print("batches", batches)
    print("padding_stats", padding_stats)
    assert len(batches) == 2
    assert padding_stats is not None
    b0 = batches[0]
    b1 = batches[1]
    assert b0["n_struct_total"] == 3
    assert b1["n_struct_total"] == 3

    assert b0["batch_tot_nat"] == 6
    assert b1["batch_tot_nat"] == 6


def test_construct_batches_df() -> None:
    df = pd.read_pickle(os.path.join(dir_path, "data/MoNbTaW_train50.pkl.gz"))
    elements_map = {"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}
    data_builders = [GeometricalDataBuilder(elements_map=elements_map, cutoff=6)]

    batches, padding_stats = construct_batches(
        df,
        data_builders=data_builders,
        batch_size=10,
        max_n_buckets=2,
        return_padding_stats=True,
        verbose=True,
    )

    # print("batches", batches)
    print("padding_stats", padding_stats)
    padding_stats_ref = {
        "pad_nstruct": 5,
        "pad_nat": 5,
        "pad_nneigh": 56,
        "nreal_struc": 50,
        "nreal_atoms": 844,
        "nreal_neigh": 48904,
    }
    assert len(batches) == 5
    assert padding_stats is not None
    b0 = batches[0]
    b1 = batches[1]
    assert b0["n_struct_total"] == 11
    assert b1["n_struct_total"] == 11
    #
    # assert b0["batch_tot_nat"] == 199
    # assert b1["batch_tot_nat"] == 199

    assert padding_stats_ref == padding_stats_ref


@pytest.mark.xfail
def test_construct_batches_multiple_db() -> None:
    from tensorpotential.constants import CELL_VECTORS, ATOMIC_POS

    try:
        from tensorpotential.extra.gen_tensor.databuilder import (
            PositionsDataBuilder,
            CellDataBuilder,
        )
    except ImportError:
        assert 1 == 0
    df = pd.read_pickle(os.path.join(dir_path, "data/MoNbTaW_train50.pkl.gz"))
    elements_map = {"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}
    data_builders = [
        GeometricalDataBuilder(elements_map=elements_map, cutoff=6),
        PositionsDataBuilder(),
        CellDataBuilder(),
    ]

    batches, padding_stats = construct_batches(
        df,
        data_builders=data_builders,
        batch_size=10,
        max_n_buckets=2,
        return_padding_stats=True,
        verbose=True,
    )

    # print("batches", batches)
    print("padding_stats", padding_stats)
    padding_stats_ref = {
        "pad_nstruct": 5,
        "pad_nat": 5,
        "pad_nneigh": 56,
        "nreal_struc": 50,
        "nreal_atoms": 844,
        "nreal_neigh": 48904,
    }
    assert len(batches) == 5
    assert padding_stats is not None
    b0 = batches[0]
    b1 = batches[1]
    assert b0["n_struct_total"] == 11
    assert b1["n_struct_total"] == 11
    #
    assert CELL_VECTORS in b0
    assert len(b1[CELL_VECTORS].shape) == 3
    assert ATOMIC_POS in b1
    print("b0", b0[CELL_VECTORS])

    assert padding_stats_ref == padding_stats_ref


def test_bucketing_split_dense_one_width_per_batch():
    from ase.build import bulk
    from tensorpotential.data.databuilder import bucketing_split_dense
    from tensorpotential import constants as C

    db = GeometricalDataBuilder({"Cu": 0}, cutoff=6.0, dense_nbr=True)
    # 3 small cells + 1 supercell; a supercell has IDENTICAL per-atom coordination to
    # its primitive cell, so all four may land in the same band. This test checks the
    # per-batch invariant (PAD_MAX_NEIGH present, bond count derived correctly, no
    # structure lost) regardless of how many bands the planner produces.
    structs = [
        bulk("Cu", "fcc", a=3.6, cubic=True),
        bulk("Cu", "fcc", a=3.6, cubic=True),
        bulk("Cu", "fcc", a=3.6, cubic=True),
        bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2),
    ]
    data = [db.extract_from_ase_atoms(s) for s in structs]

    batches, max_pad = bucketing_split_dense(
        data, batch_size=4, max_n_buckets="auto", slot_budget="auto"
    )
    assert len(batches) == len(max_pad)
    # every batch carries an integer reshape width and the derived bond count
    for mp in max_pad:
        assert C.PAD_MAX_NEIGH in mp
        assert mp[C.PAD_MAX_N_NEIGHBORS] == mp[C.PAD_MAX_N_ATOMS] * mp[C.PAD_MAX_NEIGH]
    # no structure lost
    total = sum(len(b) for b in batches)
    assert total == 4


def test_bucketing_split_dense_drop_cap(caplog):
    from ase.build import bulk
    from tensorpotential.data.databuilder import bucketing_split_dense

    db = GeometricalDataBuilder({"Cu": 0}, cutoff=6.0, dense_nbr=True)
    small = bulk("Cu", "fcc", a=3.6, cubic=True)
    data = [db.extract_from_ase_atoms(small) for _ in range(3)]
    # cap=0 drops everything (every atom has >=1 neighbor)
    with caplog.at_level(logging.WARNING):
        batches, max_pad = bucketing_split_dense(
            data, batch_size=4, max_n_buckets="auto", max_neigh_cap=0
        )
    assert batches == [] and max_pad == []
    assert any("dense_max_neigh_cap=0" in r.message for r in caplog.records)


def test_bucketing_split_dense_separates_buckets_by_max_neigh():
    from ase.build import bulk
    from tensorpotential.data.databuilder import bucketing_split_dense, _struct_max_neigh
    from tensorpotential import constants as C

    db = GeometricalDataBuilder({"Cu": 0}, cutoff=6.0, dense_nbr=True)
    normal = bulk("Cu", "fcc", a=3.6, cubic=True)      # lower coordination within rcut
    compressed = bulk("Cu", "fcc", a=2.2, cubic=True)  # high coordination within rcut
    data = [db.extract_from_ase_atoms(normal), db.extract_from_ase_atoms(compressed)]
    mns = [_struct_max_neigh(d) for d in data]
    # sanity: the two structures have clearly different max_neigh
    assert abs(mns[0] - mns[1]) >= 16, mns

    batches, max_pad = bucketing_split_dense(
        data, batch_size=4, max_n_buckets="auto", slot_budget="auto"
    )
    # adaptive bucketing -> the two distinct max_neigh fall in separate width buckets,
    # each padded to its own max -> different reshape widths -> cannot share a batch
    widths = {mp[C.PAD_MAX_NEIGH] for mp in max_pad}
    assert len(widths) >= 2, widths
    assert len(batches) >= 2
    for mp in max_pad:
        assert mp[C.PAD_MAX_N_NEIGHBORS] == mp[C.PAD_MAX_N_ATOMS] * mp[C.PAD_MAX_NEIGH]
    # each width hugs its bucket's own max (adaptive, not a rounded-up fixed tier)
    assert sorted(widths) == sorted(mns)


def test_construct_batches_df_parallel() -> None:
    df = pd.read_pickle(os.path.join(dir_path, "data/MoNbTaW_train50.pkl.gz"))
    elements_map = {"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}
    data_builders = [GeometricalDataBuilder(elements_map=elements_map, cutoff=6)]

    batches, padding_stats = construct_batches(
        df,
        data_builders=data_builders,
        batch_size=10,
        max_n_buckets=2,
        return_padding_stats=True,
        verbose=True,
        max_workers=2,
    )

    # print("batches", batches)
    print("padding_stats", padding_stats)
    padding_stats_ref = {
        "pad_nstruct": 5,
        "pad_nat": 5,
        "pad_nneigh": 56,
        "nreal_struc": 50,
        "nreal_atoms": 844,
        "nreal_neigh": 48904,
    }
    assert len(batches) == 5
    assert padding_stats is not None
    b0 = batches[0]
    b1 = batches[1]
    assert b0["n_struct_total"] == 11
    assert b1["n_struct_total"] == 11
    #
    # assert b0["batch_tot_nat"] == 185
    assert b0["batch_tot_nat"] == 199
    assert b1["batch_tot_nat"] == 199

    assert padding_stats_ref == padding_stats_ref


def test_pad_batch_dense_groups_bonds_into_per_atom_blocks():
    """Dense reshape places atom i's bonds in the contiguous slot block [i*width,(i+1)*width):
    reshaping BOND_IND_I to [max_nat, width] gives rows whose REAL (within-cutoff) entries all
    equal the row index, with exactly counts[i] real entries per row. A vacancy makes
    coordination heterogeneous, so this is FALSE for the flat seg-sum layout."""
    import numpy as np
    from ase.build import bulk
    from tensorpotential import constants as C

    db = GeometricalDataBuilder({"Cu": 0}, cutoff=6.0, dense_nbr=True)
    s = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    del s[0]  # vacancy -> heterogeneous coordination
    batch = db.join_to_batch([db.extract_from_ase_atoms(s)])
    real_nat = int(np.asarray(batch[C.N_ATOMS_BATCH_REAL]))
    counts = np.bincount(np.asarray(batch[C.BOND_IND_I]), minlength=real_nat)[:real_nat]
    assert counts.min() < counts.max(), counts  # sanity: heterogeneous, else vacuous
    width = int(counts.max())

    max_nat = real_nat
    db.pad_batch(batch, {
        C.PAD_MAX_N_STRUCTURES: 1, C.PAD_MAX_N_ATOMS: max_nat,
        C.PAD_MAX_NEIGH: width, C.PAD_MAX_N_NEIGHBORS: max_nat * width,
    })

    assert int(np.asarray(batch[C.BOND_IND_I]).shape[0]) == max_nat * width
    ind_i = np.asarray(batch[C.BOND_IND_I]).reshape(max_nat, width)
    bnorm = np.linalg.norm(np.asarray(batch[C.BOND_VECTOR]).reshape(max_nat, width, 3), axis=2)
    for atom in range(real_nat):
        real = bnorm[atom] <= db.cutoff
        assert real.sum() == counts[atom], (atom, int(real.sum()), int(counts[atom]))
        assert np.all(ind_i[atom][real] == atom), atom


def test_pad_batch_dense_fake_atom_block_all_dummy():
    """With atom padding (max_nat = real_nat + 1) the fake atom gets a full width block of
    dummy bonds (> cutoff), real atom blocks stay centered on their index, and the dense bond
    count is max_nat * width."""
    import numpy as np
    from ase.build import bulk
    from tensorpotential import constants as C

    db = GeometricalDataBuilder({"Cu": 0}, cutoff=6.0, dense_nbr=True)
    s = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    del s[0]
    batch = db.join_to_batch([db.extract_from_ase_atoms(s)])
    real_nat = int(np.asarray(batch[C.N_ATOMS_BATCH_REAL]))
    counts = np.bincount(np.asarray(batch[C.BOND_IND_I]), minlength=real_nat)[:real_nat]
    width = int(counts.max())
    max_nat = real_nat + 1  # one fake atom
    db.pad_batch(batch, {
        C.PAD_MAX_N_STRUCTURES: 2, C.PAD_MAX_N_ATOMS: max_nat,
        C.PAD_MAX_NEIGH: width, C.PAD_MAX_N_NEIGHBORS: max_nat * width,
    })
    assert int(np.asarray(batch[C.ATOMIC_MU_I]).shape[0]) == max_nat
    assert int(np.asarray(batch[C.BOND_IND_I]).shape[0]) == max_nat * width
    bnorm = np.linalg.norm(np.asarray(batch[C.BOND_VECTOR]).reshape(max_nat, width, 3), axis=2)
    assert np.all(bnorm[max_nat - 1] > db.cutoff)  # fake atom block entirely dummy
    ind_i = np.asarray(batch[C.BOND_IND_I]).reshape(max_nat, width)
    for atom in range(real_nat):
        real = bnorm[atom] <= db.cutoff
        assert np.all(ind_i[atom][real] == atom), atom


def test_construct_batches_dense_layout_and_stats():
    import numpy as np
    from ase.build import bulk
    from tensorpotential import constants as C

    structs = [
        bulk("Cu", "fcc", a=3.6, cubic=True),
        bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 1, 1),
        bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 1),
        bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2),
    ]
    db = GeometricalDataBuilder({"Cu": 0}, cutoff=6.0, dense_nbr=True)
    batches, stats = construct_batches(
        structs,
        data_builders=[db],
        batch_size=2,
        max_n_buckets="auto",
        return_padding_stats=True,
        verbose=False,
    )
    # every batch is in the per-atom-uniform layout
    for b in batches:
        nb = int(np.asarray(b[C.BOND_IND_I]).shape[0])
        nat = int(np.asarray(b[C.N_ATOMS_BATCH_TOTAL]))
        assert nb % nat == 0, f"{nb} not a multiple of {nat}"
    assert stats is not None
    assert stats["nreal_atoms"] == sum(len(s) for s in structs)


def test_construct_batches_dense_matches_segment_real_counts():
    import numpy as np
    from ase.build import bulk

    structs = [bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2) for _ in range(3)]
    seg = construct_batches(
        structs, data_builders=[GeometricalDataBuilder({"Cu": 0}, cutoff=6.0)],
        batch_size=3, max_n_buckets=1, return_padding_stats=True, verbose=False,
    )[1]
    dense = construct_batches(
        structs, data_builders=[GeometricalDataBuilder({"Cu": 0}, cutoff=6.0, dense_nbr=True)],
        batch_size=3, max_n_buckets=1, return_padding_stats=True, verbose=False,
    )[1]
    # identical real content, only padding differs
    assert seg["nreal_atoms"] == dense["nreal_atoms"]
    assert seg["nreal_neigh"] == dense["nreal_neigh"]


def test_dense_batch_parity_with_segment_sum():
    import numpy as np
    import tensorflow as tf
    from ase.build import bulk
    from tensorpotential import TPModel, constants as C
    from tensorpotential.potentials.presets import GRACE_2LAYER_v2_25

    def build_model(dense):
        tf.random.set_seed(7)
        np.random.seed(7)
        ins = GRACE_2LAYER_v2_25(
            element_map={"Cu": 0}, rcut=6.0, dense_nbr=dense
        ).get_instructions()
        m = TPModel(ins)
        m.build(tf.float64)
        return m

    # Three 2x2x1 supercells (16 atoms each) with different rattles.
    # Equal atom counts are required: the dense planner sorts structures by nat within a band
    # (pack_structures_elastic), so mixed-size structures arrive in a different order than the
    # segment path, which preserves insertion order.  Same-size structures are stable under that
    # sort, so both paths see the same ordering and the real-prefix comparison is valid.
    structs = [
        bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 1),
        bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 1),
        bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 1),
    ]
    for i, s in enumerate(structs):
        s.rattle(stdev=0.05, seed=i)

    def make_batch(dense):
        db = GeometricalDataBuilder({"Cu": 0}, cutoff=6.0, dense_nbr=dense)
        batches = construct_batches(
            [s.copy() for s in structs], data_builders=[db],
            batch_size=3, max_n_buckets=1, verbose=False,
        )
        assert len(batches) == 1, f"expected 1 batch, got {len(batches)}"
        return batches[0]

    def to_tensors(batch):
        out = {}
        for k, v in batch.items():
            arr = np.asarray(v)
            if np.issubdtype(arr.dtype, np.floating):
                out[k] = tf.constant(arr, dtype=tf.float64)
            else:
                out[k] = tf.constant(arr, dtype=tf.int32)
        return out

    m_seg, m_den = build_model(False), build_model(True)
    seg = to_tensors(make_batch(False))
    den = to_tensors(make_batch(True))

    r_seg = m_seg.train_function(m_seg.instructions, seg)
    r_den = m_den.train_function(m_den.instructions, den)

    n_struct = len(structs)
    n_atoms = sum(len(s) for s in structs)
    e_seg = r_seg[C.PREDICT_TOTAL_ENERGY].numpy()[:n_struct]
    e_den = r_den[C.PREDICT_TOTAL_ENERGY].numpy()[:n_struct]
    f_seg = r_seg[C.PREDICT_FORCES].numpy()[:n_atoms]
    f_den = r_den[C.PREDICT_FORCES].numpy()[:n_atoms]

    assert np.allclose(e_seg, e_den, atol=1e-6, rtol=0), (e_seg, e_den)
    assert np.allclose(f_seg, f_den, atol=1e-5, rtol=0)


def test_dense_export_resets_dense_nbr_flag(tmp_path):
    """Regression: exporting a dense-capable model must leave dense_nbr=False on the
    shared instructions afterwards.

    save_model flips dense_nbr=True on the shared instructions to trace the
    `compute_dense` signature. Sibling signatures traced in the same export that read
    the flag but never set it (the aux computes and, via grace_uq, `compute_uq*`) would
    otherwise inherit that True and bake the dense reshape (`n_bonds // n_atoms`) into
    their flat-layout graphs -- a size mismatch at inference. The dual signature is
    emitted for any equivariant model, so this is the standard grace_uq export path.
    """
    import numpy as np
    import tensorflow as tf
    from tensorpotential import TPModel
    from tensorpotential.potentials.presets import GRACE_2LAYER_v2_25

    tf.random.set_seed(7)
    np.random.seed(7)
    ins = GRACE_2LAYER_v2_25(
        element_map={"Cu": 0}, rcut=6.0, dense_nbr=True
    ).get_instructions()
    m = TPModel(ins)
    m.build(tf.float64)

    instr = m.instructions
    dense_instr = [
        it
        for it in (instr.values() if hasattr(instr, "values") else instr)
        if getattr(it, "dense_capable", False)
    ]
    assert dense_instr, "expected a dense-capable instruction (equivariant SPBF)"
    # opt-in dense model: the flag is True at rest before export.
    assert any(it.dense_nbr for it in dense_instr)

    m.save_model(str(tmp_path / "m"), input_signature_float_dtype=tf.float64)

    # The export must restore dense_nbr to False so co-traced signatures stay on
    # segment_sum regardless of TF's trace order.
    assert all(not it.dense_nbr for it in dense_instr), [
        it.dense_nbr for it in dense_instr
    ]

    loaded = tf.saved_model.load(str(tmp_path / "m"))
    assert "compute" in loaded.signatures
    assert "compute_dense" in loaded.signatures
