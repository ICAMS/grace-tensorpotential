import numpy as np
import pytest

from tensorpotential.data.dense_nbr import (
    auto_slot_budget,
    bucket_by_max_neigh,
    pack_structures_elastic,
    plan_dense_batches,
    dense_padding_fractions,
)


def test_auto_slot_budget_is_typical_batch_footprint():
    nat = np.array([10, 10, 10, 10])
    max_neigh = np.array([20, 20, 20, 20])
    # batch_size * median(nat) * median(max_neigh) = 2 * 10 * 20
    assert auto_slot_budget(nat, max_neigh, batch_size=2) == 400


def test_bucket_by_max_neigh_explicit_k_minimizes_padding():
    # 4 single-atom structures; K=2 -> DP groups {10,12} and {40,44} (pad each to bucket max).
    nat = np.array([1, 1, 1, 1])
    max_neigh = np.array([10, 12, 40, 44])
    n_neigh = np.array([10, 12, 40, 44])
    width, K = bucket_by_max_neigh(nat, max_neigh, n_neigh, n_buckets=2)
    assert K == 2
    assert width.tolist() == [12, 12, 44, 44]  # each padded to its bucket's own max


def test_bucket_by_max_neigh_auto_hits_padding_target():
    # 1 bucket -> width 40 -> pad 37.5%; 2 buckets -> {10,10},{40,40} -> pad 0% <= 15%.
    nat = np.array([1, 1, 1, 1])
    max_neigh = np.array([10, 10, 40, 40])
    n_neigh = np.array([10, 10, 40, 40])
    width, K = bucket_by_max_neigh(nat, max_neigh, n_neigh, n_buckets="auto", max_padding=0.15)
    assert K == 2
    assert sorted(set(width.tolist())) == [10, 40]


def test_bucket_by_max_neigh_single_value_one_bucket():
    nat = np.array([1, 1, 1])
    max_neigh = np.array([20, 20, 20])
    n_neigh = np.array([20, 20, 20])
    width, K = bucket_by_max_neigh(nat, max_neigh, n_neigh, n_buckets="auto")
    assert K == 1
    assert width.tolist() == [20, 20, 20]


def test_pack_groups_by_assigned_width():
    # width-group 12 has structs {0,1}; width-group 20 has {2}
    nat = np.array([4, 4, 4])
    width_per_struct = np.array([12, 12, 20])
    batches = pack_structures_elastic(nat, width_per_struct, batch_size=8, slot_budget=10_000)
    assert sorted(b["width"] for b in batches) == [12, 20]
    assert sorted(len(b["structure_ind"]) for b in batches) == [1, 2]


def test_pack_respects_slot_budget():
    # width 20, budget 200 -> at most 10 atoms/batch -> three size-1 batches of 6 atoms
    nat = np.array([6, 6, 6])
    width_per_struct = np.array([20, 20, 20])
    batches = pack_structures_elastic(nat, width_per_struct, batch_size=99, slot_budget=200)
    for b in batches:
        total_nat = sum(int(nat[i]) for i in b["structure_ind"])
        assert total_nat * b["width"] <= 200 or len(b["structure_ind"]) == 1


def test_pack_lone_outlier_is_size_one():
    nat = np.array([50])
    width_per_struct = np.array([4000])
    batches = pack_structures_elastic(nat, width_per_struct, batch_size=8, slot_budget=100)
    assert len(batches) == 1
    assert len(batches[0]["structure_ind"]) == 1


def test_plan_drops_over_cap_and_reports():
    nat = np.array([4, 4, 4])
    max_neigh = np.array([10, 10, 5000])
    n_neigh = np.array([40, 40, 200])
    batches, dropped = plan_dense_batches(
        nat, max_neigh, n_neigh, batch_size=8, slot_budget=10_000, max_neigh_cap=100
    )
    assert dropped == [2]
    kept = sorted(i for b in batches for i in b["structure_ind"])
    assert kept == [0, 1]


def test_plan_no_structure_lost_without_cap():
    nat = np.array([3, 5, 7, 9])
    max_neigh = np.array([10, 12, 40, 44])
    n_neigh = np.array([30, 60, 280, 396])
    batches, dropped = plan_dense_batches(
        nat, max_neigh, n_neigh, batch_size=2, slot_budget="auto"
    )
    assert dropped == []
    kept = sorted(i for b in batches for i in b["structure_ind"])
    assert kept == [0, 1, 2, 3]
    for b in batches:
        assert isinstance(b["max_neigh"], int) and b["max_neigh"] > 0


def test_dense_padding_fractions():
    stats = {
        "pad_nneigh": 15, "nreal_neigh": 85,
        "pad_nat": 1, "nreal_atoms": 99,
        "pad_nstruct": 0, "nreal_struc": 10,
    }
    f = dense_padding_fractions(stats)
    assert f["neigh"] == pytest.approx(0.15)
    assert f["atoms"] == pytest.approx(0.01)
    assert f["struct"] == pytest.approx(0.0)


def test_plan_max_nat_is_batch_total_atoms():
    # max_nat is the per-batch TOTAL real atom count (flat atom axis, matching
    # N_ATOMS_BATCH_TOTAL / _dense_reshape_einsum's n_atoms) -- NOT a per-structure max.
    # Force 2 width buckets {10,12},{40,44}; huge budget -> one batch per width group, no fake.
    nat = np.array([3, 5, 7, 9])
    max_neigh = np.array([10, 12, 40, 44])
    n_neigh = np.array([30, 60, 280, 396])
    batches, _ = plan_dense_batches(
        nat, max_neigh, n_neigh, batch_size=2, slot_budget=10_000, n_neigh_buckets=2
    )
    for b in batches:
        total_real = sum(int(nat[i]) for i in b["structure_ind"])
        assert b["max_nat"] == total_real, (b["max_nat"], total_real)


def test_plan_max_nat_pads_with_one_fake_atom():
    # Two size-1 batches (batch_size=1) of different totals, same width, forced into ONE
    # nat-bucket (max_n_buckets=1) -> share max_nat = max(totals) + 1 fake atom.
    nat = np.array([4, 6])
    max_neigh = np.array([20, 20])
    n_neigh = np.array([20, 20])
    batches, _ = plan_dense_batches(
        nat, max_neigh, n_neigh, batch_size=1, slot_budget=10_000, max_n_buckets=1
    )
    assert sorted(b["max_nat"] for b in batches) == [7, 7]


def test_plan_net_padding_lands_at_target():
    # the unified net_padding target drives BOTH width and nat bucketing so the achieved net
    # neighbor padding (over max_nat*width slots) lands at/under the target. Uses a realistic
    # CLUSTERED max_neigh distribution with low intra-structure spread (~5% floor) so the target
    # is achievable within the default bucket cap.
    rng = np.random.RandomState(0)
    nat = rng.randint(5, 40, size=200)
    max_neigh = rng.choice([40, 50, 60, 70, 90, 120], size=200)
    n_neigh = (nat * (max_neigh * 0.95).astype(int)).astype(int)  # avg ~0.95*max -> ~5% floor
    for tgt in (0.15, 0.10):
        b, _ = plan_dense_batches(
            nat, max_neigh, n_neigh, batch_size=16, slot_budget="auto", net_padding=tgt
        )
        slots = sum(x["max_nat"] * x["max_neigh"] for x in b)
        net = 1 - n_neigh.sum() / slots
        assert net <= tgt + 0.03, (tgt, net)  # lands at/under target (small per-group slack)


def test_dense_padding_fractions_warning_threshold():
    stats = {
        "pad_nneigh": 30, "nreal_neigh": 70,
        "pad_nat": 0, "nreal_atoms": 100,
        "pad_nstruct": 0, "nreal_struc": 10,
    }
    f = dense_padding_fractions(stats)
    assert f["neigh"] > 0.15
