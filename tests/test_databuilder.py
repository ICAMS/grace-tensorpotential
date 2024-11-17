import pandas as pd
from ase.build import bulk

from tensorpotential.data.databuilder import construct_batches, GeometricalDataBuilder

import os

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
    assert b0["batch_tot_nat"] == 199
    assert b1["batch_tot_nat"] == 199

    assert padding_stats_ref == padding_stats_ref
