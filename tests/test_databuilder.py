from ase.build import bulk

from tensorpotential.data.databuilder import construct_batches, GeometricalDataBuilder


def test_construct_batches_df() -> None:
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
