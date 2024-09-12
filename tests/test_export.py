import numpy as np
import tensorflow as tf
import yaml

from tensorpotential.potentials.presets import FS
from tensorpotential.export import export_to_yaml


def test_export_to_yaml():
    np.random.seed(1)
    tf.random.set_seed(1)

    fs_ins = FS(
        element_map={"Al": 0, "Li": 1, "H": 2},
        lmax=2,
        n_rad_base=4,
        n_rad_max=12,
        embedding_size=8,
        max_order=3,
    )

    # BUILD
    for ins in fs_ins:
        ins.build(tf.float64)

    export_to_yaml(fs_ins, "test_export_to_yaml.yaml")

    with open("test_export_to_yaml.yaml") as f:
        pot_yaml = yaml.load(f, yaml.SafeLoader)

    print(f"pot_yaml.keys={pot_yaml.keys()}")

    for k in ["elements", "emb_spec", "radial_basis", "functions"]:
        assert k in pot_yaml

    assert pot_yaml["elements"] == ["Al", "Li", "H"]
    assert pot_yaml["emb_spec"] == {
        "params": [1.0, 1.0, 1.0, 0.5, 1.0, 2.0, 1.0, 0.75],
        "type": "FinnisSinclairShiftedScaled",
    }
    print("radial_basis.keys=", pot_yaml["radial_basis"].keys())
    for k in ["Z", "crad", "cutoff", "n_rad_base", "nradmax", "radbasename"]:
        assert k in pot_yaml["radial_basis"]
    assert np.shape(pot_yaml["radial_basis"]["Z"]) == (12 * 3,)

    # shape = [n_rad_max, lmax + 1, n_rad_base]
    assert np.shape(pot_yaml["radial_basis"]["crad"]) == (12 * 3 * 4,)

    assert len(pot_yaml["functions"]) == 3
    assert len(pot_yaml["functions"][0]) == 108

    tdbfunc = pot_yaml["functions"][0][12]
    print(f"tdbfunc={tdbfunc}")
    for k, v in {
        "mu0": 0,
        "ns": 1,
        "ls": [0, 0],
        "ms_combs": [0, 0],
        "gen_cgs": [1.0],
        "ndensity": 4,
        "coeff": [-0.00653062, 0.2612635, -0.33575857, 0.86569158],
        "rank": 2,
    }.items():
        assert k in tdbfunc
        assert tdbfunc[k] == v or np.allclose(tdbfunc[k], v)
