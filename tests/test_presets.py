import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pathlib import Path

from tensorpotential.instructions.base import load_instructions_list
from tensorpotential.potentials.presets import *

prefix = Path(__file__).parent.resolve()


def test_load_saved_model_yaml():
    model_name = str(prefix / "model_grace.yaml")
    instructions = load_instructions_list(model_name)
    assert len(instructions) == 18


def test_preset_FS_HEA25_with_simplification():
    kwargs = dict(
        lmax=(5, 5, 4, 3),
        Lmax=(None, 3, 0, 0),
        max_sum_l=(None, None, 6, 4),
        lmax_hist=(None, None, None, 3),
        max_order=4,
        simplify_prod=False,
    )
    pot = FS(element_map={"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}, **kwargs)
    print(len(pot))
    assert len(pot) == 14

    A_names = ["A", "AA", "AAA", "AAAA"]
    A_ins_dict = {ins.name: ins for ins in pot if ins.name in A_names}

    for n in A_names:
        print(n, A_ins_dict[n].coupling_meta_data.shape[0])

    assert [A_ins_dict[n].coupling_meta_data.shape[0] for n in A_names] == [
        36,
        115,
        7,
        5,
    ]

    # with simplification

    kwargs["simplify_prod"] = True

    pot2 = FS(element_map={"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}, **kwargs)
    print(len(pot2))
    assert len(pot2) == 14

    A_ins_dict = {ins.name: ins for ins in pot2 if ins.name in A_names}

    for n in A_names:
        print(n, A_ins_dict[n].coupling_meta_data.shape[0])

    assert [A_ins_dict[n].coupling_meta_data.shape[0] for n in A_names] == [
        36,
        115,
        7,
        5,
    ]
