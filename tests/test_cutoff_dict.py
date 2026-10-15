"""process_cutoff_dict: explicit pairs win over wildcard pairs; presets; clear errors."""

import pytest

from tensorpotential.utils import process_cutoff_dict

ELEMENTS = {"Mo": 0, "Nb": 1, "Ta": 2}


@pytest.mark.parametrize(
    "cutoff_dict",
    [
        {"MoNb": 3, "Mo*": 4},
        {"Mo*": 4, "MoNb": 3},  # order of the keys does not matter
        {"Mo*": 4, "NbMo": 3},  # nor the order of the elements in the pair
    ],
)
def test_explicit_pair_wins_over_wildcard(cutoff_dict):
    cut = process_cutoff_dict(cutoff_dict, ELEMENTS)
    assert cut[("Mo", "Nb")] == 3 and cut[("Nb", "Mo")] == 3
    assert cut[("Mo", "Ta")] == 4 and cut[("Ta", "Mo")] == 4
    assert cut[("Mo", "Mo")] == 4
    assert ("Nb", "Ta") not in cut  # neither given nor covered by Mo*: uses `cutoff`


def test_single_element_key_and_input_not_modified():
    cutoff_dict = {"Ta": 5, "Mo*": 4}
    cut = process_cutoff_dict(cutoff_dict, ELEMENTS)
    assert cut[("Ta", "Ta")] == 5
    assert cutoff_dict == {"Ta": 5, "Mo*": 4}


def test_preset_takes_the_mean_of_the_element_radii():
    cut = process_cutoff_dict("CUTOFF_2L", ELEMENTS)
    for a in ELEMENTS:
        for b in ELEMENTS:
            assert cut[(a, b)] == pytest.approx((cut[(a, a)] + cut[(b, b)]) / 2)


def test_unknown_preset_name_is_a_clear_error():
    with pytest.raises(ValueError, match="Unknown cutoff_dict preset 'CUTOFF_3L'"):
        process_cutoff_dict("CUTOFF_3L", ELEMENTS)
