"""Tests for `predict_structures` — the shared bulk-evaluation loop.

Guards the two properties callers actually depend on:
  - structures are evaluated LARGEST FIRST (so the widest XLA shape compiles
    once), but results come back in the CALLER's order, and
  - the caller's Atoms are never given a calculator or otherwise mutated.
plus the extra-key harvesting and the error policies.
"""
import logging

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, PropertyNotImplementedError

from tensorpotential.calculator import predict_structures


class RecordingCalc(Calculator):
    """Energy = n_atoms, forces = zeros. Records the order it was called in and
    exposes a `gamma` in `results`, mimicking a UQ-enabled model."""

    # `stress` deliberately undeclared: ASE then raises its own
    # PropertyNotImplementedError, which is what a real stress-less calculator
    # does and what `predict_structures` tolerates
    implemented_properties = ["energy", "forces"]

    def __init__(self, fail_on=(), gamma_scale=0.5, **kw):
        super().__init__(**kw)
        self.seen = []
        self.fail_on = set(fail_on)
        self.gamma_scale = gamma_scale

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        n = len(atoms)
        self.seen.append(n)
        if n in self.fail_on:
            raise RuntimeError(f"boom for n_atoms={n}")
        self.results = {
            "energy": float(n),
            "forces": np.zeros((n, 3)),
            "gamma": np.full(n, self.gamma_scale * n),
        }


def _structures(sizes):
    return [Atoms("H" * n, positions=np.zeros((n, 3))) for n in sizes]


def test_results_are_in_input_order_but_evaluated_largest_first():
    sizes = [1, 5, 2, 4, 3]
    atoms = _structures(sizes)
    calc = RecordingCalc()
    out = predict_structures(atoms, calc, properties=("energy",))
    # energy == n_atoms, so this pins the alignment to the INPUT order
    assert out["energy"] == [float(n) for n in sizes]
    # ... while evaluation actually went largest -> smallest
    assert calc.seen == sorted(sizes, reverse=True)


def test_sort_can_be_disabled():
    sizes = [1, 5, 2]
    calc = RecordingCalc()
    predict_structures(_structures(sizes), calc, properties=("energy",),
                       sort_by_natoms=False)
    assert calc.seen == sizes


def test_caller_atoms_are_not_mutated():
    atoms = _structures([2, 3])
    predict_structures(atoms, RecordingCalc(), properties=("energy",))
    assert all(a.calc is None for a in atoms), "a calculator was left attached"


def test_extra_keys_are_harvested_and_missing_ones_are_none():
    atoms = _structures([2, 3])
    out = predict_structures(atoms, RecordingCalc(), properties=("energy",),
                             extra=("gamma", "not_provided_by_this_model"))
    assert [g.tolist() for g in out["gamma"]] == [[1.0, 1.0], [1.5, 1.5, 1.5]]
    assert out["not_provided_by_this_model"] == [None, None]


def test_optional_property_failure_yields_none_not_an_error():
    # stress is optional by default: a non-periodic cell has none, and that is
    # not a failure of the structure
    out = predict_structures(_structures([2]), RecordingCalc(),
                             properties=("energy", "stress"))
    assert out["energy"] == [2.0]
    assert out["stress"] == [None]


def test_non_optional_property_failure_propagates():
    with pytest.raises(PropertyNotImplementedError):
        predict_structures(_structures([2]), RecordingCalc(),
                           properties=("energy", "stress"), optional=())


def test_on_error_warn_records_none_and_keeps_going():
    sizes = [1, 5, 2]
    out = predict_structures(_structures(sizes), RecordingCalc(fail_on=(5,)),
                             properties=("energy",), on_error="warn")
    assert out["energy"] == [1.0, None, 2.0]   # the failure did not shift the rest


def test_on_error_raise_is_the_default():
    with pytest.raises(RuntimeError):
        predict_structures(_structures([1, 5]), RecordingCalc(fail_on=(5,)),
                           properties=("energy",))


def test_rejects_unknown_property_and_policy():
    with pytest.raises(ValueError, match="unknown propert"):
        predict_structures(_structures([1]), RecordingCalc(), properties=("nrg",))
    with pytest.raises(ValueError, match="on_error"):
        predict_structures(_structures([1]), RecordingCalc(), on_error="explode")


def test_empty_input_returns_empty_columns():
    out = predict_structures([], RecordingCalc(), properties=("energy",), extra=("gamma",))
    assert out == {"energy": [], "gamma": []}


def test_progress_is_called_once_per_structure():
    seen = []
    predict_structures(_structures([1, 2, 3]), RecordingCalc(), properties=("energy",),
                       progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_failure_warnings_are_capped_then_summarised(caplog):
    """The old grace_predict capped error output at 3 lines; a per-structure
    warning for every row of a broken dataset is unusable."""
    sizes = list(range(1, 11))
    with caplog.at_level(logging.WARNING, logger="tensorpotential.calculator.bulk"):
        predict_structures(_structures(sizes), RecordingCalc(fail_on=sizes),
                           properties=("energy",), on_error="warn")
    per_structure = [r for r in caplog.records if "n_atoms=" in r.getMessage()]
    assert len(per_structure) == 3, "per-structure warnings were not capped"
    assert any("failed for 10 of 10 structures" in r.getMessage() for r in caplog.records)


def test_a_real_not_implemented_error_is_not_mistaken_for_a_missing_property():
    """`optional` tolerates only ASE's PropertyNotImplementedError. A bare
    NotImplementedError from inside the model is a real failure, not "this
    calculator has no stress"."""

    class BrokenStress(RecordingCalc):
        implemented_properties = ["energy", "forces", "stress"]

        def get_stress(self, atoms=None):
            raise NotImplementedError("unimplemented branch deep in the model")

    with pytest.raises(NotImplementedError):
        predict_structures(_structures([2]), BrokenStress(),
                           properties=("energy", "stress"))


def test_postprocess_reduces_extra_values_per_structure():
    """postprocess sees one structure's extra values and may drop or add keys; new keys
    are aligned with the input order (None where it did not set them)."""
    sizes = [1, 4, 2]
    seen = []

    def post(values):
        seen.append(len(values["gamma"]))
        g = values.pop("gamma")
        if len(g) > 1:
            values["big"] = g[:1]
        return values

    out = predict_structures(_structures(sizes), RecordingCalc(), properties=("energy",),
                             extra=("gamma",), postprocess=post)
    assert seen == [4, 2, 1]  # largest first, one structure at a time
    assert out["gamma"] == [None, None, None]
    assert out["big"][0] is None
    np.testing.assert_allclose(out["big"][1], [2.0])
    np.testing.assert_allclose(out["big"][2], [1.0])
