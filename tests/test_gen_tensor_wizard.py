"""Tests for tensorpotential.extra.gen_tensor.wizard — verification & YAML generation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ase import Atoms

import tensorpotential.cli.wizard as _cli_wizard
from tensorpotential.extra.gen_tensor.wizard import (
    _SYMMETRY_CHECKS,
    _VOIGT6_TO_FULL9_IDX,
    _WizardStateTensor,
    _apply_state_tensor,
    _check_tensor_shapes,
    _check_tensor_symmetry,
    _collect_tensor_arrays,
    _generate_tensor_property_interactive,
    _section_verify_dataset,
    _verify_one_df,
)


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------

def _atoms(n=4):
    return Atoms("H" * n, positions=np.random.randn(n, 3), cell=3 * np.eye(3), pbc=True)


def _make_state(components, *, per_structure=False, compute_energy=False, compute_forces=False):
    s = _WizardStateTensor()
    s.tensor_components = components
    s.tensor_rank = 1 if components == [1] else 2
    s.per_structure = per_structure
    s.compute_energy = compute_energy
    s.compute_forces = compute_forces
    return s


def _symmetric_traceless(n):
    """Return (n, 9) array of symmetric traceless matrices."""
    out = []
    for _ in range(n):
        M = np.random.randn(3, 3)
        M = (M + M.T) / 2
        M -= np.eye(3) * np.trace(M) / 3
        out.append(M.flatten())
    return np.array(out)


def _symmetric(n):
    """Return (n, 9) array of symmetric (non-traceless) matrices."""
    out = []
    for _ in range(n):
        M = np.random.randn(3, 3)
        M = (M + M.T) / 2
        out.append(M.flatten())
    return np.array(out)


def _antisymmetric(n):
    """Return (n, 9) array of antisymmetric (traceless) matrices."""
    out = []
    for _ in range(n):
        M = np.random.randn(3, 3)
        M = (M - M.T) / 2
        out.append(M.flatten())
    return np.array(out)


def _df_with_tensor(tensor_fn, n_structs=5, n_atoms=4, extra_cols=None):
    rows = []
    for _ in range(n_structs):
        at = _atoms(n_atoms)
        row = {"ase_atoms": at, "tensor_property": tensor_fn(n_atoms)}
        if extra_cols:
            row.update(extra_cols(at))
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def silence_cli(monkeypatch):
    """Silence all CLI output helpers."""
    monkeypatch.setattr(_cli_wizard, "_info", lambda msg: None)
    monkeypatch.setattr(_cli_wizard, "_success", lambda msg: None)
    monkeypatch.setattr(_cli_wizard, "_section", lambda title: None)
    monkeypatch.setattr(_cli_wizard, "_ask_confirm", lambda msg, default=True: default)
    monkeypatch.setattr(_cli_wizard, "_ask_select", lambda msg, choices, default=None: default)
    monkeypatch.setattr(_cli_wizard, "_ask_text", lambda msg, default=None: default)


# ---------------------------------------------------------------------------
# _VOIGT6_TO_FULL9_IDX  — Voigt expansion correctness
# ---------------------------------------------------------------------------

class TestVoigtConversion:
    def test_indices_length(self):
        assert len(_VOIGT6_TO_FULL9_IDX) == 9

    def test_known_values(self):
        # ASE Voigt: [xx, yy, zz, yz, xz, xy] → full [xx, xy, xz, yx, yy, yz, zx, zy, zz]
        voigt = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])  # xx=1 yy=2 zz=3 yz=.1 xz=.2 xy=.3
        full = voigt[:, _VOIGT6_TO_FULL9_IDX]
        expected = np.array([[1.0, 0.3, 0.2, 0.3, 2.0, 0.1, 0.2, 0.1, 3.0]])
        np.testing.assert_allclose(full, expected)

    def test_symmetric_expansion_passes_symmetry_check(self):
        """Tensor expanded from Voigt should be symmetric."""
        rng = np.random.default_rng(0)
        voigt = rng.standard_normal((10, 6))
        full = voigt[:, _VOIGT6_TO_FULL9_IDX]  # (10, 9)
        T3 = full.reshape(10, 3, 3)
        diff = np.abs(T3 - T3.transpose(0, 2, 1)).max()
        assert diff < 1e-12, "Expanded Voigt tensor should be symmetric"

    def test_diagonal_preserved(self):
        """Diagonal elements (indices 0, 4, 8) should equal Voigt indices 0, 1, 2."""
        voigt = np.array([[10.0, 20.0, 30.0, 0.0, 0.0, 0.0]])
        full = voigt[:, _VOIGT6_TO_FULL9_IDX]
        assert full[0, 0] == 10.0  # xx
        assert full[0, 4] == 20.0  # yy
        assert full[0, 8] == 30.0  # zz


# ---------------------------------------------------------------------------
# _SYMMETRY_CHECKS  — mapping completeness
# ---------------------------------------------------------------------------

class TestSymmetryChecksMap:
    def test_all_tensor_types_present(self):
        for key in ("[1]", "[2]", "[0, 2]", "[1, 2]", "[0, 1, 2]"):
            assert key in _SYMMETRY_CHECKS

    def test_vector_no_checks(self):
        traceless, symmetric, antisymmetric = _SYMMETRY_CHECKS[str([1])]
        assert not any([traceless, symmetric, antisymmetric])

    def test_bec_no_checks(self):
        traceless, symmetric, antisymmetric = _SYMMETRY_CHECKS[str([0, 1, 2])]
        assert not any([traceless, symmetric, antisymmetric])

    def test_efg_symmetric_and_traceless(self):
        traceless, symmetric, antisymmetric = _SYMMETRY_CHECKS[str([2])]
        assert traceless and symmetric and not antisymmetric

    def test_stress_symmetric_only(self):
        traceless, symmetric, antisymmetric = _SYMMETRY_CHECKS[str([0, 2])]
        assert not traceless and symmetric and not antisymmetric

    def test_antisym_traceless(self):
        traceless, symmetric, antisymmetric = _SYMMETRY_CHECKS[str([1, 2])]
        assert traceless and not symmetric and antisymmetric


# ---------------------------------------------------------------------------
# _collect_tensor_arrays
# ---------------------------------------------------------------------------

class TestCollectTensorArrays:
    def test_stacks_rows(self):
        df = _df_with_tensor(lambda n: np.ones((n, 9)), n_structs=3, n_atoms=4)
        result = _collect_tensor_arrays(df, "tensor_property", max_rows=3)
        assert result.shape == (12, 9)

    def test_1d_per_structure_row(self):
        rows = [{"tensor_property": np.ones(9)} for _ in range(3)]
        df = pd.DataFrame(rows)
        result = _collect_tensor_arrays(df, "tensor_property")
        assert result.shape == (3, 9)

    def test_3d_arrays_flattened(self):
        rows = [{"tensor_property": np.ones((4, 3, 3))} for _ in range(2)]
        df = pd.DataFrame(rows)
        result = _collect_tensor_arrays(df, "tensor_property")
        assert result.shape == (8, 9)

    def test_max_rows_respected(self):
        df = _df_with_tensor(lambda n: np.ones((n, 9)), n_structs=10, n_atoms=2)
        result = _collect_tensor_arrays(df, "tensor_property", max_rows=3)
        assert result.shape[0] == 6  # 3 structs × 2 atoms


# ---------------------------------------------------------------------------
# _check_tensor_shapes
# ---------------------------------------------------------------------------

class TestCheckTensorShapes:
    def test_rank1_per_atom_correct(self):
        df = _df_with_tensor(lambda n: np.random.randn(n, 3), n_atoms=4)
        s = _make_state([1])
        ok, issues = _check_tensor_shapes(df, s)
        assert ok
        assert issues == []

    def test_rank2_per_atom_correct(self):
        df = _df_with_tensor(lambda n: np.random.randn(n, 9), n_atoms=4)
        s = _make_state([2])
        ok, issues = _check_tensor_shapes(df, s)
        assert ok

    def test_rank1_per_atom_wrong_cols(self):
        # rank-1 expects (N,3) but gets (N,9)
        df = _df_with_tensor(lambda n: np.random.randn(n, 9), n_atoms=4)
        s = _make_state([1])
        ok, issues = _check_tensor_shapes(df, s)
        assert not ok
        assert len(issues) > 0

    def test_rank2_per_atom_wrong_cols(self):
        df = _df_with_tensor(lambda n: np.random.randn(n, 3), n_atoms=4)
        s = _make_state([2])
        ok, issues = _check_tensor_shapes(df, s)
        assert not ok

    def test_rank2_per_structure_correct(self):
        rows = [{"ase_atoms": _atoms(4), "tensor_property": np.random.randn(1, 9)}
                for _ in range(5)]
        df = pd.DataFrame(rows)
        s = _make_state([0, 2], per_structure=True)
        ok, issues = _check_tensor_shapes(df, s)
        assert ok

    def test_rank2_per_structure_wrong_n(self):
        # Per-structure expects N=1 but gets N=4
        rows = [{"ase_atoms": _atoms(4), "tensor_property": np.random.randn(4, 9)}
                for _ in range(5)]
        df = pd.DataFrame(rows)
        s = _make_state([0, 2], per_structure=True)
        ok, issues = _check_tensor_shapes(df, s)
        assert not ok

    def test_flat_1d_accepted_for_per_structure(self):
        # shape (9,) should be normalised to (1,9) → correct for per-structure
        rows = [{"ase_atoms": _atoms(4), "tensor_property": np.random.randn(9)}
                for _ in range(5)]
        df = pd.DataFrame(rows)
        s = _make_state([0, 2], per_structure=True)
        ok, _ = _check_tensor_shapes(df, s)
        assert ok

    def test_max_rows_limits_check(self):
        # Only 2 valid rows + 3 invalid; with max_rows=2 should see 0 issues
        valid = [{"ase_atoms": _atoms(4), "tensor_property": np.random.randn(4, 9)}
                 for _ in range(2)]
        invalid = [{"ase_atoms": _atoms(4), "tensor_property": np.random.randn(4, 3)}
                   for _ in range(3)]
        df = pd.DataFrame(valid + invalid)
        s = _make_state([2])
        ok, _ = _check_tensor_shapes(df, s, max_rows=2)
        assert ok


# ---------------------------------------------------------------------------
# _check_tensor_symmetry
# ---------------------------------------------------------------------------

class TestCheckTensorSymmetry:
    # ── Types with no checks ───────────────────────────────────────────────

    def test_rank1_vector_no_checks(self):
        df = _df_with_tensor(lambda n: np.random.randn(n, 3))
        s = _make_state([1])
        assert _check_tensor_symmetry(df, s) == []

    def test_bec_no_checks(self):
        df = _df_with_tensor(lambda n: np.random.randn(n, 9))
        s = _make_state([0, 1, 2])
        assert _check_tensor_symmetry(df, s) == []

    # ── [2]  EFG: symmetric + traceless ───────────────────────────────────

    def test_efg_valid_passes(self):
        df = _df_with_tensor(_symmetric_traceless)
        s = _make_state([2])
        results = _check_tensor_symmetry(df, s)
        labels = {r[0] for r in results}
        assert "Traceless |trace|" in labels
        assert "Symmetric |T−Tᵀ|" in labels
        for _, mx, _, passed in results:
            assert passed, f"Expected pass but max violation = {mx:.2e}"

    def test_efg_random_fails_both(self):
        rng = np.random.default_rng(42)
        df = _df_with_tensor(lambda n: rng.standard_normal((n, 9)))
        s = _make_state([2])
        results = _check_tensor_symmetry(df, s)
        assert len(results) == 2
        for _, _, _, passed in results:
            assert not passed

    def test_efg_symmetric_but_not_traceless_fails_trace(self):
        df = _df_with_tensor(_symmetric)  # symmetric but NOT traceless
        s = _make_state([2])
        results = _check_tensor_symmetry(df, s)
        passed_by_label = {r[0]: r[3] for r in results}
        assert passed_by_label["Symmetric |T−Tᵀ|"]
        assert not passed_by_label["Traceless |trace|"]

    # ── [0, 2]  Stress: symmetric only ────────────────────────────────────

    def test_stress_symmetric_passes(self):
        df = _df_with_tensor(_symmetric)
        s = _make_state([0, 2])
        results = _check_tensor_symmetry(df, s)
        assert len(results) == 1
        assert results[0][0] == "Symmetric |T−Tᵀ|"
        assert results[0][3]

    def test_stress_random_fails_symmetry(self):
        rng = np.random.default_rng(7)
        df = _df_with_tensor(lambda n: rng.standard_normal((n, 9)))
        s = _make_state([0, 2])
        results = _check_tensor_symmetry(df, s)
        assert len(results) == 1
        assert not results[0][3]

    def test_stress_does_not_check_traceless(self):
        df = _df_with_tensor(_symmetric)  # symmetric, non-traceless
        s = _make_state([0, 2])
        labels = {r[0] for r in _check_tensor_symmetry(df, s)}
        assert "Traceless |trace|" not in labels

    # ── [1, 2]  Antisymmetric + traceless ─────────────────────────────────

    def test_antisym_valid_passes(self):
        df = _df_with_tensor(_antisymmetric)
        s = _make_state([1, 2])
        results = _check_tensor_symmetry(df, s)
        assert len(results) == 2
        for _, mx, _, passed in results:
            assert passed, f"Expected pass, max={mx:.2e}"

    def test_antisym_random_fails(self):
        rng = np.random.default_rng(3)
        df = _df_with_tensor(lambda n: rng.standard_normal((n, 9)))
        s = _make_state([1, 2])
        for _, _, _, passed in _check_tensor_symmetry(df, s):
            assert not passed

    def test_antisym_checks_both_traceless_and_antisymmetric(self):
        df = _df_with_tensor(_antisymmetric)
        s = _make_state([1, 2])
        labels = {r[0] for r in _check_tensor_symmetry(df, s)}
        assert "Traceless |trace|" in labels
        assert "Antisymmetric |T+Tᵀ|" in labels

    # ── Per-structure tensors work too ────────────────────────────────────

    def test_per_structure_symmetric_passes(self):
        rows = []
        for _ in range(10):
            at = _atoms(4)
            M = np.random.randn(3, 3)
            M = (M + M.T) / 2
            rows.append({"ase_atoms": at, "tensor_property": M.reshape(1, 9)})
        df = pd.DataFrame(rows)
        s = _make_state([0, 2], per_structure=True)
        results = _check_tensor_symmetry(df, s)
        assert results[0][3]


# ---------------------------------------------------------------------------
# _generate_tensor_property_interactive
# ---------------------------------------------------------------------------

class TestGenerateTensorProperty:
    """All tests patch I/O helpers in tensorpotential.cli.wizard."""

    def _patch(self, monkeypatch, select_return=None, confirm_return=True):
        monkeypatch.setattr(_cli_wizard, "_info", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_success", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_ask_confirm", lambda msg, default=True: confirm_return)
        monkeypatch.setattr(
            _cli_wizard, "_ask_select",
            lambda msg, choices, default=None: select_return if select_return is not None else (choices[0] if choices else default),
        )

    def test_rank1_from_Nx3_source(self, monkeypatch):
        self._patch(monkeypatch, select_return="src")
        n = 4
        src = np.random.randn(n, 3)
        df = pd.DataFrame([{"ase_atoms": _atoms(n), "src": src}])
        s = _make_state([1])
        df_out = _generate_tensor_property_interactive(df, s)
        assert df_out is not None
        tp = np.asarray(df_out["tensor_property"].iloc[0])
        assert tp.shape == (n, 3)

    def test_rank2_from_Nx9_source(self, monkeypatch):
        self._patch(monkeypatch, select_return="src")
        n = 4
        df = pd.DataFrame([{"ase_atoms": _atoms(n), "src": np.random.randn(n, 9)}])
        s = _make_state([0, 1, 2])
        df_out = _generate_tensor_property_interactive(df, s)
        assert df_out is not None
        assert np.asarray(df_out["tensor_property"].iloc[0]).shape == (n, 9)

    def test_rank2_from_Nx3x3_source(self, monkeypatch):
        self._patch(monkeypatch, select_return="src")
        n = 4
        df = pd.DataFrame([{"ase_atoms": _atoms(n), "src": np.random.randn(n, 3, 3)}])
        s = _make_state([0, 2])
        df_out = _generate_tensor_property_interactive(df, s)
        assert df_out is not None
        assert np.asarray(df_out["tensor_property"].iloc[0]).shape == (n, 9)

    def test_rank2_from_3x3_per_structure(self, monkeypatch):
        self._patch(monkeypatch, select_return="src")
        at = _atoms(4)
        df = pd.DataFrame([{"ase_atoms": at, "src": np.random.randn(3, 3)}])
        s = _make_state([0, 2], per_structure=True)
        df_out = _generate_tensor_property_interactive(df, s)
        assert df_out is not None
        assert np.asarray(df_out["tensor_property"].iloc[0]).shape == (1, 9)

    def test_rank2_from_flat9_per_structure(self, monkeypatch):
        self._patch(monkeypatch, select_return="src")
        at = _atoms(4)
        df = pd.DataFrame([{"ase_atoms": at, "src": np.random.randn(9)}])
        s = _make_state([0, 2], per_structure=True)
        df_out = _generate_tensor_property_interactive(df, s)
        assert df_out is not None
        assert np.asarray(df_out["tensor_property"].iloc[0]).shape == (1, 9)

    def test_voigt6_expansion(self, monkeypatch):
        """Voigt (N,6) should expand to symmetric (N,9)."""
        self._patch(monkeypatch, select_return="src", confirm_return=True)
        n = 6
        voigt = np.zeros((n, 6))
        voigt[:, 0] = 1.0  # xx=1
        voigt[:, 1] = 2.0  # yy=2
        voigt[:, 2] = 3.0  # zz=3
        df = pd.DataFrame([{"ase_atoms": _atoms(n), "src": voigt}])
        s = _make_state([0, 2])
        df_out = _generate_tensor_property_interactive(df, s)
        assert df_out is not None
        full = np.asarray(df_out["tensor_property"].iloc[0])
        assert full.shape == (n, 9)
        # off-diagonal should be 0, diagonal [1,2,3]
        np.testing.assert_allclose(full[:, 0], 1.0)  # xx
        np.testing.assert_allclose(full[:, 4], 2.0)  # yy
        np.testing.assert_allclose(full[:, 8], 3.0)  # zz
        # result must be symmetric
        T3 = full.reshape(n, 3, 3)
        np.testing.assert_allclose(T3, T3.transpose(0, 2, 1), atol=1e-12)

    def test_voigt6_per_structure(self, monkeypatch):
        self._patch(monkeypatch, select_return="src", confirm_return=True)
        at = _atoms(4)
        df = pd.DataFrame([{"ase_atoms": at, "src": np.zeros(6)}])
        s = _make_state([0, 2], per_structure=True)
        df_out = _generate_tensor_property_interactive(df, s)
        assert df_out is not None
        assert np.asarray(df_out["tensor_property"].iloc[0]).shape == (1, 9)

    def test_scale_by_volume(self, monkeypatch):
        """Per-structure source normalised per volume → multiply by cell volume."""
        call_log = []

        def fake_select(msg, choices, default=None):
            if "normali" in msg.lower() or "source data" in msg.lower():
                call_log.append("scale")
                return "Intensive per volume (e.g. eV/Å³) → multiply by cell volume"
            return choices[0] if choices else default

        monkeypatch.setattr(_cli_wizard, "_info", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_success", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_ask_confirm", lambda msg, default=True: True)
        monkeypatch.setattr(_cli_wizard, "_ask_select", fake_select)

        at = _atoms(4)
        vol = at.get_volume()
        unit_tensor = np.ones(9)  # intensive: 1.0 per Å³
        df = pd.DataFrame([{"ase_atoms": at, "src": unit_tensor}])
        s = _make_state([0, 2], per_structure=True)
        df_out = _generate_tensor_property_interactive(df, s)
        assert df_out is not None
        result = np.asarray(df_out["tensor_property"].iloc[0])
        np.testing.assert_allclose(result, vol * np.ones((1, 9)), rtol=1e-10)
        assert "scale" in call_log

    def test_scale_by_natom(self, monkeypatch):
        """Per-structure source normalised per atom → multiply by n_atoms."""
        def fake_select(msg, choices, default=None):
            if "normali" in msg.lower() or "source data" in msg.lower():
                return "Intensive per atom (e.g. eV/atom) → multiply by number of atoms"
            return choices[0] if choices else default

        monkeypatch.setattr(_cli_wizard, "_info", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_success", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_ask_confirm", lambda msg, default=True: True)
        monkeypatch.setattr(_cli_wizard, "_ask_select", fake_select)

        n = 5
        at = _atoms(n)
        df = pd.DataFrame([{"ase_atoms": at, "src": np.ones(9)}])
        s = _make_state([0, 2], per_structure=True)
        df_out = _generate_tensor_property_interactive(df, s)
        assert df_out is not None
        result = np.asarray(df_out["tensor_property"].iloc[0])
        np.testing.assert_allclose(result, float(n) * np.ones((1, 9)), rtol=1e-10)

    def test_no_candidate_columns_returns_none(self, monkeypatch):
        monkeypatch.setattr(_cli_wizard, "_info", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_success", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_ask_confirm", lambda msg, default=True: True)
        monkeypatch.setattr(_cli_wizard, "_ask_select", lambda msg, choices, default=None: default)
        # Only ase_atoms and tensor_property — both excluded → no candidates
        df = pd.DataFrame([{"ase_atoms": _atoms(4), "tensor_property": np.random.randn(4, 9)}])
        s = _make_state([2])
        result = _generate_tensor_property_interactive(df, s)
        assert result is None

    def test_energy_forces_stress_columns_are_offered(self, monkeypatch):
        """energy, forces, stress etc. should appear as selectable source columns."""
        offered = []

        def fake_select(msg, choices, default=None):
            offered.extend(choices)
            return choices[0] if choices else default

        monkeypatch.setattr(_cli_wizard, "_info", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_success", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_ask_confirm", lambda msg, default=True: True)
        monkeypatch.setattr(_cli_wizard, "_ask_select", fake_select)

        n = 4
        df = pd.DataFrame([{
            "ase_atoms": _atoms(n),
            "energy": -1.0,
            "forces": np.random.randn(n, 3),
            "stress": np.zeros(6),
            "my_custom": np.random.randn(n, 9),
        }])
        s = _make_state([2])
        _generate_tensor_property_interactive(df, s)
        assert "energy" in offered
        assert "forces" in offered
        assert "stress" in offered
        assert "my_custom" in offered

    def test_unrecognised_shape_returns_none(self, monkeypatch):
        self._patch(monkeypatch, select_return="src")
        # shape (4, 7) — not recognised for rank-2
        df = pd.DataFrame([{"ase_atoms": _atoms(4), "src": np.random.randn(4, 7)}])
        s = _make_state([2])
        result = _generate_tensor_property_interactive(df, s)
        assert result is None

    def test_voigt_confirm_declined_returns_none(self, monkeypatch):
        """User declines the Voigt convention → None."""
        confirm_calls = [False]  # first call is the Voigt confirmation

        def fake_confirm(msg, default=True):
            return confirm_calls.pop(0) if confirm_calls else default

        monkeypatch.setattr(_cli_wizard, "_info", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_success", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_ask_confirm", fake_confirm)
        monkeypatch.setattr(_cli_wizard, "_ask_select", lambda msg, choices, default=None: "src")
        n = 4
        df = pd.DataFrame([{"ase_atoms": _atoms(n), "src": np.random.randn(n, 6)}])
        s = _make_state([2])
        result = _generate_tensor_property_interactive(df, s)
        assert result is None


# ---------------------------------------------------------------------------
# _verify_one_df
# ---------------------------------------------------------------------------

class TestVerifyOneDf:
    """Tests for the single-file verification helper."""

    def test_all_present_passes(self, silence_cli):
        df = _df_with_tensor(_symmetric_traceless)
        df["energy"] = -1.0
        df["forces"] = [np.zeros((4, 3))] * len(df)
        s = _make_state([2], compute_energy=True, compute_forces=True)
        _, ok = _verify_one_df(df, "dummy.pckl.gz", s, label="train")
        assert ok

    def test_missing_energy_fails(self, silence_cli):
        df = _df_with_tensor(_symmetric_traceless)
        s = _make_state([2], compute_energy=True)
        _, ok = _verify_one_df(df, "dummy.pckl.gz", s, label="train")
        assert not ok

    def test_missing_forces_fails(self, silence_cli):
        df = _df_with_tensor(_symmetric_traceless)
        df["energy"] = -1.0
        s = _make_state([2], compute_energy=True, compute_forces=True)
        _, ok = _verify_one_df(df, "dummy.pckl.gz", s, label="train")
        assert not ok

    def test_missing_tensor_property_user_declines_fails(self, monkeypatch):
        monkeypatch.setattr(_cli_wizard, "_info", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_success", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_ask_confirm", lambda msg, default=True: False)
        df = pd.DataFrame([{"ase_atoms": _atoms(4), "src": np.random.randn(4, 9)}])
        s = _make_state([2])
        _, ok = _verify_one_df(df, "dummy.pckl.gz", s, label="train")
        assert not ok

    def test_wrong_shape_fails(self, silence_cli):
        # rank-2 but (N, 3) data
        df = _df_with_tensor(lambda n: np.random.randn(n, 3))
        s = _make_state([2])
        _, ok = _verify_one_df(df, "dummy.pckl.gz", s, label="train")
        assert not ok

    def test_symmetry_violation_fails(self, silence_cli):
        rng = np.random.default_rng(99)
        df = _df_with_tensor(lambda n: rng.standard_normal((n, 9)))
        s = _make_state([2])  # expects symmetric + traceless
        _, ok = _verify_one_df(df, "dummy.pckl.gz", s, label="train")
        assert not ok

    def test_filename_unchanged_when_no_generation(self, silence_cli):
        df = _df_with_tensor(_symmetric)
        s = _make_state([0, 2])
        new_fname, _ = _verify_one_df(df, "original.pckl.gz", s, label="train")
        assert new_fname == "original.pckl.gz"

    def test_generates_and_saves_tensor_property(self, monkeypatch, tmp_path):
        """When tensor_property is missing and user accepts, file is saved and filename updated."""
        saved_to = tmp_path / "out.pckl.gz"

        def fake_confirm(msg, default=True):
            return True  # accept all prompts

        def fake_select(msg, choices, default=None):
            if choices:
                return choices[0]
            return default

        def fake_text(msg, default=None):
            return str(saved_to)

        monkeypatch.setattr(_cli_wizard, "_info", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_success", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_ask_confirm", fake_confirm)
        monkeypatch.setattr(_cli_wizard, "_ask_select", fake_select)
        monkeypatch.setattr(_cli_wizard, "_ask_text", fake_text)

        n = 4
        df = pd.DataFrame([{"ase_atoms": _atoms(n), "src": np.random.randn(n, 9)}])
        s = _make_state([2])
        new_fname, ok = _verify_one_df(df, "original.pckl.gz", s, label="train")

        assert ok
        assert new_fname == str(saved_to)
        assert saved_to.exists()
        df_saved = pd.read_pickle(str(saved_to))
        assert "tensor_property" in df_saved.columns


# ---------------------------------------------------------------------------
# _section_verify_dataset
# ---------------------------------------------------------------------------

class TestSectionVerifyDataset:
    """Integration tests for the top-level verification section."""

    def _base_monkeypatch(self, monkeypatch, do_verify=True):
        """Patch output helpers; set do_verify for the top-level confirm."""
        monkeypatch.setattr(_cli_wizard, "_info", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_success", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_section", lambda title: None)
        first_call = [True]

        def fake_confirm(msg, default=True):
            if first_call[0]:
                first_call[0] = False
                return do_verify
            return default

        monkeypatch.setattr(_cli_wizard, "_ask_confirm", fake_confirm)
        monkeypatch.setattr(_cli_wizard, "_ask_select", lambda msg, choices, default=None: default)
        monkeypatch.setattr(_cli_wizard, "_ask_text", lambda msg, default=None: default)

    def test_user_declines_returns_unchanged(self, monkeypatch, tmp_path):
        self._base_monkeypatch(monkeypatch, do_verify=False)
        s = _make_state([2])
        s.train_filename = "nonexistent.pckl.gz"
        result = _section_verify_dataset(s)
        assert result.train_filename == "nonexistent.pckl.gz"

    def test_train_only_verified(self, monkeypatch, tmp_path):
        """No separate test file: only train dataset is checked."""
        self._base_monkeypatch(monkeypatch)

        df = _df_with_tensor(_symmetric_traceless)
        train_path = tmp_path / "train.pckl.gz"
        df.to_pickle(str(train_path))

        s = _make_state([2])
        s.train_filename = str(train_path)
        s.use_separate_test = False

        result = _section_verify_dataset(s)
        assert result.train_filename == str(train_path)

    def test_train_and_test_both_verified(self, monkeypatch, tmp_path):
        """Both train and test files are loaded and verified."""
        self._base_monkeypatch(monkeypatch)

        df = _df_with_tensor(_symmetric_traceless)
        train_path = tmp_path / "train.pckl.gz"
        test_path = tmp_path / "test.pckl.gz"
        df.to_pickle(str(train_path))
        df.to_pickle(str(test_path))

        s = _make_state([2])
        s.train_filename = str(train_path)
        s.use_separate_test = True
        s.test_filename = str(test_path)

        result = _section_verify_dataset(s)
        assert result.train_filename == str(train_path)
        assert result.test_filename == str(test_path)

    def test_bad_train_path_survives_gracefully(self, monkeypatch):
        self._base_monkeypatch(monkeypatch)
        s = _make_state([2])
        s.train_filename = "/nonexistent/path.pckl.gz"
        s.use_separate_test = False
        # Should not raise; just report an issue
        result = _section_verify_dataset(s)
        assert result is s  # state returned unchanged on load failure

    def test_test_filename_updated_when_tensor_generated(self, monkeypatch, tmp_path):
        """If tensor_property is generated for test set, s.test_filename is updated."""
        # train has tensor_property; test does not
        df_train = _df_with_tensor(_symmetric_traceless)
        df_test = pd.DataFrame([{"ase_atoms": _atoms(4), "src": np.random.randn(4, 9)}])
        train_path = tmp_path / "train.pckl.gz"
        test_path = tmp_path / "test.pckl.gz"
        new_test_path = tmp_path / "test_with_tensor.pckl.gz"
        df_train.to_pickle(str(train_path))
        df_test.to_pickle(str(test_path))

        call_count = [0]

        def fake_confirm(msg, default=True):
            call_count[0] += 1
            return True  # accept everything

        def fake_select(msg, choices, default=None):
            if choices:
                return choices[0]
            return default

        def fake_text(msg, default=None):
            # Route the save-path prompt to our new_test_path
            return str(new_test_path)

        monkeypatch.setattr(_cli_wizard, "_info", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_success", lambda msg: None)
        monkeypatch.setattr(_cli_wizard, "_section", lambda title: None)
        monkeypatch.setattr(_cli_wizard, "_ask_confirm", fake_confirm)
        monkeypatch.setattr(_cli_wizard, "_ask_select", fake_select)
        monkeypatch.setattr(_cli_wizard, "_ask_text", fake_text)

        s = _make_state([2])
        s.train_filename = str(train_path)
        s.use_separate_test = True
        s.test_filename = str(test_path)

        result = _section_verify_dataset(s)

        assert result.test_filename == str(new_test_path)
        assert new_test_path.exists()


# ---------------------------------------------------------------------------
# _apply_state_tensor — YAML generation
# ---------------------------------------------------------------------------

class TestApplyStateTensor:
    def _base_state(self, **kwargs):
        s = _WizardStateTensor()
        s.train_filename = "data.pckl.gz"
        s.test_size = 0.05
        s.tensor_components = [0, 1, 2]
        s.tensor_rank = 2
        s.per_structure = False
        s.preset = "TENSOR_2L"
        s.cutoff = 6.0
        s.compute_energy = False
        s.compute_forces = False
        s.loss_type = "huber"
        s.tensor_delta = 0.1
        s.tensor_weight = 10.0
        s.energy_delta = 0.01
        s.energy_weight = 1.0
        s.forces_weight = 5.0
        s.batch_size = 10
        s.target_total_updates = 5000
        for k, v in kwargs.items():
            setattr(s, k, v)
        return s

    def test_no_placeholders_remain(self):
        s = self._base_state()
        yaml = _apply_state_tensor(s)
        assert "{" not in yaml or "{{" not in yaml  # only single-brace YAML dicts
        # Specifically check no unfilled Python f-string escapes remain
        assert "{{" not in yaml

    def test_cutoff_and_preset_present(self):
        s = self._base_state(cutoff=5.5, preset="TENSOR_1L")
        yaml = _apply_state_tensor(s)
        assert "cutoff: 5.5" in yaml
        assert "preset: TENSOR_1L" in yaml

    def test_tensor_components_written(self):
        s = self._base_state(tensor_components=[2])
        yaml = _apply_state_tensor(s)
        assert "tensor_components: [2]" in yaml

    def test_compute_energy_false_no_energy_loss(self):
        s = self._base_state(compute_energy=False)
        yaml = _apply_state_tensor(s)
        assert "energy:" not in yaml or "compute_energy: False" in yaml
        # The loss block should not have a top-level energy key
        loss_block = yaml.split("loss:")[1].split("target_total_updates:")[0]
        assert "energy:" not in loss_block

    def test_compute_energy_true_adds_energy_loss(self):
        s = self._base_state(compute_energy=True, compute_forces=False)
        yaml = _apply_state_tensor(s)
        loss_block = yaml.split("loss:")[1].split("target_total_updates:")[0]
        assert "energy:" in loss_block

    def test_compute_forces_adds_forces_loss(self):
        s = self._base_state(compute_energy=True, compute_forces=True)
        yaml = _apply_state_tensor(s)
        loss_block = yaml.split("loss:")[1].split("target_total_updates:")[0]
        assert "forces:" in loss_block

    def test_huber_delta_in_tensor_loss(self):
        s = self._base_state(loss_type="huber", tensor_delta=0.05)
        yaml = _apply_state_tensor(s)
        assert "delta: 0.05" in yaml

    def test_square_loss_no_delta(self):
        s = self._base_state(loss_type="square")
        yaml = _apply_state_tensor(s)
        # WeightedTensorLoss line should not contain delta
        tensor_loss_line = [
            l for l in yaml.splitlines() if "WeightedTensorLoss" in l
        ][0]
        assert "delta" not in tensor_loss_line

    def test_square_loss_energy_no_delta(self):
        s = self._base_state(loss_type="square", compute_energy=True)
        yaml = _apply_state_tensor(s)
        energy_line = [l for l in yaml.splitlines() if "energy:" in l and "weight" in l][0]
        assert "delta" not in energy_line

    def test_separate_test_file(self):
        s = self._base_state(use_separate_test=True, test_filename="test.pckl.gz")
        yaml = _apply_state_tensor(s)
        assert "test_filename: test.pckl.gz" in yaml
        assert "save_dataset: False" in yaml

    def test_split_test_fraction(self):
        s = self._base_state(use_separate_test=False, test_size=0.1)
        yaml = _apply_state_tensor(s)
        assert "test_size: 0.1" in yaml
        assert "save_dataset: True" in yaml

    def test_per_structure_propagated(self):
        s = self._base_state(per_structure=True)
        yaml = _apply_state_tensor(s)
        assert "per_structure: True" in yaml

    def test_batch_sizes(self):
        s = self._base_state(batch_size=8)
        yaml = _apply_state_tensor(s)
        assert "batch_size: 8" in yaml
        assert "test_batch_size: 40" in yaml

    def test_target_total_updates(self):
        s = self._base_state(target_total_updates=12345)
        yaml = _apply_state_tensor(s)
        assert "target_total_updates: 12345" in yaml

    def test_compute_forces_only_in_config_not_in_kwargs(self):
        """compute_forces must appear in compute_function_config but NOT in potential.kwargs."""
        s = self._base_state(compute_energy=True, compute_forces=True)
        yaml = _apply_state_tensor(s)
        kwargs_block = yaml.split("kwargs:")[1].split("}")[0]
        assert "compute_forces" not in kwargs_block
        config_block = yaml.split("compute_function_config:")[1].split("}")[0]
        assert "compute_forces" in config_block

    def test_rank1_vector(self):
        s = self._base_state(tensor_components=[1], tensor_rank=1)
        yaml = _apply_state_tensor(s)
        assert "tensor_components: [1]" in yaml
        assert "tensor_rank: 1" in yaml

    def test_weighted_tensor_loss_block_always_present(self):
        for components in ([1], [2], [0, 2], [1, 2], [0, 1, 2]):
            s = self._base_state(tensor_components=components)
            yaml = _apply_state_tensor(s)
            assert "WeightedTensorLoss" in yaml
