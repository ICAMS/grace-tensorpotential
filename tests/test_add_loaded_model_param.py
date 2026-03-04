import logging
import os
import tempfile

import pytest
import yaml

from tensorpotential.cli.gracemaker import add_loaded_model_parameter
from tensorpotential import constants as tc


def _write_model_yaml(instructions, tmpdir):
    """Write a minimal model.yaml from a list of instruction dicts."""
    path = os.path.join(tmpdir, "model.yaml")
    with open(path, "w") as f:
        yaml.dump({ins["name"]: ins for ins in instructions}, f)
    return path


def _base_args_yaml(cutoff="n/a"):
    return {
        tc.INPUT_CUTOFF: cutoff,
        tc.INPUT_POTENTIAL_SECTION: {},
        tc.INPUT_FIT_SECTION: {},
    }


class TestAddLoadedModelParameter:
    """Tests for cutoff extraction in add_loaded_model_parameter."""

    def test_rcut_extracted(self, tmp_path):
        """Standard case: instruction has 'rcut' key (e.g. RadialBasis)."""
        instructions = [
            {
                "__cls__": "tensorpotential.instructions.compute.RadialBasis",
                "name": "RadialBasis",
                "rcut": 7.0,
            }
        ]
        model_path = _write_model_yaml(instructions, tmp_path)
        args = _base_args_yaml(cutoff="n/a")
        args = add_loaded_model_parameter(model_path, args)
        assert args[tc.INPUT_CUTOFF] == 7.0

    def test_cutoff_extracted_when_no_rcut(self, tmp_path):
        """New case: instruction has 'cutoff' (not 'rcut'), e.g. SMAX models."""
        instructions = [
            {
                "__cls__": "tensorpotential.instructions.compute.BondSpecificRadialBasisFunction",
                "name": "BondSpecificRadialBasisFunction",
                "cutoff": 6.0,
                "cutoff_dict": "CUTOFF_2L",
            }
        ]
        model_path = _write_model_yaml(instructions, tmp_path)
        args = _base_args_yaml(cutoff="n/a")
        args = add_loaded_model_parameter(model_path, args)
        assert args[tc.INPUT_CUTOFF] == 6.0

    def test_cutoff_dict_extracted(self, tmp_path):
        """cutoff_dict should be propagated to args_yaml."""
        instructions = [
            {
                "__cls__": "tensorpotential.instructions.compute.BondSpecificRadialBasisFunction",
                "name": "BondSpecificRadialBasisFunction",
                "cutoff": 6.0,
                "cutoff_dict": "CUTOFF_2L",
            }
        ]
        model_path = _write_model_yaml(instructions, tmp_path)
        args = _base_args_yaml(cutoff="n/a")
        args = add_loaded_model_parameter(model_path, args)
        assert args[tc.INPUT_CUTOFF_DICT] == "CUTOFF_2L"

    def test_user_cutoff_overridden_with_warning(self, tmp_path, caplog):
        """User provides a numeric cutoff that differs from model — should warn."""
        instructions = [
            {
                "__cls__": "tensorpotential.instructions.compute.BondSpecificRadialBasisFunction",
                "name": "BondSpecificRadialBasisFunction",
                "cutoff": 6.0,
            }
        ]
        model_path = _write_model_yaml(instructions, tmp_path)
        args = _base_args_yaml(cutoff=5.0)  # user sets different cutoff
        with caplog.at_level(logging.WARNING):
            args = add_loaded_model_parameter(model_path, args)
        assert args[tc.INPUT_CUTOFF] == 6.0  # model wins
        assert "differs from foundation model cutoff" in caplog.text

    def test_user_cutoff_same_no_warning(self, tmp_path, caplog):
        """User provides same cutoff as model — no warning."""
        instructions = [
            {
                "__cls__": "tensorpotential.instructions.compute.BondSpecificRadialBasisFunction",
                "name": "BondSpecificRadialBasisFunction",
                "cutoff": 6.0,
            }
        ]
        model_path = _write_model_yaml(instructions, tmp_path)
        args = _base_args_yaml(cutoff=6.0)
        with caplog.at_level(logging.WARNING):
            args = add_loaded_model_parameter(model_path, args)
        assert args[tc.INPUT_CUTOFF] == 6.0
        assert "differs" not in caplog.text

