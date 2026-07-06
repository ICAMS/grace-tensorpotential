import os
import sys
from unittest.mock import patch

import numpy as np
import tensorflow as tf
import pytest

from tensorpotential.scripts import grace_utils
from tensorpotential.potentials import get_preset
from tensorpotential.instructions.base import save_instructions_dict


@pytest.fixture
def model_yaml(tmp_path):
    # Create a simple 1L model
    preset = get_preset("GRACE_1LAYER_latest")
    config = {
        "element_map": {"H": 0},
        "rcut": 4.5,
        "n_rad_max": 4,
        "lmax": 2,
    }
    instructor = preset(**config)
    instructions = instructor.get_instructions()
    path = tmp_path / "model.yaml"
    save_instructions_dict(str(path), instructions, param_dtype=tf.float32)
    return path


@pytest.fixture
def mock_checkpoint(tmp_path, model_yaml):
    # Create a minimal checkpoint
    from tensorpotential import TensorPotential

    tp = TensorPotential(potential=str(model_yaml), param_dtype=tf.float32)
    cp_path = tmp_path / "checkpoint"
    tp.save_checkpoint(checkpoint_name=str(cp_path))
    return cp_path


def test_export_aux_functions(tmp_path, model_yaml, mock_checkpoint):
    output_model = tmp_path / "exported_model"

    # Run grace_utils export
    args = [
        "-p",
        str(model_yaml),
        "-c",
        str(mock_checkpoint),
        "export",
        "-n",
        str(output_model),
    ]

    import sys
    from unittest.mock import patch

    with patch.object(sys, "argv", ["grace_utils"] + args):
        grace_utils.main()

    # Load and verify SaveModel signatures
    assert os.path.exists(output_model)
    loaded = tf.saved_model.load(str(output_model))

    # Verify primary compute is there
    assert "compute" in loaded.signatures

    # Verify auxiliary compute_energy is there
    # (save_model_with_aux_computes adds compute_energy if CreateOutputTarget atomic_energy exists)
    assert "compute_energy" in loaded.signatures


def test_export_kokkos_1l(tmp_path, model_yaml, mock_checkpoint):
    output = tmp_path / "weights.npz"
    args = [
        "-p", str(model_yaml),
        "-c", str(mock_checkpoint),
        "export_kokkos",
        "-o", str(output),
    ]
    with patch.object(sys, "argv", ["grace_utils"] + args):
        grace_utils.main()

    assert output.exists()
    data = np.load(str(output), allow_pickle=False)
    files = set(data.files)

    # Layout sentinels
    assert "n_elements" in files
    assert "element_names" in files
    assert "nn_dtype" in files
    # 1L flat-prefix layout — top-level mlp_rad_* keys
    assert "mlp_rad_n_layers" in files
    # 2L-only sentinels MUST be absent
    assert "mlp_rad_names" not in files
    assert "rms_names" not in files

    # Float arrays were promoted to float64 regardless of model dtype
    assert int(data["nn_dtype"][0]) == 0
    weight_keys = [k for k in files if data[k].dtype.kind == "f"]
    assert weight_keys, "expected at least one float weight array"
    for k in weight_keys:
        assert data[k].dtype == np.float64, f"{k} not promoted to float64"


def test_collapse_shifts_captures_non_unit_scale_and_pre_scales_upstream_shifts():
    """`_collapse_shifts` must return the non-unit scale and pre-multiply
    contributions of `TrainableShiftTarget` instructions that sit upstream
    of the scaling instruction (since the live model would apply scale to
    them). Shifts downstream stay unscaled."""
    from tensorpotential.scripts._kokkos_export import _collapse_shifts

    class CreateOutputTarget:
        def __init__(self, name):
            self.name = name
            self.target = None

    class LinMLPOut2ScalarTarget:
        def __init__(self, target):
            self.target = target

    class TrainableShiftTarget:
        def __init__(self, target, at_shifts):
            self.target = target
            self._a = np.asarray(at_shifts, dtype=np.float64).reshape(-1, 1)

        @property
        def at_shifts(self):
            class _T:
                def __init__(self, x):
                    self._x = x
                def numpy(self):
                    return self._x
            return _T(self._a)

    class _Scalar:
        def __init__(self, v):
            self._v = float(v)
        def numpy(self):
            return np.array(self._v)

    class ConstantScaleShiftTarget:
        def __init__(self, target, scale):
            self.target = target
            self.scale = _Scalar(scale)
            self.constant_shift = 0
            self.atomic_shift_map = None
            self.chemical_embedding = None

    out = CreateOutputTarget("atomic_energy")
    linmlp = LinMLPOut2ScalarTarget(target=out)
    upstream_shift = TrainableShiftTarget(target=linmlp, at_shifts=[1.0, 2.0])
    scaler = ConstantScaleShiftTarget(target=upstream_shift, scale=2.5)
    downstream_shift = TrainableShiftTarget(target=scaler, at_shifts=[10.0, 20.0])

    instr = {
        "atomic_energy": out,
        "LinMLPOut2ScalarTarget": linmlp,
        "TrainableShiftTarget_upstream": upstream_shift,
        "ConstantScaleShiftTarget": scaler,
        "TrainableShiftTarget_downstream": downstream_shift,
    }

    shift_values, contributions, output_scale = _collapse_shifts(
        instr, n_elements=2, chem_embedding_np=None
    )
    assert output_scale == 2.5
    # upstream shift pre-multiplied by scale, downstream shift kept as-is
    np.testing.assert_array_equal(shift_values, [1.0 * 2.5 + 10.0, 2.0 * 2.5 + 20.0])
    assert any("upstream" in c and "× scale" in c for c in contributions)
    assert any("downstream" in c for c in contributions)


def _run_export_kokkos(output, model_yaml, mock_checkpoint, extra_args=()):
    args = [
        "-p", str(model_yaml),
        "-c", str(mock_checkpoint),
        "export_kokkos",
        "-o", str(output),
        *extra_args,
    ]
    with patch.object(sys, "argv", ["grace_utils"] + args):
        grace_utils.main()


def _make_uq_artifact(path, feature_dim, n_clusters=2, n_model_elems=1, rp_dim=None):
    """Write a minimal basis-RP (schema-v3) gmm_artifacts.npz for element H(0).

    The feature dim ``D`` is the basis-RP projection out_dim, decoupled from any
    readout. ``rp_dim`` defaults to ``feature_dim`` (the consistent case); pass a
    different value to fabricate an inconsistent artifact (D != stored rp_dim).
    """
    from tensorpotential.uq.artifact_builder import GMMUQArtifactBuilder
    from tensorpotential.uq import constants as uqc

    rng = np.random.default_rng(0)
    K, D = n_clusters, feature_dim
    rp_dim = D if rp_dim is None else rp_dim
    centroids = rng.standard_normal((K, D)).astype(np.float64)
    inv_cov = np.broadcast_to(np.eye(D), (K, D, D)).astype(np.float64).copy()
    counts = np.full(K, 100, dtype=np.int64)
    artifacts = {
        0: {
            uqc.CENTROIDS: centroids,
            uqc.INV_COV: inv_cov,
            uqc.COUNTS: counts,
        }
    }
    # basis-RP spec: a verbatim projection R[D_basis, rp_dim] (D_basis arbitrary
    # for the export, which just copies R through) + feature mode / dim / seed.
    d_basis = 32
    R = (rng.standard_normal((d_basis, rp_dim)) / np.sqrt(rp_dim)).astype(np.float32)
    GMMUQArtifactBuilder.save_artifacts(
        str(path),
        artifacts,
        interp_thresholds=np.full((n_model_elems, K), 3.0, dtype=np.float64),
        element_map=np.array(["H"], dtype="S2"),
        # basis-RP self-describing spec (schema v3)
        **{
            uqc.UQ_FEATURE_MODE: np.array(uqc.FEATURE_MODE_BASIS_RP),
            uqc.UQ_RP_MATRIX: R,
            uqc.UQ_RP_DIM: np.int64(rp_dim),
            uqc.UQ_RP_SEED: np.int64(42),
        },
    )


def test_export_kokkos_with_uq(tmp_path, model_yaml, mock_checkpoint):
    """`--uq-artifacts` bakes dense uq_* arrays + the basis-RP matrix into the
    kokkos .npz."""
    from tensorpotential.uq import constants as uqc

    D = 16  # basis-RP projection out_dim (decoupled from the readout)
    K = 2
    artifact = tmp_path / "gmm_artifacts.npz"
    _make_uq_artifact(artifact, feature_dim=D, n_clusters=K)

    output = tmp_path / "weights_uq.npz"
    _run_export_kokkos(
        output, model_yaml, mock_checkpoint,
        extra_args=["--uq-artifacts", str(artifact)],
    )

    data = np.load(str(output), allow_pickle=False)
    files = set(data.files)

    # Dense UQ arrays present with the expected shapes.
    assert data["uq_centroids"].shape == (1, K, D)
    assert data["uq_inv_cov"].shape == (1, K, D, D)
    assert data["uq_interp_thresholds"].shape == (1, K)
    assert int(data["uq_n_clusters"][0]) == K
    assert int(data["uq_feature_dim"][0]) == D
    # _make_uq_artifact stamps a linear basis-RP spec (no normalize/density/transform
    # flags) → schema v3, regardless of the latest writer default (uqv6).
    assert int(data["uq_schema_version"][0]) == uqc.SCHEMA_VERSION_LINEAR
    assert int(data["uq_n_elements"][0]) == 1
    assert int(data["uq_max_clusters"][0]) == K

    # basis-RP projection baked in: R[D_basis, rp_dim] + dim + seed.
    assert int(data["uq_rp_dim"][0]) == D
    assert data["uq_rp_matrix"].shape[1] == D
    assert int(data["uq_rp_seed"][0]) == 42

    # UQ float arrays are stored float32 (the C++ npz_get_double must be
    # word_size-aware to read them; see the LAMMPS task spec).
    assert data["uq_centroids"].dtype == np.float32
    assert data["uq_inv_cov"].dtype == np.float32
    assert data["uq_interp_thresholds"].dtype == np.float32
    assert data["uq_rp_matrix"].dtype == np.float32

    # The error model has been removed: no eps_hat / tau_e / a_k,c_k in the
    # kokkos export.
    assert "uq_has_error_model" not in files
    assert "uq_a_per_cluster" not in files
    assert "uq_c_per_cluster" not in files
    assert "uq_tau_e" not in files
    assert "uq_tau_e_quantile" not in files

    # Values round-trip vs a freshly built GMMUQModel.
    from tensorpotential.uq.gmmuq import GMMUQModel

    m = GMMUQModel(str(artifact))
    # Stored float32, so compare against the float32 view of the model stats.
    np.testing.assert_allclose(data["uq_centroids"], m.centroids.numpy().astype(np.float32))
    np.testing.assert_allclose(data["uq_inv_cov"], m.inv_covs.numpy().astype(np.float32))
    np.testing.assert_allclose(
        data["uq_interp_thresholds"], m.interp_thresholds.numpy().astype(np.float32)
    )
    np.testing.assert_allclose(
        data["uq_rp_matrix"], np.asarray(m.extra_data[uqc.UQ_RP_MATRIX]).astype(np.float32)
    )


def test_export_kokkos_uq_dim_mismatch_raises(tmp_path, model_yaml, mock_checkpoint):
    """A basis-RP artifact whose feature dim != the stored projection rp_dim is
    rejected (an inconsistent artifact)."""
    D = 16
    artifact = tmp_path / "gmm_bad_dim.npz"
    _make_uq_artifact(artifact, feature_dim=D, rp_dim=D + 1)  # D != stored rp_dim

    output = tmp_path / "weights_bad.npz"
    with pytest.raises(ValueError, match="feature dim"):
        _run_export_kokkos(
            output, model_yaml, mock_checkpoint,
            extra_args=["--uq-artifacts", str(artifact)],
        )


def test_export_kokkos_uq_rejects_hidden_artifact(tmp_path, model_yaml, mock_checkpoint):
    """A legacy artifact lacking the basis-RP spec is rejected with a rebuild
    message (the hidden-layer UQ feature is retired)."""
    from tensorpotential.uq.artifact_builder import GMMUQArtifactBuilder
    from tensorpotential.uq import constants as uqc

    D, K = 16, 2
    rng = np.random.default_rng(0)
    artifacts = {
        0: {
            uqc.CENTROIDS: rng.standard_normal((K, D)).astype(np.float64),
            uqc.INV_COV: np.broadcast_to(np.eye(D), (K, D, D)).astype(np.float64).copy(),
            uqc.COUNTS: np.full(K, 100, dtype=np.int64),
        }
    }
    artifact = tmp_path / "gmm_hidden.npz"
    GMMUQArtifactBuilder.save_artifacts(
        str(artifact),
        artifacts,
        interp_thresholds=np.full((1, K), 3.0, dtype=np.float64),
        element_map=np.array(["H"], dtype="S2"),
    )  # no uq_feature_mode / uq_rp_matrix

    output = tmp_path / "weights_hidden.npz"
    with pytest.raises(ValueError, match="basis-RP"):
        _run_export_kokkos(
            output, model_yaml, mock_checkpoint,
            extra_args=["--uq-artifacts", str(artifact)],
        )


def test_export_kokkos_arch_override_routes_to_2l(
    tmp_path, model_yaml, mock_checkpoint
):
    """`--arch 2l` forces the 2L exporter regardless of auto-detect.
    On a 1L model that means we get the named-prefix layout with empty *_names
    index arrays — confirming the override is wired through to the dispatch."""
    output = tmp_path / "forced_2l.npz"
    args = [
        "-p", str(model_yaml),
        "-c", str(mock_checkpoint),
        "export_kokkos",
        "-o", str(output),
        "--arch", "2l",
    ]
    with patch.object(sys, "argv", ["grace_utils"] + args):
        grace_utils.main()
    assert output.exists()
    data = np.load(str(output), allow_pickle=False)
    # 2L exporter always writes the empty *_names index arrays
    assert "mlp_rad_names" in data.files
    assert "rms_names" in data.files
