import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from tensorpotential.uq.feature_extraction import (
    FeatureBuffer,
    extract_features,
    extract_features_bulk,
    batch_feature_chunks,
    setup_feature_calculator,
)
from tensorpotential.uq import constants as uqc


def test_feature_buffer():
    """Test FeatureBuffer capacity growth and slicing."""
    dim = 4
    buf = FeatureBuffer(feature_dim=dim, capacity=10)

    # Fill partially
    f1 = np.ones((6, dim))
    e1 = np.zeros(6, dtype=np.int32)
    buf.append(f1, e1)
    assert len(buf) == 6

    # Trigger growth
    f2 = np.ones((10, dim)) * 2
    e2 = np.ones(10, dtype=np.int32)
    buf.append(f2, e2)
    assert len(buf) == 16
    assert buf._capacity >= 16

    # Verify contents
    assert np.allclose(buf.features[6:16], f2)
    assert np.all(buf.elements[6:16] == 1)

    # Test iteration
    chunks = list(buf.iter_chunks(chunk_size=7))
    assert len(chunks) == 3  # 7, 7, 2
    assert chunks[0][0].shape == (7, dim)
    assert chunks[2][0].shape == (2, dim)


def test_extract_features(uq_setup):
    """Test feature extraction generators with a mocked-checkpoint model."""
    calc = setup_feature_calculator(
        uq_setup["model_yaml"],
        uq_setup["checkpoint"],
        feature_spec=uq_setup["feature_spec"],
    )
    atoms = uq_setup["atoms"][:3]

    # 1. extract_features generator
    gen = extract_features(calc, atoms, element_map=uq_setup["element_map"])
    results = list(gen)
    assert len(results) == 3
    assert results[0][0].shape[1] == uq_setup["feature_dim"]

    # 2. extract_features_bulk collector
    all_f, all_e = extract_features_bulk(
        calc, atoms, element_map=uq_setup["element_map"]
    )
    total_at = sum(len(at) for at in atoms)
    assert all_f.shape == (total_at, uq_setup["feature_dim"])
    assert all_e.shape == (total_at,)


def test_batch_feature_chunks():
    """Test re-batching of per-structure features into fixed-size chunks."""

    def mock_gen():
        yield np.ones((3, 2)), np.zeros(3)
        yield np.ones((5, 2)), np.zeros(5)
        yield np.ones((2, 2)), np.zeros(2)

    chunks = list(batch_feature_chunks(mock_gen(), chunk_size=7))
    # 1st chunk gathers 3 + 5 = 8 atoms
    # 2nd chunk gathers remaining 2 atoms
    assert len(chunks) == 2
    assert chunks[0][0].shape == (8, 2)
    assert chunks[1][0].shape == (2, 2)


def test_feature_transform_contract(uq_setup):
    """Pin the feature-transform contract.

    - ``make_basis_rp_spec`` defaults to ``DEFAULT_FEATURE_TRANSFORM`` (``None`` —
      the production uqv6 feature uses no element-wise transform), so the
      ``uq_feature_transform`` key is omitted by default.
    - ``feature_transform="asinh"`` (the uqv4 feature) stamps the key verbatim.
    - ``feature_transform=None`` (default / legacy linear) omits the key entirely
      so old artifacts keep their exact key set.
    - ``_basis_rp_spec_from_artifact`` reads the key back; an absent key -> None.
    - The instruction applies element-wise asinh (vs identity for None) and
      rejects unknown transforms.
    """
    from tensorpotential.uq.factories import (
        make_basis_rp_spec,
        _basis_rp_spec_from_artifact,
    )
    from tensorpotential.uq.constants import DEFAULT_FEATURE_TRANSFORM
    from tensorpotential.uq.instructions import RandomProjectedBasisFeatures

    # The default transform is None (uqv6) -> key omitted.
    assert DEFAULT_FEATURE_TRANSFORM is None
    spec = make_basis_rp_spec(uq_setup["model_yaml"], rp_dim=16)
    assert uqc.UQ_FEATURE_TRANSFORM not in spec
    # Explicit asinh (uqv4) stamps the key ...
    spec_asinh = make_basis_rp_spec(
        uq_setup["model_yaml"], rp_dim=16, feature_transform="asinh"
    )
    assert str(np.asarray(spec_asinh[uqc.UQ_FEATURE_TRANSFORM]).item()) == "asinh"
    # ... and explicit linear omits the key (legacy artifacts).
    spec_linear = make_basis_rp_spec(
        uq_setup["model_yaml"], rp_dim=16, feature_transform=None
    )
    assert uqc.UQ_FEATURE_TRANSFORM not in spec_linear

    # Round-trip through the artifact reader (uses .extra_data on the model).
    class _Stub:
        def __init__(self, extra):
            self.extra_data = extra

    asinh_spec = _basis_rp_spec_from_artifact(_Stub(spec_asinh))
    assert asinh_spec["transform"] == "asinh"
    default_spec = _basis_rp_spec_from_artifact(_Stub(spec))
    assert default_spec["transform"] is None  # absent key -> None
    linear_spec = _basis_rp_spec_from_artifact(_Stub(spec_linear))
    assert linear_spec["transform"] is None  # absent key -> legacy linear

    # The instruction applies asinh element-wise vs identity for None.
    x = tf.constant([[-3.0, 0.0, 2.0, 50.0]], dtype=tf.float64)
    asinh_instr = RandomProjectedBasisFeatures([], out_dim=4, feature_transform="asinh")
    asinh_instr.projection = tf.eye(4, dtype=tf.float64)  # stub dtype carrier
    np.testing.assert_allclose(
        asinh_instr._apply_feature_transform(x).numpy(), np.arcsinh(x.numpy())
    )
    linear_instr = RandomProjectedBasisFeatures([], out_dim=4, feature_transform=None)
    np.testing.assert_array_equal(
        linear_instr._apply_feature_transform(x).numpy(), x.numpy()
    )

    # Unknown transforms are rejected at construction.
    import pytest

    with pytest.raises(ValueError, match="feature_transform must be one of"):
        RandomProjectedBasisFeatures([], out_dim=4, feature_transform="rmsblock_asinh")


def test_batched_feature_iterator(uq_setup):
    """Test the optimized batched iterator (requires StreamingDatasetWrapper logic)."""
    from tensorpotential.uq.feature_extraction import batched_feature_iterator
    from tensorpotential.instructions import load_instructions
    from tensorpotential.tensorpot import TensorPotential

    # We need a decorated model for this iterator
    from tensorpotential.uq.factories import patch_instructions_for_basis_rp_features

    instructions = load_instructions(uq_setup["model_yaml"])
    # UQ feature = basis-RP projection of the invariant energy-path basis,
    # written under the canonical FEATURES key by the appended instruction.
    patch_instructions_for_basis_rp_features(
        instructions, out_dim=uq_setup["feature_spec"]["out_dim"]
    )

    tp = TensorPotential(instructions, param_dtype=tf.float64)
    tp.load_checkpoint(uq_setup["checkpoint"])
    tp.model.decorate_compute_function()
    # Ensure the compute function returns the feature in the output dict
    tp.model.compute_function.extra_return_keys = [uqc.FEATURES]

    atoms = uq_setup["atoms"][:5]
    it = batched_feature_iterator(
        atoms, tp.model, element_map=uq_setup["element_map"], cutoff=6.0, verbose=False
    )

    results = list(it)
    assert len(results) > 0
    total_collected = sum(len(f) for f, e, w in results)
    total_expected = sum(len(at) for at in atoms)
    assert total_collected == total_expected
    assert results[0][0].shape[1] == uq_setup["feature_dim"]
    # Default weight is 1.0 when atoms.info has no UQ_WEIGHT tag.
    assert all(np.all(w == 1.0) for _, _, w in results)
