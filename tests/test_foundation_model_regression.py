"""
Regression tests for foundation model prediction consistency.

These tests verify that:
1. Foundation models (grace_fm) produce consistent predictions matching hardcoded references.
2. Checkpoint reconstruction matches the frozen saved_model predictions.

Tests are skipped if the model files aren't cached locally.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest
from ase.build import bulk

from tensorpotential.calculator.foundation_models import (
    FOUNDATION_CACHE_DIR,
    FOUNDATION_CHECKPOINTS_CACHE_DIR,
    grace_fm,
)
from tensorpotential.calculator.asecalculator import TPCalculator
from tensorpotential.instructions import load_instructions
from tensorpotential.tensorpot import TensorPotential

import tensorflow as tf
tf.experimental.numpy.experimental_enable_numpy_behavior(dtype_conversion_mode="all")

# Reference values computed from frozen saved_model.pb exports.
# Test structure: Al FCC (a=4.05, cubic=True), atom 0 displaced by [0.1, 0, 0].
REFERENCE = {
    "GRACE-FS-OMAT": {
        "energy": -14.891636071538885,
        "forces": np.array([
            [-3.70896073e-01, -3.78169718e-16, -2.46764414e-16],
            [-3.43625022e-02, 2.63677968e-16, 1.28369537e-16],
            [2.02629288e-01, -4.81385765e-17, -1.93638508e-16],
            [2.02629288e-01, 1.60245081e-16, 3.25911173e-16],
        ]),
        "stress": np.array([
            -8.10779749e-03, -7.74958458e-03, -7.74958458e-03,
            -6.73728458e-18, 1.65037359e-17, -2.97693970e-18,
        ]),
    },
    "GRACE-1L-OMAT": {
        "energy": -14.964122964711601,
        "forces": np.array([
            [-4.23401486e-01, -1.12540186e-16, -2.19008839e-17],
            [-3.55439213e-02, 6.70036943e-17, 7.49617382e-16],
            [2.29472704e-01, 1.93421668e-16, -4.87023616e-16],
            [2.29472704e-01, -1.69243959e-16, -2.24646690e-16],
        ]),
        "stress": np.array([
            5.69415248e-03, 6.31726303e-03, 6.31726303e-03,
            -3.14406614e-17, -1.42057473e-17, 1.80836030e-17,
        ]),
    },
    "GRACE-2L-OMAT": {
        "energy": -14.955221575533013,
        "forces": np.array([
            [-4.06163420e-01, -1.83582533e-16, 2.60208521e-16],
            [-4.44705368e-02, 1.70653422e-16, -5.78530279e-16],
            [2.25316979e-01, -3.18755439e-17, 4.39156090e-16],
            [2.25316979e-01, 3.86789125e-17, -1.37910516e-16],
        ]),
        "stress": np.array([
            2.49289461e-03, 3.19660193e-03, 3.19660192e-03,
            6.22807121e-18, -6.47614952e-18, -2.17264371e-17,
        ]),
    },
    "GRACE-1L-OMAT-medium-base": {
        "energy": -14.94301636333494,
        "forces": np.array([
            [-4.23891176e-01, 3.73168835e-17, -6.48352899e-17],
            [-4.64184755e-02, 2.85348459e-17, 2.47652105e-16],
            [2.35154826e-01, -1.41217333e-17, -5.69477191e-17],
            [2.35154826e-01, -5.85536936e-17, -1.24683250e-16],
        ]),
        "stress": np.array([
            3.41204468e-03, 4.06133352e-03, 4.06133352e-03,
            -1.12483928e-17, 1.88017244e-18, 2.13086210e-17,
        ]),
    },
    "GRACE-1L-OMAT-medium-ft-E": {
        "energy": -14.974611895114293,
        "forces": np.array([
            [-4.28068105e-01, -4.54543290e-17, -1.21864324e-16],
            [-4.57890978e-02, 2.04160351e-17, 1.63340987e-16],
            [2.36928601e-01, -9.96517322e-17, -8.99599812e-17],
            [2.36928601e-01, 1.27691064e-16, 4.69459541e-17],
        ]),
        "stress": np.array([
            3.67613706e-03, 4.31993694e-03, 4.31993694e-03,
            1.06281970e-17, -7.92544911e-18, -1.07587645e-17,
        ]),
    },
    "GRACE-1L-OMAT-medium-ft-AM": {
        "energy": -14.958577909458537,
        "forces": np.array([
            [-4.07035691e-01, 9.01514106e-17, -1.73472348e-17],
            [-3.87803735e-02, 8.06104315e-17, 1.05411556e-16],
            [2.22908032e-01, -1.22677476e-16, -2.43674438e-17],
            [2.22908032e-01, -5.33969570e-17, -7.26415456e-17],
        ]),
        "stress": np.array([
            1.14274725e-03, 1.75807271e-03, 1.75807271e-03,
            1.13528468e-17, -5.73974865e-17, 2.75758625e-17,
        ]),
    },
    "GRACE-1L-OMAT-large-base": {
        "energy": -14.955965920493544,
        "forces": np.array([
            [-4.07736352e-01, 9.97465999e-18, 1.28749008e-16],
            [-4.89794354e-02, -2.89102509e-16, 2.10335221e-17],
            [2.28357894e-01, 2.69912131e-16, -1.74122869e-16],
            [2.28357894e-01, 1.21430643e-17, 1.52601456e-17],
        ]),
        "stress": np.array([
            -5.07941247e-05, 5.36487389e-04, 5.36487386e-04,
            -3.54099143e-17, 2.65835492e-17, -4.32439661e-17,
        ]),
    },
    "GRACE-1L-OMAT-large-ft-E": {
        "energy": -14.982298591442234,
        "forces": np.array([
            [-4.11598858e-01, -2.52185425e-16, -1.06739704e-16],
            [-5.10731565e-02, -6.61363325e-18, 5.05238212e-17],
            [2.31336007e-01, 2.06892880e-16, 2.13370988e-16],
            [2.31336007e-01, 5.13911830e-17, -1.65774512e-16],
        ]),
        "stress": np.array([
            3.38757720e-05, 6.05452554e-04, 6.05452551e-04,
            1.40020620e-16, 1.20383263e-16, 1.76736209e-16,
        ]),
    },
    "GRACE-1L-OMAT-large-ft-AM": {
        "energy": -14.956031256170562,
        "forces": np.array([
            [-3.85692051e-01, 9.08561421e-17, 1.21213803e-16],
            [-4.74309114e-02, -1.67292395e-16, 8.76035355e-17],
            [2.16561481e-01, 5.98479599e-17, -1.22514845e-16],
            [2.16561481e-01, 1.30104261e-17, -9.77950360e-17],
        ]),
        "stress": np.array([
            9.03889033e-04, 1.44880523e-03, 1.44880522e-03,
            -5.30887579e-17, 2.04312072e-16, 1.75482761e-17,
        ]),
    },
    "GRACE-2L-OMAT-medium-base": {
        "energy": -14.965543502708845,
        "forces": np.array([
            [-4.05457617e-01, 1.07923856e-16, 2.60394869e-17],
            [-4.14835478e-02, -1.15792792e-16, 1.12323345e-16],
            [2.23470582e-01, 1.29236899e-16, 9.15066634e-17],
            [2.23470582e-01, -1.14692496e-16, -2.25040560e-16],
        ]),
        "stress": np.array([
            3.55945661e-03, 4.20875950e-03, 4.20875950e-03,
            4.49152305e-18, -1.20252696e-17, -2.86204027e-17,
        ]),
    },
    "GRACE-2L-OMAT-medium-ft-E": {
        "energy": -14.982405575748693,
        "forces": np.array([
            [-4.07563290e-01, 6.04578236e-17, 7.34818022e-17],
            [-4.24890735e-02, -2.73218947e-17, -1.77375475e-16],
            [2.25026182e-01, -5.55111512e-17, -5.63785130e-18],
            [2.25026182e-01, 1.51788304e-17, 1.14044516e-16],
        ]),
        "stress": np.array([
            3.75185062e-03, 4.39624360e-03, 4.39624360e-03,
            1.91150865e-17, 7.82099508e-18, -2.19353451e-17,
        ]),
    },
    "GRACE-2L-OMAT-medium-ft-AM": {
        "energy": -14.97817488263803,
        "forces": np.array([
            [-3.95304229e-01, 1.25686137e-16, -1.68864488e-17],
            [-3.66284603e-02, -5.55111512e-17, 1.07986536e-16],
            [2.15966344e-01, 1.79543880e-16, 5.63785130e-18],
            [2.15966344e-01, -2.50260966e-16, -9.04224612e-17],
        ]),
        "stress": np.array([
            2.54533341e-03, 3.15553226e-03, 3.15553226e-03,
            -7.57291678e-18, 1.37879312e-17, 2.06818968e-17,
        ]),
    },
    "GRACE-2L-OMAT-large-base": {
        "energy": -14.961046641889862,
        "forces": np.array([
            [-4.04372966e-01, -1.58618778e-16, -1.00288701e-17],
            [-3.96250266e-02, 4.27175656e-17, 9.10729825e-18],
            [2.21998996e-01, 6.24500451e-17, -1.21647484e-16],
            [2.21998996e-01, 5.04154010e-17, 1.23219577e-16],
        ]),
        "stress": np.array([
            1.86253550e-03, 2.57143405e-03, 2.57143405e-03,
            3.62455465e-17, 3.32947203e-17, -4.12593397e-17,
        ]),
    },
    "GRACE-2L-OMAT-large-ft-E": {
        "energy": -14.964531406619901,
        "forces": np.array([
            [-4.05205236e-01, -2.20201461e-16, -1.43548368e-16],
            [-3.92384887e-02, 2.31585584e-16, 3.44776291e-17],
            [2.22221863e-01, 1.54607230e-16, 8.26162055e-17],
            [2.22221863e-01, -1.78025997e-16, 3.39355280e-17],
        ]),
        "stress": np.array([
            1.71413475e-03, 2.44355860e-03, 2.44355859e-03,
            -9.28596278e-17, 2.28101476e-17, 5.84942537e-18,
        ]),
    },
    "GRACE-2L-OMAT-large-ft-AM": {
        "energy": -14.976071534026135,
        "forces": np.array([
            [-3.93845375e-01, -1.37232890e-16, 5.23127548e-18],
            [-3.76806909e-02, 1.39428399e-16, 2.19008839e-16],
            [2.15763033e-01, -1.27935856e-17, -2.43294968e-16],
            [2.15763033e-01, 1.00559751e-17, 1.85398571e-17],
        ]),
        "stress": np.array([
            2.54965018e-03, 3.23997778e-03, 3.23997777e-03,
            -3.68200436e-17, -2.36392514e-17, 3.13362073e-18,
        ]),
    },
    "GRACE-1L-SMAX-large": {
        "energy": -14.971632820958067,
        "forces": np.array([[-4.23692278e-01, -4.14788646e-16,  4.59755931e-16],
       [-5.11025451e-02, -1.69745403e-16, -3.85433872e-17],
       [ 2.37397412e-01,  1.21864324e-16, -3.29922721e-16],
       [ 2.37397412e-01,  4.59322250e-16, -8.95822045e-17]]),
        "stress": np.array([ 1.55941038e-03,  2.08034596e-03,  2.08034596e-03, -2.17623432e-17,
        3.91702592e-19,  3.86479891e-17]),
    },
    "GRACE-1L-SMAX-OMAT-large": {
        "energy": -14.99741470718766,
        "forces": np.array([[-4.21008065e-01, -1.02714603e-16, -4.46555770e-17],
       [-4.48163245e-02,  1.01264483e-16, -1.24683250e-18],
       [ 2.32912195e-01, -5.05238212e-17,  1.22514845e-17],
       [ 2.32912195e-01,  4.68993673e-17,  3.32587487e-17]]),
        "stress": np.array([ 1.00909098e-03,  1.62683351e-03,  1.62683351e-03, -1.35920799e-17,
       -4.42362794e-17, -5.32715525e-18]),
    },
    "GRACE-2L-SMAX-large": {
        "energy": -14.969561997984645,
        "forces": np.array([[-4.12379417e-01,  5.63514079e-17,  1.68864488e-17],
       [-5.43035863e-02, -1.35958952e-16,  1.72604986e-16],
       [ 2.33341501e-01,  3.64291930e-17, -1.06685494e-16],
       [ 2.33341501e-01,  4.18773089e-17, -7.16657636e-17]]),
        "stress": np.array([ 1.85533469e-03,  2.46253353e-03,  2.46253352e-03, -1.01842674e-18,
       -2.23923315e-18, -1.19077588e-17]),
    },
    "GRACE-2L-SMAX-medium": {
        "energy": -14.97005430700654,
        "forces": np.array([[-4.19510346e-01, -2.49366500e-17, -8.51640806e-17],
       [-5.08216892e-02, -5.55111512e-17, -1.39211559e-16],
       [ 2.35166018e-01, -2.29850861e-17,  1.36392633e-16],
       [ 2.35166018e-01,  1.05113401e-16,  1.00342911e-16]]),
        "stress": np.array([2.11967092e-03, 2.69176084e-03, 2.69176084e-03, 2.47294903e-17,
       2.47294903e-17, 8.98304611e-18]),
    },
    "GRACE-2L-SMAX-OMAT-large": {
        "energy": -14.970213995109782,
        "forces": np.array([[-3.98903191e-01,  4.62954328e-17, -7.95262294e-17],
       [-4.52734967e-02,  9.23740251e-17,  1.08637058e-16],
       [ 2.22088344e-01, -1.07986536e-16, -1.53956708e-16],
       [ 2.22088344e-01, -3.54534110e-17,  1.19099609e-16]]),
        "stress": np.array([ 2.84228395e-03,  3.50542219e-03,  3.50542218e-03,  1.56681037e-19,
        2.19353451e-18, -1.30567531e-17]),
    },
    "GRACE-2L-SMAX-OMAT-medium": {
        "energy": -14.978684650015197,
        "forces": np.array([[-4.02579279e-01,  2.45029691e-17, -2.18846209e-16],
       [-4.02532889e-02, -1.34441069e-17, -3.51281504e-17],
       [ 2.21416284e-01, -4.07660017e-17,  3.25260652e-17],
       [ 2.21416284e-01,  2.15756232e-17,  2.33808198e-16]]),
        "stress": np.array([ 2.46527537e-03,  3.10845267e-03,  3.10845267e-03, -7.72959781e-18,
       -4.70043110e-18,  7.52068976e-18]),
    },
}

ALL_MODELS = list(REFERENCE.keys())
SMAX_MODELS = [
    "GRACE-1L-SMAX-large",
    "GRACE-1L-SMAX-OMAT-large",
    "GRACE-2L-SMAX-large",
    "GRACE-2L-SMAX-medium",
    "GRACE-2L-SMAX-OMAT-large",
    "GRACE-2L-SMAX-OMAT-medium",
]


def make_test_structure():
    """Create a deterministic Al FCC test structure with one atom displaced."""
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.positions[0] += [0.1, 0.0, 0.0]
    return atoms


def saved_model_exists(model_name):
    """Check if the saved_model for a foundation model is cached locally."""
    model_path = os.path.join(FOUNDATION_CACHE_DIR, model_name)
    return os.path.isdir(model_path) and os.path.isfile(
        os.path.join(model_path, "saved_model.pb")
    )


def checkpoint_exists(model_name):
    """Check if the checkpoint for a foundation model is cached locally."""
    checkpoint_dir = os.path.join(FOUNDATION_CHECKPOINTS_CACHE_DIR, model_name)
    return os.path.isdir(checkpoint_dir) and os.path.isfile(
        os.path.join(checkpoint_dir, "model.yaml")
    )


# ---- Test 1: grace_fm prediction regression ----


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_grace_fm_prediction(model_name):
    """Verify grace_fm predictions match hardcoded reference values."""
    if not saved_model_exists(model_name):
        pytest.skip(f"Saved model for {model_name} not cached locally")

    calc = grace_fm(model_name)
    atoms = make_test_structure()
    atoms.calc = calc

    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    s = atoms.get_stress()

    ref = REFERENCE[model_name]
    assert np.allclose(e, ref["energy"], atol=1e-6), (
        f"Energy mismatch for {model_name}: {e} vs {ref['energy']}"
    )
    assert np.allclose(f, ref["forces"], atol=1e-6), (
        f"Forces mismatch for {model_name}"
    )
    assert np.allclose(s, ref["stress"], atol=1e-6), (
        f"Stress mismatch for {model_name}"
    )


# ---- Test 2: Checkpoint reconstruction matches grace_fm ----


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_checkpoint_matches_grace_fm(model_name):
    """Verify checkpoint-reconstructed model matches frozen saved_model predictions."""
    if not saved_model_exists(model_name):
        pytest.skip(f"Saved model for {model_name} not cached locally")
    if not checkpoint_exists(model_name):
        pytest.skip(f"Checkpoint for {model_name} not cached locally")

    # Path 1: grace_fm (frozen saved_model)
    calc_fm = grace_fm(model_name)
    atoms = make_test_structure()
    atoms.calc = calc_fm
    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    s0 = atoms.get_stress()

    # Path 2: reconstruct from checkpoint
    checkpoint_dir = os.path.join(FOUNDATION_CHECKPOINTS_CACHE_DIR, model_name)
    model_yaml = os.path.join(checkpoint_dir, "model.yaml")
    checkpoint_name = os.path.join(checkpoint_dir, "checkpoint")

    ins = load_instructions(model_yaml)
    tp = TensorPotential(ins, float_dtype=tf.float64)
    tp.load_checkpoint(checkpoint_name=checkpoint_name)

    calc_ckpt = TPCalculator(model=tp.model)
    atoms2 = make_test_structure()
    atoms2.calc = calc_ckpt
    e1 = atoms2.get_potential_energy()
    f1 = atoms2.get_forces()
    s1 = atoms2.get_stress()

    assert np.allclose(e0, e1, atol=1e-5), (
        f"Energy mismatch for {model_name}: saved_model={e0} vs checkpoint={e1}"
    )
    assert np.allclose(f0, f1, atol=1e-5), (
        f"Forces mismatch for {model_name}"
    )
    assert np.allclose(s0, s1, atol=1e-5), (
        f"Stress mismatch for {model_name}"
    )

@pytest.mark.parametrize("model_name", SMAX_MODELS)
def test_checkpoint_matches_grace_smax_fm(model_name):
    """Verify checkpoint-reconstructed SMAX model matches frozen saved_model predictions."""
    if not saved_model_exists(model_name):
        pytest.skip(f"Saved model for {model_name} not cached locally")
    if not checkpoint_exists(model_name):
        pytest.skip(f"Checkpoint for {model_name} not cached locally")

    # Path 1: grace_fm (frozen saved_model)
    calc_fm = grace_fm(model_name)
    atoms = make_test_structure()
    atoms.calc = calc_fm
    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    s0 = atoms.get_stress()

    # Path 2: reconstruct from checkpoint
    checkpoint_dir = os.path.join(FOUNDATION_CHECKPOINTS_CACHE_DIR, model_name)
    model_yaml = os.path.join(checkpoint_dir, "model.yaml")
    checkpoint_name = os.path.join(checkpoint_dir, "checkpoint")

    ins = load_instructions(model_yaml)
    tp = TensorPotential(ins, float_dtype=tf.float64)
    tp.load_checkpoint(checkpoint_name=checkpoint_name)

    calc_ckpt = TPCalculator(model=tp.model)
    atoms2 = make_test_structure()
    atoms2.calc = calc_ckpt
    e1 = atoms2.get_potential_energy()
    f1 = atoms2.get_forces()
    s1 = atoms2.get_stress()

    assert np.allclose(e0, e1, atol=1e-5), (
        f"Energy mismatch for {model_name}: saved_model={e0} vs checkpoint={e1}"
    )
    assert np.allclose(f0, f1, atol=1e-5), (
        f"Forces mismatch for {model_name}"
    )
    assert np.allclose(s0, s1, atol=1e-5), (
        f"Stress mismatch for {model_name}"
    )
