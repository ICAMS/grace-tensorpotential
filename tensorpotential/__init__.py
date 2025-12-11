import os
import sys
import warnings

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("tensorpotential")
except (ImportError,PackageNotFoundError) as e:
    # If the package is not installed, don't crash.
    __version__ = "unknown"

def _configure_keras_backend(verbose=True):
    """
    Sets TF_USE_LEGACY_KERAS=1 and informs the user.
    Must be run before 'import tensorflow'.
    """
    target_val = "1"
    env_key = "TF_USE_LEGACY_KERAS"

    existing_val = os.environ.get(env_key)

    # CRITICAL CHECK: Is TensorFlow already loaded?
    if 'tensorflow' in sys.modules and existing_val!=target_val:
        warnings.warn(
            f"TensorFlow was imported before {__name__} could set {env_key}={target_val}. "
            "The flag may be ignored. Please import this package first or continue at your own risk.",
            RuntimeWarning,
            stacklevel=2
        )
        return

    if existing_val is None or existing_val=='':
        # It is missing, set it and inform.
        os.environ[env_key] = target_val
        if verbose:
            msg = f"[{__name__}] Info: Environment variable {env_key} is automatically set to '{target_val}'."
            print(msg)
    elif existing_val not in [target_val, 'true']:
        if verbose:
            msg = f"[{__name__}] Warning: Environment variable {env_key} is already set to '{existing_val}', but tensorpotential requires '{target_val}'. Do it at your own risk"
            print(msg)

# Run immediately on import
_configure_keras_backend(verbose=True)

from tensorpotential.tensorpot import TensorPotential
from tensorpotential.tpmodel import TPModel
from tensorpotential.loss import LossFunction, L2Loss

__all__ = ["TensorPotential", "TPModel", "LossFunction", "L2Loss"]
