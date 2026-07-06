import os
import sys
import warnings

try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("tensorpotential")
except (ImportError, PackageNotFoundError):
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
    if "tensorflow" in sys.modules and existing_val != target_val:
        warnings.warn(
            f"TensorFlow was imported before {__name__} could set {env_key}={target_val}. "
            "The flag may be ignored. Please import this package first or continue at your own risk.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    if existing_val is None or existing_val == "":
        # It is missing, set it and inform.
        os.environ[env_key] = target_val
        if verbose:
            msg = f"[{__name__}] Info: Environment variable {env_key} is automatically set to '{target_val}'."
            print(msg)
    elif existing_val not in [target_val, "true"]:
        if verbose:
            msg = f"[{__name__}] Warning: Environment variable {env_key} is already set to '{existing_val}', but tensorpotential requires '{target_val}'. Do it at your own risk"
            print(msg)


def _configure_tf_options(verbose=True):
    """
    Globally disables TensorFloat-32 execution for accurate mathematical operations.
    Runs immediately on package initialize.
    """
    try:
        import tensorflow as tf

        try:
            tf.experimental.numpy.experimental_enable_numpy_behavior(
                dtype_conversion_mode="all"
            )
        except TypeError:
            # Fallback for older TF versions or those that don't support the kwarg
            tf.experimental.numpy.experimental_enable_numpy_behavior()
            
        tf.config.experimental.enable_tensor_float_32_execution(False)
        if verbose:
            print(f"[{__name__}] Info: tf.experimental.numpy behavior enabled.")
            print(f"[{__name__}] Info: TensorFloat-32 execution disabled.")
    except ImportError:
        pass


# Run immediately on import
_configure_keras_backend(verbose=True)
_configure_tf_options(verbose=True)

from tensorpotential.tensorpot import TensorPotential  # noqa: E402
from tensorpotential.tpmodel import TPModel  # noqa: E402
from tensorpotential.loss import LossFunction, L2Loss  # noqa: E402

__all__ = ["TensorPotential", "TPModel", "LossFunction", "L2Loss"]
