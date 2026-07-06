"""Helpers for setting up tf.distribute strategies for gracemaker.

Single source of truth for MultiWorkerMirroredStrategy (MWMS) construction
with NCCL forced. Used by both:

- the multi-worker SLURM path, where TF_CONFIG is provided externally by the
  launcher, and
- the single-host auto-upgrade path, where TF_CONFIG is not set and we
  synthesise a one-worker localhost cluster so MWMS can be used in place of
  MirroredStrategy.

Why MWMS on single host: MirroredStrategy's default CrossDeviceOps routes
scalar `strategy.reduce(SUM, scalar, axis=None)` calls (emitted per training
step in `tensorpot.reduce_dict`) through `ReductionToOneDevice` to CPU:0.
That host roundtrip serialises step boundaries and gaps every replica except
the one holding the variables. MWMS uses fused NCCL `CollectiveOps` for the
same reductions, with no host hop.
"""

import json
import logging
import os

log = logging.getLogger(__name__)

DEFAULT_TF_PORT = 12355


def ensure_single_worker_tf_config(port: int | None = None) -> None:
    """Set TF_CONFIG to a one-worker localhost cluster if not already set.

    Idempotent: a TF_CONFIG provided by an external launcher (SLURM/k8s)
    takes precedence and is left untouched. The port can be overridden via
    the TF_PORT environment variable; otherwise DEFAULT_TF_PORT is used.
    """
    if "TF_CONFIG" in os.environ:
        return
    if port is None:
        port = int(os.environ.get("TF_PORT", DEFAULT_TF_PORT))
    tf_config = {
        "cluster": {"worker": [f"localhost:{port}"]},
        "task": {"type": "worker", "index": 0},
    }
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    log.info(
        f"Auto-set TF_CONFIG to single-worker localhost:{port} "
        f"to enable MultiWorkerMirroredStrategy on single host."
    )


def make_mwms_strategy():
    """Construct MultiWorkerMirroredStrategy with NCCL forced.

    AUTO falls back to gRPC ring over TCP, which on PC2/Noctua2 has been
    measured ~3x slower than native NCCL-over-IB at 2 nodes; on a single
    host with NVLink the gap is even larger.
    """
    import tensorflow as tf

    comm_options = tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
    )
    return tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=comm_options
    )
