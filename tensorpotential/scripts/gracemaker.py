#!/usr/bin/env python

import logging
import os
import sys

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()


# argparse flags from cli.gracemaker.build_parser that consume the next argv
# token as their value. Used by _detect_multigpu_intent below to find the
# YAML positional argument without importing tensorflow (importing the real
# parser would pull in the whole tensorpotential package and thus TF).
# Must stay in sync with cli.gracemaker.build_parser;
# tests/test_detect_multigpu_intent.py cross-checks this list against it.
_VALUE_TAKING_FLAGS = {
    "-l",
    "--log",
    "-p",
    "--potential",
    "-rs",
    "--restart-suffix",
    "--seed",
    "-cn",
    "--checkpoint-name",
}


def _detect_multigpu_intent(argv):
    """Return True iff the user is asking for multi-GPU (MWMS) mode.

    Checks both `-m` / `--multigpu` on argv and the input YAML's
    `fit.strategy: mirrored` field. Runs before any tensorflow import so
    that scripts.main can set TF_CONFIG and let MWMS configure collective
    ops at program startup, as required by tf.distribute.
    """
    if "-m" in argv or "--multigpu" in argv:
        return True

    # The input YAML is a positional argument with default "input.yaml".
    yaml_path = "input.yaml"
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in _VALUE_TAKING_FLAGS:
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        yaml_path = arg
        break

    try:
        import yaml as _yaml

        with open(yaml_path) as f:
            cfg = _yaml.safe_load(f) or {}
    except (OSError, _yaml.YAMLError, AttributeError):
        return False
    return cfg.get("fit", {}).get("strategy") == "mirrored"


def main():
    argv = sys.argv[1:]

    strategy = None
    strategy_desc = ""

    # MultiWorkerMirroredStrategy must be constructed before any tf.config.*
    # call (it configures collective ops at program startup, and the eager
    # context locks once initialised). Detect intent early so we can set
    # TF_CONFIG and route through the existing MWMS branch below.
    wants_multigpu = _detect_multigpu_intent(argv)

    # NUM_VIRTUAL_DEVICES (CPU-only debug mode) initialises the TF context
    # via tf.config.set_logical_device_configuration, after which MWMS
    # cannot configure collective ops. They are mutually exclusive at the
    # tf.distribute API level. Fall back to MirroredStrategy in that case
    # (constructed later by cli.gracemaker.main); MirroredStrategy has no
    # program-startup constraint.
    auto_mwms = wants_multigpu and "NUM_VIRTUAL_DEVICES" not in os.environ
    if wants_multigpu and "NUM_VIRTUAL_DEVICES" in os.environ:
        log.warning(
            "NUM_VIRTUAL_DEVICES is set; auto-MWMS upgrade is incompatible "
            "with virtual device setup. Falling back to MirroredStrategy. "
            "(MirroredStrategy keeps the per-step CPU-reduce bubble; the "
            "auto-MWMS perf win only applies to real multi-GPU runs.)"
        )
    if auto_mwms and "TF_CONFIG" not in os.environ:
        from tensorpotential.cli.distribute import ensure_single_worker_tf_config

        ensure_single_worker_tf_config()

    # NUM_VIRTUAL_DEVICES forces the MirroredStrategy fallback (see above), so
    # the MWMS branch must be skipped even when TF_CONFIG is supplied
    # externally (e.g. by a SLURM launcher or left in the shell). Otherwise
    # make_mwms_strategy() initialises the TF context and the virtual-device
    # setup below fails with "Virtual devices cannot be modified after being
    # initialized".
    if "TF_CONFIG" in os.environ and "NUM_VIRTUAL_DEVICES" not in os.environ:
        tf_config = os.environ["TF_CONFIG"]
        log.info(f"TF_CONFIG detected: {tf_config}")
        log.info("Switching to MultiWorker distributed strategy")

        from tensorpotential.cli.distribute import make_mwms_strategy

        strategy = make_mwms_strategy()
        strategy_desc = (
            "Single host/multi GPU (MWMS auto-TF_CONFIG)"
            if auto_mwms
            else "Multi Host / Multi GPU"
        )

    # for debugging data-parallel
    if "NUM_VIRTUAL_DEVICES" in os.environ:
        # Simulate multiple CPUs with virtual devices
        N_VIRTUAL_DEVICES = int(os.environ["NUM_VIRTUAL_DEVICES"])
        log.info(f"Setting {N_VIRTUAL_DEVICES} virtual devices")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import tensorflow as tf

        physical_devices = tf.config.list_physical_devices("CPU")
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [
                tf.config.LogicalDeviceConfiguration()  # memory_limit=3024
                for _ in range(N_VIRTUAL_DEVICES)
            ],
        )
        log.info("Available devices:")
        devices = tf.config.list_logical_devices()
        for i, device in enumerate(devices):
            log.info(f"{i}) {device}")

    from tensorpotential.cli.gracemaker import main as gracemaker_main

    gracemaker_main(sys.argv[1:], strategy=strategy, strategy_desc=strategy_desc)


if __name__ == "__main__":
    main()
