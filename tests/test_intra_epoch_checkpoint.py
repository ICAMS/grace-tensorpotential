"""Tests for mid-epoch checkpointing.

Covers:
- The intra_epoch_save flag persists through tf.train.Checkpoint write/read
  with the value passed to save_checkpoint(is_mid_epoch=...).
- Default save_checkpoint() leaves the flag False.
- Backward compat: a checkpoint written without the flag (simulating a
  pre-feature checkpoint) loads cleanly into a Checkpoint that has the
  flag — the in-memory variable stays at its default False.
- The ff_skip + epoch rewind logic in train_adam computes the right values
  for varied (epoch, step, steps_per_epoch) combinations.
"""

import os
import shutil
import tempfile
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf


def _make_ckpt(with_flag: bool, initial_flag_value: bool = False):
    """Build a minimal tf.train.Checkpoint with step/epoch/(intra_epoch_save)."""
    objs = {
        "step": tf.Variable(0, dtype=tf.int32, trainable=False),
        "epoch": tf.Variable(0, dtype=tf.int32, trainable=False),
    }
    if with_flag:
        objs["intra_epoch_save"] = tf.Variable(
            initial_flag_value, dtype=tf.bool, trainable=False
        )
    return tf.train.Checkpoint(**objs), objs


def test_flag_round_trip_true():
    """Save with intra_epoch_save=True, load, verify it persists as True."""
    tmp = tempfile.mkdtemp(prefix="tp_test_")
    try:
        ckpt_a, objs_a = _make_ckpt(with_flag=True)
        objs_a["step"].assign(123)
        objs_a["epoch"].assign(4)
        objs_a["intra_epoch_save"].assign(True)
        path = os.path.join(tmp, "ckpt")
        ckpt_a.write(path)

        ckpt_b, objs_b = _make_ckpt(with_flag=True)
        ckpt_b.read(path).expect_partial()
        assert int(objs_b["step"].numpy()) == 123
        assert int(objs_b["epoch"].numpy()) == 4
        assert bool(objs_b["intra_epoch_save"].numpy()) is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_flag_round_trip_false():
    """Default behavior: flag stays False through save/load."""
    tmp = tempfile.mkdtemp(prefix="tp_test_")
    try:
        ckpt_a, objs_a = _make_ckpt(with_flag=True, initial_flag_value=False)
        objs_a["step"].assign(50)
        objs_a["epoch"].assign(2)
        # intra_epoch_save left at default False
        path = os.path.join(tmp, "ckpt")
        ckpt_a.write(path)

        ckpt_b, objs_b = _make_ckpt(with_flag=True, initial_flag_value=True)
        ckpt_b.read(path).expect_partial()
        assert bool(objs_b["intra_epoch_save"].numpy()) is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_backward_compat_old_checkpoint():
    """Old checkpoint (pre-feature) loaded into a Checkpoint with the flag.

    The flag variable should keep its in-memory default value (False) since
    no key in the file matches it.
    """
    tmp = tempfile.mkdtemp(prefix="tp_test_")
    try:
        ckpt_old, objs_old = _make_ckpt(with_flag=False)
        objs_old["step"].assign(7)
        objs_old["epoch"].assign(1)
        path = os.path.join(tmp, "ckpt")
        ckpt_old.write(path)

        ckpt_new, objs_new = _make_ckpt(with_flag=True, initial_flag_value=False)
        # expect_partial avoids "unmatched object" complaints; the codebase
        # uses assert_consumed=False at the equivalent call site.
        ckpt_new.read(path).expect_partial()
        assert int(objs_new["step"].numpy()) == 7
        assert int(objs_new["epoch"].numpy()) == 1
        assert bool(objs_new["intra_epoch_save"].numpy()) is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _compute_ff(saved_step: int, saved_epoch: int, steps_per_epoch: int):
    """Mirror of the rewind logic in train_adam, for unit-testing."""
    epoch_start_step = (saved_epoch - 1) * steps_per_epoch
    ff_skip = max(0, saved_step - epoch_start_step)
    new_epoch = max(0, saved_epoch - 1)
    return new_epoch, epoch_start_step, ff_skip


def test_ff_skip_first_epoch_partial():
    """Crash 60% into epoch 1: rewind to epoch 0, ff 20500 batches."""
    new_epoch, new_step, ff = _compute_ff(20500, 1, 34371)
    assert new_epoch == 0
    assert new_step == 0
    assert ff == 20500


def test_ff_skip_late_epoch_partial():
    """Crash 60% into epoch 6 with the same per-epoch step count."""
    new_epoch, new_step, ff = _compute_ff(192000, 6, 34371)
    assert new_epoch == 5
    assert new_step == 5 * 34371
    assert ff == 192000 - 5 * 34371


def test_ff_skip_at_epoch_boundary():
    """Saved exactly at epoch start (step == epoch_start_step): ff_skip=0."""
    new_epoch, new_step, ff = _compute_ff(34371, 1, 34371)
    assert ff == 34371  # full epoch worth - happens if save fired at last step
    new_epoch, new_step, ff = _compute_ff(0, 1, 34371)
    assert ff == 0


def test_ff_skip_clamped_when_step_below_epoch_start():
    """Defensive: corrupted state shouldn't go negative."""
    new_epoch, new_step, ff = _compute_ff(saved_step=10, saved_epoch=5, steps_per_epoch=100)
    # epoch_start_step = 400, saved_step = 10 -> max(0, -390) = 0
    assert ff == 0
    assert new_epoch == 4


def test_ff_skip_zero_epoch_clamp():
    """Defensive: epoch=0 (impossible in practice) clamps to 0, not negative."""
    new_epoch, new_step, ff = _compute_ff(saved_step=5, saved_epoch=0, steps_per_epoch=100)
    assert new_epoch == 0


def test_fast_forward_with_callback_list():
    """train_one_epoch with skip_first_n_steps must not crash Keras callbacks.

    Regression: on_batch_end without a prior on_batch_begin left
    _batch_start_time as None, causing a TypeError in Keras internals.
    The crash requires a callback with on_batch_end (like the LR scheduler),
    _check_timing enabled, and batch >= 1.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from tensorflow.keras.callbacks import Callback, CallbackList

    from tensorpotential.cli.train import train_one_epoch

    n_batches = 6
    skip = 6  # skip all — we only need to exercise the fast-forward path

    # Minimal tp mock: needs .step, .increment_step(), .distributed_train_step()
    tp = MagicMock()
    tp.step = 0

    def _increment():
        tp.step += 1

    tp.increment_step = _increment
    tp.distributed_train_step.return_value = {}

    # Plain list dataset (non-tf), each element is a single batch dict
    dataset = [{"dummy": i} for i in range(n_batches)]

    # Strategy with 1 replica
    strategy = SimpleNamespace(num_replicas_in_sync=1)

    # A callback with on_batch_end triggers _should_call_train_batch_hooks,
    # which activates the timing code path that reads _batch_start_time.
    class LRLikeCallback(Callback):
        def on_batch_end(self, batch, logs=None):
            pass

    callback_list = CallbackList([LRLikeCallback()])
    callback_list._check_timing = True

    # This must not raise
    train_one_epoch(
        tp=tp,
        train_dataset=dataset,
        distr_strategy=strategy,
        skip_first_n_steps=skip,
        callback_list=callback_list,
        progress_bar=False,
    )


def _run_shuffled_epoch(seed, epoch, n_batches, skip=0):
    """Run train_one_epoch with shuffle=True on a mock tp; return the ids of
    the batches that were actually trained (fast-forwarded ones excluded)."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from tensorpotential.cli.train import train_one_epoch

    trained = []
    tp = MagicMock()
    tp.seed = seed
    tp.epoch = epoch
    tp.step = 0

    def _increment():
        tp.step += 1

    tp.increment_step = _increment

    def _train(batch):
        trained.append(batch[0]["id"])
        return {"nat/per_struct": [1.0]}

    tp.distributed_train_step = _train

    train_one_epoch(
        tp=tp,
        train_dataset=[{"id": i} for i in range(n_batches)],
        distr_strategy=SimpleNamespace(num_replicas_in_sync=1),
        shuffle=True,
        skip_first_n_steps=skip,
        progress_bar=False,
    )
    return trained


def test_shuffle_order_reproducible_across_mid_epoch_resume():
    """The epoch shuffle must depend only on (seed, epoch) so that a mid-epoch
    resume fast-forwards past exactly the batches already trained.

    Regression: the shuffle used the global np.random stream, whose state at
    epoch N depended on the N-1 shuffles drawn before it. A resumed process
    does not replay those draws, so for any crash after the first epoch the
    reconstructed order differed and the resume duplicated some batches while
    omitting others.
    """
    seed, n_batches, crash_epoch, crash_step = 42, 8, 3, 4

    # Original run: epochs 1..crash_epoch, interrupted at crash_step.
    orders = [
        _run_shuffled_epoch(seed, ep, n_batches)
        for ep in range(1, crash_epoch + 1)
    ]
    trained_prefix = orders[-1][:crash_step]  # encoded in the checkpoint

    # Sanity: consecutive epochs must still get different orders.
    assert orders[0] != orders[1]

    # Resumed run: fresh process, same seed, fast-forward in crash_epoch.
    resumed = _run_shuffled_epoch(seed, crash_epoch, n_batches, skip=crash_step)

    # Together they must cover every batch exactly once.
    assert sorted(trained_prefix + resumed) == list(range(n_batches))


def test_shuffle_order_differs_for_different_seeds():
    orders = [_run_shuffled_epoch(seed, 1, 32) for seed in (1, 2)]
    assert orders[0] != orders[1]
    assert sorted(orders[0]) == sorted(orders[1]) == list(range(32))


def test_streaming_fast_forward_rejected():
    """skip_first_n_steps > 0 with a streaming dataset must raise.

    Fast-forward needs a replayable batch order and a known batches-per-epoch;
    streaming provides neither (len() is only an estimate until one full epoch
    has run). train_adam restarts the interrupted epoch from the beginning
    instead of fast-forwarding.
    """
    import pytest

    from tensorpotential.cli.train import train_one_epoch
    from tensorpotential.data.streaming import (
        StreamingConfig,
        StreamingDatasetWrapper,
    )

    ds = StreamingDatasetWrapper([], [], StreamingConfig())

    with pytest.raises(ValueError, match="not supported for streaming"):
        train_one_epoch(
            tp=None,
            train_dataset=ds,
            distr_strategy=SimpleNamespace(num_replicas_in_sync=1),
            skip_first_n_steps=3,
            progress_bar=False,
        )
