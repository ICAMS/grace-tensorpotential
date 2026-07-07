from __future__ import annotations

import itertools
import json
import logging
import os.path
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import CallbackList
from scipy.optimize import minimize
from tqdm import tqdm

from tensorpotential import TensorPotential
from tensorpotential.cli.data import (
    regroup_similar_batches,
    is_tf_distr_dataset,
    regroup_dataset_to_iterator,
)
from tensorpotential.data.streaming import StreamingDatasetWrapper
from tensorpotential.cli.metrics import (
    addup_metrics,
    aggregate_metrics,
    process_accumulated_metrics,
    normalize_group_metrics,
    concatenate_per_structure_metrics,
)
from tensorpotential.cli.train_callbacks import (
    LRSchedulerFactory,
    CustomReduceLROnPlateau,
)
from tensorpotential.utils import NumpyEncoder, is_chief

LEGACY_SCHEDULER_PARAMS = "learning_rate_reduction"
SCHEDULER_PARAMS = "scheduler_params"

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()

MININTERVAL = 1  # min interval for progress bar, in seconds
eV_A3_to_GPa = 160.2176621


def dump_metrics(filename, metrics, strategy=None):
    if not is_chief(strategy):
        return
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    json_repr = json.dumps(metrics, cls=NumpyEncoder)
    with open(filename, "at") as f:
        print("-", json_repr, file=f)


def metrics_to_string(metrics):
    return " ".join(
        [
            f"{k}: {v:.3e}"
            for k, v in metrics.items()
            if k not in ["mae/de", "rmse/de", "mae/virial", "rmse/virial"]
        ]
    )


def test_one_epoch(
    tp,
    dataset,
    distr_strategy,
    cycle=True,
    compute_concat_per_structure_metrics=False,
    chunk_batches=False,
    regroup_window_factor=None,
):
    is_tf_dataset = is_tf_distr_dataset(dataset)
    is_streaming = isinstance(dataset, StreamingDatasetWrapper)
    n_batch = get_dataset_num_batches(dataset)

    if chunk_batches and distr_strategy.num_replicas_in_sync > 1:
        if is_tf_dataset and dataset.distribute_values and regroup_window_factor:
            dataset = regroup_dataset_to_iterator(
                dataset,
                n_group=distr_strategy.num_replicas_in_sync,
                regroup_window_factor=regroup_window_factor,
            )
            is_tf_dataset = False

    accumulated_metrics = {}
    data_iter = iter(dataset)
    if not is_streaming:
        if cycle or distr_strategy.num_replicas_in_sync > 1:
            if not is_tf_dataset:
                data_iter = itertools.cycle(data_iter)
        steps = int(
            np.round(np.ceil(n_batch / distr_strategy.num_replicas_in_sync))
        )  # ! extra batch will be processed
        steps = max(steps, 1)
    num_processed_batches = 0
    total_time = 0
    agg_concat_per_structure_metrics = defaultdict(list)

    def _process_batch(batch):
        nonlocal num_processed_batches, total_time
        num_processed_batches += distr_strategy.num_replicas_in_sync
        t_start = time.perf_counter()
        step_results = tp.distributed_test_step(batch)
        metrics = aggregate_metrics(step_results)
        total_time += time.perf_counter() - t_start
        addup_metrics(metrics, accumulated_metrics)
        if compute_concat_per_structure_metrics:
            concatenate_per_structure_metrics(
                step_results, batch, agg_concat_per_structure_metrics
            )
        del batch

    if is_streaming:
        # Exhaust streaming iterator — len() is only an estimate
        for batch_dict in data_iter:
            _process_batch([batch_dict])
    else:
        for i in range(steps):
            if is_tf_dataset:
                batch = next(data_iter)
            else:
                batch = [
                    next(data_iter) for _ in range(distr_strategy.num_replicas_in_sync)
                ]
            _process_batch(batch)

    del data_iter
    accumulated_metrics["total_time/test"] = total_time
    if compute_concat_per_structure_metrics:
        return accumulated_metrics, agg_concat_per_structure_metrics
    else:
        return accumulated_metrics


def train_one_epoch(
    tp,
    train_dataset,
    distr_strategy,
    progress_bar=False,
    desc=None,
    shuffle=False,
    cycle=False,
    compute_concat_per_structure_metrics=False,
    chunk_batches=False,
    regroup_window_factor=None,
    callback_list=None,
    intra_epoch_checkpoint_steps=None,
    intra_epoch_checkpoint_fn=None,
    skip_first_n_steps=0,
):
    is_tf_dataset = is_tf_distr_dataset(train_dataset)
    n_batch = get_dataset_num_batches(train_dataset)
    accumulated_metrics = {}

    is_streaming = isinstance(train_dataset, StreamingDatasetWrapper)

    if shuffle or distr_strategy.num_replicas_in_sync > 1:
        if is_streaming:
            pass  # shuffling is handled internally by StreamingDatasetWrapper
        elif is_tf_dataset:
            # TODO: shuffle train_dataset
            pass
        else:
            train_dataset = list(train_dataset)
            # Deterministic per-(seed, epoch) order: a mid-epoch resume must
            # reconstruct the interrupted epoch's batch order to fast-forward
            # past exactly the batches already baked into the checkpoint. The
            # global np.random stream cannot provide that — its state at epoch
            # N depends on the N-1 shuffles a resumed process never replays.
            np.random.default_rng([tp.seed, tp.epoch]).shuffle(train_dataset)
    if chunk_batches and distr_strategy.num_replicas_in_sync > 1:
        # group similar buckets together
        if not is_tf_dataset:
            train_dataset = regroup_similar_batches(
                train_dataset, n_group=distr_strategy.num_replicas_in_sync
            )
        elif (
            is_tf_dataset and train_dataset.distribute_values and regroup_window_factor
        ):
            train_dataset = regroup_dataset_to_iterator(
                train_dataset,
                n_group=distr_strategy.num_replicas_in_sync,
                regroup_window_factor=regroup_window_factor,
            )
            is_tf_dataset = False

    data_iter = iter(train_dataset)
    if not is_streaming:
        if cycle or distr_strategy.num_replicas_in_sync > 1:
            if not is_tf_dataset:
                data_iter = itertools.cycle(data_iter)

        # TODO: decide the strategy for batches: add extra or remove tail?
        # steps = int(np.round(np.ceil(n_batch / distr_strategy.num_replicas_in_sync))) # ! extra batch will be processed
        steps = (
            n_batch // distr_strategy.num_replicas_in_sync
        )  # tail batch will be dropped !
        steps = max(steps, 1)

    num_processed_batches = 0
    total_time = 0
    agg_concat_per_structure_metrics = defaultdict(list)

    def _fast_forward_batch(batch):
        # Mid-epoch resume fast-forward: advance step counter and LR
        # scheduler, but do NOT call distributed_train_step. Gradients
        # for these batches are already baked into the restored
        # model/optimizer state from the saved checkpoint.
        tp.increment_step()
        if callback_list is not None:
            callback_list.on_batch_begin(tp.step)
            callback_list.on_batch_end(tp.step, {})
        del batch

    def _process_train_batch(batch):
        nonlocal num_processed_batches, total_time
        num_processed_batches += distr_strategy.num_replicas_in_sync
        if callback_list is not None:
            callback_list.on_batch_begin(tp.step)
        t_start = time.perf_counter()
        step_results = tp.distributed_train_step(batch)
        total_time += time.perf_counter() - t_start
        metrics = aggregate_metrics(step_results)
        addup_metrics(metrics, accumulated_metrics)
        if compute_concat_per_structure_metrics:
            concatenate_per_structure_metrics(
                step_results, batch, agg_concat_per_structure_metrics
            )
        tp.increment_step()
        if callback_list is not None:
            callback_list.on_batch_end(tp.step, metrics)
        if (
            intra_epoch_checkpoint_steps is not None
            and intra_epoch_checkpoint_fn is not None
            and tp.step % intra_epoch_checkpoint_steps == 0
        ):
            intra_epoch_checkpoint_fn()
        del batch

    if is_streaming:
        if skip_first_n_steps:
            # Fast-forward needs a replayable batch order and a known
            # batches-per-epoch; streaming provides neither (len() is an
            # estimate until one full epoch has run). Callers must redo the
            # interrupted epoch instead — see the mid-epoch resume block in
            # train_adam.
            raise ValueError(
                f"skip_first_n_steps={skip_first_n_steps} is not supported "
                "for streaming datasets"
            )
        # Exhaust streaming iterator — len() is only an estimate
        stream_iter = data_iter
        pbar = None
        if progress_bar:
            tqdm_kwargs = dict(desc=desc, mininterval=MININTERVAL)
            if train_dataset._estimated_n_batches is not None:
                tqdm_kwargs["total"] = len(train_dataset)
            pbar = tqdm(data_iter, **tqdm_kwargs)
            stream_iter = pbar
        is_first_streaming_epoch = train_dataset._estimated_n_batches is None
        for batch_dict in stream_iter:
            _process_train_batch([batch_dict])
            # Dynamically refresh tqdm total during the first epoch
            if pbar is not None and is_first_streaming_epoch:
                new_total = len(train_dataset)
                if pbar.total != new_total:
                    pbar.total = new_total
                    pbar.refresh()
    else:
        if progress_bar:
            pbar = tqdm(range(steps), total=steps, desc=desc, mininterval=MININTERVAL)
        else:
            pbar = range(steps)
        for i, _ in enumerate(pbar):
            # Always advance the iterator — even during fast-forward, the data
            # must be consumed so the iterator state matches what training
            # would have seen.
            if is_tf_dataset:
                batch = next(data_iter)
            else:
                batch = [
                    next(data_iter) for _ in range(distr_strategy.num_replicas_in_sync)
                ]
            if i < skip_first_n_steps:
                _fast_forward_batch(batch)
            else:
                _process_train_batch(batch)

    del data_iter
    accumulated_metrics["total_time/train"] = total_time
    if compute_concat_per_structure_metrics:
        return accumulated_metrics, agg_concat_per_structure_metrics
    else:
        return accumulated_metrics


def train_bfgs_one_epoch(
    tp,
    tuple_of_datasets,
    distr_strategy,
    progress_bar=False,
    desc=None,
    cycle=False,
    compute_concat_per_structure_metrics=False,
):
    from tensorpotential.data.streaming import StreamingDatasetWrapper

    is_streaming = isinstance(tuple_of_datasets, StreamingDatasetWrapper)
    accumulated_metrics = {}
    if not is_streaming and distr_strategy.num_replicas_in_sync > 1:
        tuple_of_datasets = list(tuple_of_datasets)
        np.random.shuffle(tuple_of_datasets)
    data_iter = iter(tuple_of_datasets)
    if not is_streaming:
        if cycle or distr_strategy.num_replicas_in_sync > 1:
            data_iter = itertools.cycle(data_iter)
        n_batch = len(tuple_of_datasets)
        steps = (
            n_batch // distr_strategy.num_replicas_in_sync
        )  # tail batch will be dropped !
        steps = max(steps, 1)
    pbar = range(steps) if not is_streaming else None
    if not is_streaming and progress_bar:
        pbar = tqdm(pbar, total=steps, desc=desc, mininterval=MININTERVAL)
    total_time = 0
    agg_concat_per_structure_metrics = defaultdict(list)

    def _process_bfgs_batch(batch):
        nonlocal total_time
        t_start = time.perf_counter()
        step_results = tp.distributed_bfgs_train_step(batch)
        metrics = aggregate_metrics(step_results)
        total_time += time.perf_counter() - t_start
        addup_metrics(metrics, accumulated_metrics)
        if compute_concat_per_structure_metrics:
            concatenate_per_structure_metrics(
                step_results, batch, agg_concat_per_structure_metrics
            )

    if is_streaming:
        stream_iter = data_iter
        if progress_bar:
            tqdm_kwargs = dict(desc=desc, mininterval=MININTERVAL)
            if tuple_of_datasets._estimated_n_batches is not None:
                tqdm_kwargs["total"] = len(tuple_of_datasets)
            stream_iter = tqdm(data_iter, **tqdm_kwargs)
        for batch_dict in stream_iter:
            _process_bfgs_batch([batch_dict])
    else:
        for _ in pbar:
            batch = [
                next(data_iter) for _ in range(distr_strategy.num_replicas_in_sync)
            ]
            _process_bfgs_batch(batch)
    accumulated_metrics["total_time/train"] = total_time
    if compute_concat_per_structure_metrics:
        return accumulated_metrics, agg_concat_per_structure_metrics
    else:
        return accumulated_metrics


def get_dataset_num_batches(dataset):

    if dataset is not None:
        return dataset.n_batch if is_tf_distr_dataset(dataset) else len(dataset)


def train_adam(
    tp: TensorPotential,
    fit_config: dict,
    train_ds,
    test_ds=None,
    strategy=None,
    seed=None,
    train_grouping_df=None,
    test_grouping_df=None,
    intra_epoch_redo: bool = True,
):
    callbacks = []

    compute_init_stats = fit_config.get("eval_init_stats", False)
    epochs = fit_config.get("maxiter", 500)
    checkpoint_freq = fit_config.get("checkpoint_freq", 10)
    checkpoint_freq_steps = fit_config.get("checkpoint_freq_steps", None)
    if checkpoint_freq_steps is not None and checkpoint_freq_steps <= 0:
        log.warning(
            f"Ignoring non-positive checkpoint_freq_steps={checkpoint_freq_steps}"
        )
        checkpoint_freq_steps = None
    progress_bar = fit_config.get("progressbar", False)
    group_similar_batches = fit_config.get("group_similar_batches", True)
    regroup_window_factor = fit_config.get("regroup_window_factor", 128)
    init_flat_vars = tp.model.get_flat_trainable_variables()
    log.info(f"Number of trainable parameters: {len(init_flat_vars)}")
    best_test_loss = 1e99

    try:
        shed_params = fit_config[LEGACY_SCHEDULER_PARAMS]
    except KeyError:
        shed_params = fit_config[SCHEDULER_PARAMS]

    resume_lr = shed_params.get("resume_lr", True)

    n_batches = get_dataset_num_batches(train_ds)
    steps_per_epoch = max(n_batches // strategy.num_replicas_in_sync, 1)

    # reset_optimizer must run BEFORE the mid-epoch rewind below: the rewind
    # exists to "complete" gradient updates already baked into optimizer
    # moments, so once we drop the moments the rewind has no purpose. Without
    # this ordering, a coefficient-only restart (e.g. finetuning on a different
    # dataset) would still try to fast-forward through the saved iterator
    # position. reset_optimizer() also clears intra_epoch_save so the block
    # below becomes a no-op.
    if fit_config.get("reset_optimizer", False):
        log.info(
            "Resetting optimizers by reinitialization "
            "(also clears mid-epoch resume state)"
        )
        tp.reset_optimizer()

    # Mid-epoch resume: if the loaded checkpoint was a mid-epoch save, rewind
    # epoch and step to the start of the interrupted epoch and compute how many
    # batches to fast-forward in the first train_one_epoch call. The model and
    # optimizer state already encode the gradients of the skipped batches; we
    # only need the iterator to land at the right position and the step-based
    # LR scheduler to advance through those steps without training.
    ff_skip = 0
    if tp.intra_epoch_save and isinstance(train_ds, StreamingDatasetWrapper):
        # Streaming datasets do not support fast-forward: steps_per_epoch is
        # only an estimate until a full epoch has run, so the rewind/skip
        # arithmetic below would land at an arbitrary position. Instead, redo
        # the interrupted epoch from its beginning (regardless of
        # intra_epoch_redo): batches trained before the mid-epoch checkpoint
        # get their gradients applied a second time, which is inconsistent
        # but strictly better than training on a wrong slice of the epoch.
        # tp.step is kept as saved so the step-based LR scheduler continues
        # from where it left off.
        new_epoch = max(0, tp.epoch - 1)
        log.warning(
            f"Mid-epoch resume with a streaming dataset: fast-forward is not "
            f"supported, restarting the interrupted epoch from the beginning "
            f"(epoch {tp.epoch} -> {new_epoch}, step {tp.step} kept). Batches "
            f"already trained before the mid-epoch checkpoint will be "
            f"trained again."
        )
        tp.epoch = new_epoch
        tp.intra_epoch_save = False
    elif tp.intra_epoch_save and intra_epoch_redo:
        saved_step = tp.step
        epoch_start_step = (tp.epoch - 1) * steps_per_epoch
        ff_skip = max(0, saved_step - epoch_start_step)
        new_epoch = max(0, tp.epoch - 1)
        log.info(
            f"Mid-epoch resume: rewinding epoch {tp.epoch} -> {new_epoch}, "
            f"step {tp.step} -> {epoch_start_step}; will fast-forward "
            f"{ff_skip} batches in epoch {new_epoch + 1} "
            f"(steps_per_epoch={steps_per_epoch})."
        )
        tp.epoch = new_epoch
        tp.step = epoch_start_step
        tp.intra_epoch_save = False
    elif tp.intra_epoch_save and not intra_epoch_redo:
        # Skip the interrupted epoch's tail batches entirely, but bump tp.step
        # to where it would have been at a clean end of that epoch. This keeps
        # the step-based LR scheduler aligned with the nominal maxiter budget
        # (final step == maxiter * steps_per_epoch, same as a fresh run) at the
        # cost of under-training by (nominal_end - saved_step) gradient updates.
        nominal_end_step = tp.epoch * steps_per_epoch
        skipped = max(0, nominal_end_step - tp.step)
        log.info(
            f"Mid-epoch resume with --no-intra-epoch-redo: bumping step "
            f"{tp.step} -> {nominal_end_step} so the LR scheduler aligns with "
            f"the nominal end of epoch {tp.epoch}. The interrupted epoch's "
            f"remaining {skipped} batches will NOT be trained (skipped)."
        )
        tp.step = nominal_end_step
        tp.intra_epoch_save = False

    test_n_batches = get_dataset_num_batches(test_ds)

    with strategy.scope():
        lr_scheduler = LRSchedulerFactory.create_lr_scheduler(
            tp, n_batches, fit_config, strategy
        )
        if lr_scheduler is not None:
            callbacks.append(lr_scheduler)
        callback_list = CallbackList(callbacks)
    # callback_list.on_train_begin()

    if group_similar_batches and strategy.num_replicas_in_sync > 1:
        n_group = strategy.num_replicas_in_sync
        log.info(f"Regroup similar TRAIN batches by group of {n_group}")
        train_ds = regroup_similar_batches(train_ds, n_group=n_group)
        if test_ds is not None:
            log.info(f"Regroup similar TEST batches by group of {n_group}")
            test_ds = regroup_similar_batches(test_ds, n_group=n_group)

    if not resume_lr:
        lr = fit_config["opt_params"]["learning_rate"]
        current_lr = tp.optimizer.learning_rate.numpy()
        log.info(
            f"Resetting learning rate from {current_lr} to initial {lr} because 'resume_lr' is set to False"
        )
        tp.optimizer.learning_rate.assign(lr)

    loss_weights_switch = None

    new_loss_params = {}
    if "loss" in fit_config and "switch" in fit_config["loss"]:
        # TODO: implement multiple switches
        if not isinstance(lr_scheduler, CustomReduceLROnPlateau):
            error_msg = f"ERROR: Switching loss weights is implemented for CustomReduceLROnPlateau only. It will be ignored for your scheduler {lr_scheduler.__class__.__name__}"
            log.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            switch_params = fit_config["loss"]["switch"]
            loss_weights_switch = switch_params.pop("after_iter")
            new_loss_params = switch_params

    metrics_normalization_spec = tp.compute_metrics.get_normalization_spec()

    # If checkpoint was saved with EMA weights in model, swap back for training.
    # Safe for old checkpoints (model==shadow → swap is identity).
    tp.prepare_for_training()

    #### INIT STATS ####
    if compute_init_stats:
        epoch = tp.epoch

        with tp.ema_scope():
            train_accumulated_metrics, train_agg_concat_per_structure_metrics = (
                test_one_epoch(
                    tp,
                    train_ds,
                    strategy,
                    cycle=fit_config.get("train_cycle", False),
                    compute_concat_per_structure_metrics=True,
                    chunk_batches=group_similar_batches,
                    regroup_window_factor=regroup_window_factor,
                )
            )
            initial_train_metrics = process_accumulated_metrics(
                train_accumulated_metrics,
                n_batches=n_batches,
                normalization_spec=metrics_normalization_spec,
            )
            initial_train_metrics["epoch"] = epoch
            if train_grouping_df is not None:
                compute_per_group_metrics(
                    train_grouping_df,
                    train_agg_concat_per_structure_metrics,
                    initial_train_metrics,
                    n_batches=n_batches,
                    metrics_normalization_spec=metrics_normalization_spec,
                )
            # replace "test" with "train" for consistency:
            initial_train_metrics = {
                k.replace("test", "train"): v for k, v in initial_train_metrics.items()
            }
            dump_metrics(
                filename=os.path.join(tp.output_dir, "train_metrics.yaml"),
                metrics=initial_train_metrics,
                strategy=tp.strategy,
            )
            if test_ds is not None:
                test_accumulated_metrics, test_agg_concat_per_structure_metrics = (
                    test_one_epoch(
                        tp,
                        test_ds,
                        strategy,
                        cycle=fit_config.get("test_cycle", False),
                        compute_concat_per_structure_metrics=True,
                        chunk_batches=group_similar_batches,
                        regroup_window_factor=regroup_window_factor,
                    )
                )
                initial_test_metrics = process_accumulated_metrics(
                    test_accumulated_metrics,
                    n_batches=test_n_batches,
                    normalization_spec=metrics_normalization_spec,
                )
                initial_test_metrics["epoch"] = epoch

                if test_grouping_df is not None:
                    compute_per_group_metrics(
                        test_grouping_df,
                        test_agg_concat_per_structure_metrics,
                        initial_test_metrics,
                        n_batches=test_n_batches,
                        metrics_normalization_spec=metrics_normalization_spec,
                    )

                msg = generate_train_test_message(
                    initial_train_metrics, initial_test_metrics, epoch, epochs
                )
                log.info(msg)
                # save test metrics to JSON-like file
                dump_metrics(
                    filename=os.path.join(tp.output_dir, "test_metrics.yaml"),
                    metrics=initial_test_metrics,
                    strategy=tp.strategy,
                )
            else:  # test_ds is None
                log.info(
                    f"Iteration #{epoch}/{epochs}: TRAIN stats: {metrics_to_string(initial_train_metrics)}"
                )
    #### END INIT STATS ####
    # After INIT STATS, streaming wrappers know their true batch count
    if isinstance(train_ds, StreamingDatasetWrapper):
        n_batches = get_dataset_num_batches(train_ds)
    if test_ds is not None and isinstance(test_ds, StreamingDatasetWrapper):
        test_n_batches = get_dataset_num_batches(test_ds)

    if loss_weights_switch is not None and tp.epoch >= loss_weights_switch:
        log.info(
            f"Switching loss weights to {new_loss_params}, because current loaded epoch ({tp.epoch}) >= {loss_weights_switch} (switch epoch)"
        )
        tp.loss_function.set_loss_component_params(new_loss_params)
        best_test_loss = 1e99
        if isinstance(lr_scheduler, CustomReduceLROnPlateau):
            log.info("Reset best metric for CustomReduceLROnPlateau")
            lr_scheduler.reset_best()

    def _intra_epoch_save():
        log.info(f"Intra-epoch checkpointing at step {tp.step}")
        with tp.ema_scope():
            tp.save_checkpoint(is_mid_epoch=True)

    callback_list.on_train_begin()
    while tp.epoch < epochs:
        # E-F WEIGHTS SWITCHING: loss component switch
        if loss_weights_switch is not None and tp.epoch == loss_weights_switch:
            # TODO: need better names
            do_loss_switch(tp, loss_weights_switch, new_loss_params, test_ds)
            best_test_loss = 1e99  # reset best test loss after switch
            # reset callback self.best
            if isinstance(lr_scheduler, CustomReduceLROnPlateau):
                log.info("Reset best metric for CustomReduceLROnPlateau")
                lr_scheduler.reset_best()

        tp.increment_epoch()  # old tp.epoch = 1

        epoch_begin_lr = tp.optimizer.learning_rate.numpy()
        # -----------------TRAIN---------------------
        train_accumulated_metrics, train_agg_concat_per_structure_metrics = (
            train_one_epoch(
                tp,
                train_ds,
                strategy,
                progress_bar,
                desc=f"Iteration #{tp.epoch}/{epochs}",
                shuffle=fit_config.get("train_shuffle", False),
                cycle=fit_config.get("train_cycle", False),
                compute_concat_per_structure_metrics=True,
                chunk_batches=group_similar_batches,
                regroup_window_factor=regroup_window_factor,
                callback_list=callback_list,
                intra_epoch_checkpoint_steps=checkpoint_freq_steps,
                intra_epoch_checkpoint_fn=(
                    _intra_epoch_save if checkpoint_freq_steps else None
                ),
                skip_first_n_steps=ff_skip,
            )
        )
        ff_skip = 0  # only the first epoch after a mid-epoch resume gets ff

        # --- EMA scope: evaluation and checkpoints use smoothed weights ---
        with tp.ema_scope():
            # post_process agg_metrics
            final_train_metrics = process_accumulated_metrics(
                train_accumulated_metrics,
                n_batches=n_batches,
                normalization_spec=metrics_normalization_spec,
            )
            final_train_metrics["epoch"] = tp.epoch
            final_train_metrics["step"] = tp.step
            final_train_metrics["lr_epoch_begin"] = float(epoch_begin_lr)
            final_train_metrics["lr_epoch_end"] = float(
                tp.optimizer.learning_rate.numpy()
            )
            # TODO: compute per-group metrics from train_agg_concat_per_structure_metrics and train_grouping_df
            if train_grouping_df is not None:
                compute_per_group_metrics(
                    train_grouping_df,
                    train_agg_concat_per_structure_metrics,
                    final_train_metrics,
                    n_batches=n_batches,
                    metrics_normalization_spec=metrics_normalization_spec,
                )
            dump_metrics(
                filename=os.path.join(tp.output_dir, "train_metrics.yaml"),
                metrics=final_train_metrics,
                strategy=tp.strategy,
            )
            # -----------------TEST---------------------

            # TODO: maybe tune its frequency
            if test_ds is not None:
                test_accumulated_metrics, test_agg_concat_per_structure_metrics = (
                    test_one_epoch(
                        tp,
                        test_ds,
                        strategy,
                        cycle=fit_config.get("test_cycle", False),
                        compute_concat_per_structure_metrics=True,
                        chunk_batches=group_similar_batches,
                        regroup_window_factor=regroup_window_factor,
                    )
                )
                final_test_metrics = process_accumulated_metrics(
                    test_accumulated_metrics,
                    n_batches=test_n_batches,
                    normalization_spec=metrics_normalization_spec,
                )
                final_test_metrics["epoch"] = tp.epoch
                final_test_metrics["step"] = tp.step

                if test_grouping_df is not None:
                    compute_per_group_metrics(
                        test_grouping_df,
                        test_agg_concat_per_structure_metrics,
                        final_test_metrics,
                        n_batches=test_n_batches,
                        metrics_normalization_spec=metrics_normalization_spec,
                    )

                final_test_metrics["lr_epoch_begin"] = float(epoch_begin_lr)
                final_test_metrics["lr_epoch_end"] = float(
                    tp.optimizer.learning_rate.numpy()
                )

                msg = generate_train_test_message(
                    final_train_metrics, final_test_metrics, tp.epoch, epochs
                )
                log.info(msg)
                # save test metrics to JSON-like file
                dump_metrics(
                    filename=os.path.join(tp.output_dir, "test_metrics.yaml"),
                    metrics=final_test_metrics,
                    strategy=tp.strategy,
                )

                # best test loss checkpoint
                current_test_loss = final_test_metrics["total_loss/test"]
                if current_test_loss < best_test_loss:
                    log.info(
                        f"New best test loss found ({current_test_loss:.5e}), checkpointing"
                    )
                    best_test_loss = current_test_loss
                    tp.save_checkpoint(suffix=".best_test_loss")

            else:  # test_ds is None
                log.info(
                    f"Iteration #{tp.epoch}/{epochs}: TRAIN stats: {metrics_to_string(final_train_metrics)}"
                )

            epoch_end_metrics = {
                "train_loss": final_train_metrics["total_loss/train"],
            }
            if test_ds is not None:
                epoch_end_metrics["test_loss"] = final_test_metrics["total_loss/test"]

            callback_list.on_epoch_end(tp.epoch, epoch_end_metrics)

            # regular checkpoint (saved with EMA weights in model)
            if tp.epoch % checkpoint_freq == 0:
                log.info("Regular checkpointing")
                if fit_config.get("save_all_regular_checkpoints", False):
                    tp.save_checkpoint(suffix=f".epoch_{tp.epoch}")
                else:
                    tp.save_checkpoint()
        # --- end EMA scope: training weights restored ---

        if tp.stop_training:
            if loss_weights_switch is not None and tp.epoch < loss_weights_switch:
                tp.epoch = loss_weights_switch
                tp.stop_training = False
            else:
                break


def do_loss_switch(tp, loss_weights_switch, new_loss_params, test_ds):
    log.info(
        f"Switching loss weights to {new_loss_params} after {loss_weights_switch} epochs"
    )
    lr_reduction_factor = new_loss_params.pop("learning_rate_reduction_factor", None)
    new_lr = new_loss_params.pop("learning_rate", None)
    tp.loss_function.set_loss_component_params(new_loss_params)
    cur_epoch = tp.epoch
    cur_step = tp.step
    if test_ds is not None:  # has test set
        try_load_checkpoint(tp, restart_best_test=True)
        # save stage1 best checkpoint (model=EMA as loaded from checkpoint)
        tp.save_checkpoint(suffix=".stage1.best_test_loss")
        # swap back to training weights for continued training
        tp.prepare_for_training()
    # set new learning rate
    if lr_reduction_factor is not None:
        current_lr = float(tp.optimizer.learning_rate.numpy())
        new_lr_value = current_lr * lr_reduction_factor
        log.info(
            f"Applying learning_rate_reduction_factor={lr_reduction_factor}: "
            f"{current_lr:.3e} → {new_lr_value:.3e}"
        )
        tp.optimizer.learning_rate.assign(new_lr_value)
    elif new_lr:
        log.info(f"Set new learning rate to {new_lr:.3e}")
        tp.optimizer.learning_rate.assign(new_lr)
    tp.epoch = cur_epoch
    tp.step = cur_step


def train_bfgs(
    tp: TensorPotential,
    fit_config: dict,
    train_ds,
    test_ds=None,
    strategy=None,
    seed=None,
    train_grouping_df=None,
    test_grouping_df=None,
):
    compute_init_stats = fit_config.get("eval_init_stats", False)
    epochs = fit_config.get("maxiter", 500)
    checkpoint_freq = fit_config.get("checkpoint_freq", 10)

    progress_bar = fit_config.get("progressbar", False)
    group_similar_batches = fit_config.get("group_similar_batches", True)
    init_flat_vars = tp.model.get_flat_trainable_variables()
    log.info(f"Number of trainable parameters: {len(init_flat_vars)}")
    best_test_loss = 1e16
    epoch = tp.epoch

    if group_similar_batches and strategy.num_replicas_in_sync > 1:
        n_group = strategy.num_replicas_in_sync
        log.info(f"Regroup similar TRAIN batches by group of {n_group}")
        train_ds = regroup_similar_batches(train_ds, n_group=n_group)
        if test_ds is not None:
            log.info(f"Regroup similar TEST batches by group of {n_group}")
            test_ds = regroup_similar_batches(test_ds, n_group=n_group)

    metrics_norm_spec = tp.compute_metrics.get_normalization_spec()
    #### INIT STATS ####
    if compute_init_stats:
        train_accumulated_metrics, train_agg_concat_per_structure_metrics = (
            test_one_epoch(
                tp, train_ds, strategy, compute_concat_per_structure_metrics=True
            )
        )
        initial_train_metrics = process_accumulated_metrics(
            train_accumulated_metrics,
            n_batches=len(train_ds),
            normalization_spec=metrics_norm_spec,
        )
        initial_train_metrics["epoch"] = epoch
        initial_train_metrics = {
            k.replace("test", "train"): v for k, v in initial_train_metrics.items()
        }
        dump_metrics(
            filename=os.path.join(tp.output_dir, "train_metrics.yaml"),
            metrics=initial_train_metrics,
            strategy=tp.strategy,
        )
        if test_ds is not None:
            test_accumulated_metrics, test_agg_concat_per_structure_metrics = (
                test_one_epoch(
                    tp, test_ds, strategy, compute_concat_per_structure_metrics=True
                )
            )
            initial_test_metrics = process_accumulated_metrics(
                test_accumulated_metrics,
                n_batches=len(test_ds),
                normalization_spec=metrics_norm_spec,
            )
            initial_test_metrics["epoch"] = epoch
            if test_grouping_df is not None:
                compute_per_group_metrics(
                    test_grouping_df,
                    test_agg_concat_per_structure_metrics,
                    initial_test_metrics,
                    n_batches=len(test_ds),
                    metrics_normalization_spec=metrics_norm_spec,
                )
            dump_metrics(
                filename=os.path.join(tp.output_dir, "test_metrics.yaml"),
                metrics=initial_test_metrics,
                strategy=tp.strategy,
            )
            msg = generate_train_test_message(
                initial_train_metrics, initial_test_metrics, epoch, epochs
            )
            log.info(msg)
            current_test_loss = initial_test_metrics["total_loss/test"]
            best_test_loss = current_test_loss
        else:
            log.info(
                f"Iteration #{epoch}/{epochs}: TRAIN stats: {metrics_to_string(initial_train_metrics)}"
            )
    #### END INIT STATS ####

    opt = fit_config["optimizer"]
    final_train_metrics = {}

    def fit_func(x):
        tp.model.set_flat_trainable_variables(x)
        # -----------------TRAIN---------------------
        train_accumulated_metrics, train_agg_concat_per_structure_metrics = (
            train_bfgs_one_epoch(
                tp,
                train_ds,
                strategy,
                progress_bar=progress_bar,
                compute_concat_per_structure_metrics=True,
            )
        )
        tot_loss = train_accumulated_metrics["total_loss/train"]
        tot_jac = train_accumulated_metrics["total_jac"]
        # post_process agg_metrics
        nonlocal final_train_metrics
        final_train_metrics = process_accumulated_metrics(
            train_accumulated_metrics,
            n_batches=len(train_ds),
            normalization_spec=metrics_norm_spec,
        )
        return tot_loss, tot_jac

    def callback(x):
        tp.increment_epoch()
        epoch = tp.epoch
        nonlocal final_train_metrics, best_test_loss
        final_train_metrics["epoch"] = epoch
        # TODO: compute per-group metrics from train_agg_concat_per_structure_metrics and train_grouping_df
        # if train_grouping_df is not None:
        #     compute_per_group_metrics(
        #         train_grouping_df,
        #         train_agg_concat_per_structure_metrics,
        #         final_train_metrics,
        #     )
        dump_metrics(
            filename=os.path.join(tp.output_dir, "train_metrics.yaml"),
            metrics=final_train_metrics,
            strategy=tp.strategy,
        )
        if test_ds is not None:
            tp.model.set_flat_trainable_variables(x)
            test_accumulated_metrics, test_agg_concat_per_structure_metrics = (
                test_one_epoch(
                    tp, test_ds, strategy, compute_concat_per_structure_metrics=True
                )
            )
            final_test_metrics = process_accumulated_metrics(
                test_accumulated_metrics,
                n_batches=len(test_ds),
                normalization_spec=metrics_norm_spec,
            )
            final_test_metrics["epoch"] = epoch
            if test_grouping_df is not None:
                compute_per_group_metrics(
                    test_grouping_df,
                    test_agg_concat_per_structure_metrics,
                    final_test_metrics,
                    n_batches=len(test_ds),
                    metrics_normalization_spec=metrics_norm_spec,
                )
            dump_metrics(
                filename=os.path.join(tp.output_dir, "test_metrics.yaml"),
                metrics=final_test_metrics,
                strategy=tp.strategy,
            )
            msg = generate_train_test_message(
                final_train_metrics, final_test_metrics, epoch, epochs
            )
            log.info(msg)

            # best test loss  checkpoint
            current_test_loss = final_test_metrics["total_loss/test"]
            if current_test_loss < best_test_loss:
                log.info(
                    f"New best test loss found ({current_test_loss:.5e}), checkpointing"
                )
                best_test_loss = current_test_loss
                tp.save_checkpoint(suffix=".best_test_loss")

        else:
            log.info(
                f"Iteration #{epoch}/{epochs}: TRAIN stats: {metrics_to_string(final_train_metrics)}"
            )

        if epoch % checkpoint_freq == 0:
            log.info("Regular checkpointing")
            tp.model.set_flat_trainable_variables(x)
            if fit_config.get("save_all_regular_checkpoints", False):
                tp.save_checkpoint(suffix=f".epoch_{epoch}")
            else:
                tp.save_checkpoint()

    optimizer_options = fit_config.get(
        "opt_params", {"gtol": 1e-8, "disp": False, "maxcor": 200, "iprint": -1}
    )
    optimizer_options["maxiter"] = epochs
    log.info(f"Optimization options: {optimizer_options}")
    res_opt = minimize(
        fit_func,
        init_flat_vars,
        method=opt,
        jac=True,
        options=optimizer_options,
        callback=callback,
    )
    tp.model.set_flat_trainable_variables(res_opt.x)
    log.info(
        f"{opt} optimization results: success={res_opt.success}, message={res_opt.message}"
    )
    tp.save_checkpoint()


def try_load_checkpoint(
    tp,
    restart_best_test=True,
    restart_latest=False,
    restart_suffix=None,
    expect_partial=False,
    checkpoint_name=None,
    verbose=True,
    assert_consumed=False,
):
    if checkpoint_name is not None:
        log.info(f"Trying to load explicit checkpoint {checkpoint_name}")
        tp.load_checkpoint(
            checkpoint_name=checkpoint_name,
            expect_partial=expect_partial,
            verbose=verbose,
            assert_consumed=assert_consumed,
        )
    elif restart_latest:
        log.info("Trying to load latest regular checkpoint")
        tp.load_checkpoint(
            expect_partial=expect_partial,
            verbose=verbose,
            assert_consumed=assert_consumed,
        )
    elif restart_best_test:
        log.info("Trying to load best test checkpoint")
        tp.load_checkpoint(
            suffix=".best_test_loss",
            expect_partial=expect_partial,
            verbose=verbose,
            assert_consumed=assert_consumed,
        )
    elif restart_suffix is not None:
        log.info(f"Trying to load checkpoint with suffix {restart_suffix}")
        tp.load_checkpoint(
            suffix=restart_suffix,
            expect_partial=expect_partial,
            verbose=verbose,
            assert_consumed=assert_consumed,
        )


def generate_train_test_message(final_train_metrics, final_test_metrics, epoch, epochs):
    METRICS_SKIP_LIST = [
        "mae/de",
        "rmse/de",
        "rmse/virial",
        "mae/virial",
        "epoch",
        "step",
        "per_group_metrics",
        "lr_epoch_begin",
        "lr_epoch_end",
    ]
    if epoch == 0:
        msg = "INITIAL    TRAIN(TEST): "
    else:
        msg = f"Iteration #{epoch}/{epochs} TRAIN(TEST): "
    for k in final_train_metrics:
        if k in METRICS_SKIP_LIST or "total_time" in k or "loss_component" in k:
            continue
        if k == "total_loss/train":
            msg += f"total_loss: {final_train_metrics[k]:.3e} "
            msg += f"({final_test_metrics['total_loss/test']:.3e}) "
        elif "stress" in k:
            # convert eV/A^3 -> GPa
            msg += f"{k}(GPa): {final_train_metrics[k] * eV_A3_to_GPa:.3e} "
            if k in final_test_metrics:
                msg += f"({final_test_metrics[k] * eV_A3_to_GPa:.3e}) "
        else:  # general case
            msg += f"{k}: {final_train_metrics[k]:.3e} "
            if k in final_test_metrics:
                msg += f"({final_test_metrics[k]:.3e}) "

    if "total_time/train/per_atom" in final_train_metrics:
        msg += f"Time(mcs/at): {final_train_metrics['total_time/train/per_atom'] * 1e6:.0f}"
        if "total_time/test/per_atom" in final_test_metrics:
            msg += f" ({final_test_metrics['total_time/test/per_atom'] * 1e6:.0f})"
    return msg


def compute_per_group_metrics(
    grouping_df,
    agg_concat_per_structure_metrics,
    final_metrics,
    n_batches,
    metrics_normalization_spec,
):
    agg_concat_per_structure_metrics_df = pd.DataFrame(
        agg_concat_per_structure_metrics
    ).set_index("structure_id")

    group_cols_dict = {
        col.split("group__")[-1]: col
        for col in grouping_df.columns
        if col.startswith("group__")
    }

    per_group_metrics = {}
    for group_name, group_col in group_cols_dict.items():
        cur_group_df = grouping_df[grouping_df[group_col]]
        mdf = pd.merge(
            cur_group_df,
            agg_concat_per_structure_metrics_df,
            left_index=True,
            right_index=True,
        )
        # drop duplicates by index (if duplication of data due to batch cycle)
        mdf = mdf.loc[mdf.index.drop_duplicates()]
        # NEED number_of_atoms and number_of_structures
        num_struct = len(mdf)
        num_atoms = sum(mdf["NUMBER_OF_ATOMS"])

        cur_metrics = {
            col: np.sum(mdf[col]) for col in agg_concat_per_structure_metrics_df.columns
        }
        norm_cur_metrics = normalize_group_metrics(
            cur_metrics,
            num_struct=num_struct,
            num_atoms=num_atoms,
            n_batches=n_batches,
            normalization_spec=metrics_normalization_spec,
        )
        norm_cur_metrics["num_struct"] = num_struct
        norm_cur_metrics["num_atoms"] = num_atoms
        per_group_metrics[group_name] = norm_cur_metrics

    final_metrics["per_group_metrics"] = per_group_metrics
