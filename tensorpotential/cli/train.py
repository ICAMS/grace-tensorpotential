from __future__ import annotations

import itertools
import json
import logging
import os.path
from collections import defaultdict

import numpy as np
import time

import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

from tensorpotential import constants
import tensorflow as tf

LEARNING_RATE_REDUCTION = "learning_rate_reduction"

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()

MININTERVAL = 1  # min interval for progress bar, in seconds
eV_A3_to_GPa = 160.2176621


def dump_metrics(filename, metrics):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    json_repr = json.dumps(metrics)
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
    tuple_of_datasets,
    distr_strategy,
    cycle=True,
    compute_concat_per_structure_metrics=False,
):
    acc_metrics = {}
    data_iter = iter(tuple_of_datasets)
    if cycle or distr_strategy.num_replicas_in_sync > 1:
        data_iter = itertools.cycle(data_iter)
    n_batch = len(tuple_of_datasets)
    steps = int(
        np.round(np.ceil(n_batch / distr_strategy.num_replicas_in_sync))
    )  # ! extra batch will be processed
    # steps = n_batch // distr_strategy.num_replicas_in_sync  # tail batch will be dropped !
    steps = max(steps, 1)
    num_processed_batches = 0
    total_time = 0
    agg_concat_per_structure_metrics = defaultdict(list)
    for i in range(steps):
        b_data = [
            next(data_iter)  # .get_single_element()
            for _ in range(distr_strategy.num_replicas_in_sync)
        ]  # list of num_replicas_in_sync batch_dicts
        num_processed_batches += len(b_data)
        # TODO: filter out unnecessary data, not to submit to GPU
        t_start = time.perf_counter()
        out = tp.distributed_test_step(b_data)
        # agg metrics from out[".../per_struct"]
        metrics = aggregate_per_struct_metrics(out)
        total_time += time.perf_counter() - t_start
        #  aggregate "de", "df"
        acc_metrics = accumulate_metrics(metrics, acc_metrics)
        if compute_concat_per_structure_metrics:
            # append predictions b_data[DATA_STRUCTURE_ID] and out[".../per_struct"] to agg_concat_per_structure_metrics
            concatenate_per_structure_metrics(
                out, b_data, agg_concat_per_structure_metrics
            )
    # log.info(f"[TEST]: processed batches/total batches: {num_processed_batches}/{n_batch}")
    acc_metrics["total_time/test"] = total_time
    if compute_concat_per_structure_metrics:
        return acc_metrics, agg_concat_per_structure_metrics
    else:
        return acc_metrics


def train_one_epoch(
    tp,
    tuple_of_datasets,
    distr_strategy,
    progress_bar=False,
    desc=None,
    shuffle=False,
    cycle=False,
    compute_concat_per_structure_metrics=False,
):
    acc_metrics = {}
    if shuffle or distr_strategy.num_replicas_in_sync > 1:
        tuple_of_datasets = list(tuple_of_datasets)
        np.random.shuffle(tuple_of_datasets)
    data_iter = iter(tuple_of_datasets)
    if cycle or distr_strategy.num_replicas_in_sync > 1:
        data_iter = itertools.cycle(data_iter)
    n_batch = len(tuple_of_datasets)
    # TODO: decide the strategy for batches: add extra or remove tail?
    # steps = int(np.round(np.ceil(n_batch / distr_strategy.num_replicas_in_sync))) # ! extra batch will be processed
    steps = (
        n_batch // distr_strategy.num_replicas_in_sync
    )  # tail batch will be dropped !
    steps = max(steps, 1)
    if progress_bar:
        pbar = tqdm(range(steps), total=steps, desc=desc, mininterval=MININTERVAL)
    else:
        pbar = range(steps)
    num_processed_batches = 0
    total_time = 0
    agg_concat_per_structure_metrics = defaultdict(list)
    for i in pbar:
        b_data = [
            next(data_iter)  # .get_single_element()
            for _ in range(distr_strategy.num_replicas_in_sync)
        ]
        # sort b_data by number of total neighbours, in order to rebalance load in multi-GPU env
        # as GPU:0 should synchronize  grads and apply them
        # if len(b_data) > 1:
        #     b_data = sorted(b_data, key=lambda b: len(b[constants.BOND_IND_I]))
        num_processed_batches += len(b_data)
        # TODO: filter out unnecessary data, not to submit to GPU
        t_start = time.perf_counter()
        out = tp.distributed_train_step(b_data)
        # agg metrics from out[".../per_struct"]
        metrics = aggregate_per_struct_metrics(out)
        total_time += time.perf_counter() - t_start
        acc_metrics = accumulate_metrics(metrics, acc_metrics)
        if compute_concat_per_structure_metrics:
            # append predictions b_data[DATA_STRUCTURE_ID] and out[".../per_struct"] to agg_concat_per_structure_metrics
            concatenate_per_structure_metrics(
                out, b_data, agg_concat_per_structure_metrics
            )
    # log.info(f"[TRAIN]: processed batches/total batches: {num_processed_batches}/{n_batch}")
    acc_metrics["total_time/train"] = total_time
    if compute_concat_per_structure_metrics:
        return acc_metrics, agg_concat_per_structure_metrics
    else:
        return acc_metrics


def concatenate_per_structure_metrics(out, b_data, agg_concat_per_structure_metrics):
    # cur_structure_ids = []
    for b in b_data:
        # cur_structure_ids += list(b[constants.DATA_STRUCTURE_ID])
        agg_concat_per_structure_metrics[constants.DATA_STRUCTURE_ID] += [
            ind
            for ind in b[constants.DATA_STRUCTURE_ID].numpy().reshape(-1)
            if ind != -1
        ]

    # cur_metrics_per_struct_dict = defaultdict(list)
    for metric_name, metric_values in out.items():
        if metric_name.endswith("per_struct"):
            # cur_metrics_per_struct_dict[metric_name] += list(metric_values.numpy())
            agg_concat_per_structure_metrics[metric_name] += list(
                metric_values.numpy().reshape(-1)
            )


def train_bfgs_one_epoch(
    tp,
    tuple_of_datasets,
    distr_strategy,
    progress_bar=False,
    desc=None,
    cycle=False,
    compute_concat_per_structure_metrics=False,
):
    acc_metrics = {}
    if distr_strategy.num_replicas_in_sync > 1:
        tuple_of_datasets = list(tuple_of_datasets)
        np.random.shuffle(tuple_of_datasets)
    data_iter = iter(tuple_of_datasets)
    if cycle or distr_strategy.num_replicas_in_sync > 1:
        data_iter = itertools.cycle(data_iter)
    n_batch = len(tuple_of_datasets)
    steps = (
        n_batch // distr_strategy.num_replicas_in_sync
    )  # tail batch will be dropped !
    steps = max(steps, 1)
    pbar = range(steps)
    if progress_bar:
        pbar = tqdm(pbar, total=steps, desc=desc, mininterval=MININTERVAL)
    total_time = 0
    agg_concat_per_structure_metrics = defaultdict(list)
    for i in pbar:
        b_data = [
            next(data_iter)  # .get_single_element()
            for _ in range(distr_strategy.num_replicas_in_sync)
        ]
        # TODO: filter out unnecessary data, not to submit to GPU
        t_start = time.perf_counter()
        out = tp.distributed_bfgs_train_step(b_data)
        # agg metrics from out[".../per_struct"]
        metrics = aggregate_per_struct_metrics(out)
        total_time += time.perf_counter() - t_start
        acc_metrics = accumulate_metrics(metrics, acc_metrics)
        if compute_concat_per_structure_metrics:
            # append predictions b_data[DATA_STRUCTURE_ID] and out[".../per_struct"] to agg_concat_per_structure_metrics
            concatenate_per_structure_metrics(
                out, b_data, agg_concat_per_structure_metrics
            )
    acc_metrics["total_time/train"] = total_time
    if compute_concat_per_structure_metrics:
        return acc_metrics, agg_concat_per_structure_metrics
    else:
        return acc_metrics


def train_adam(
    tp,
    fit_config,
    train_ds,
    test_ds=None,
    strategy=None,
    seed=None,
    train_grouping_df=None,
    test_grouping_df=None,
    compute_init_stats: bool = False,
):
    epochs = fit_config.get("maxiter", 500)
    checkpoint_freq = fit_config.get("checkpoint_freq", 10)
    progress_bar = fit_config.get("progressbar", False)
    init_flat_vars = tp.model.get_flat_trainable_variables()
    log.info(f"Number of trainable parameters: {len(init_flat_vars)}")
    best_test_loss = 1e99
    test_loss_history = []

    lr_stop_at_min = False
    patience = None
    lr_reduction_factor = 1.0
    lr_min = None
    resume_lr = True
    loss_explosion_threshold = None

    if fit_config.get("reset_optimizer", False):
        log.info("Resetting optimizers by reinitialization")
        tp.reset_optimizer()

    if LEARNING_RATE_REDUCTION in fit_config:
        lr_red_params = fit_config[LEARNING_RATE_REDUCTION]
        patience = lr_red_params.get("patience", None)
        lr_reduction_factor = lr_red_params.get("factor", 0.8)
        lr_min = lr_red_params.get("min", 1.0e-4)
        lr_stop_at_min = lr_red_params.get("stop_at_min", False)
        log.info(
            f"Learning rate reduction will be used with parameters: {lr_red_params}"
        )
        resume_lr = lr_red_params.get("resume_lr")
        loss_explosion_threshold = lr_red_params.get("loss_explosion_threshold")

    if not resume_lr:
        lr = fit_config["opt_params"]["learning_rate"]
        current_lr = tp.optimizer.learning_rate.numpy()
        log.info(
            f"Resetting learning rate from {current_lr} to initial {lr} because 'resume_lr' is set to False"
        )
        tp.optimizer.learning_rate.assign(lr)

    loss_weights_switch = None

    new_loss_params = {}
    if "loss_weights_switch" in fit_config:
        log.warning("DEPRECATION WARNING!!! Use new loss switch params")
        loss_weights_switch = fit_config.get("loss_weights_switch")
        new_loss_params = {
            "energy": {"weight": fit_config["energy_loss_weight_2"]},
            "forces": {"weight": fit_config["forces_loss_weight_2"]},
        }
    elif "loss" in fit_config and "switch" in fit_config["loss"]:
        # TODO: implement multiple switches
        switch_params = fit_config["loss"]["switch"]
        loss_weights_switch = switch_params.pop("after_iter")
        new_loss_params = switch_params

    #### INIT STATS ####
    if compute_init_stats:
        tp.epoch = 0

        train_acc_metrics, train_agg_concat_per_structure_metrics = test_one_epoch(
            tp,
            train_ds,
            strategy,
            cycle=fit_config.get("train_cycle", False),
            compute_concat_per_structure_metrics=True,
        )
        initial_train_metrics = process_acc_metrics(
            train_acc_metrics, n_batches=len(train_ds)
        )
        initial_train_metrics["epoch"] = tp.epoch
        if train_grouping_df is not None:
            compute_per_group_metrics(
                train_grouping_df,
                train_agg_concat_per_structure_metrics,
                initial_train_metrics,
                n_batches=len(train_ds),
            )
        # replace "test" with "train" for consistency:
        initial_train_metrics = {
            k.replace("test", "train"): v for k, v in initial_train_metrics.items()
        }
        dump_metrics(
            filename=os.path.join(tp.output_dir, "train_metrics.yaml"),
            metrics=initial_train_metrics,
        )
        if test_ds is not None:
            test_acc_metrics, test_agg_concat_per_structure_metrics = test_one_epoch(
                tp,
                test_ds,
                strategy,
                cycle=fit_config.get("test_cycle", False),
                compute_concat_per_structure_metrics=True,
            )
            # tp.on_test_end()
            initial_test_metrics = process_acc_metrics(
                test_acc_metrics,
                n_batches=len(test_ds),
            )
            initial_test_metrics["epoch"] = tp.epoch

            if test_grouping_df is not None:
                compute_per_group_metrics(
                    test_grouping_df,
                    test_agg_concat_per_structure_metrics,
                    initial_test_metrics,
                    n_batches=len(test_ds),
                )

            msg = generate_train_test_message(
                initial_train_metrics, initial_test_metrics, tp.epoch, epochs
            )
            log.info(msg)
            # save test metrics to JSON-like file
            dump_metrics(
                filename=os.path.join(tp.output_dir, "test_metrics.yaml"),
                metrics=initial_test_metrics,
            )
        else:  # test_ds is None
            log.info(
                f"Iteration #{tp.epoch}/{epochs}: TRAIN stats: {metrics_to_string(initial_train_metrics)}"
            )
    #### END INIT STATS ####

    tp.epoch = 1
    last_lr_reduction_epoch = 0
    min_lr_achieved = False
    while tp.epoch <= epochs:
        # tp.on_epoch_begin()
        # -----------------TRAIN---------------------
        train_acc_metrics, train_agg_concat_per_structure_metrics = train_one_epoch(
            tp,
            train_ds,
            strategy,
            progress_bar,
            desc=f"Iteration #{tp.epoch}/{epochs}",
            shuffle=fit_config.get("train_shuffle", False),
            cycle=fit_config.get("train_cycle", False),
            compute_concat_per_structure_metrics=True,
        )

        # manually rewrite values with EMA after each epoch before applying (use_ema is checking inside)
        tp.optimizer.finalize_variable_values(tp.model.trainable_variables)
        # post_process agg_metrics
        final_train_metrics = process_acc_metrics(
            train_acc_metrics, n_batches=len(train_ds)
        )
        final_train_metrics["epoch"] = tp.epoch
        # TODO: compute per-group metrics from train_agg_concat_per_structure_metrics and train_grouping_df
        if train_grouping_df is not None:
            compute_per_group_metrics(
                train_grouping_df,
                train_agg_concat_per_structure_metrics,
                final_train_metrics,
                n_batches=len(train_ds),
            )
        dump_metrics(
            filename=os.path.join(tp.output_dir, "train_metrics.yaml"),
            metrics=final_train_metrics,
        )
        # -----------------TEST---------------------

        # TODO: maybe tune its frequency
        if test_ds is not None:
            # tp.on_test_begin()
            test_acc_metrics, test_agg_concat_per_structure_metrics = test_one_epoch(
                tp,
                test_ds,
                strategy,
                cycle=fit_config.get("test_cycle", False),
                compute_concat_per_structure_metrics=True,
            )
            # tp.on_test_end()
            final_test_metrics = process_acc_metrics(
                test_acc_metrics,
                n_batches=len(test_ds),
            )
            final_test_metrics["epoch"] = tp.epoch

            if test_grouping_df is not None:
                compute_per_group_metrics(
                    test_grouping_df,
                    test_agg_concat_per_structure_metrics,
                    final_test_metrics,
                    n_batches=len(test_ds),
                )

            msg = generate_train_test_message(
                final_train_metrics, final_test_metrics, tp.epoch, epochs
            )
            log.info(msg)
            # save test metrics to JSON-like file
            dump_metrics(
                filename=os.path.join(tp.output_dir, "test_metrics.yaml"),
                metrics=final_test_metrics,
            )

            # best test loss  checkpoint
            current_test_loss = final_test_metrics["total_loss/test"]
            test_loss_history.append(current_test_loss)
            if current_test_loss < best_test_loss:
                log.info(
                    f"New best test loss found ({current_test_loss:.5e}), checkpointing"
                )
                best_test_loss = current_test_loss
                tp.save_checkpoint(suffix=".best_test_loss")

            # LOSS EXPLOSION: rollback to last best test loss, reduce LR and reset optimizer
            if (
                loss_explosion_threshold is not None
                and current_test_loss >= best_test_loss * loss_explosion_threshold
            ):
                #  rollback  and restart optimizer
                log.info(
                    f"Current test loss increased by factor {(current_test_loss)/best_test_loss:.2f} "
                    f"(threshold={loss_explosion_threshold}). "
                    f"Rollback to latest best test loss checkpoint, reset optimizer and reduce learning rate"
                )
                try_load_checkpoint(tp, restart_best_test=True)
                restart_optimizer(tp.optimizer)
                current_lr = tp.optimizer.learning_rate.numpy()
                new_lr = current_lr * lr_reduction_factor
                tp.optimizer.learning_rate.assign(new_lr)
                log.info(f"Reducing learning rate: {current_lr:.5e} -> {new_lr:.5e}")

            # POOR-MAN's LRReduceOnPlateau
            elif patience is not None and not min_lr_achieved:
                last_min_test_loss = np.argmin(test_loss_history)
                if (
                    len(test_loss_history) - (last_min_test_loss + 1) > patience
                    and tp.epoch - last_lr_reduction_epoch > patience
                ):
                    current_lr = tp.optimizer.learning_rate.numpy()
                    new_lr = current_lr * lr_reduction_factor
                    if lr_min is not None:
                        if new_lr < lr_min and lr_stop_at_min:
                            log.info(
                                f"Minimal value of learning rate is achieved  ({new_lr:.3e} < {lr_min:.3e}), stopping"
                            )
                            if (
                                loss_weights_switch is not None
                                and tp.epoch < loss_weights_switch
                            ):  # to avoid multiple switches
                                tp.epoch = loss_weights_switch  # jump to new epoch
                                log.info(f"Jump to epoch #{tp.epoch}")
                            else:
                                break  # stop training loop
                        else:  # not lr_stop_at_min or new_lr > lr_min
                            if new_lr < lr_min:
                                new_lr = lr_min
                                min_lr_achieved = True
                                log.info(
                                    "Minimal value of learning rate is achieved, no further reduction"
                                )

                            last_lr_reduction_epoch = tp.epoch
                            tp.optimizer.learning_rate.assign(new_lr)
                            log.info(
                                f"Reducing learning rate: {current_lr:.5e} -> {new_lr:.5e}"
                            )
                    else:  # lr_min is None
                        last_lr_reduction_epoch = tp.epoch
                        tp.optimizer.learning_rate.assign(new_lr)
                        log.info(
                            f"Reducing learning rate: {current_lr:.5e} -> {new_lr:.5e}"
                        )
        else:  # test_ds is None
            log.info(
                f"Iteration #{tp.epoch}/{epochs}: TRAIN stats: {metrics_to_string(final_train_metrics)}"
            )

        # regular checkpoint
        if (tp.epoch) % checkpoint_freq == 0:
            log.info("Regular checkpointing")
            if fit_config.get("save_all_regular_checkpoints", False):
                tp.save_checkpoint(suffix=f".epoch_{tp.epoch}")
            else:
                tp.save_checkpoint()

        # TODO: restore non-EMA weights?

        # E-F WEIGHTS SWITCHING: loss component switch
        if loss_weights_switch is not None and tp.epoch == loss_weights_switch:
            # TODO: need better names
            do_loss_switch(tp, loss_weights_switch, new_loss_params, test_ds)
            best_test_loss = 1e99  # reset best test loss after switch
            test_loss_history = [best_test_loss]  # reset test loss history

        tp.epoch += 1


def do_loss_switch(tp, loss_weights_switch, new_loss_params, test_ds):
    log.info(
        f"Switching loss weights to {new_loss_params} after {loss_weights_switch} epochs"
    )
    new_lr = new_loss_params.pop("learning_rate", None)
    tp.loss_function.set_loss_component_params(new_loss_params)
    if test_ds is not None:  # has test set
        try_load_checkpoint(tp, restart_best_test=True)
        # save best test loss for stage1
        tp.save_checkpoint(suffix=".stage1.best_test_loss")
    # set new learning rate
    if new_lr:
        log.info(f"Set new learning rate to {new_lr:.3e}")
        tp.optimizer.learning_rate.assign(new_lr)


def train_bfgs(
    tp,
    fit_config,
    train_ds,
    test_ds=None,
    strategy=None,
    seed=None,
    train_grouping_df=None,
    test_grouping_df=None,
    compute_init_stats: bool = False,
):
    epochs = fit_config.get("maxiter", 500)
    checkpoint_freq = fit_config.get("checkpoint_freq", 10)

    progress_bar = fit_config.get("progressbar", False)
    init_flat_vars = tp.model.get_flat_trainable_variables()
    log.info(f"Number of trainable parameters: {len(init_flat_vars)}")
    best_test_loss = 1e16
    tp.epoch = 0

    #### INIT STATS ####
    if compute_init_stats:
        train_acc_metrics, train_agg_concat_per_structure_metrics = test_one_epoch(
            tp, train_ds, strategy, compute_concat_per_structure_metrics=True
        )
        initial_train_metrics = process_acc_metrics(
            train_acc_metrics,
            n_batches=len(train_ds),
        )
        initial_train_metrics["epoch"] = tp.epoch
        initial_train_metrics = {
            k.replace("test", "train"): v for k, v in initial_train_metrics.items()
        }
        dump_metrics(
            filename=os.path.join(tp.output_dir, "train_metrics.yaml"),
            metrics=initial_train_metrics,
        )
        if test_ds is not None:
            test_acc_metrics, test_agg_concat_per_structure_metrics = test_one_epoch(
                tp, test_ds, strategy, compute_concat_per_structure_metrics=True
            )
            initial_test_metrics = process_acc_metrics(
                test_acc_metrics,
                n_batches=len(test_ds),
            )
            initial_test_metrics["epoch"] = tp.epoch
            if test_grouping_df is not None:
                compute_per_group_metrics(
                    test_grouping_df,
                    test_agg_concat_per_structure_metrics,
                    initial_test_metrics,
                    n_batches=len(test_ds),
                )
            dump_metrics(
                filename=os.path.join(tp.output_dir, "test_metrics.yaml"),
                metrics=initial_test_metrics,
            )
            msg = generate_train_test_message(
                initial_train_metrics, initial_test_metrics, tp.epoch, epochs
            )
            log.info(msg)
            current_test_loss = initial_test_metrics["total_loss/test"]
            best_test_loss = current_test_loss
        else:
            log.info(
                f"Iteration #{tp.epoch}/{epochs}: TRAIN stats: {metrics_to_string(initial_train_metrics)}"
            )
    #### END INIT STATS ####

    tp.epoch = 1
    opt = fit_config["optimizer"]
    final_train_metrics = {}

    def fit_func(x):
        tp.model.set_flat_trainable_variables(x)
        # -----------------TRAIN---------------------
        train_acc_metrics, train_agg_concat_per_structure_metrics = (
            train_bfgs_one_epoch(
                tp,
                train_ds,
                strategy,
                progress_bar=progress_bar,
                compute_concat_per_structure_metrics=True,
            )
        )
        tot_loss = train_acc_metrics["total_loss/train"]
        tot_jac = train_acc_metrics["total_jac"]
        # post_process agg_metrics
        nonlocal final_train_metrics
        final_train_metrics = process_acc_metrics(
            train_acc_metrics, n_batches=len(train_ds)
        )
        return tot_loss, tot_jac

    def callback(x):
        nonlocal final_train_metrics, best_test_loss
        final_train_metrics["epoch"] = tp.epoch
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
        )
        if test_ds is not None:
            tp.model.set_flat_trainable_variables(x)
            test_acc_metrics, test_agg_concat_per_structure_metrics = test_one_epoch(
                tp, test_ds, strategy, compute_concat_per_structure_metrics=True
            )
            final_test_metrics = process_acc_metrics(
                test_acc_metrics,
                n_batches=len(test_ds),
            )
            final_test_metrics["epoch"] = tp.epoch
            if test_grouping_df is not None:
                compute_per_group_metrics(
                    test_grouping_df,
                    test_agg_concat_per_structure_metrics,
                    final_test_metrics,
                    n_batches=len(test_ds),
                )
            dump_metrics(
                filename=os.path.join(tp.output_dir, "test_metrics.yaml"),
                metrics=final_test_metrics,
            )
            msg = generate_train_test_message(
                final_train_metrics, final_test_metrics, tp.epoch, epochs
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
                f"Iteration #{tp.epoch}/{epochs}: TRAIN stats: {metrics_to_string(final_train_metrics)}"
            )

        if tp.epoch % checkpoint_freq == 0:
            log.info("Regular checkpointing")
            tp.model.set_flat_trainable_variables(x)
            if fit_config.get("save_all_regular_checkpoints", False):
                tp.save_checkpoint(suffix=f".epoch_{tp.epoch}")
            else:
                tp.save_checkpoint()
        tp.epoch += 1

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


def aggregate_per_struct_metrics(metrics_per_struct):
    """
    For one batch, aggregates metrics by summation
    """
    # TODO: implement structure_id aggregation, implement generation of per-structure-group metrics
    assert (
        "nat/per_struct" in metrics_per_struct
    ), "Can't aggregate metrics without `nat/per_struct`"
    nat_per_struct = np.array(metrics_per_struct["nat/per_struct"])
    out_dict = {}
    for k, v in metrics_per_struct.items():
        if k == "nat/per_struct":
            out_dict["num_atoms"] = np.sum(v)
        elif k.endswith("/per_struct"):
            out_dict[k] = np.sum(v)
        elif "loss" in k or "jac" in k:
            if np.any(np.isnan(v)):
                print("NAN found!")
            out_dict[k] = np.array(v)  # per batch (already reduced over mini batches)
        else:
            pass  # raise ValueError(f"Unknown key to proceed: {k}")
    out_dict["num_struct"] = len(nat_per_struct)
    out_dict["num_step"] = 1
    return out_dict


def process_acc_metrics(total_metrics, n_batches):
    """
    Convert accumulated over batches metrics to total metric (by normalizing to proper number of observations)
    """
    num_struct = total_metrics["num_struct"]  # in dataset
    num_atoms = total_metrics["num_atoms"]  # in dataset

    final_metrics = normalize_group_metrics(
        total_metrics, num_struct, num_atoms, n_batches
    )
    return final_metrics


def normalize_group_metrics(total_metrics, num_struct, num_atoms, n_batches):
    final_metrics = {}
    for k, v in total_metrics.items():
        if "abs" in k:
            # TODO: infer behaviour from key
            if "depa" in k:
                final_metrics["mae/depa"] = v / num_struct
            elif "de" in k:
                final_metrics["mae/de"] = v / num_struct
            elif "df" in k:
                final_metrics["mae/f_comp"] = v / num_atoms / 3
            elif "dv" in k:
                final_metrics["mae/virial"] = v / num_struct / 6
            elif "stress" in k:
                final_metrics["mae/stress"] = v / num_struct / 6
        elif "sqr" in k:
            if "depa" in k:
                final_metrics["rmse/depa"] = np.sqrt(v / num_struct)
            elif "de" in k:
                final_metrics["rmse/de"] = np.sqrt(v / num_struct)
            elif "df" in k:
                final_metrics["rmse/f_comp"] = np.sqrt(v / num_atoms / 3)
            elif "dv" in k:
                final_metrics["rmse/virial"] = np.sqrt(v / num_struct / 6)
            elif "stress" in k:
                final_metrics["rmse/stress"] = np.sqrt(v / num_struct / 6)
        elif "loss" in k:
            # overall loss should be normalized by number of structures
            final_metrics[k] = float(v) / n_batches  # per dataset
        elif "total_time" in k:
            final_metrics[k + "/per_atom"] = v / num_atoms
    return final_metrics


def accumulate_metrics(metrics, acc_metrics):
    """Accumulate metrics over batches"""
    for k, v in metrics.items():
        if k in acc_metrics:
            acc_metrics[k] += v
        else:
            acc_metrics[k] = v
    return acc_metrics


def try_load_checkpoint(
    tp,
    restart_best_test=True,
    restart_latest=False,
    restart_suffix=None,
    expect_partial=False,
    verbose=True,
):
    if restart_latest:
        log.info("Trying to load latest regular checkpoint")
        tp.load_checkpoint(expect_partial=expect_partial, verbose=verbose)
    elif restart_best_test:
        log.info("Trying to load best test checkpoint")
        tp.load_checkpoint(
            suffix=".best_test_loss", expect_partial=expect_partial, verbose=verbose
        )
    elif restart_suffix is not None:
        log.info(f"Trying to load checkpoint with suffix {restart_suffix}")
        tp.load_checkpoint(
            suffix=restart_suffix, expect_partial=expect_partial, verbose=verbose
        )


def generate_train_test_message(final_train_metrics, final_test_metrics, epoch, epochs):
    METRICS_SKIP_LIST = [
        "mae/de",
        "rmse/de",
        "rmse/virial",
        "mae/virial",
        "epoch",
        "per_group_metrics",
    ]
    if epoch == 0:
        msg = f"INITIAL    TRAIN(TEST): "
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
            msg += f"{k}(GPa): {final_train_metrics[k]*eV_A3_to_GPa:.3e} "
            if k in final_test_metrics:
                msg += f"({final_test_metrics[k]*eV_A3_to_GPa:.3e}) "
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
    grouping_df, agg_concat_per_structure_metrics, final_metrics, n_batches
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
        )
        norm_cur_metrics["num_struct"] = num_struct
        norm_cur_metrics["num_atoms"] = num_atoms
        per_group_metrics[group_name] = norm_cur_metrics

    final_metrics["per_group_metrics"] = per_group_metrics


def restart_optimizer(optimizer: tf.keras.optimizers.Adam):
    """
    Reset the state of Keras optimizer
    """
    if isinstance(optimizer, tf.keras.optimizers.Adam):

        var_lists_name = ["_momentums", "_velocities", "_velocity_hats"]
        var_lists = []
        for name in var_lists_name:
            if hasattr(optimizer, name):
                var_lists.append(getattr(optimizer, name))
        for var_list in var_lists:
            for var in var_list:
                var.assign(tf.zeros(var.shape, dtype=var.dtype))
    else:
        raise NotImplementedError(f"Restarting {optimizer} is not implemented")
