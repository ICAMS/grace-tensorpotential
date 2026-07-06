from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile

import tensorflow as tf

from tensorpotential.loss import LossFunction
from tensorpotential.metrics import ComputeMetrics
from tensorpotential.tpmodel import (
    ComputeBatchEnergyAndForces,
    ComputeStructureEnergyAndForcesAndVirial,
    TPModel,
)
from tensorpotential.utils import is_chief


@contextlib.contextmanager
def _chief_or_discardable_path(path, strategy):
    """Yield `path` on chief; on non-chief, yield a unique temp path and rmtree after.

    MultiWorkerMirroredStrategy requires every worker to participate in save ops
    (they may invoke collectives internally), but only one worker's output should
    land on the shared filesystem. Non-chief workers write to a local temp dir
    which is removed when the context exits.
    """
    if is_chief(strategy):
        yield path
        return
    tmp_dir = tempfile.mkdtemp(prefix="tp_nonchief_")
    try:
        yield os.path.join(tmp_dir, os.path.basename(path) or "out")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def get_output_dir(seed=42):
    """
    generate name and make folder
    :param seed:
    :return:  folder name
    """
    path = os.path.join("seed", f"{seed}")
    return path


class TensorPotential:
    """
    Class for fitting tp models
    """

    def __init__(
        self,
        potential,
        fit_config=None,
        global_batch_size=None,
        global_test_batch_size=None,
        loss_function: LossFunction = None,
        regularization_loss: LossFunction = None,
        compute_metrics: ComputeMetrics = None,
        strategy: tf.distribute.Strategy = None,
        model_compute_function=ComputeStructureEnergyAndForcesAndVirial(),
        model_train_function=ComputeBatchEnergyAndForces(),
        param_dtype: tf.DType = tf.float32,
        eager_mode: bool = False,
        jit_compile: bool = True,
        seed: int = 42,
        loss_norm_by_batch_size: bool = False,
    ):
        # get default mock strategy (single GPU mode)
        if strategy is None:
            strategy = tf.distribute.get_strategy()
        self.strategy = strategy
        self.fit_config = fit_config or {}
        self.global_batch_size = global_batch_size
        self.global_test_batch_size = global_test_batch_size or self.global_batch_size
        self.loss_function = loss_function
        self.regularization_loss = regularization_loss
        self.compute_metrics = compute_metrics
        self.param_dtype = param_dtype
        self.float_dtype = tf.float64

        # global output dir (will be used in many places)
        self.seed = seed
        self.output_dir = get_output_dir(seed=seed)

        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "checkpoint")
        self.optimizer = None
        self.loss_norm_by_batch_size = loss_norm_by_batch_size

        self.stop_training = False

        with self.strategy.scope():
            self.model = TPModel(
                potential,
                train_function=model_train_function,
                compute_function=model_compute_function,
            )
            self.model.build(
                param_dtype=self.param_dtype,
                input_signature_float_dtype=self.float_dtype,
            )
            if self.loss_function is not None:
                self.loss_function.build(self.float_dtype)
            if self.regularization_loss is not None:
                self.regularization_loss.build(self.float_dtype)

            if eager_mode and jit_compile:
                logging.warning(
                    "`eager_mode` will overwrite `jit_compile` option to False"
                )
                jit_compile = False

            if not eager_mode:
                self.decorate_tf_function(jit_compile=jit_compile)
                # TODO:
                # self.model.decorate_compute_function(float_dtype = self.float_dtype, jit_compile = jit_compile)

            opt = self.fit_config.get("optimizer")
            if opt == "Adam":
                self.optimizer = tf.keras.optimizers.Adam(
                    **self.fit_config.get(
                        "opt_params", {"learning_rate": 0.01, "amsgrad": True}
                    ),
                )
                self._setup_weight_decay_exclusions()
                try:
                    self.use_ema = self.fit_config["opt_params"]["use_ema"]
                except KeyError:
                    self.use_ema = False

                self.swap_on_epoch = False
                if self.use_ema:
                    self.swap_on_epoch = True
                    self._ema_weights_in_model = False
                # self.setup_checkpoint_with_optimizer()
            # else:
            #     self.checkpoint = tf.train.Checkpoint(
            #         model=self.model,
            #         step=tf.Variable(0, dtype=tf.int32, trainable=False),
            #         epoch=tf.Variable(0, dtype=tf.int32, trainable=False),
            #     )
            self.setup_checkpoint()

    # Allowlist of variable name patterns that should have weight decay applied.
    # Only variables whose names contain at least one of these patterns will be
    # subject to weight decay. Everything else (norms, scales, shifts, gates,
    # embeddings, readout) is automatically excluded.
    WD_ALLOW_PATTERNS = ["reducing_"]

    def _setup_weight_decay_exclusions(self):
        """Exclude all variables except those matching WD_ALLOW_PATTERNS from weight decay."""
        wd = self.optimizer.weight_decay
        if not wd:
            return

        exclude_names = []
        for v in self.model.variables_to_train:
            has_wd = any(pat in v.name for pat in self.WD_ALLOW_PATTERNS)
            tag = " WD " if has_wd else "SKIP"
            logging.info(f"Weight decay: [{tag}] {v.name} {v.shape}")
            if not has_wd:
                exclude_names.append(v.name)

        self.optimizer.exclude_from_weight_decay(var_names=exclude_names)

    def setup_checkpoint(self, with_optimizer=True):
        trackable_objs = {
            "model": self.model,
            "step": tf.Variable(0, dtype=tf.int32, trainable=False),
            "epoch": tf.Variable(0, dtype=tf.int32, trainable=False),
            "intra_epoch_save": tf.Variable(False, dtype=tf.bool, trainable=False),
        }
        if self.optimizer and with_optimizer:
            trackable_objs["optimizer"] = self.optimizer
        logging.info(
            f"Setting checkpoint with trackable objects: {trackable_objs.keys()}"
        )
        self.checkpoint = tf.train.Checkpoint(**trackable_objs)

    @property
    def step(self):
        return int(self.checkpoint.step.numpy())

    @step.setter
    def step(self, value):
        self.checkpoint.step.assign(value)

    def increment_step(self):
        self.checkpoint.step.assign_add(1)

    @property
    def epoch(self):
        return int(self.checkpoint.epoch.numpy())

    @epoch.setter
    def epoch(self, value):
        self.checkpoint.epoch.assign(value)

    def increment_epoch(self):
        self.checkpoint.epoch.assign_add(1)

    @property
    def intra_epoch_save(self):
        return bool(self.checkpoint.intra_epoch_save.numpy())

    @intra_epoch_save.setter
    def intra_epoch_save(self, value):
        self.checkpoint.intra_epoch_save.assign(bool(value))

    def reset_epoch_and_step(self):
        self.checkpoint.epoch.assign(0)
        self.checkpoint.step.assign(0)
        # No mid-epoch position remains once epoch/step are zeroed.
        self.checkpoint.intra_epoch_save.assign(False)

    def reset_optimizer(self):
        with self.strategy.scope():
            if self.optimizer:
                self.optimizer = self.optimizer.__class__(**self.optimizer.get_config())
                self.setup_checkpoint()
        # Mid-epoch rewind exists to "complete" gradient updates already baked
        # into optimizer moments. With moments gone, rewind has no purpose.
        self.checkpoint.intra_epoch_save.assign(False)

    def get_model_grad_signatures(self):
        dtypes = {"int": tf.int32, "float": self.float_dtype}
        input_signature = {}
        for k, v in self.model.train_specs.items():
            input_signature[k] = tf.TensorSpec(
                shape=v["shape"], dtype=dtypes[v["dtype"]], name=k
            )
        if self.loss_function is not None:
            for k, v in self.loss_function.get_input_signatures().items():
                input_signature[k] = tf.TensorSpec(
                    shape=v["shape"], dtype=dtypes[v["dtype"]], name=k
                )
        if self.compute_metrics is not None:
            for k, v in self.compute_metrics.get_input_signatures().items():
                input_signature[k] = tf.TensorSpec(
                    shape=v["shape"], dtype=dtypes[v["dtype"]], name=k
                )
        return input_signature

    def get_model_predict_signatures(self):
        dtypes = {"int": tf.int32, "float": self.float_dtype}
        input_signature = {}
        for k, v in self.model.train_specs.items():
            input_signature[k] = tf.TensorSpec(
                shape=v["shape"], dtype=dtypes[v["dtype"]], name=k
            )
        for k, v in self.compute_metrics.get_input_signatures().items():
            input_signature[k] = tf.TensorSpec(
                shape=v["shape"], dtype=dtypes[v["dtype"]], name=k
            )
        return input_signature

    def decorate_tf_function(self, jit_compile=False):
        self.model_grad = tf.function(
            self.model_grad,
            jit_compile=jit_compile,
            reduce_retracing=True,
            input_signature=[self.get_model_grad_signatures()],
        )
        self.predict = tf.function(
            self.predict,
            jit_compile=jit_compile,
            reduce_retracing=True,
            input_signature=[self.get_model_grad_signatures()],
        )
        self.distributed_train_step = tf.function(
            self.distributed_train_step, reduce_retracing=True
        )
        self.distributed_bfgs_train_step = tf.function(
            self.distributed_bfgs_train_step, reduce_retracing=True
        )
        self.distributed_test_step = tf.function(
            self.distributed_test_step, reduce_retracing=True
        )

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def load_checkpoint(
        self,
        checkpoint_name=None,
        suffix="",
        expect_partial: bool = False,
        verbose=False,
        raise_errors=True,
        assert_consumed=False,
        assert_existing_objects_matched=False,
        model_only: bool = False,
    ):
        """load checkpoint from checkpoint_dir if exists.

        ``model_only=True`` reads only the ``model`` (weights) subtree, ignoring
        the training-bookkeeping trackables (step/epoch/intra_epoch_save) and
        optimizer slots. Use it for inference / UQ loads where those are
        irrelevant and a strict object match must not fail on checkpoints from
        older fits that never wrote a given auxiliary trackable.
        """
        if checkpoint_name is None:
            checkpoint_name = self.checkpoint_prefix + suffix
        if os.path.exists(f"{os.path.join(checkpoint_name)}.index"):
            ckpt = (
                tf.train.Checkpoint(model=self.model)
                if model_only
                else self.checkpoint
            )
            with self.strategy.scope():
                status = ckpt.read(checkpoint_name)
                if expect_partial:
                    status.expect_partial()
            if assert_consumed:
                status.assert_consumed()
            elif assert_existing_objects_matched:
                status.assert_existing_objects_matched()
            if verbose:
                logging.info(f"Loaded checkpoint from {checkpoint_name}")
        else:
            if verbose:
                logging.info(
                    f"FAILED Loaded checkpoint from {checkpoint_name} (path does not exist)"
                )
            if raise_errors:
                raise ValueError(
                    f"No checkpoint index found at {checkpoint_name}.index"
                )

    def save_checkpoint(
        self, suffix="", checkpoint_name=None, verbose=False, is_mid_epoch=False
    ):
        """save checkpoint to checkpoint_dir, overwrite if exists.

        Sets the on-disk `intra_epoch_save` flag to `is_mid_epoch` so the
        resume path can distinguish a mid-epoch save (caller passes True
        from the intra-epoch save closure) from an epoch-boundary save
        (default). All other call sites inherit False.
        """
        self.checkpoint.intra_epoch_save.assign(bool(is_mid_epoch))
        checkpoint_name = checkpoint_name or (self.checkpoint_prefix + suffix)
        with _chief_or_discardable_path(checkpoint_name, self.strategy) as write_name:
            if verbose and is_chief(self.strategy):
                logging.info(f"Writing checkpoint to {write_name}.*")
            if os.path.dirname(write_name):
                os.makedirs(os.path.dirname(write_name), exist_ok=True)
            self.checkpoint.write(write_name)

    def save_model(self, model_name=None, jit_compile=True, exact_path=None):
        """Saves model for serving"""
        exact_path = exact_path or os.path.join(self.output_dir, model_name)
        with _chief_or_discardable_path(exact_path, self.strategy) as write_path:
            if os.path.dirname(write_path):
                os.makedirs(os.path.dirname(write_path), exist_ok=True)
            self.model.save_model(
                write_path,
                jit_compile=jit_compile,
                input_signature_float_dtype=self.float_dtype,
            )

    def save_model_with_aux_computes(
        self,
        model_name=None,
        jit_compile=True,
        exact_path=None,
        communicated_keys=None,
        gmm_uq_model=None,
    ):
        """Saves model with all auxiliary compute functions"""
        exact_path = exact_path or os.path.join(self.output_dir, model_name)
        with _chief_or_discardable_path(exact_path, self.strategy) as exact_path:
            self._save_model_with_aux_computes_impl(
                exact_path=exact_path,
                model_name=model_name,
                jit_compile=jit_compile,
                communicated_keys=communicated_keys,
                gmm_uq_model=gmm_uq_model,
            )

    def _save_model_with_aux_computes_impl(
        self,
        exact_path,
        model_name,
        jit_compile,
        communicated_keys,
        gmm_uq_model=None,
    ):
        if os.path.dirname(exact_path):
            os.makedirs(os.path.dirname(exact_path), exist_ok=True)

        try:
            from tensorpotential.tpmodel import ComputeEnergy
            from tensorpotential.instructions.output import CreateOutputTarget
            from tensorpotential import constants as _tc

            # Decide whether to use 2L parallel split
            from tensorpotential.instructions.compute import (
                SingleParticleBasisFunctionEquivariantInd,
            )

            use_2l_parallel = False
            # When exporting with UQ, use standard compute for the 'compute' signature
            # so its tf.function captures no GMM variables (TF cannot track them through
            # the non-tf.Module compute_function attribute). 'compute_uq' is added
            # separately by save_model and is the only signature that uses the GMM model.
            export_compute_fn = (
                ComputeStructureEnergyAndForcesAndVirial()
                if gmm_uq_model is not None
                else self.model.compute_function
            )
            has_custom_compute = not isinstance(
                export_compute_fn, ComputeStructureEnergyAndForcesAndVirial
            )
            has_2layer = any(
                isinstance(ins, SingleParticleBasisFunctionEquivariantInd)
                for ins in self.model.instructions.values()
            )
            model_type = "2L" if has_2layer else "1L"

            extra_aux_computes = {}
            if not has_custom_compute:
                has_energy_output = any(
                    isinstance(ins, CreateOutputTarget)
                    and ins.name == _tc.PREDICT_ATOMIC_ENERGY
                    for ins in self.model.instructions.values()
                )
                if has_energy_output:
                    extra_aux_computes["compute_energy"] = ComputeEnergy()

            if not has_custom_compute and has_2layer and communicated_keys is None:
                # Auto-detect communicated_keys for 2L models
                try:
                    from tensorpotential.instructions.instruction_graph_utils import (
                        infer_communicated_keys,
                    )

                    communicated_keys = infer_communicated_keys(self.model.instructions)
                    logging.info(f"Auto-detected communicated_keys={communicated_keys}")
                except Exception as e:
                    logging.warning(f"Could not auto-detect communicated_keys: {e}")

            if has_custom_compute:
                logging.info(
                    f"Custom compute_function detected for {model_type} model, skipping extra aux functions."
                )
            elif communicated_keys and len(communicated_keys) >= 2:
                use_2l_parallel = True
            elif communicated_keys:
                logging.warning(
                    f"communicated_keys={communicated_keys} has fewer than 2 keys, "
                    "skipping 2L parallel aux functions."
                )

            if use_2l_parallel:
                from tensorpotential.instructions.instruction_graph_utils import (
                    build_split_tpmodel,
                )

                logging.info(
                    f"Saving 2L model with parallel aux functions. Communicated keys: {communicated_keys}"
                )
                with self.strategy.scope():
                    m_aux = build_split_tpmodel(
                        self.model.instructions,
                        communicated_keys=communicated_keys,
                        param_dtype=self.param_dtype,
                        input_signature_float_dtype=self.float_dtype,
                        jit_compile=jit_compile,
                        extra_aux_computes=extra_aux_computes,
                    )
                    m_aux.save_model(
                        exact_path,
                        jit_compile=jit_compile,
                        input_signature_float_dtype=self.float_dtype,
                        gmm_uq_model=gmm_uq_model,
                    )
            else:
                with self.strategy.scope():
                    m_aux = TPModel(
                        instructions=self.model.instructions,
                        compute_function=export_compute_fn,
                        train_function=self.model.train_function,
                        aux_compute=extra_aux_computes,
                    )
                    m_aux.build(param_dtype=self.param_dtype, jit_compile=jit_compile)
                    m_aux.save_model(
                        exact_path,
                        jit_compile=jit_compile,
                        input_signature_float_dtype=self.float_dtype,
                        gmm_uq_model=gmm_uq_model,
                    )
        except Exception:
            logging.exception("Failed to save model with auxiliary compute functions")
            if gmm_uq_model is not None:
                # Fallback save_model uses the original model which may have a UQ
                # compute_function capturing untracked GMM variables — don't attempt it.
                raise
            logging.warning(
                "Falling back to standard save_model without auxiliary functions."
            )
            self.save_model(
                model_name=model_name,
                jit_compile=jit_compile,
                exact_path=exact_path,
            )

    def export_to_yaml(self, model_name=None, exact_filename=None):
        """Export FS-model to YAML file"""
        exact_filename = exact_filename or os.path.join(self.output_dir, model_name)
        if os.path.dirname(exact_filename):
            os.makedirs(os.path.dirname(exact_filename), exist_ok=True)
        self.model.export_to_yaml(exact_filename)

    @staticmethod
    def load_model(path: str) -> tf.saved_model:
        return tf.saved_model.load(path)

    def compute_loss(
        self,
        input_data: dict,
        model_predictions: dict,
    ):
        loss = self.loss_function(
            input_data=input_data,
            predictions=model_predictions,
        )
        loss = {k: tf.reduce_sum(v) for k, v in loss.items()}
        total_loss = loss[self.loss_function.name]
        loss = {k: v for k, v in loss.items() if k != self.loss_function.name}
        return total_loss, loss

    # @tf.function(jit_compile=True)
    def model_grad(self, input_data):
        with tf.GradientTape() as tape:
            model_prediction = self.model(input_data, training=True)
            total_loss, losses_dict = self.compute_loss(
                input_data=input_data,
                model_predictions=model_prediction,
            )
            losses_dict = {
                k: v / self.global_batch_size for k, v in losses_dict.items()
            }
            if self.loss_norm_by_batch_size:
                # THIS IS NOT CORRECT, it is divided effectively by compute_loss already
                total_loss /= self.global_batch_size

            if self.regularization_loss is not None:
                reg_los = self.regularization_loss(model=self.model)[
                    self.regularization_loss.name
                ]
                total_loss += reg_los / self.strategy.num_replicas_in_sync

        # custom filtered variables to train
        gradients = tape.gradient(total_loss, self.model.variables_to_train)

        return total_loss, model_prediction, gradients, losses_dict

    def train_step(self, input_data):  # on each node, get mini_batch
        result = {}
        # TODO: Possibly filter data here?
        total_loss, model_prediction, gradients, losses_dict = self.model_grad(
            input_data
        )  # under XLA
        metrics = self.compute_metrics(input_data, model_prediction)
        result.update(
            {
                f"{self.loss_function.name}/train": total_loss,
            }
        )
        result.update(metrics)
        self.optimizer.apply_gradients(zip(gradients, self.model.variables_to_train))
        for k, v in losses_dict.items():
            result[f"loss_component/{k}/train"] = v
        return result

    def train_bfgs_step(self, input_data):
        result = {}
        total_loss, model_prediction, gradients, losses_dict = self.model_grad(
            input_data
        )
        metrics = self.compute_metrics(input_data, model_prediction)

        flat_grad = []
        for i, gvar in enumerate(gradients):
            gv = tf.reshape(gvar, [-1])
            flat_grad += [gv]
        flat_grad = tf.concat(flat_grad, axis=0)

        result.update(
            {f"{self.loss_function.name}/train": total_loss, "total_jac": flat_grad}
        )
        for k, v in losses_dict.items():
            result[f"loss_component/{k}/train"] = v
        result.update(metrics)

        return result

    # @tf.function
    def predict(self, input_data, training: bool = False):
        model_prediction = self.model(input_data, training=training)
        return model_prediction

    def reduce_dict(self, d: dict):  # on root node
        """reduce items in dictionary d"""
        # TODO:  cleaner solution ?
        if self.strategy.num_replicas_in_sync > 1:
            for k, v in d.items():
                # dtype = v.values[0].dtype
                if k.endswith("/per_struct"):
                    try:
                        d[k] = tf.concat(v.values, axis=0)  # over all replics
                    except AttributeError:
                        d[k] = tf.concat(v, axis=0)
                elif "loss" in k or "jac" in k:
                    d[k] = self.strategy.reduce(
                        tf.distribute.ReduceOp.SUM, v, axis=None
                    )
                else:
                    pass  # raise ValueError(f"Unknown key: {k}")

    # @tf.function
    def distributed_train_step(self, input_data):  # on root node
        def value_fn(ctx):
            return input_data[ctx.replica_id_in_sync_group]

        if isinstance(input_data, list):  # manual list/tuple of batches
            distributed_values = (
                self.strategy.experimental_distribute_values_from_function(value_fn)
            )
        else:  # already dict of PerReplica or of tensors
            distributed_values = input_data  #

        results = self.strategy.run(self.train_step, args=(distributed_values,))
        self.reduce_dict(results)
        return results

    # @tf.function
    def distributed_bfgs_train_step(self, input_data):
        def value_fn(ctx):
            return input_data[ctx.replica_id_in_sync_group]

        distributed_values = self.strategy.experimental_distribute_values_from_function(
            value_fn
        )

        results = self.strategy.run(self.train_bfgs_step, args=(distributed_values,))
        self.reduce_dict(results)
        # tf.print(results, 'results')
        return results

    def test_step(self, input_data):
        result = {}
        model_prediction = self.predict(input_data, training=False)
        metrics = self.compute_metrics(input_data, model_prediction)

        total_loss, losses_dict = self.compute_loss(
            input_data=input_data,
            model_predictions=model_prediction,
        )

        losses_dict = {
            k: v / self.global_test_batch_size for k, v in losses_dict.items()
        }
        if self.loss_norm_by_batch_size:
            # THIS IS NOT CORRECT, it is divided effectively by compute_loss already
            total_loss /= self.global_test_batch_size

        result.update(
            {
                f"{self.loss_function.name}/test": total_loss,
            }
        )
        for k, v in losses_dict.items():
            result[f"loss_component/{k}/test"] = v
        result.update(metrics)
        return result

    # @tf.function
    def distributed_test_step(self, input_data):
        def value_fn(ctx):
            return input_data[ctx.replica_id_in_sync_group]

        if isinstance(input_data, list):  # manual list/tuple of batches
            distributed_values = (
                self.strategy.experimental_distribute_values_from_function(value_fn)
            )
        else:  # already dict of PerReplica or of tensors
            distributed_values = input_data

        results = self.strategy.run(self.test_step, args=(distributed_values,))
        self.reduce_dict(results)
        return results

    def _get_ema_optimizer(self):
        """Return the optimizer that holds EMA state, unwrapping LossScaleOptimizer if needed."""
        if hasattr(self.optimizer, "inner_optimizer"):
            return self.optimizer.inner_optimizer
        return self.optimizer

    def _swap_variables(self, optimizer):
        """Swap model weights with EMA shadow weights in-place.

        Uses numpy as intermediate storage. For MirroredVariables,
        .numpy() reads from the primary replica and .assign() broadcasts to all.
        Avoids strategy.extended.update which can be flaky with NCCL on multi-GPU.
        """
        for var, average_var in zip(
            self.model.variables_to_train,
            optimizer._model_variables_moving_average,
        ):
            tmp = var.numpy()
            var.assign(average_var)
            average_var.assign(tmp)

    def _swap_variables_on_device(self, optimizer):
        """Swap model weights with EMA shadow weights in-place (on-device variant).

        Uses tf.identity + strategy.extended.update to stay on GPU.
        Exact precision, single update call per variable pair.
        May have issues with NCCL synchronization on some multi-GPU setups.
        """
        strategy = optimizer._distribution_strategy
        for var, average_var in zip(
            self.model.variables_to_train,
            optimizer._model_variables_moving_average,
        ):

            def swap_fn(v, avg):
                tmp = tf.identity(v)
                v.assign(avg)
                avg.assign(tmp)

            strategy.extended.update(var, swap_fn, args=(average_var,))

    def swap_variables(self):
        optimizer = self._get_ema_optimizer()
        if not hasattr(optimizer, "_model_variables_moving_average"):
            raise ValueError(
                "swap_variables requires use_ema=True on the optimizer. "
                f"Received: use_ema={getattr(optimizer, 'use_ema', False)}"
            )
        self._swap_variables(optimizer)

    @contextlib.contextmanager
    def ema_scope(self):
        """Swap EMA weights into model on enter, swap back on exit.

        No-op when use_ema=False or optimizer not yet built.
        Safe for nested calls (inner scope is a no-op).
        """
        optimizer = self._get_ema_optimizer()
        should_swap = (
            self.use_ema
            and hasattr(optimizer, "_model_variables_moving_average")
            and not self._ema_weights_in_model
        )
        if should_swap:
            self._swap_variables(optimizer)
            self._ema_weights_in_model = True
        try:
            yield
        finally:
            if should_swap:
                self._swap_variables(optimizer)
                self._ema_weights_in_model = False

    def prepare_for_training(self):
        """After loading a checkpoint (which stores model=EMA), swap back to training state.

        No-op when use_ema=False or optimizer not built.
        Safe for old checkpoints where model==shadow (swap is identity).
        """
        if not self.use_ema:
            return
        optimizer = self._get_ema_optimizer()
        if not hasattr(optimizer, "_model_variables_moving_average"):
            return
        self._swap_variables(optimizer)
        self._ema_weights_in_model = False

    def enable_lora_adaptation(self, lora_config=None):
        raise NotImplementedError("This functionality is not yet fully implemented")
        self.model.enable_lora_adaptation(lora_config=lora_config)
        with self.strategy.scope():
            self.setup_checkpoint()

    def finalize_lora_update(self):
        raise NotImplementedError("This functionality is not yet fully implemented")
        self.model.finalize_lora_update()
        logging.info("Resetting optimizer after reducing LORA")
        self.reset_optimizer()

    def set_trainable_variables(self, only_trainable_names, verbose=False):
        with self.strategy.scope():
            self.model.set_trainable_variables(
                only_trainable_names=only_trainable_names, verbose=verbose
            )

    def is_lora_enabled(self):
        return self.model.is_lora_enabled()
