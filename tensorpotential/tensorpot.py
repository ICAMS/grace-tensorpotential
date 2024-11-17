from __future__ import annotations

import logging
import os
import tensorflow as tf

from tensorpotential.metrics import ComputeMetrics
from tensorpotential.tpmodel import (
    TPModel,
    ComputeBatchEnergyAndForces,
    ComputeStructureEnergyAndForcesAndVirial,
)
from tensorpotential.loss import LossFunction


def get_output_dir(seed=42):
    """
    generate name and make folder
    :param seed:
    :return:  folder name
    """
    path = os.path.join("seed", f"{seed}")
    os.makedirs(path, exist_ok=True)
    return path


# TODO: this is fit class, should be renamed?
class TensorPotential:
    """
    Class for fitting tp models
    """

    def __init__(
        self,
        potential,
        fit_config,
        global_batch_size=None,
        global_test_batch_size=None,
        loss_function: LossFunction = None,
        regularization_loss: LossFunction = None,
        compute_metrics: ComputeMetrics = None,
        strategy: tf.distribute.Strategy = None,
        model_compute_function=ComputeStructureEnergyAndForcesAndVirial,
        model_train_function=ComputeBatchEnergyAndForces,
        float_dtype: tf.DType = tf.float64,
        eager_mode: bool = False,
        jit_compile: bool = True,
        seed: int = 42,
        loss_norm_by_batch_size: bool = False,
    ):
        # get default mock strategy (single GPU mode)
        if strategy is None:
            strategy = tf.distribute.get_strategy()
        self.strategy = strategy
        self.fit_config = fit_config
        self.global_batch_size = global_batch_size
        self.global_test_batch_size = global_test_batch_size or self.global_batch_size
        self.loss_function = loss_function
        self.regularization_loss = regularization_loss
        self.compute_metrics = compute_metrics
        self.float_dtype = float_dtype

        # global output dir (will be used in many places)
        self.output_dir = get_output_dir(seed=seed)

        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "checkpoint")
        self.optimizer = None
        self.loss_norm_by_batch_size = loss_norm_by_batch_size

        with self.strategy.scope():
            self.model = TPModel(
                potential,
                train_function=model_train_function,
                compute_function=model_compute_function,
            )
            self.model.build(self.float_dtype)
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

            opt = self.fit_config.get("optimizer", None)
            if opt == "Adam":
                self.optimizer = tf.keras.optimizers.Adam(
                    **self.fit_config.get(
                        "opt_params", {"learning_rate": 0.01, "amsgrad": True}
                    ),
                )
                # self.optimizer.exclude_from_weight_decay(var_names=['no_decay'])
                # self.optimizer.exclude_from_weight_decay(var_names=["reducing", "FC"])
                try:
                    self.use_ema = self.fit_config["opt_params"]["use_ema"]
                except KeyError:
                    self.use_ema = False

                self.swap_on_epoch = False
                if self.use_ema:
                    self.swap_on_epoch = True
                    self._ema_weights_in_model = False

                self.setup_checkpoint_with_optimizer()
            else:
                self.checkpoint = tf.train.Checkpoint(model=self.model)

    def setup_checkpoint_with_optimizer(self):
        self.checkpoint = tf.train.Checkpoint(
            model=self.model, optimizer=self.optimizer
        )

    def reset_optimizer(self):
        with self.strategy.scope():
            self.optimizer = self.optimizer.__class__(**self.optimizer.get_config())
            self.setup_checkpoint_with_optimizer()

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

    def save_checkpoint(self, suffix=""):
        """save checkpoint to checkpoint_dir, overwrite if exists"""
        checkpoint_name = self.checkpoint_prefix + suffix
        self.checkpoint.write(checkpoint_name)

    def load_checkpoint(self, suffix="", expect_partial: bool = False, verbose=False):
        """load checkpoint from checkpoint_dir if exists"""
        checkpoint_name = self.checkpoint_prefix + suffix
        if os.path.exists(f"{os.path.join(checkpoint_name)}.index"):
            with self.strategy.scope():
                if expect_partial:
                    self.checkpoint.read(checkpoint_name).expect_partial()
                else:
                    self.checkpoint.read(checkpoint_name)
            if verbose:
                logging.info(f"Loaded checkpoint from {checkpoint_name}")
        else:
            if verbose:
                logging.info(f"FAILED Loaded checkpoint from {checkpoint_name}")

    def save_model(self, path, jit_compile=True):
        """saves model for serving"""
        self.model.save_model(
            os.path.join(self.output_dir, path),
            jit_compile=jit_compile,
            float_dtype=self.float_dtype,
        )

    def export_to_yaml(self, filename):
        """Export FS-model to YAML file"""
        self.model.export_to_yaml(os.path.join(self.output_dir, filename))

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
        gradients = tape.gradient(total_loss, self.model.trainable_variables)

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
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
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

        distributed_values = self.strategy.experimental_distribute_values_from_function(
            value_fn
        )

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

        distributed_values = self.strategy.experimental_distribute_values_from_function(
            value_fn
        )

        results = self.strategy.run(self.test_step, args=(distributed_values,))
        self.reduce_dict(results)
        return results

    def _swap_variables(self, optimizer):
        for var, average_var in zip(
            self.model.trainable_variables,
            optimizer._model_variables_moving_average,
        ):
            # if isinstance(var, tf.Variable):
            #     var = var.value
            # if isinstance(average_var, tf.Variable):
            #     average_var = average_var.value
            # swap using addition to prevent variable creation
            optimizer._distribution_strategy.extended.update(
                var,
                lambda a, b: a.assign_add(b),
                args=(average_var,),
            )
            optimizer._distribution_strategy.extended.update(
                var,
                lambda a, b: b.assign(a - b),
                args=(average_var,),
            )
            optimizer._distribution_strategy.extended.update(
                var,
                lambda a, b: a.assign(a - b),
                args=(average_var,),
            )

    def swap_variables(self):
        # if hasattr(self.optimizer, "inner_optimizer"):
        #     # LossScaleOptimizer
        #     optimizer = self.optimizer.inner_optimizer
        # else:
        optimizer = self.optimizer
        if not hasattr(optimizer, "_model_variables_moving_average"):
            raise ValueError(
                "SwapEMAWeights must be used when "
                "`use_ema=True` is set on the optimizer. "
                f"Received: use_ema={optimizer.use_ema}"
            )

        self._swap_variables(optimizer)

    def on_epoch_begin(self):
        if self.use_ema and self.swap_on_epoch and self._ema_weights_in_model:
            self.swap_variables()
            self._ema_weights_in_model = False

    # def on_epoch_end(self, epoch, logs=None):
    #     if self.swap_on_epoch and not self._ema_weights_in_model:
    #         self._swap_variables()
    #         self._ema_weights_in_model = True
    #         # We need to recover EMA weights from the previously swapped weights
    #         # in the last epoch. This is because, at the end of the fitting,
    #         # `finalize_variable_values` will be called to assign
    #         # `_model_variables_moving_average` to `trainable_variables`.
    #         if epoch == self.params["epochs"] - 1:
    #             self._finalize_ema_values()
    # def _tf_finalize_ema_values(self, optimizer):
    #     for var, average_var in zip(
    #         self.model.trainable_variables,
    #         optimizer._model_variables_moving_average,
    #     ):
    #         if isinstance(var, backend.Variable):
    #             var = var.value
    #         if isinstance(average_var, backend.Variable):
    #             average_var = average_var.value
    #         optimizer._distribution_strategy.extended.update(
    #             average_var,
    #             lambda a, b: a.assign(b),
    #             args=(var,),
    #         )

    def on_test_begin(self):
        if self.use_ema and not self._ema_weights_in_model:
            self.swap_variables()
            self._ema_weights_in_model = True

    def on_test_end(self):
        if self.use_ema and self._ema_weights_in_model:
            self.swap_variables()
            self._ema_weights_in_model = False

    def finalize_ema_values(self):
        if hasattr(self.optimizer, "inner_optimizer"):
            # LossScaleOptimizer
            optimizer = self.optimizer.inner_optimizer
        else:
            optimizer = self.optimizer
        if not hasattr(optimizer, "_model_variables_moving_average"):
            raise ValueError(
                "SwapEMAWeights must be used when "
                "`use_ema=True` is set on the optimizer. "
                f"Received: use_ema={optimizer.use_ema}"
            )
        if self.use_ema:
            self._finalize_ema_values(optimizer)

    def _finalize_ema_values(self, optimizer):
        for var, average_var in zip(
            self.model.trainable_variables,
            optimizer._model_variables_moving_average,
        ):
            # if isinstance(var, tf.Variable):
            #     var = var.value
            # if isinstance(average_var, tf.Variable):
            #     average_var = average_var.value
            optimizer._distribution_strategy.extended.update(
                average_var,
                lambda a, b: a.assign(b),
                args=(var,),
            )
