import logging
from abc import abstractmethod
from typing import Any, Dict, Union, Protocol

import numpy as np
import tensorflow as tf


from tensorpotential import TensorPotential

log = logging.getLogger(__name__)


class TPCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model: TensorPotential,
        initial_lr: float,
        decay_steps: int,
        warmup_target: float = None,
        warmup_steps: int = 0,
        min_lr: float = 1e-6,
        logfile: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps

        self.warmup_target = warmup_target
        self.warmup_steps = warmup_steps if warmup_target is not None else 0
        self.total_steps = (
            self.decay_steps + self.warmup_steps
            if self._is_warmup()
            else self.decay_steps
        )

        self.min_lr = min_lr
        self.logfile = logfile

        self._log_lr(0, initial_lr)

    # required because tf.Callback.model is a getter whereas TPCallbacks requires also a setter
    def model(self):
        return self.model

    def on_train_begin(self, logs: Dict = None) -> None:
        if self.warmup_steps > self.model.step:
            lr_0 = self.initial_lr if self._is_warmup() else self.warmup_target
            self.model.optimizer.learning_rate.assign(lr_0)

    def on_batch_end(self, batch: int, logs: dict = None) -> None:
        if self.warmup_target is None:
            lr = self._decay_function(batch, self.initial_lr)
        else:
            lr = (
                self._warmup_function(batch)
                if batch < self.warmup_steps
                else self._decay_function(batch - self.warmup_steps, self.warmup_target)
            )
        lr = self.min_lr if batch >= self.total_steps else lr
        self.model.optimizer.learning_rate.assign(lr)
        self._log_lr(batch, lr)

    @abstractmethod
    def _decay_function(self, step: int, decay_from_lr: float):
        pass

    def _warmup_function(self, batch: int):
        completed_fraction = batch / self.warmup_steps
        total_step_delta = self.warmup_target - self.initial_lr
        return total_step_delta * completed_fraction + self.initial_lr

    def _is_warmup(self):
        return self.warmup_steps > 0 and self.warmup_target is not None

    def _log_lr(self, batch: int, learning_rate: float):
        if self.logfile is None:
            return

        with open(self.logfile, "a") as f:
            msg = f"{self.__class__.__name__}: "
            msg += f"lr = {learning_rate:.3e} for step {batch:5d}"
            msg += "\n" if batch < self.total_steps else " # training completed\n"
            f.write(msg)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, initial_lr={self.initial_lr}, decay_steps={self.decay_steps}, warmup_target={self.warmup_target}, warmup_steps={self.warmup_steps}, min_lr={self.min_lr}, logfile={self.logfile})"


class CustomReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(
        self,
        model: TensorPotential,
        factor=0.1,
        patience=10,
        mode="auto",
        min_delta=1e-10,
        cooldown=0,
        min_lr=0.0,
        monitor="train_loss",
        stop_on_min_lr=False,
        **kwargs,
    ):
        super().__init__(
            monitor,
            factor,
            patience,
            0,
            mode,
            min_delta,
            cooldown,
            min_lr,
            **kwargs,
        )
        self.model = model
        self.stop_on_min_lr = stop_on_min_lr
        if self.best == 0.0:
            self.best = 1e99

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        logs = logs or {}
        current = logs.get(self.monitor)
        current_lr = self.model.optimizer.learning_rate.numpy()

        if current is None:
            log.warning(
                f"{self.__class__.__name__}-Callback Warning: {self.monitor} not found in logs. Skipping the call"
            )
            return
        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = self.model.optimizer.learning_rate.numpy()
                if old_lr > np.float32(self.min_lr):
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    self.model.optimizer.learning_rate.assign(new_lr)
                    log.info(
                        f"Reducing learning rate: {current_lr:.5e} -> {new_lr:.5e}"
                    )
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
                elif self.stop_on_min_lr:
                    self.model.stop_training = True
                    log.info(
                        f"Minimum value of learning rate {self.min_lr} is achieved  {self.patience} times - stopping"
                    )
                else:
                    log.info(
                        f"Minimum value of learning rate {self.min_lr} is achieved, no further reduction"
                    )

    def model(self):
        return self.model

    def reset_best(self):
        self.best = 1e99


class ExponentialDecay(TPCallback):
    def __init__(
        self,
        model: TensorPotential,
        initial_lr: float,
        decay_rate: float,
        decay_steps: int,
        staircase=False,
        warmup_target=None,
        warmup_steps=0,
        logfile=None,
        min_lr=1e-6,
        **kwargs,
    ):

        super().__init__(
            model,
            initial_lr,
            decay_steps,
            warmup_target,
            warmup_steps,
            min_lr,
            logfile,
            **kwargs,
        )
        self.decay_rate = decay_rate
        self.staircase = staircase

    def _decay_function(self, step: int, decay_from_lr: float):
        p = step / max(self.decay_steps - 1, 1)
        return decay_from_lr * np.power(self.decay_rate, p)


class CosineDecay(TPCallback):
    def __init__(
        self,
        model: TensorPotential,
        initial_lr: float,
        decay_steps: int,
        min_lr: float = 1e-6,
        warmup_target: float = None,
        warmup_steps: int = 0,
        logfile: str = None,
        **kwargs,
    ):
        super().__init__(
            model,
            initial_lr,
            decay_steps,
            warmup_target,
            warmup_steps,
            min_lr,
            logfile,
            **kwargs,
        )
        # self.alpha = (
        #     min_lr / warmup_target if self._is_warmup() else min_lr / initial_lr
        # )
        self.alpha = (
            min_lr / warmup_target
        )

    def _decay_function(self, step: int, decay_from_lr: float):
        completed_fraction = step / max(self.decay_steps - 1, 1)
        cosine_decayed = 0.5 * (1.0 + np.cos(np.pi * completed_fraction))
        decayed = (1 - self.alpha) * cosine_decayed + self.alpha
        return decay_from_lr * decayed


class LinearDecay(TPCallback):
    def __init__(
        self,
        model: TensorPotential,
        initial_lr: float,
        decay_steps: int,
        min_lr: float = 1e-6,
        warmup_target: float = None,
        warmup_steps: int = 0,
        logfile: str = None,
        **kwargs,
    ):
        super().__init__(
            model,
            initial_lr,
            decay_steps,
            warmup_target,
            warmup_steps,
            min_lr,
            logfile,
            **kwargs,
        )

        self.min_lr = min_lr

    def _decay_function(self, step: int, decay_from_lr: float):
        completed_fraction = step / max(self.decay_steps - 1, 1)
        return (
            1 - completed_fraction
        ) * decay_from_lr + completed_fraction * self.min_lr


class OnBatchEndProtocol(Protocol):
    def on_batch_end(self, batch: int, logs=None) -> None: ...


class OnEpochEndProtocol(Protocol):
    def on_epoch_end(self, epoch: int, logs=None) -> None: ...


class LRSchedulerFactory:
    OPTIMIZER_PARAMS_FIELD: str = "opt_params"
    LR_SCHEDULER_FIELD: str = "scheduler"
    LR_SCHEDULER_PARAMS_FIELD: str = "scheduler_params"
    LEGACY_LR_REDUCER_FIELD: str = "learning_rate_reduction"

    EXPONENTIAL_DECAY_FIELD: str = "exponential_decay"
    COSINE_DECAY_FIELD: str = "cosine_decay"
    LINEAR_DECAY_FIELD: str = "linear_decay"
    REDUCE_ON_PLATEAU_FIELD: str = "reduce_on_plateau"

    DEFAULT_VALUES: Dict[str, Any] = {
        # shared by all shedulers
        "maxiter": 500,
        "minimum_learning_rate": 1e-4,
        # exponential, cosine, linear decay
        "warmup_epochs": 0,
        "cold_learning_rate": 1e-7,
        # exponential decay
        "staircase": False,
        # reduce lr on plateau
        "reduction_factor": 0.8,
        "stop_at_min": False,
        "patience": 10,
        "monitor": "test_loss",
        "cooldown": 0,
        "resume_lr": True,
    }

    @staticmethod
    def create_lr_scheduler(
        model: TensorPotential,
        n_batches: int,
        fit_config: Dict,
        distr_strategy: tf.distribute.Strategy,
    ) -> Union[OnEpochEndProtocol, OnBatchEndProtocol, None]:
        num_replicas = distr_strategy.num_replicas_in_sync
        n_batches = n_batches // num_replicas

        cls = LRSchedulerFactory
        if not cls.if_lr_scheduler(fit_config):
            return None
        if cls.LEGACY_LR_REDUCER_FIELD in fit_config:
            log.warning(
                "DEPRECATION WARNING: input.fit.learning_rate_reduction is now deprecated. Configure it with input.fit.scheduler and input.fit.scheduler_params."
            )
            kwargs = cls.read_legacy_reduce_on_plateau_kwargs(fit_config)
            return CustomReduceLROnPlateau(model, **kwargs)
        if fit_config[cls.LR_SCHEDULER_FIELD] == cls.REDUCE_ON_PLATEAU_FIELD:
            kwargs = cls.read_reduce_on_plateau_kwargs(fit_config)
            return CustomReduceLROnPlateau(model, **kwargs)

        kwargs = cls.read_decay_kwargs(fit_config, n_batches)
        kwargs["model"] = model
        if fit_config[cls.LR_SCHEDULER_FIELD] == cls.COSINE_DECAY_FIELD:
            return CosineDecay(**kwargs)
        if fit_config[cls.LR_SCHEDULER_FIELD] == cls.LINEAR_DECAY_FIELD:
            return LinearDecay(**kwargs)
        if fit_config[cls.LR_SCHEDULER_FIELD] == cls.EXPONENTIAL_DECAY_FIELD:
            kwargs["decay_rate"] = kwargs["min_lr"] / kwargs["warmup_target"]
            kwargs["staircase"] = fit_config.get(
                "staircase", cls.DEFAULT_VALUES["staircase"]
            )
            return ExponentialDecay(**kwargs)

        raise ValueError(
            f"Unsupported learning rate scheduler: {fit_config[cls.LR_SCHEDULER_FIELD]}"
        )

    @staticmethod
    def if_lr_scheduler(fit_config: Dict) -> bool:
        cls = LRSchedulerFactory
        if (
            cls.LR_SCHEDULER_FIELD in fit_config
            and cls.LEGACY_LR_REDUCER_FIELD in fit_config
        ):
            raise ValueError(
                "both old and new scheduler cannot be present in fit input configuration"
            )
        return (
            cls.LR_SCHEDULER_FIELD in fit_config
            or cls.LEGACY_LR_REDUCER_FIELD in fit_config
        )

    @staticmethod
    def read_legacy_reduce_on_plateau_kwargs(fit_config: Dict) -> Dict[str, Any]:
        kwargs = dict()
        cls = LRSchedulerFactory
        lr_red_params = fit_config[cls.LEGACY_LR_REDUCER_FIELD]

        kwargs["factor"] = lr_red_params.get(
            "factor", cls.DEFAULT_VALUES["reduction_factor"]
        )
        adjusted_patience = (
            lr_red_params.get("patience", cls.DEFAULT_VALUES["patience"]) + 1
        )

        kwargs["patience"] = adjusted_patience
        kwargs["min_lr"] = lr_red_params.get(
            "min", cls.DEFAULT_VALUES["minimum_learning_rate"]
        )
        kwargs["monitor"] = cls.DEFAULT_VALUES["monitor"]
        kwargs["stop_on_min_lr"] = lr_red_params.get(
            "stop_at_min", cls.DEFAULT_VALUES["stop_at_min"]
        )
        kwargs["resume_lr"] = lr_red_params.get(
            "resume_lr", cls.DEFAULT_VALUES["resume_lr"]
        )

        return kwargs

    @staticmethod
    def read_reduce_on_plateau_kwargs(fit_config: Dict) -> Dict[str, Any]:
        kwargs = dict()
        cls = LRSchedulerFactory

        kwargs["factor"] = fit_config[cls.LR_SCHEDULER_PARAMS_FIELD].get(
            "reduction_factor", cls.DEFAULT_VALUES["reduction_factor"]
        )
        adjusted_patience = (
            fit_config[cls.LR_SCHEDULER_PARAMS_FIELD].get(
                "patience", cls.DEFAULT_VALUES["patience"]
            )
            + 1
        )
        kwargs["patience"] = adjusted_patience
        kwargs["min_lr"] = fit_config[cls.LR_SCHEDULER_PARAMS_FIELD].get(
            "minimum_learning_rate", cls.DEFAULT_VALUES["minimum_learning_rate"]
        )
        kwargs["monitor"] = fit_config[cls.LR_SCHEDULER_PARAMS_FIELD].get(
            "monitor", cls.DEFAULT_VALUES["monitor"]
        )
        kwargs["stop_on_min_lr"] = fit_config.get(
            "stop_at_min", cls.DEFAULT_VALUES["stop_at_min"]
        )
        kwargs["cooldown"] = fit_config[cls.LR_SCHEDULER_PARAMS_FIELD].get(
            "cooldown", cls.DEFAULT_VALUES["cooldown"]
        )
        kwargs["resume_lr"] = fit_config[cls.LR_SCHEDULER_PARAMS_FIELD].get(
            "resume_lr", cls.DEFAULT_VALUES["resume_lr"]
        )

        return kwargs

    @staticmethod
    def read_decay_kwargs(fit_config: Dict, n_batches: int) -> Dict[str, Any]:
        kwargs = dict()

        cls = LRSchedulerFactory
        maxiter = fit_config.get("maxiter", cls.DEFAULT_VALUES["maxiter"])
        lr = fit_config[cls.OPTIMIZER_PARAMS_FIELD]["learning_rate"]
        min_lr = fit_config[cls.LR_SCHEDULER_PARAMS_FIELD].get(
            "minimum_learning_rate", cls.DEFAULT_VALUES["minimum_learning_rate"]
        )

        warmup_epochs = fit_config[cls.LR_SCHEDULER_PARAMS_FIELD].get(
            "warmup_epochs", cls.DEFAULT_VALUES["warmup_epochs"]
        )
        assert warmup_epochs >= 0, ":warmup_epochs: cannot be negative"

        warmup_steps = int(warmup_epochs * n_batches)
        decay_steps = maxiter * n_batches - warmup_steps
        cold_lr = fit_config[cls.LR_SCHEDULER_PARAMS_FIELD].get(
            "cold_learning_rate", cls.DEFAULT_VALUES["cold_learning_rate"]
        )

        kwargs["initial_lr"] = cold_lr
        kwargs["decay_steps"] = decay_steps
        kwargs["min_lr"] = min_lr
        kwargs["warmup_target"] = lr
        kwargs["warmup_steps"] = warmup_steps

        return kwargs
