"""fit.scheduler_params: the keys gracemaker -t writes are the keys the schedulers read.

gracemaker -t wrote ``minimal_learning_rate`` (ignored: the cosine decay ended at the 1e-4
default) and put ``stop_at_min`` into ``scheduler_params``, where reduce_on_plateau did not
look for it.
"""

import logging

import yaml

from tensorpotential.cli.train_callbacks import LRSchedulerFactory
from tensorpotential.cli.wizard import _apply_state
from tests.test_wizard import _MINIMAL_TEMPLATE, _make_state


def _fit_section(**state):
    return yaml.safe_load(_apply_state(_make_state(**state), _MINIMAL_TEMPLATE))["fit"]


def test_wizard_cosine_decay_writes_a_known_min_lr_key(caplog):
    fit = _fit_section(optimizer="Adam", learning_rate=0.006, use_switch=False)
    assert fit["scheduler"] == "cosine_decay"
    assert fit["scheduler_params"]["minimum_learning_rate"] == 0.006 / 12
    with caplog.at_level(logging.WARNING):
        LRSchedulerFactory.warn_unknown_scheduler_params(fit)
    assert "unknown key" not in caplog.text


def test_wizard_switch_stops_at_min_lr():
    fit = _fit_section(optimizer="Adam", learning_rate=0.006, use_switch=True,
                       switch_after=0.75, lr_reduction_factor=0.1,
                       new_energy_w=128, new_force_w=32)
    assert fit["scheduler"] == "reduce_on_plateau"
    kwargs = LRSchedulerFactory.read_reduce_on_plateau_kwargs(fit)
    assert kwargs["stop_on_min_lr"] is True
    assert kwargs["min_lr"] == 0.006 / 12


def test_stop_at_min_on_the_fit_level_still_works():
    fit = {"scheduler": "reduce_on_plateau", "scheduler_params": {}, "stop_at_min": True}
    assert LRSchedulerFactory.read_reduce_on_plateau_kwargs(fit)["stop_on_min_lr"] is True
    fit = {"scheduler": "reduce_on_plateau", "scheduler_params": {}}
    assert LRSchedulerFactory.read_reduce_on_plateau_kwargs(fit)["stop_on_min_lr"] is False


def test_scheduler_params_win_over_the_fit_level():
    fit = {"scheduler_params": {"stop_at_min": False}, "stop_at_min": True}
    assert LRSchedulerFactory.scheduler_or_fit_param(fit, "stop_at_min") is False


def test_unknown_scheduler_param_is_reported_with_a_hint(caplog):
    fit = {"scheduler": "cosine_decay", "scheduler_params": {"minimal_learning_rate": 1e-3}}
    with caplog.at_level(logging.WARNING):
        LRSchedulerFactory.warn_unknown_scheduler_params(fit)
    assert "'minimal_learning_rate' is ignored" in caplog.text
    assert "Did you mean 'minimum_learning_rate'?" in caplog.text


def test_gen_tensor_wizard_template_uses_the_known_key():
    import inspect

    from tensorpotential.extra.gen_tensor import wizard as gt_wizard

    src = inspect.getsource(gt_wizard._apply_state_tensor)
    assert "minimum_learning_rate" in src and "minimal_learning_rate" not in src
