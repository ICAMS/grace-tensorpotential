import os
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from tensorpotential.utils import is_chief


def test_is_chief_none_strategy():
    assert is_chief(None) is True


def test_is_chief_default_strategy():
    assert is_chief(tf.distribute.get_strategy()) is True


def test_is_chief_mirrored_strategy():
    assert is_chief(tf.distribute.MirroredStrategy()) is True


def test_is_chief_mwms_worker_zero():
    fake = SimpleNamespace(
        cluster_resolver=SimpleNamespace(task_type="worker", task_id=0)
    )
    assert is_chief(fake) is True


def test_is_chief_mwms_worker_nonzero():
    fake = SimpleNamespace(
        cluster_resolver=SimpleNamespace(task_type="worker", task_id=1)
    )
    assert is_chief(fake) is False

    fake3 = SimpleNamespace(
        cluster_resolver=SimpleNamespace(task_type="worker", task_id=3)
    )
    assert is_chief(fake3) is False


def test_is_chief_mwms_chief_type():
    fake = SimpleNamespace(
        cluster_resolver=SimpleNamespace(task_type="chief", task_id=0)
    )
    assert is_chief(fake) is True
