import json
import os
import shutil
import sys

import pytest
import subprocess
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


prefix = Path(__file__).parent.resolve()


TF_DATASET = "tf_dataset"
DATA_DISTRIB = "data_distrib"


def test_compute_distributed_data_and_distrib_fit():
    tf_dataset_path = prefix / DATA_DISTRIB / TF_DATASET
    if os.path.isdir(tf_dataset_path):
        shutil.rmtree(tf_dataset_path)

    tf_dataset_stats_json_path = (
        prefix / DATA_DISTRIB / TF_DATASET / "stage3" / "stats.json"
    )

    assert not os.path.isfile(tf_dataset_stats_json_path)
    script_name = "compute_distributed_data.sh"
    subprocess.run(["bash", script_name], cwd=str(prefix / DATA_DISTRIB), check=True)
    assert os.path.isfile(tf_dataset_stats_json_path)

    with open(tf_dataset_stats_json_path, "r") as f:
        stats = json.load(f)

    assert "scale" in stats
    assert "avg_n_neigh" in stats
    assert "total_num_of_neighs" in stats
    assert "total_num_of_atoms" in stats
    assert "total_num_structures" in stats
    assert "total_num_of_batches" in stats
    assert "cutoff" in stats
    assert "cutoff_dict" in stats
    assert "batch_size" in stats
    assert stats["total_num_of_atoms"] == 844
    assert stats["total_num_of_neighs"] == 48904
    assert stats["total_num_structures"] == 50
    assert stats["total_num_of_batches"] == 14

    seed_path = prefix / DATA_DISTRIB / "seed"
    if os.path.isdir(seed_path):
        shutil.rmtree(seed_path)
    assert not os.path.isdir(seed_path)

    current_env = os.environ.copy()
    current_env["NUM_VIRTUAL_DEVICES"] = "2"
    current_env["CUDA_VISIBLE_DEVICES"] = "-1"
    current_env["TF_USE_LEGACY_KERAS"] = "1"

    subprocess.run(
        "gracemaker -m",
        cwd=str(prefix / DATA_DISTRIB),
        check=True,
        shell=True,
        env=current_env,
        # stderr=sys.stderr,
        # stdout=sys.stdout,
    )

    test_metrics_path = prefix / DATA_DISTRIB / "seed" / "1" / "test_metrics.yaml"
    assert os.path.isfile(test_metrics_path)
    # shutil.rmtree(seed_path)
