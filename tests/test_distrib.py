import json
import os
import shutil

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

    # Regression (scripts/gracemaker.py): an externally-supplied TF_CONFIG -- as
    # set by a SLURM launcher or left over in the shell -- must NOT route the
    # NUM_VIRTUAL_DEVICES debug path through MultiWorkerMirroredStrategy. MWMS
    # initialises the TF context, after which the virtual-device setup crashes
    # with "Virtual devices cannot be modified after being initialized". With
    # NUM_VIRTUAL_DEVICES set, gracemaker must fall back to MirroredStrategy and
    # ignore TF_CONFIG. Re-run the fit with TF_CONFIG present and confirm it
    # still completes (reusing the tf_dataset already built above).
    shutil.rmtree(seed_path)
    assert not os.path.isdir(seed_path)

    tf_config_env = current_env.copy()
    tf_config_env["TF_CONFIG"] = json.dumps(
        {
            "cluster": {"worker": ["localhost:12345"]},
            "task": {"type": "worker", "index": 0},
        }
    )
    subprocess.run(
        "gracemaker -m",
        cwd=str(prefix / DATA_DISTRIB),
        check=True,
        shell=True,
        env=tf_config_env,
    )
    assert os.path.isfile(test_metrics_path)
    # shutil.rmtree(seed_path)
