"""Resolve model.yaml, checkpoint and CALM artifact from one model directory."""

from __future__ import annotations

import os

# Searched in this order, relative to the model directory. Within one layout the first
# hit wins (best_test_loss before the last checkpoint); hits in two layouts are ambiguous.
MODEL_YAML_CANDIDATES = (("model.yaml",),)
CHECKPOINT_CANDIDATES = (
    ("checkpoint.index",),                                    # examples/data/<model>/
    ("checkpoints/checkpoint.best_test_loss.index",
     "checkpoints/checkpoint.index"),                         # gracemaker fit directory
)
ARTIFACT_CANDIDATES = (("gmm_artifacts.npz",), ("UQ/gmm_artifacts.npz",))

SEARCH_ORDER_HELP = (
    "model.yaml; checkpoint: checkpoint.index, then checkpoints/checkpoint.best_test_loss.index, "
    "checkpoints/checkpoint.index; CALM artifact: gmm_artifacts.npz, UQ/gmm_artifacts.npz"
)


def _find(model_dir, layouts, what, flag):
    tried, hits = [], []
    for layout in layouts:
        for rel in layout:
            p = os.path.join(model_dir, rel)
            tried.append(p)
            if os.path.exists(p):
                hits.append(p)
                break  # first hit within a layout wins
    if not hits:
        raise FileNotFoundError(
            f"--model-dir {model_dir}: no {what} found; tried " + ", ".join(tried)
            + f". Pass {flag} explicitly."
        )
    if len(hits) > 1:
        raise ValueError(
            f"--model-dir {model_dir}: ambiguous {what}: " + " and ".join(hits)
            + f". Pass {flag} explicitly."
        )
    return hits[0]


def resolve_model_dir(model_dir, model_yaml=None, checkpoint=None, artifact_path=None, *,
                      require=("model_yaml", "checkpoint", "artifact"),
                      flags=("--model-yaml", "--checkpoint", "--artifact-path")):
    """Return ``(model_yaml, checkpoint, artifact_path)`` for a model directory.

    Explicit arguments win over the search. The checkpoint is returned without the
    ``.index`` suffix (the TensorFlow checkpoint prefix). Layouts: ``examples/data/<model>``
    (model.yaml, checkpoint.*, gmm_artifacts.npz) and a gracemaker fit directory
    (model.yaml, checkpoints/checkpoint.best_test_loss.*, UQ/gmm_artifacts.npz or
    gmm_artifacts.npz). Raises FileNotFoundError (with every path tried) or ValueError
    (ambiguous) otherwise. Items not in ``require`` are None when not found or
    ambiguous; ``flags`` are the option names the messages point to
    (``grace_uq select`` uses its own).
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"--model-dir {model_dir} is not a directory")
    errors = []
    out = {}
    for key, given, layouts, what, flag in (
        ("model_yaml", model_yaml, MODEL_YAML_CANDIDATES, "model.yaml", flags[0]),
        ("checkpoint", checkpoint, CHECKPOINT_CANDIDATES, "checkpoint", flags[1]),
        ("artifact", artifact_path, ARTIFACT_CANDIDATES, "CALM artifact", flags[2]),
    ):
        if given is not None:
            out[key] = given
            continue
        try:
            out[key] = _find(model_dir, layouts, what, flag)
        except FileNotFoundError as exc:
            out[key] = None
            if key in require:
                errors.append(str(exc))
        except ValueError:
            if key in require:
                raise
            out[key] = None
    if errors:
        raise FileNotFoundError("\n".join(errors))
    ckpt = out["checkpoint"]
    if checkpoint is None and ckpt is not None and ckpt.endswith(".index"):
        ckpt = ckpt[: -len(".index")]
    return out["model_yaml"], ckpt, out["artifact"]
