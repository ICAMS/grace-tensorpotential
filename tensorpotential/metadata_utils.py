"""Utilities for model metadata and parameter dtype resolution."""

from __future__ import annotations
import os
import yaml
import logging
from tensorflow.dtypes import float32, float64, DType


def read_model_metadata(filename: str) -> dict:
    """Read only the metadata block from a model.yaml without deserializing instructions.
    Supports both new wrapped format and old flat-dict format (returns empty dict for old models).
    """
    if not os.path.exists(filename):
        return {}
    with open(filename, "rt") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        return data.get("metadata", {})
    return {}


def get_dtype_by_name(dtype_name: str) -> DType:
    """Resolve string name to TensorFlow DType."""
    param_dtype = {
        "float64": float64,
        "float32": float32,
    }.get(dtype_name, None)
    if param_dtype:
        return param_dtype
    else:
        raise KeyError(f"Unknown dtype name {dtype_name}")


def resolve_param_dtype(model_yaml: str, param_dtype: DType | None = None) -> DType:
    """Resolution logic for param_dtype, prioritizing manual override then YAML metadata."""
    if param_dtype is None:
        if os.path.exists(model_yaml):
            meta = read_model_metadata(model_yaml)
            dtype_name = meta.get("param_dtype")
            if dtype_name:
                dtype = get_dtype_by_name(dtype_name)
                logging.info(
                    f"Inferred param_dtype: {dtype_name} ({dtype}) from {model_yaml}"
                )
                return dtype
            else:
                logging.info(
                    "No param_dtype in model.yaml (old model) — using float64 for backward compatibility"
                )
                return float64
        else:
            logging.info(f"model.yaml {model_yaml} not found — using default float64")
            return float64
    return param_dtype
