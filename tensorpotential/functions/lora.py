from __future__ import annotations

import tensorflow as tf

MODE_LORA = "lora"  # default
MODE_FULL_ADDITIVE = "full_additive"
MODE_ADDITIVE = "additive"


def initialize_lora_tensors(w, lora_config, name=None):
    """
    Generate tensors/Variables for fine-tuning.
    If mode is "lora", generates low-rank matrices.
    If mode is "additive", generates a full matrix of the same shape as w.

    w: tf.Tensor (original weight matrix)
    lora_config: dict containing configuration.
                 Requires "mode": "lora" or "additive".
                 For "lora": "rank", "alpha", optional "keep_dims".
                 For "additive": no other specific keys needed from this function.
    name: Optional name prefix.
    """

    # https://github.com/hkproj/pytorch-lora/blob/main/lora.ipynb
    if lora_config is None:
        raise ValueError("lora_config must be provided for lora_reconstruction")

    name = name or ""
    if name:
        name = name + "/"

    mode = lora_config.get(
        "mode", MODE_LORA
    )  # Default to lora if mode is not specified
    if mode in [MODE_FULL_ADDITIVE, MODE_ADDITIVE]:
        # For additive mode, we create one trainable tensor of the same shape as w, initialized to zeros.
        additive_tensor = tf.Variable(
            tf.zeros(shape=w.shape, dtype=w.dtype),
            trainable=True,
            name=name + "ADDITIVE/delta_W",
        )
        # Set original Variable/tensor to non-trainable
        w._trainable = False
        return [additive_tensor]
    elif mode == MODE_LORA:
        symbols = "ABCDEFG"

        rank = lora_config["rank"]
        alpha = lora_config["alpha"]
        keep_dims = lora_config.get("keep_dims", 0)
        dims = w.shape.as_list()
        assert len(dims) - keep_dims >= 2, f"Tensor {w.name} has too small rank"
        assert len(dims) - keep_dims >= 2, f"Tensor {w.name} has too small rank"

        scale = (alpha / rank) ** (1 / (len(dims) - 1 - keep_dims))

        lora_tensors = []
        dims_to_keep = dims[:keep_dims]
        for i in range(keep_dims, len(dims) - 1):  # all except last:
            lora_tensors.append(
                tf.Variable(
                    tf.random.normal(
                        shape=dims_to_keep + [dims[i], rank],
                        dtype=w.dtype,
                        mean=0.0,
                        stddev=scale,
                    ),
                    trainable=True,
                    name=name + f"LORA/{symbols[i-keep_dims]}",
                )
            )

        lora_tensors.append(
            tf.Variable(
                tf.zeros(shape=dims_to_keep + [dims[-1], rank], dtype=w.dtype),
                trainable=True,
                name=name + f"LORA/{symbols[len(dims)-1-keep_dims]}",
            )
        )

        # set orignal Variable/tensor to non-trainable
        w._trainable = False

        return lora_tensors
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Supported modes are 'lora' and 'additive'."
        )


def lora_reconstruction(*inputs, lora_config=None):
    """
    Compute delta_W, given the fine-tuning tensors `inputs`.

    *inputs: list of LORA or Additive tensors.
    lora_config: dict containing configuration, including "mode".
    return: delta_W
    """
    if lora_config is None:
        raise ValueError("lora_config must be provided for lora_reconstruction")

    mode = lora_config.get("mode", MODE_LORA)

    if mode in [MODE_FULL_ADDITIVE, MODE_ADDITIVE]:
        # For additive mode, inputs is a list containing the single delta_W tensor.
        if not inputs or len(inputs) != 1:
            raise ValueError(
                "For additive mode, 'inputs' should be a list containing one tensor."
            )
        return inputs[0]
    elif mode == MODE_LORA:
        keep_dims = lora_config.get("keep_dims", 0) if lora_config else 0
        symbols = "abcdefgh"  # max
        keep_dims_symbols = "zyxw"  # max
        assert len(inputs) <= len(symbols)
        # A_ar B_br C_cr ... -> W_abcd , if keep_dims == 0
        # A_zar B_zbr C_zcr ... -> W_zabc, if keep_dims > 0
        keep_dims_prefix = "".join(keep_dims_symbols[:keep_dims])
        eq_in = ",".join(
            [f"{keep_dims_prefix}{axis}r" for axis in symbols[: len(inputs)]]
        )
        eq_out = keep_dims_symbols[:keep_dims] + symbols[: len(inputs)]
        return tf.linalg.einsum(f"{eq_in}->{eq_out}", *inputs)


def apply_lora_update(w, *inputs, lora_config=None):
    """
    Merges the learned delta_W back into the original weight tensor w.
    This is typically done after training is complete.

    w: tf.Variable (original weight matrix to be updated)
    *inputs: list of LORA or Additive tensors.
    lora_config: dict containing configuration.
    """
    w.assign(w + lora_reconstruction(*inputs, lora_config=lora_config))
    w._trainable = True
