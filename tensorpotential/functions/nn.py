from __future__ import annotations

from typing import Callable, Optional, Any

import tensorflow as tf

from tensorpotential.functions.lora import (
    initialize_lora_tensors,
    lora_reconstruction,
    apply_lora_update,
)

ACTIVATION_DICT = {"tanh": tf.nn.tanh, "silu": tf.nn.silu, "sigmoid": tf.nn.sigmoid}


class Linear(tf.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        name: str = "Linear",
        no_weight_decay: bool = False,
        use_bias: bool = False,
        init_type: str = "normal",
        normalize: bool = True,
        lora_config=None,
    ):
        super().__init__(name=name)
        self.n_in = n_in
        self.n_out = n_out

        self.init_type = init_type
        assert init_type in ["normal", "uniform", "zeros"]

        self.normalize = normalize
        self.norm = 1.0

        if no_weight_decay:
            self.weight_decay = "no_decay"
        else:
            self.weight_decay = "_"
        self.is_built = False
        self.use_bias = use_bias

        self.lora_config = lora_config
        self.lora = lora_config is not None

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            var_name = f"Linear_{self.name}_{self.weight_decay}"
            shape = [self.n_in, self.n_out]
            b_shape = [1, self.n_out]
            if self.normalize:
                init_value = 1.0
                self.norm = 1.0 / self.n_in**0.5
            else:
                init_value = 1.0 / self.n_in**0.5
            if self.init_type == "normal":
                self.w = tf.Variable(
                    tf.random.normal(shape=shape, stddev=init_value, dtype=float_dtype),
                    name=var_name,
                )
                if self.use_bias:
                    self.b = tf.Variable(
                        tf.random.normal(
                            shape=b_shape, stddev=init_value, dtype=float_dtype
                        ),
                        name=var_name + "_bias",
                    )
            if self.init_type == "uniform":
                self.w = tf.Variable(
                    tf.random.uniform(
                        minval=-init_value,
                        maxval=init_value,
                        shape=shape,
                        dtype=float_dtype,
                    ),
                    name=var_name,
                )
                if self.use_bias:
                    self.b = tf.Variable(
                        tf.random.uniform(
                            minval=-init_value,
                            maxval=init_value,
                            shape=b_shape,
                            dtype=float_dtype,
                        ),
                        name=var_name + "_bias",
                    )
            elif self.init_type == "zeros":
                self.w = tf.Variable(
                    tf.zeros(shape=shape, dtype=float_dtype),
                    name=var_name,
                )
                if self.use_bias:
                    self.b = tf.Variable(
                        tf.zeros(shape=b_shape, dtype=float_dtype),
                    )
            self.norm = tf.convert_to_tensor(self.norm, dtype=float_dtype)

            if self.lora:
                self.enable_lora_adaptation(self.lora_config)

            self.is_built = True

    @tf.Module.with_name_scope
    def enable_lora_adaptation(self, lora_config):
        # x. upd _init_args,
        # 2. add new trainable LORA weights,
        # 3. set main weights as non-trainable
        self.lora = True
        self.lora_config = lora_config
        self.lora_tensors = initialize_lora_tensors(self.w, lora_config)

    def finalize_lora_update(self):
        apply_lora_update(self.w, *self.lora_tensors, lora_config=self.lora_config)
        del self.lora_tensors
        self.lora = False
        self.lora_config = None

    @tf.Module.with_name_scope
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        w = self.w
        if self.lora:
            w = w + lora_reconstruction(
                *self.lora_tensors, lora_config=self.lora_config
            )
        w = w * self.norm
        x = tf.einsum("...k,...kn->...n", x, w)
        if self.use_bias:
            x += self.b * self.norm

        return x


class DenseLayer(tf.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        activation: Optional[Callable] = None,
        name: str = "DenseLayer",
        no_weight_decay: bool = False,
        use_bias: bool = False,
        lora_config=None,
    ):
        super().__init__(name=name)
        self.activation = activation
        self.n_in = n_in
        self.n_out = n_out
        self.norm = 1.0
        if no_weight_decay:
            self.weight_decay = "no_decay"
        else:
            self.weight_decay = "_"
        self.is_built = False
        self.use_bias = use_bias

        self.lora_config = lora_config
        self.lora = lora_config is not None

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            var_name = f"DenseLayer_{self.name}_{self.weight_decay}"
            self.w = tf.Variable(
                tf.random.normal(shape=[self.n_in, self.n_out], dtype=float_dtype),
                name=var_name,
            )
            if self.use_bias and self.activation is not None:
                self.b = tf.Variable(
                    tf.zeros(shape=[self.n_out], dtype=float_dtype),
                    name=var_name + "_bias",
                )
            self.norm = tf.convert_to_tensor(1 / self.n_in**0.5, dtype=float_dtype)

            if self.lora:
                self.enable_lora_adaptation(self.lora_config)

            self.is_built = True

    @tf.Module.with_name_scope
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        w = self.w
        if self.lora:
            w = w + lora_reconstruction(
                *self.lora_tensors, lora_config=self.lora_config
            )
        w = w * self.norm
        x = tf.einsum("...k,...kn->...n", x, w)
        if self.use_bias:
            x += self.b
        if self.activation is not None:
            # TODO: apply only if self.activation. Code above is always executed
            x = self.activation(x)
        return x

    @tf.Module.with_name_scope
    def enable_lora_adaptation(self, lora_config):
        # x. upd _init_args,
        # 2. add new trainable LORA weights,
        # 3. set main weights as non-trainable
        self.lora = True
        self.lora_config = lora_config
        self.lora_tensors = initialize_lora_tensors(self.w, lora_config)

    def finalize_lora_update(self):
        apply_lora_update(self.w, *self.lora_tensors, lora_config=self.lora_config)
        del self.lora_tensors
        self.lora = False
        self.lora_config = None


def silu_n2norm(x):
    return tf.nn.silu(x) * 1.6759


class FullyConnectedMLP(tf.Module):
    """Fully-connected MLP

    Parameters
    ----------
    input_size: int
        Value specifying shape[-1] of the input

    hidden_layers : list[int]
        Sizes of the hidden layers

    output_size: int
        Value specifying shape[-1] of the output

    activation : callable
        Activation function. Default silu_n2norm

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: list,
        activation: callable = None,
        out_act: bool = False,
        name: str = "FullyConnectedMLP",
        no_weight_decay: bool = False,
        use_bias: bool = False,
        lora_config=None,
    ):
        super().__init__(name=name)
        self.layers_config = [input_size] + hidden_layers + [output_size]
        self.nlayers = 0
        self.is_built = False
        self.use_bias = use_bias
        self.lora_config = lora_config
        if activation is None:
            activation = silu_n2norm
        elif activation in ACTIVATION_DICT:
            activation = ACTIVATION_DICT[activation]
        else:
            raise ValueError(
                f"activation must be a string ({list(ACTIVATION_DICT.keys())})"
            )

        for i, (n_in, n_out) in enumerate(
            zip(self.layers_config, self.layers_config[1:])
        ):
            if i == len(self.layers_config) - 2:
                a = activation if out_act else None
            else:
                a = activation

            layer = DenseLayer(
                n_in,
                n_out,
                activation=a,
                no_weight_decay=no_weight_decay,
                use_bias=use_bias,
                lora_config=self.lora_config,
                name=f"{self.name}_layer{i}",
            )
            # layer.build()
            setattr(self, f"layer{i}", layer)
            self.nlayers += 1

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        for sm in self.submodules:
            if hasattr(sm, "build"):
                if not hasattr(sm, "is_built") or not sm.is_built:
                    sm.build(float_dtype)
        self.is_built = True

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        for i in range(self.nlayers):
            layer = getattr(self, f"layer{i}")
            x = layer(x)

        return x

    def __repr__(self):
        return f"{self.name}{self.layers_config}"

    @tf.Module.with_name_scope
    def enable_lora_adaptation(self, lora_config: dict[str, Any]):
        # 1. upd _init_args,
        # 2. add new trainable LORA weights,
        # 3. set main weights as non-trainable
        for i in range(self.nlayers):
            layer = getattr(self, f"layer{i}")
            layer.enable_lora_adaptation(lora_config)

    def finalize_lora_update(self):
        for i in range(self.nlayers):
            layer = getattr(self, f"layer{i}")
            layer.finalize_lora_update()


def scalar_LN(x, scale: tf.Variable, shift: tf.Variable = None, r_map=None):
    var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
    m = tf.reduce_mean(x, axis=-1, keepdims=True)
    x = scale * (x - m) * tf.math.rsqrt(var + 1e-16)
    if shift is not None:
        x += shift
    if r_map is not None:
        x = tf.where(r_map, x, tf.zeros_like(x))
    return x


def scalar_rms_ln(x, scale: tf.Variable, r_map=None):
    xx = x**2
    rms = tf.math.rsqrt(tf.reduce_mean(xx, axis=-1, keepdims=True) + 1e-12)
    if r_map is not None:
        rms = tf.where(r_map, rms, tf.zeros_like(rms))

    return x * scale * rms
