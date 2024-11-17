from __future__ import annotations

import tensorflow as tf

from typing import Callable, Optional


ACTIVATION_DICT = {"tanh": tf.nn.tanh}


class DenseLayer(tf.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        activation: Optional[Callable] = None,
        name: str = "DenseLayer",
        no_weight_decay: bool = False,
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

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        var_name = f"DenseLayer_{self.name}_{self.weight_decay}"
        self.w = tf.Variable(
            tf.random.normal(shape=[self.n_in, self.n_out], dtype=float_dtype),
            name=var_name,
        )
        self.norm = tf.convert_to_tensor(1 / self.n_in**0.5, dtype=float_dtype)
        self.is_built = True

    @tf.Module.with_name_scope
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        if self.activation is not None:
            w = self.w * self.norm
            x = tf.einsum("...k,...kn->...n", x, w)
            x = self.activation(
                x
            )  # TODO: apply only if self.activation. Code above is always executed
        else:
            w = self.w * self.norm
            x = tf.einsum("...k,...kn->...n", x, w)
        return x


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
    ):
        super().__init__(name=name)
        self.layers_config = [input_size] + hidden_layers + [output_size]
        self.nlayers = 0
        self.is_built = False

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

            layer = DenseLayer(n_in, n_out, a, no_weight_decay=no_weight_decay)
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
