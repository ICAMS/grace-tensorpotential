from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from tensorpotential import constants
from tensorpotential.functions.nn import (
    ElementDependentLinear,
    FullyConnectedMLP,
    Linear,
    scalar_rms_ln,
    ACTIVATION_DICT,
)
from tensorpotential.instructions.base import (
    capture_init_args,
    TPInstruction,
    LORAInstructionMixin,
    ElementsReduceInstructionMixin,
)
from tensorpotential.instructions.compute import (
    FunctionReduce,
    FunctionReduceParticular,
    ScalarChemicalEmbedding,
)


@capture_init_args
class CreateOutputTarget(TPInstruction):
    def __init__(self, name: str, initial_value: float = 0.0, l: int = 0):  # noqa: E741
        super().__init__(name=name)
        self.l = l
        if initial_value is None:
            self.value = 0.0
        else:
            self.value = initial_value

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            self.value = tf.reshape(tf.constant(self.value, dtype=float_dtype), [])
            self.is_built = True

    def frwrd(self, input_data, training=False, local=False):
        return self.value


@capture_init_args
class TPOutputInstruction(TPInstruction):
    _INSTRUCTION_TYPE = "TPOutputInstruction"

    def __init__(self, name: str, target: TPInstruction, l: int = 0):  # noqa: E741
        super(TPOutputInstruction, self).__init__(name=name)
        self.target = target
        # TODO: Improve this
        self.l = l

    def assert_l_compatibility(self, target: CreateOutputTarget):
        assert (
            self.l == target.l
        ), f"Target l={target.l} does not match origin l={self.l}"

    def build(self, float_dtype):
        pass

    def __call__(
        self, input_data: dict, training: bool = False, local: bool = False
    ) -> dict:
        output = self.frwrd(input_data, training=training, local=local)

        input_data[self.target.name] = output

        return input_data


@capture_init_args
class LinearOut2Target(TPOutputInstruction):
    """
    Adds origin to target without transformation
    """

    def __init__(
        self,
        origin: list[FunctionReduce],
        target: CreateOutputTarget,
        name="LinearOut2ScalarTarget",
        trainable: bool = False,
        init_type: str = "zeros",
        n_out: int = 1,
        l: int = 0,  # noqa: E741
    ):
        super(LinearOut2Target, self).__init__(name=name, target=target, l=l)
        self.origin = origin
        self.trainable = trainable
        self.n_out = n_out
        try:
            lmax = np.max([ins.lmax for ins in self.origin])
        except AttributeError:
            lmax = 0
        assert self.l == lmax, f"lmax > target l={target.l} for some instructions"
        self.assert_l_compatibility(target)

        out_shapes = [ins.n_out for ins in self.origin]
        assert len(set(out_shapes)) == 1, "Not all shapes are the same"
        if self.trainable:
            self.lins = [
                Linear(n_in=ins.n_out, n_out=self.n_out, init_type=init_type)
                for ins in self.origin
            ]

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            if self.trainable:
                for lin in self.lins:
                    lin.build(float_dtype)
            self.is_built = True

    def frwrd(self, input_data, training=False, local=False):
        target = input_data[f"{self.target.name}"]

        origin = 0.0
        for i, ins in enumerate(self.origin):
            if self.trainable:
                oi = input_data[f"{ins.name}"][:, :, 0]
                origin += self.lins[i](oi)
            else:
                origin += input_data[f"{ins.name}"][:, :, 0]

        return target + origin


@capture_init_args
class FSOut2ScalarTarget(TPOutputInstruction):
    """
    Adds origin to target without transformation
    """

    def __init__(
        self,
        origin: list[FunctionReduce],
        target: TPInstruction,
        fs_parameters: list[tuple[float, float]] = ((1.0, 1.0), (1.0, 0.5)),
        name="FSOut2ScalarTarget",
        l: int = 0,  # noqa: E741,
    ):
        super(FSOut2ScalarTarget, self).__init__(name=name, target=target, l=l)
        self.origin = origin
        self.target = target
        try:
            lmax = np.max([ins.lmax for ins in self.origin])
        except AttributeError:
            lmax = 0
        assert lmax == 0, "Trying to output non-scalar origin to a scalar target"

        out_shapes = [ins.n_out for ins in self.origin]
        assert len(set(out_shapes)) == 1, "Not all shapes are the same"
        n_out = out_shapes[0]
        assert all(
            len(s) == 2 for s in fs_parameters
        ), "`fs_parameters` should be [[c1, mexp1], [c2, mexp2],...]"
        assert n_out == len(
            fs_parameters
        ), "`n_out` if FSOut2ScalarTarget.origin should be equal to len(fs_parameters)"
        self.fs_parameters = fs_parameters

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            self.fs_parameters = [
                (
                    tf.constant(coeff, dtype=float_dtype),
                    tf.constant(mexp, dtype=float_dtype),
                )
                for coeff, mexp in self.fs_parameters
            ]
            self.is_built = True

    @staticmethod
    def f_exp_shsc(rho, mexp):
        eps = tf.constant(1e-10, dtype=rho.dtype)
        cond = tf.abs(
            tf.ones_like(rho, dtype=rho.dtype) * mexp
            - tf.constant(1.0, dtype=rho.dtype)
        )
        mask = tf.where(
            tf.less(cond, eps),
            tf.ones_like(rho, dtype=tf.bool),
            tf.zeros_like(rho, dtype=tf.bool),
        )

        arho = tf.abs(rho)
        # func = tf.where(mask, rho, tf.sign(rho) * (tf.sqrt(tf.abs(arho + 0.25 * tf.exp(-arho))) - 0.5 * tf.exp(-arho)))
        exprho = tf.exp(-arho)
        nx = 1.0 / mexp
        xoff = tf.pow(nx, (nx / (1.0 - nx))) * exprho
        yoff = tf.pow(nx, (1 / (1.0 - nx))) * exprho
        func = tf.where(mask, rho, tf.sign(rho) * (tf.pow(xoff + arho, mexp) - yoff))

        return func

    def frwrd(self, input_data, training=False, local=False):
        target = input_data[f"{self.target.name}"]

        origin = 0.0
        for ins in self.origin:
            origin += input_data[f"{ins.name}"][:, :, 0]

        contrib = 0
        for dens, (coeff, mexp) in enumerate(self.fs_parameters):
            contrib += coeff * self.f_exp_shsc(origin[:, dens], mexp)
        contrib = tf.reshape(contrib, (-1, 1))
        return target + contrib


@capture_init_args
class MLPOut2ScalarTarget(TPOutputInstruction):
    """
    Adds origin to target via MLP transformation
    """

    input_tensor_spec = {
        constants.N_ATOMS_BATCH_REAL: {"shape": [], "dtype": "int"},
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
    }

    def __init__(
        self,
        origin: list[FunctionReduce],
        target: CreateOutputTarget,
        hidden_layers: list[int] = None,
        n_out: int = 1,
        name="MLPOut2ScalarTarget",
        normalize: str = None,
        activation: str = None,
        l: int = 0,  # noqa: E741
        **kwargs,
    ):
        super(MLPOut2ScalarTarget, self).__init__(name=name, target=target, l=l)
        self.origin = origin
        self.n_out = n_out
        self.normalize = normalize
        if self.normalize is not None:
            assert self.normalize in ["layer"]

        if hidden_layers is None:
            self.hidden_layers = [32]
        else:
            self.hidden_layers = hidden_layers

        try:
            lmax = np.max([ins.lmax for ins in self.origin])
        except AttributeError:
            lmax = 0
        assert lmax == 0, "Trying to output non-scalar origin to a scalar target"
        self.assert_l_compatibility(target)

        out_shapes = [ins.n_out for ins in self.origin]
        assert len(set(out_shapes)) == 1, "Not all shapes are the same"

        self.mlp = FullyConnectedMLP(
            input_size=out_shapes[0],
            hidden_layers=self.hidden_layers,
            output_size=self.n_out,
            activation=activation,
            name=self.name + "_MLP",
        )

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.mlp.is_built:
            self.mlp.build(float_dtype)
        if self.normalize is not None:
            shape = [1, self.origin[0].n_out]
            # self.scale = tf.Variable(tf.zeros(shape, dtype=float_dtype))
            self.scale = tf.Variable(
                tf.random.normal(shape, stddev=1e-16, dtype=float_dtype), name="scale"
            )
        self.is_built = True

    def rmsln(self, x, input_data):
        n_at_b_real = input_data[constants.N_ATOMS_BATCH_REAL]
        n_at_b_total = input_data[constants.N_ATOMS_BATCH_TOTAL]
        xx = x**2
        rms = tf.math.rsqrt(tf.reduce_mean(xx, axis=-1, keepdims=True) + self.epsilon)
        r_map = tf.reshape(
            tf.range(n_at_b_total, delta=1, dtype=tf.int32, name="r_map"), [-1, 1]
        )
        rms = tf.where(r_map < n_at_b_real, rms, tf.zeros_like(rms))

        return x * self.scale * rms

    def frwrd(self, input_data, training=False, local=False):
        target = input_data[f"{self.target.name}"]

        origin = 0.0
        for ins in self.origin:
            origin += input_data[f"{ins.name}"][:, :, 0]
        if self.normalize == "layer":
            # origin = self.rmsln(origin, input_data)
            n_at_b_real = input_data[constants.N_ATOMS_BATCH_REAL]
            n_at_b_total = input_data[constants.N_ATOMS_BATCH_TOTAL]
            r_map = tf.reshape(
                tf.range(n_at_b_total, delta=1, dtype=tf.int32, name="r_map"), [-1, 1]
            )
            r_map = r_map < n_at_b_real
            origin = scalar_rms_ln(origin, self.scale, r_map=r_map)
        origin = self.mlp(origin)

        return target + origin


@capture_init_args
class LinMLPOut2ScalarTarget(TPOutputInstruction, LORAInstructionMixin):
    """
    Adds origin to target via MLP transformation and a single element without transformation
    """

    input_tensor_spec = {
        constants.N_ATOMS_BATCH_REAL: {"shape": [], "dtype": "int"},
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
    }

    def __init__(
        self,
        origin: list[FunctionReduce],
        target: CreateOutputTarget,
        hidden_layers: list[int] = None,
        name="LinMLPOut2ScalarTarget",
        n_out: int = 1,
        normalize: str = None,
        activation: str = None,
        l: int = 0,  # noqa: E741
        lora_config: dict[str, Any] = None,
        return_hidden_target: str = None,
        **kwargs,
    ):
        super(LinMLPOut2ScalarTarget, self).__init__(name=name, target=target, l=l)
        LORAInstructionMixin.__init__(self, lora_config)  # explicitly call

        self.origin = origin
        self.n_out = n_out
        self.normalize = normalize
        self.return_hidden_target = return_hidden_target
        if self.normalize is not None:
            assert self.normalize in ["layer"]

        if hidden_layers is None:
            self.hidden_layers = [32]
        else:
            self.hidden_layers = hidden_layers
        try:
            lmax = np.max([ins.lmax for ins in self.origin])
        except AttributeError:
            lmax = 0
        assert lmax == 0, "Trying to output non-scalar origin to a scalar target"
        self.assert_l_compatibility(target)
        out_shapes = [ins.n_out for ins in self.origin]
        assert len(set(out_shapes)) == 1, "Not all shapes are the same"

        self.mlp = FullyConnectedMLP(
            input_size=self.origin[0].n_out - 1,
            hidden_layers=self.hidden_layers,
            activation=activation,
            output_size=self.n_out,
            lora_config=lora_config,
            name=self.name + "_MLP",
        )

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.mlp.is_built:
            self.mlp.build(float_dtype)
        if self.normalize is not None:
            shape = [1, self.origin[0].n_out - 1]
            # self.scale = tf.Variable(tf.zeros(shape, dtype=float_dtype))
            self.scale = tf.Variable(
                tf.random.normal(shape, stddev=1e-16, dtype=float_dtype), name="scale"
            )

        if self.lora_config:
            self.enable_lora_adaptation(self.lora_config)

        self.is_built = True

    @tf.Module.with_name_scope
    def enable_lora_adaptation(self, lora_config: dict[str, Any]):
        # 1. upd _init_args,
        # 2. add new trainable LORA weights,
        # 3. set main weights as non-trainable
        super().enable_lora_adaptation(lora_config)
        self.mlp.enable_lora_adaptation(lora_config)
        if self.normalize is not None:
            self.scale._trainable = False

    def finalize_lora_update(self):
        # common part
        super().finalize_lora_update()
        self.mlp.finalize_lora_update()
        if self.normalize is not None:
            self.scale._trainable = True

    def frwrd(self, input_data, training=False, local=False):
        target = input_data[f"{self.target.name}"]

        transformed_origin = 0.0
        lin_origin = 0.0
        for ins in self.origin:
            x = input_data[f"{ins.name}"]
            if getattr(ins, "lm_first", False):
                # [lm, atoms, n_out] -> [atoms, n_out, lm]
                x = tf.transpose(x, [1, 2, 0])
            full_scalar = x[:, :, 0]
            transformed_origin += full_scalar[:, 1:]
            lin_origin += tf.reshape(full_scalar[:, 0], [-1, 1])
        if self.normalize == "layer":
            n_at_b_real = input_data[constants.N_ATOMS_BATCH_REAL]
            n_at_b_total = input_data[constants.N_ATOMS_BATCH_TOTAL]
            r_map = tf.reshape(
                tf.range(n_at_b_total, delta=1, dtype=tf.int32, name="r_map"), [-1, 1]
            )
            r_map = r_map < n_at_b_real
            transformed_origin = scalar_rms_ln(
                transformed_origin, self.scale, r_map=r_map
            )

        if self.return_hidden_target:
            transformed_origin, hidden = self.mlp(
                transformed_origin, return_hidden=True
            )
            hidden = tf.concat([lin_origin, hidden], axis=1)
            # NOTE: weak contract violation — frwrd() is expected to expose outputs
            # via its return value (stored by the manager under self.name), but here
            # we additionally write `hidden` directly into input_data so downstream
            # consumers (UQ feature extraction) can pick it up without restructuring
            # the instruction graph.
            input_data[self.return_hidden_target] = hidden
        else:
            transformed_origin = self.mlp(transformed_origin)
        norm_out = transformed_origin + lin_origin

        return target + norm_out


@capture_init_args
class LinMLPScalarReadOut(TPOutputInstruction):
    """Per-origin lin/non-lin scalar readout.

    Each origin's first scalar feature is treated as a linear passthrough with a
    learnable scalar prefactor (init 1 for the first origin, 0 for the rest).
    The remaining features are individually RMS-normalized with a learnable
    per-origin scale vector (init 0) and fed through an MLP.

    ``mlp_mode``:
      - ``"per_input"`` (default): one MLP per origin, outputs are summed.
      - ``"shared"``: normalized non-linear parts are summed first and fed
        through a single MLP (requires all origins to share ``n_out``).

    ``element_dependent``:
      - ``False`` (default): standard MLP weights shared across atoms.
      - ``True``: per-element MLP weights gathered via ``atomic_mu_i``.

    At init the model collapses to ``target + lin_0`` (pure linear passthrough
    of the first origin's first feature).
    """

    input_tensor_spec = {
        constants.N_ATOMS_BATCH_REAL: {"shape": [], "dtype": "int"},
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
    }

    def __init__(
        self,
        origin: list[FunctionReduce],
        target: CreateOutputTarget,
        hidden_layers: list[int] = None,
        name: str = "LinMLPScalarReadOut",
        n_out: int = 1,
        mlp_mode: str = "per_input",
        element_dependent: bool = False,
        number_of_atom_types: int = None,
        activation: str | list[str] = "silu",
        l: int = 0,  # noqa: E741
        **kwargs,
    ):
        super().__init__(name=name, target=target, l=l)

        assert mlp_mode in (
            "per_input",
            "shared",
        ), f"mlp_mode must be 'per_input' or 'shared', got '{mlp_mode}'"
        if element_dependent:
            assert (
                number_of_atom_types is not None and number_of_atom_types > 0
            ), "element_dependent=True requires number_of_atom_types > 0"

        self.origin = origin
        self.n_out = n_out
        self.mlp_mode = mlp_mode
        self.element_dependent = element_dependent
        self.number_of_atom_types = number_of_atom_types

        if hidden_layers is None:
            hidden_layers = [32]
        self.hidden_layers = hidden_layers
        n_hidden = len(hidden_layers)

        if isinstance(activation, str):
            self.activation = [activation] * n_hidden
        else:
            assert (
                len(activation) == n_hidden
            ), f"activation list length {len(activation)} != n_hidden {n_hidden}"
            self.activation = list(activation)

        try:
            lmax = np.max([ins.lmax for ins in self.origin])
        except AttributeError:
            lmax = 0
        assert lmax == 0, "Trying to output non-scalar origin to a scalar target"
        self.assert_l_compatibility(target)

        out_shapes = [ins.n_out for ins in self.origin]
        nonlin_sizes = [s - 1 for s in out_shapes]
        self.nonlin_sizes = nonlin_sizes

        if mlp_mode == "shared":
            assert len(set(out_shapes)) == 1, (
                "mlp_mode='shared' requires all origins to have identical n_out; "
                f"got {out_shapes}"
            )
            input_sizes = [nonlin_sizes[0]]
            mlp_name_fmt = lambda i: f"{self.name}_MLP_shared"  # noqa: E731
        else:
            input_sizes = nonlin_sizes
            mlp_name_fmt = lambda i: f"{self.name}_MLP_{i}"  # noqa: E731

        self.mlp_layers = []
        for mi, in_size in enumerate(input_sizes):
            layer_sizes = [in_size] + list(hidden_layers) + [n_out]
            layers = []
            for li, (n_in, n_o) in enumerate(zip(layer_sizes, layer_sizes[1:])):
                layer_name = f"{mlp_name_fmt(mi)}_Linear_{li}"
                if element_dependent:
                    layers.append(
                        ElementDependentLinear(
                            n_in=n_in,
                            n_out=n_o,
                            n_types=number_of_atom_types,
                            name=layer_name,
                            use_bias=False,
                            init_type="normal",
                            normalize=True,
                        )
                    )
                else:
                    layers.append(
                        Linear(
                            n_in=n_in,
                            n_out=n_o,
                            name=layer_name,
                            use_bias=False,
                            init_type="normal",
                            normalize=True,
                        )
                    )
            self.mlp_layers.append(layers)

        if element_dependent:
            self.input_tensor_spec = {
                **LinMLPScalarReadOut.input_tensor_spec,
                constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
            }

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if self.is_built:
            return
        self.alpha = [
            tf.Variable(
                tf.ones([], dtype=float_dtype)
                if i == 0
                else tf.zeros([], dtype=float_dtype),
                name=f"alpha_{i}",
            )
            for i in range(len(self.origin))
        ]
        self.scales = [
            tf.Variable(
                tf.zeros([1, n], dtype=float_dtype),
                name=f"scale_{i}",
            )
            for i, n in enumerate(self.nonlin_sizes)
        ]
        for layer_list in self.mlp_layers:
            for layer in layer_list:
                layer.build(float_dtype)
        self.is_built = True

    def _mlp_forward(self, x, layer_idx: int, at_mu_i=None):
        """Activation between hidden layers; no activation on the output layer."""
        layers = self.mlp_layers[layer_idx]
        for act_name, layer in zip(self.activation, layers[:-1]):
            x = layer(x, at_mu_i) if self.element_dependent else layer(x)
            x = ACTIVATION_DICT[act_name](x)
        last = layers[-1]
        x = last(x, at_mu_i) if self.element_dependent else last(x)
        return x

    def frwrd(self, input_data, training=False, local=False):
        target = input_data[self.target.name]

        n_at_b_real = input_data[constants.N_ATOMS_BATCH_REAL]
        n_at_b_total = input_data[constants.N_ATOMS_BATCH_TOTAL]
        r_map = tf.reshape(
            tf.range(n_at_b_total, delta=1, dtype=tf.int32, name="r_map"), [-1, 1]
        )
        r_map = r_map < n_at_b_real

        nonlin_normed = []
        lin_sum = 0.0
        for i, ins in enumerate(self.origin):
            x = input_data[ins.name]
            if getattr(ins, "lm_first", False):
                # [lm, atoms, n_out] -> [atoms, n_out, lm]
                x = tf.transpose(x, [1, 2, 0])
            full_scalar = x[:, :, 0]
            lin_i = tf.reshape(full_scalar[:, 0], [-1, 1])
            nonlin_i = full_scalar[:, 1:]

            rms_i = tf.math.rsqrt(
                tf.reduce_mean(nonlin_i**2, axis=-1, keepdims=True) + 1e-16
            )
            rms_i = tf.where(r_map, rms_i, tf.zeros_like(rms_i))
            nonlin_i = nonlin_i * rms_i * tf.cast(self.scales[i], nonlin_i.dtype)
            nonlin_normed.append(nonlin_i)

            lin_sum = lin_sum + tf.cast(self.alpha[i], lin_i.dtype) * lin_i

        at_mu_i = None
        if self.element_dependent:
            at_mu_i = (
                input_data[constants.ATOMIC_MU_I_LOCAL]
                if local
                else input_data[constants.ATOMIC_MU_I]
            )

        if self.mlp_mode == "shared":
            x = tf.add_n(nonlin_normed)
            nonlin_out = self._mlp_forward(x, layer_idx=0, at_mu_i=at_mu_i)
        else:
            nonlin_out = 0.0
            for i, x in enumerate(nonlin_normed):
                nonlin_out = nonlin_out + self._mlp_forward(
                    x, layer_idx=i, at_mu_i=at_mu_i
                )

        return target + lin_sum + nonlin_out


@capture_init_args
class ConstantScaleShiftTarget(TPOutputInstruction, ElementsReduceInstructionMixin):
    """
    Scales and shifts target by a (atom type dependent) constant
    """

    input_tensor_spec = {
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
        constants.N_ATOMS_BATCH_REAL: {"shape": [], "dtype": "int"},
    }

    def __init__(
        self,
        target: TPInstruction,
        scale: float = 1.0,
        shift: float = 0.0,
        atomic_shift_map: dict[int, float] = None,
        chemical_embedding: ScalarChemicalEmbedding = None,
        name: str = "ConstantScaleShiftTarget",
        l: int = 0,  # noqa: E741
    ):
        super(ConstantScaleShiftTarget, self).__init__(name=name, target=target, l=l)
        self.scale = scale
        self.constant_shift = shift
        if atomic_shift_map is not None:
            self.atomic_shift_map = np.array(
                [v for v in dict(sorted(atomic_shift_map.items())).values()]
            )
        else:
            self.atomic_shift_map = None
        self.chemical_embedding = chemical_embedding

        if (
            self.constant_shift == 0
            and self.atomic_shift_map is None
            and self.chemical_embedding is None
        ):
            self.apply_shift = False
        else:
            if self.atomic_shift_map is not None or self.chemical_embedding is not None:
                assert self.constant_shift == 0, (
                    "If :atomic_shift_map: or :chemical_embedding: is not None "
                    "constant shift must be 0, but is not"
                )
            self.apply_shift = True

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        self.scale = tf.constant(self.scale, dtype=float_dtype)
        if self.constant_shift != 0:
            self.constant_shift = tf.constant(self.constant_shift, dtype=float_dtype)
        if self.atomic_shift_map is not None:
            self.atomic_shift_map = tf.reshape(
                tf.constant(self.atomic_shift_map, dtype=float_dtype), [-1, 1]
            )
        if self.chemical_embedding is not None:
            self.embedding_shift = tf.Variable(
                tf.random.normal(
                    [self.chemical_embedding.embedding_size, 1], dtype=float_dtype
                ),
                name="embedding_shift",
            )
            self.embedding_norm = tf.math.rsqrt(
                tf.cast(self.chemical_embedding.embedding_size, dtype=float_dtype)
            )

    def frwrd(self, input_data, training=False, local=False):
        target = input_data[f"{self.target.name}"]

        atomic_mu_i = (
            input_data[constants.ATOMIC_MU_I_LOCAL]
            if local
            else input_data[constants.ATOMIC_MU_I]
        )
        n_at_b_total = tf.shape(atomic_mu_i)[0]
        n_at_b_real = input_data[constants.N_ATOMS_BATCH_REAL]
        if self.apply_shift:
            # Updating only real atoms, fake stay at 0.
            r_map = tf.reshape(
                tf.range(n_at_b_total, delta=1, dtype=tf.int32, name="r_map"), [-1, 1]
            )
            zero_shift = tf.zeros_like(target)
            total_shift = tf.zeros_like(target)

            if self.atomic_shift_map is not None:
                real_shift = tf.gather(self.atomic_shift_map, atomic_mu_i, axis=0)
                real_shift = tf.where(r_map < n_at_b_real, real_shift, zero_shift)
                total_shift += tf.cast(real_shift, total_shift.dtype)

            if self.chemical_embedding is not None:
                real_shift = (
                    tf.einsum(
                        "ae,eo->ao",
                        tf.gather(
                            input_data[self.chemical_embedding.name],
                            atomic_mu_i,
                            axis=0,
                        ),
                        self.embedding_shift,
                    )
                    * self.embedding_norm
                )
                real_shift = tf.where(r_map < n_at_b_real, real_shift, zero_shift)
                total_shift += tf.cast(real_shift, total_shift.dtype)

            if self.constant_shift != 0:
                real_shift = tf.ones_like(target) * self.constant_shift
                real_shift = tf.where(r_map < n_at_b_real, real_shift, zero_shift)
                total_shift += tf.cast(real_shift, total_shift.dtype)

            return target * tf.cast(self.scale, target.dtype) + total_shift
        else:
            return target * tf.cast(self.scale, target.dtype)

    def prepare_variables_for_selected_elements(self, index_to_select):
        result = {}
        if self.atomic_shift_map is not None:
            result["atomic_shift_map"] = tf.gather(
                self.atomic_shift_map, index_to_select, axis=0
            )
        return result

    def upd_init_args_new_elements(self, new_element_map):
        if self._init_args.get("atomic_shift_map") is not None:
            # After patching, self.atomic_shift_map has only the selected values
            # Re-index them as 0..N-1
            new_map = {}
            for i in range(len(new_element_map)):
                new_map[i] = float(self.atomic_shift_map[i].numpy().item())
            self._init_args["atomic_shift_map"] = new_map


@capture_init_args
class LinearOut2EquivarTarget(TPOutputInstruction):
    """
    Adds origin to target without transformation
    """

    def __init__(
        self,
        origin: list[FunctionReduceParticular],
        l: int,  # noqa: E741
        target: CreateOutputTarget,
        name="LinearEquivarOut2Target",
        full_r2_form: bool = False,
    ):
        super(LinearOut2EquivarTarget, self).__init__(name=name, target=target, l=l)
        self.origin = origin
        self.full_r2_form = full_r2_form

        lmax = np.max([ins.lmax for ins in self.origin])
        assert lmax == self.l, f"Origin l={lmax} does not match target l={self.l}"
        self.assert_l_compatibility(target)

        out_shapes = [ins.n_out for ins in self.origin]
        assert len(set(out_shapes)) == 1, "Not all shapes are the same"
        if self.l > 2:
            raise NotImplementedError(
                "LinearEquivarOut2Target can only do vectors and matrices for now"
            )
        if self.l == 2:
            a = -0.5 / np.sqrt(3)
            b = 1 / np.sqrt(3)
            if full_r2_form:
                self.transform = np.array(
                    [
                        [0, 0.5, 0, 0.5, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0.5, 0, 0.5, 0],
                        [a, 0, 0, 0, a, 0, 0, 0, b],
                        [0, 0, 0.5, 0, 0, 0, 0.5, 0, 0],
                        [0.5, 0, 0, 0, -0.5, 0, 0, 0, 0],
                    ]
                ).reshape(5, 9)
            else:
                self.transform = np.array(
                    [
                        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                        [a, a, b, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
                        [0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
                    ]
                ).reshape(5, 6)

    def frwrd(self, input_data, training=False, local=False):
        target = input_data[f"{self.target.name}"]

        for ins in self.origin:
            tensor = input_data[f"{ins.name}"]
            if self.l == 1:
                r_tensor = tf.roll(tensor, shift=1, axis=2)
                target += r_tensor
            elif self.l == 2:
                trnsfrm = tf.constant(self.transform, dtype=tensor.dtype)
                r_tensor = tf.einsum("...b,bc->...c", tensor, trnsfrm)
                target += r_tensor
            else:
                raise NotImplementedError(
                    "LinearEquivarOut2Target can only do vectors and matrices for now"
                )

        return target


@capture_init_args
class TrainableShiftTarget(TPOutputInstruction, ElementsReduceInstructionMixin):
    input_tensor_spec = {
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
        constants.N_ATOMS_BATCH_REAL: {"shape": [], "dtype": "int"},
    }

    def __init__(
        self,
        target: TPInstruction,
        number_of_atom_types: int,
        name: str = "TrainableShiftTarget",
        l: int = 0,  # noqa: E741
    ):
        super(TrainableShiftTarget, self).__init__(name=name, target=target, l=l)
        self.number_of_atom_types = number_of_atom_types

    def build(self, float_dtype):
        if not self.is_built:
            self.at_shifts = tf.Variable(
                tf.zeros([self.number_of_atom_types, 1], dtype=float_dtype),
                trainable=True,
                name="tr_atomic_shift",
            )
        self.is_built = True

    def frwrd(self, input_data: dict, training: bool = False, local: bool = False):
        target = input_data[f"{self.target.name}"]

        at_mu_i = (
            input_data[constants.ATOMIC_MU_I_LOCAL]
            if local
            else input_data[constants.ATOMIC_MU_I]
        )
        n_at_b_total = tf.shape(at_mu_i)[0]
        n_at_b_real = input_data[constants.N_ATOMS_BATCH_REAL]

        # Updating only real atoms, fake stay at 0.
        r_map = tf.reshape(
            tf.range(n_at_b_total, delta=1, dtype=tf.int32, name="r_map"), [-1, 1]
        )
        real_shift = tf.gather(self.at_shifts, at_mu_i, axis=0)
        real_shift = tf.where(r_map < n_at_b_real, real_shift, tf.zeros_like(target))
        return target + tf.cast(real_shift, target.dtype)

    def prepare_variables_for_selected_elements(self, index_to_select):
        return {
            "at_shifts": tf.Variable(
                tf.gather(self.at_shifts, index_to_select, axis=0)
            ),
        }

    def upd_init_args_new_elements(self, new_element_map):
        self._init_args["number_of_atom_types"] = len(new_element_map)


@capture_init_args
class TrainableShiftTarget_v2(TPOutputInstruction):
    """Trainable energy shift as a projection of a chemical embedding vector.

    Instead of learning an independent scalar per element, this projects the
    shared chemical embedding to a scalar shift:  shift_i = W @ z[mu_i].
    Projection weights are initialized to zeros so the initial model is
    identical to one without the shift.
    """

    input_tensor_spec = {
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
        constants.N_ATOMS_BATCH_REAL: {"shape": [], "dtype": "int"},
    }

    def __init__(
        self,
        target: TPInstruction,
        chemical_embedding: ScalarChemicalEmbedding,
        name: str = "TrainableShiftTarget_v2",
        l: int = 0,  # noqa: E741
    ):
        super().__init__(name=name, target=target, l=l)
        self.chemical_embedding = chemical_embedding

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            self.proj = tf.Variable(
                tf.zeros(
                    [self.chemical_embedding.embedding_size, 1], dtype=float_dtype
                ),
                trainable=True,
                name="shift_proj",
            )
            self.is_built = True

    def frwrd(self, input_data: dict, training: bool = False, local: bool = False):
        target = input_data[self.target.name]

        at_mu_i = (
            input_data[constants.ATOMIC_MU_I_LOCAL]
            if local
            else input_data[constants.ATOMIC_MU_I]
        )
        n_at_b_total = tf.shape(at_mu_i)[0]
        n_at_b_real = input_data[constants.N_ATOMS_BATCH_REAL]

        # z: [n_types, emb_size], proj: [emb_size, 1] -> shifts: [n_types, 1]
        z = input_data[self.chemical_embedding.name]
        z = tf.cast(z, self.proj.dtype)
        shifts = z @ self.proj  # [n_types, 1]

        real_shift = tf.gather(shifts, at_mu_i, axis=0)  # [n_atoms, 1]

        # Zero out padding atoms
        r_map = tf.reshape(
            tf.range(n_at_b_total, delta=1, dtype=tf.int32, name="r_map"), [-1, 1]
        )
        real_shift = tf.where(r_map < n_at_b_real, real_shift, tf.zeros_like(target))
        return target + tf.cast(real_shift, target.dtype)
