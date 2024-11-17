from __future__ import annotations

import numpy as np
import tensorflow as tf

from tensorpotential import constants
from tensorpotential.functions.nn import FullyConnectedMLP
from tensorpotential.instructions.base import (
    capture_init_args,
    TPInstruction,
)
from tensorpotential.instructions.compute import (
    FunctionReduce,
    FunctionReduceN,
    FunctionReduceParticular,
    ScalarChemicalEmbedding,
)


@capture_init_args
class TPOutputInstruction(TPInstruction):
    def __init__(self, name: str, target: TPInstruction, l: int = 0):
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

    def __call__(self, input_data: dict, training: bool = False) -> dict:
        output = self.frwrd(input_data, training=training)

        input_data[self.target.name] = output

        return input_data


@capture_init_args
class CreateOutputTarget(TPInstruction):
    def __init__(self, name: str, initial_value: float = 0.0, l: int = 0):
        super().__init__(name=name)
        self.l = l
        if initial_value is None:
            self.value = 0.0
        else:
            self.value = initial_value

    def build(self, float_dtype):
        if not self.is_built:
            self.value = tf.reshape(tf.constant(self.value, dtype=float_dtype), [])
            self.is_built = True

    def frwrd(self, input_data, training=False):
        return self.value


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
        l: int = 0,
    ):
        super(LinearOut2Target, self).__init__(name=name, target=target, l=l)
        self.origin = origin

        try:
            lmax = np.max([ins.lmax for ins in self.origin])
        except AttributeError:
            lmax = 0
        assert self.l == lmax, f"lmax > target l={target.l} for some instructions"
        self.assert_l_compatibility(target)

        out_shapes = [ins.n_out for ins in self.origin]
        assert len(set(out_shapes)) == 1, "Not all shapes are the same"

    def build(self, float_dtype):
        pass

    def frwrd(self, input_data, training=False):
        target = input_data[f"{self.target.name}"]

        origin = 0.0
        for ins in self.origin:
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
        l: int = 0,
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
        ), f"`n_out` if FSOut2ScalarTarget.origin should be equal to len(fs_parameters)"
        self.fs_parameters = fs_parameters

    def build(self, float_dtype):
        self.fs_parameters = [
            (
                tf.constant(coeff, dtype=float_dtype),
                tf.constant(mexp, dtype=float_dtype),
            )
            for coeff, mexp in self.fs_parameters
        ]

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

    def frwrd(self, input_data, training=False):
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

    def __init__(
        self,
        origin: list[FunctionReduce],
        target: CreateOutputTarget,
        hidden_layers: list[int] = None,
        n_out: int = 1,
        name="MLPOut2ScalarTarget",
        normalize: bool = False,
        l: int = 0,
    ):
        super(MLPOut2ScalarTarget, self).__init__(name=name, target=target)
        self.origin = origin
        self.n_out = n_out
        self.normalize = normalize
        if self.normalize:
            self.norm_n_origins = 1 / len(origin) ** 0.5
        else:
            self.norm_n_origins = 1

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
        )

    def build(self, float_dtype):
        if not self.mlp.is_built:
            self.mlp.build(float_dtype)
        self.norm_n_origins = tf.constant(self.norm_n_origins, dtype=float_dtype)
        self.is_built = True

    def frwrd(self, input_data, training=False):
        target = input_data[f"{self.target.name}"]

        origin = 0.0
        for ins in self.origin:
            origin += input_data[f"{ins.name}"][:, :, 0]
        origin = self.mlp(origin)

        return target + origin * self.norm_n_origins


@capture_init_args
class LinMLPOut2ScalarTarget(MLPOut2ScalarTarget):
    """
    Adds origin to target via MLP transformation and a single element without transformation
    """

    def __init__(
        self,
        origin: list[FunctionReduce | FunctionReduceN],
        target: CreateOutputTarget,
        hidden_layers: list[int] = None,
        name="MLPOut2ScalarTarget",
        n_out: int = 1,
        normalize: bool = False,
        activation: callable = None,
        l: int = 0,
    ):
        super(LinMLPOut2ScalarTarget, self).__init__(
            origin=origin,
            target=target,
            hidden_layers=hidden_layers,
            name=name,
            normalize=normalize,
            n_out=n_out,
        )

        self.mlp = FullyConnectedMLP(
            input_size=self.origin[0].n_out - 1,
            hidden_layers=self.hidden_layers,
            activation=activation,
            output_size=self.n_out,
        )
        if self.normalize:
            self.norm_contr = 0.707106781186547
        else:
            self.norm_contr = 1.0

    def frwrd(self, input_data, training=False):
        target = input_data[f"{self.target.name}"]

        transformed_origin = 0.0
        lin_origin = 0.0
        for ins in self.origin:
            full_scalar = input_data[f"{ins.name}"][:, :, 0]
            transformed_origin += full_scalar[:, 1:]
            lin_origin += tf.reshape(full_scalar[:, 0], [-1, 1])
        transformed_origin = self.mlp(transformed_origin)
        norm_out = (
            (transformed_origin + lin_origin)
            * self.norm_n_origins
            * tf.constant(self.norm_contr, dtype=self.norm_n_origins.dtype)
        )

        return target + norm_out


@capture_init_args
class ConstantScaleShiftTarget(TPOutputInstruction):
    """
    Scales and shifts target by a (atom type dependent) constant
    """

    input_tensor_spec = {
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
        constants.N_ATOMS_BATCH_REAL: {"shape": [], "dtype": "int"},
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
    }

    def __init__(
        self,
        target: TPInstruction,
        scale: float = 1.0,
        shift: float = 0.0,
        atomic_shift_map: dict[int, float] = None,
        chemical_embedding: ScalarChemicalEmbedding = None,
        name: str = "ConstantScaleShiftTarget",
        l: int = 0,
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
                    "If :atomic_shift_map: or :chemical_embedding: is not None"
                    "constant shift must be 0, but is not"
                )
            self.apply_shift = True

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

    def frwrd(self, input_data, training=False):
        target = input_data[f"{self.target.name}"]

        if self.apply_shift:
            n_at_b_real = input_data[constants.N_ATOMS_BATCH_REAL]
            n_at_b_total = input_data[constants.N_ATOMS_BATCH_TOTAL]

            # Updating only real atoms, fake stay at 0.
            r_map = tf.reshape(
                tf.range(n_at_b_total, delta=1, dtype=tf.int32, name="r_map"), [-1, 1]
            )
            zero_shift = tf.zeros_like(target)
            total_shift = tf.zeros_like(target)

            if self.atomic_shift_map is not None:
                real_shift = tf.gather(
                    self.atomic_shift_map, input_data[constants.ATOMIC_MU_I], axis=0
                )
                real_shift = tf.where(r_map < n_at_b_real, real_shift, zero_shift)
                total_shift += real_shift

            if self.chemical_embedding is not None:
                real_shift = (
                    tf.einsum(
                        "ae,eo->ao",
                        tf.gather(
                            input_data[self.chemical_embedding.name],
                            input_data[constants.ATOMIC_MU_I],
                            axis=0,
                        ),
                        self.embedding_shift,
                    )
                    * self.embedding_norm
                )
                real_shift = tf.where(r_map < n_at_b_real, real_shift, zero_shift)
                total_shift += real_shift

            if self.constant_shift != 0:
                real_shift = tf.ones_like(target) * self.constant_shift
                real_shift = tf.where(r_map < n_at_b_real, real_shift, zero_shift)
                total_shift += real_shift

            return target * self.scale + total_shift
        else:
            return target * self.scale


@capture_init_args
class LinearOut2EquivarTarget(TPOutputInstruction):
    """
    Adds origin to target without transformation
    """

    def __init__(
        self,
        origin: list[FunctionReduceParticular],
        l: int,
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
                f"LinearEquivarOut2Target can only do vectors and matrices for now"
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

    def frwrd(self, input_data, training=False):
        target = input_data[f"{self.target.name}"]

        for ins in self.origin:
            tensor = input_data[f"{ins.name}"]
            if self.l == 1:
                r_tensor = tf.roll(tensor, shift=1, axis=2)
                target += r_tensor
            elif self.l == 2:
                trnsfrm = tf.constant(self.transform, dtype=target.dtype)
                r_tensor = tf.einsum("...b,bc->...c", tensor, trnsfrm)
                target += r_tensor
            else:
                raise NotImplementedError(
                    "LinearEquivarOut2Target can only do vectors and matrices for now"
                )

        return target
