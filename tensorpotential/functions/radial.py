from __future__ import annotations

import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod
from scipy.special import sici


class RadialBasisFunction(tf.Module, ABC):
    """
    Base calss for radial basis functions

    Parameters
    ----------
    :nfunc: int
        Number of radial basis functions to construct

    :rcut: float
        Maximum cutoff distance

    :name: str
        Name of the radial basis function type

    Input
    ----------

    r: tf.Tensor [None, 1]
        Array of scalar edge lengths
    """

    def __init__(self, nfunc: int, rcut: float, name: str):
        super().__init__(name=name)
        if nfunc >= 1:
            self.nfunc = nfunc
        else:
            raise ValueError(
                f"Must be at least one basis function, {nfunc=} is provided"
            )
        if rcut > 0:
            self.rcut = rcut
        else:
            raise ValueError(
                f"Cutoff distance must be larger than 0, {rcut=} is provided"
            )
        self.is_build = False

    def build(self, float_dtype):
        self.PI = tf.constant(np.pi, dtype=float_dtype, name="pi")
        self.rc = tf.Variable(
            self.rcut, dtype=float_dtype, name="cutoff", trainable=False
        )
        self.epsilon = tf.constant(1e-8, dtype=float_dtype, name="small_epsilon")
        self.is_build = True

    @abstractmethod
    def compute_basis(self, r):
        return r

    def __repr__(self):
        return f"{self.name}(nfunc={self.nfunc}, rcut={self.rcut})"

    @tf.Module.with_name_scope
    def __call__(self, r: tf.Tensor) -> tf.Tensor:
        r = tf.where(r == 0.0, r + self.epsilon, r)
        basis = self.compute_basis(r)

        return tf.where(r > self.rcut, tf.zeros_like(basis, dtype=r.dtype), basis)


class GaussianRadialBasisFunction(RadialBasisFunction):
    """ """

    def __init__(
        self,
        nfunc: int,
        rcut: float,
        p: int,
        normalized: bool = False,
        rmin: float = 0.0,
        init_gamma: float = 1,
        trainable: bool = False,
        **kwargs,
    ):
        super().__init__(nfunc, rcut, name="GaussianRadialBasisFunction")
        self.pcut = p
        self.rmin = rmin
        self.grid = np.linspace(self.rmin, self.rcut, self.nfunc).reshape(1, -1)
        self.scale = np.ones_like(self.grid) * init_gamma
        self.trainable = trainable
        self.normalized = normalized
        if self.normalized:
            self.norm = 1 / np.sqrt(2 * np.pi)

    def build(self, float_dtype):
        super().build(float_dtype)
        self.grid = tf.Variable(self.grid, dtype=float_dtype, trainable=self.trainable)
        self.scale = tf.Variable(
            self.scale, dtype=float_dtype, trainable=self.trainable
        )
        if self.normalized:
            if self.trainable:
                self.norm = tf.convert_to_tensor(self.norm, dtype=float_dtype)
            else:
                self.norm = tf.convert_to_tensor(
                    self.norm, dtype=float_dtype
                ) * tf.math.rsqrt(self.scale)
        self.is_build = True

    def compute_basis(self, r):
        # basis = tf.math.exp(-self.scale * 0.5 * (r - self.grid) ** 2)
        basis = tf.math.exp(-(self.scale**2) * (r - self.grid) ** 2)
        if self.normalized:
            if self.trainable:
                gamma_norm = tf.math.rsqrt(self.scale)
                basis = basis * self.norm * gamma_norm
            else:
                basis = basis * self.norm

        return basis * cutoff_func_p_order_poly(r / self.rc, self.pcut)


class ChebSqrRadialBasisFunction(RadialBasisFunction):
    def __init__(
        self,
        nfunc: int,
        rcut: float,
        p: int = 5,
        normalized: bool = True,
        kind: int = 1,
        reversed: bool = False,
        **kwargs,
    ):
        super().__init__(nfunc, rcut, name="ChebSqrRadialBasisFunction")
        self.normalized = normalized
        self.lmbda = 1
        self.kind = kind
        self.reversed = reversed
        self.pcut = p
        if self.normalized:
            self.norm = np.sqrt(1 / np.pi)

    def build(self, float_dtype):
        if not self.is_build:
            super().build(float_dtype)
            if self.normalized:
                self.norm = tf.convert_to_tensor(self.norm, dtype=float_dtype)
        self.is_build = True

    def compute_basis(self, r):
        if self.reversed:
            r_rescale = -2.0 * (1.0 - tf.abs(1.0 - r / self.rc) ** self.lmbda) + 1.0
        else:
            r_rescale = 2.0 * (1.0 - tf.abs(1.0 - r / self.rc) ** self.lmbda) - 1.0
        # basis = 1 - chebvander(r_rescale, self.nfunc+1)[:, 1:]
        basis = chebvander(r_rescale, self.nfunc + 1, self.kind)[:, 1:]
        basis *= cutoff_func_p_order_poly(r / self.rc, self.pcut)
        if self.normalized:
            return basis * self.norm
        else:
            return basis


class SinBesselRadialBasisFunction(RadialBasisFunction):
    """
    Sin Bessel radial basis functions with polynomial envelop cutoff function

    Parameters
    ----------
    p: int
        Max polynomial degree

    """

    __doc__ = RadialBasisFunction.__doc__ + __doc__

    def __init__(
        self, nfunc: int, rcut: float, p: int, normalized: bool = False, **kwargs
    ):
        super().__init__(nfunc, rcut, name="SinBesselRadialBasisFunction")
        self.pcut = p
        self.normalized = normalized
        if self.normalized:
            urcut = 1.0
            ns = np.arange(1, nfunc + 1)
            npi = ns * np.pi
            mu = np.sqrt(2 / urcut) * sici(npi)[0]
            sigma2 = (
                2 * npi * sici(2 * npi)[0]
                + (np.cos(2 * npi) - 1) / urcut**2
                - 2 ** (3 / 2) / np.sqrt(urcut) * mu * sici(npi)[0]
                + mu**2 * urcut
            )
            self.mu = mu
            self.sigma = np.sqrt(sigma2)

    def build(self, float_dtype):
        self.PI = tf.constant(np.pi, dtype=float_dtype, name="pi")
        self.rc = tf.Variable(
            self.rcut, dtype=float_dtype, name="cutoff", trainable=False
        )
        self.epsilon = tf.constant(1e-8, dtype=float_dtype, name="small_epsilon")
        if self.normalized:
            self.urc = tf.constant(1.0, dtype=float_dtype, name="urc")
            self.mu = tf.constant(self.mu, dtype=float_dtype, name="nbessmu")
            self.sigma = tf.constant(self.sigma, dtype=float_dtype, name="nbesssigma")
        self.is_build = True

    def compute_basis(self, r):
        n = tf.range(1, 1 + self.nfunc, delta=1, dtype=r.dtype, name="range_sinb")

        if self.normalized:
            r /= self.rc
            f = (
                tf.math.sqrt(2 / self.urc) * tf.math.sin(n * r * self.PI / self.urc) / r
                - self.mu
            ) / self.sigma
            func = f * cutoff_func_p_order_poly(r, self.pcut)
        else:
            func = (
                tf.math.sqrt(2 / self.rc)
                * tf.math.sin(n * r * self.PI / self.rc)
                / r
                * cutoff_func_p_order_poly(r / self.rc, self.pcut)
            )
        return func


class SimplifiedBesselRadialBasisFunction(RadialBasisFunction):
    """
    Simplified Spherical Bessel radial basis functions
    """

    __doc__ = RadialBasisFunction.__doc__ + __doc__

    def __init__(self, nfunc: int, rcut: float, **kwargs):
        super().__init__(nfunc, rcut, name="SimplifiedBesselRadialBasisFunction")

    def fn(self, r, rc, n):
        return (
            tf.pow(tf.constant(-1, dtype=r.dtype), n)
            * tf.math.sqrt(tf.constant(2.0, dtype=r.dtype))
            * self.PI
            / tf.pow(rc, 3.0 / 2)
            * (n + 1)
            * (n + 2)
            / tf.math.sqrt((n + 1) ** 2 + (n + 2) ** 2)
            * (sinc(r * (n + 1) * self.PI / rc) + sinc(r * (n + 2) * self.PI / rc))
        )

    def compute_basis(self, r):
        sbf = [self.fn(r, self.rc, tf.constant(0, dtype=r.dtype))]
        d = [tf.constant(1, dtype=r.dtype)]
        for i in range(1, self.nfunc):
            n = tf.constant(i, dtype=r.dtype)
            en = n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
            dn = tf.constant(1, dtype=r.dtype) - en / d[i - 1]
            d += [dn]
            sbf += [
                1
                / tf.math.sqrt(d[i])
                * (self.fn(r, self.rc, n) + tf.math.sqrt(en / d[i - 1]) * sbf[i - 1])
            ]
        sbf = tf.stack(sbf)
        sbf = tf.transpose(sbf, [1, 2, 0])

        return sbf[:, 0, :]


def cutoff_func_p_order_poly(r, p):
    return (
        1
        - (p + 1) * (p + 2) / 2 * r**p
        + p * (p + 2) * r ** (p + 1)
        - p * (p + 1) / 2 * r ** (p + 2)
    )


def sinc(x):
    return tf.where(x != 0, tf.math.sin(x) / x, tf.ones_like(x, dtype=x.dtype))


def chebvander(x, deg, kind=1):
    v = [tf.ones_like(x)]
    if deg > 0:
        x2 = 2 * x
        if kind == 1:
            v += [x]
        elif kind == 2:
            v += [2 * x]
        else:
            raise ValueError("kind must be 1 or 2")

        for i in range(2, deg):
            v += [v[i - 1] * x2 - v[i - 2]]

    v = tf.stack(v)
    v = tf.transpose(v, [1, 2, 0])

    return v[:, 0, :]


def compute_cheb_radial_basis(r, nfunc, rcut, p, kind=1):
    roverrcut = r / rcut
    r_rescale = 2.0 * (1.0 - tf.abs(1.0 - roverrcut)) - 1.0
    basis = chebvander(r_rescale, nfunc + 1, kind)[:, 1:]
    basis *= cutoff_func_p_order_poly(roverrcut, p)

    return basis
