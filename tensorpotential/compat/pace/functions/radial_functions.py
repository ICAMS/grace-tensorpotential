import tensorflow as tf
import numpy as np


def chebvander(x, deg):
    v = [tf.ones_like(x)]
    if deg > 0:
        x2 = 2 * x
        v += [x]

        for i in range(2, deg):
            v += [v[i - 1] * x2 - v[i - 2]]

    v = tf.stack(v)
    v = tf.transpose(v, [1, 2, 0])

    return v


def bessel_j(x, order):
    j = [tf.math.special.bessel_j0(x)]

    if order > 0:
        j += [tf.math.special.bessel_j1(x)]
        for v in range(2, order):
            j += [((2 * (v - 1)) / x) * j[v - 1] - j[v - 2]]

    j = tf.stack(j)
    j = tf.transpose(j, [1, 2, 0])

    return j[:, 0, :]


def bessel_y(x, order):
    y = [tf.math.special.bessel_y0(x)]

    if order > 0:
        y += [tf.math.special.bessel_y1(x)]
        for v in range(2, order):
            y += [((2 * (v - 1)) / x) * y[v - 1] - y[v - 2]]

    y = tf.stack(y)
    y = tf.transpose(y, [1, 2, 0])

    return y[:, 0, :]


def sinc(x):
    return tf.where(x != 0, tf.math.sin(x) / x, tf.ones_like(x, dtype=tf.float64))


def fn(x, rc, n):
    pi = tf.constant(np.pi, dtype=tf.float64)
    return (
        tf.pow(tf.constant(-1, dtype=tf.float64), n)
        * tf.math.sqrt(tf.constant(2.0, dtype=tf.float64))
        * pi
        / tf.pow(rc, 3.0 / 2)
        * (n + 1)
        * (n + 2)
        / tf.math.sqrt((n + 1) ** 2 + (n + 2) ** 2)
        * (sinc(x * (n + 1) * pi / rc) + sinc(x * (n + 2) * pi / rc))
    )


def simplified_bessel(x, rc, deg):
    sbf = [fn(x, rc, tf.constant(0, dtype=tf.float64))]
    d = [tf.constant(1, dtype=tf.float64)]
    if deg > 0:
        for i in range(1, deg):
            n = tf.constant(i, dtype=tf.float64)
            en = n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
            dn = 1 - en / d[i - 1]
            d += [dn]
            sbf += [
                1
                / tf.math.sqrt(d[i])
                * (fn(x, rc, n) + tf.math.sqrt(en / d[i - 1]) * sbf[i - 1])
            ]

    sbf = tf.stack(sbf)
    sbf = tf.transpose(sbf, [1, 2, 0])

    return sbf[:, 0, :]


def legendre(x, order):
    l = [tf.ones_like(x)]
    if order > 0:
        l += [x]
        for i in range(2, order):
            l += [((2 * i - 1) * l[i - 1] * x - (i - 1) * l[i - 2]) / i]

    l = tf.stack(l)
    l = tf.transpose(l, [1, 2, 0])

    return l[:, 0, :]


def cutoff_func_cos(x, rcut):
    condition1 = tf.less(x, rcut)
    cutoff = (1.0 + tf.cos(tf.constant(np.pi, dtype=tf.float64) * x / rcut)) * 0.5

    return tf.where(condition1, cutoff, tf.zeros_like(x, dtype=tf.float64))


def cutoff_func_poly(r, rin, delta):
    x = 1 - 2 * (1 + (r - rin) / (delta + 1e-8))
    f = 7.5 * (x / 4 - x**3 / 6 + x**5 / 20)
    condition1 = tf.less(rin, r)
    condition2 = tf.less(r, rin - delta)

    val1 = tf.zeros_like(r, dtype=tf.float64)
    val2 = tf.ones_like(r, dtype=tf.float64)

    return tf.where(condition1, val1, tf.where(condition2, val2, 0.5 * (1 + f)))


def cutoff_func_lin_cos(x, rcut, lin_fraq):
    condition1 = tf.less(x, rcut)
    condition2 = tf.less(x, rcut * lin_fraq)
    rcc = (x - rcut * lin_fraq) / (rcut - rcut * lin_fraq)
    cutoff = (1.0 + tf.cos(tf.constant(np.pi, dtype=tf.float64) * rcc)) * 0.5

    return tf.where(
        condition1,
        tf.where(condition2, tf.ones_like(x, dtype=tf.float64), cutoff),
        tf.zeros_like(x, dtype=tf.float64),
    )


def scale_distance(x, lmbda, rcut):
    x_scaled = 1.0 - 2.0 * (
        (tf.exp(-lmbda * (x / rcut - 1.0)) - 1.0) / (tf.exp(lmbda) - 1.0)
    )

    return x_scaled


def gaussian(x, probe, width):
    g = tf.exp(-width * ((tf.reshape(x, [-1, 1]) - probe) ** 2))

    return g
