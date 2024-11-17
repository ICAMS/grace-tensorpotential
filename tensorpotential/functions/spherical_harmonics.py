from __future__ import annotations

import tensorflow as tf

from numpy import pi as np_pi


class SphericalHarmonics(tf.Module):
    """
    Computes complex or real spherical harmonics

    Parameters
    ----------
    lmax: int
        Maximum angular momentum l

    norm: bool = False
        Normalize by a sqrt(4 * pi). Default False

    type: str = 'real'
        Type of the computed spherical harmonics.
        Available types are: real, complex

    float_prec: tf.dtypes = tf.float64
        Float precision. Default float64

    int_prec: tf.dtypes = tf.int32
        Int precision. Default int32
    """

    def __init__(self, lmax: int, norm: bool = False, type: str = "real"):
        super().__init__(name="SphericalHarmonics")
        self.lmax = lmax
        self.norm = norm
        self.type = type
        self.is_built = False

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, type={self.type})"

    def build(self, float_dtype):
        self.float_dtype = float_dtype
        self.int_dtype = tf.int32
        self.PI = tf.constant(np_pi, dtype=self.float_dtype, name="pi")
        self.alm, self.blm = self.pre_compute()
        self.factor4pi = tf.sqrt(4 * self.PI)
        self.l_tile = tf.cast(
            tf.concat([tf.ones((2 * l + 1)) * l for l in range(self.lmax + 1)], axis=0),
            self.int_dtype,
            name="l_tile",
        )
        self.is_built = True

    @staticmethod
    def lm1d(l, m):
        return m + l * (l + 1) // 2

    def lmsh(self, l, m):
        return l + abs(m) * self.lmax - abs(m) * (abs(m) - 1) // 2

    @staticmethod
    def lm_m(l, m):
        return l * (l + 1) + m

    def pre_compute(self):
        alm = [tf.convert_to_tensor(0.0, dtype=self.float_dtype)]
        blm = [tf.convert_to_tensor(0.0, dtype=self.float_dtype)]
        lindex = tf.range(0, self.lmax + 1, dtype=self.float_dtype)
        for i in range(1, self.lmax + 1):
            l = lindex[i]
            lsq = l * l
            ld = 2 * l
            l1 = 4 * lsq - 1
            l2 = lsq - ld + 1
            for j in range(0, i + 1):
                m = lindex[j]
                msq = m * m
                a = tf.sqrt(l1 / (lsq - msq))
                b = -tf.sqrt((l2 - msq) / (4 * l2 - 1))
                if i == j:
                    cl = -tf.sqrt(1.0 + 0.5 / m)
                    alm += [cl]
                    blm += [0]  # placeholder
                else:
                    alm += [a]
                    blm += [b]

        return tf.stack(alm), tf.stack(blm)

    def legendre(self, x):
        x = tf.convert_to_tensor(x, dtype=self.float_dtype)

        y00 = 1.0 * tf.sqrt(1.0 / (4.0 * self.PI))
        plm = [x * 0 + y00]
        if self.lmax > 0:
            sq3o4pi = tf.sqrt(3.0 / (4.0 * self.PI))
            sq3o8pi = tf.sqrt(3.0 / (8.0 * self.PI))

            plm += [sq3o4pi * x]  # (1,0)
            plm += [x * 0 - sq3o8pi]  # (1,1)

            for l in range(2, self.lmax + 1):
                for m in range(0, l + 1):
                    if m == l - 1:
                        dl = tf.sqrt(2.0 * m + tf.constant(3.0, dtype=self.float_dtype))
                        plm += [x * dl * plm[self.lm1d(l - 1, l - 1)]]
                    elif m == l:
                        plm += [
                            self.alm[self.lm1d(l, l)] * plm[self.lm1d(l - 1, l - 1)]
                        ]
                    else:
                        plm += [
                            self.alm[self.lm1d(l, m)]
                            * (
                                x * plm[self.lm1d(l - 1, m)]
                                + self.blm[self.lm1d(l, m)] * plm[self.lm1d(l - 2, m)]
                            )
                        ]
        plm = tf.stack(plm)

        return plm

    def compute_ylm_sqr(self, rhat):
        ylm_r, ylm_i = self._compute_sph_harm(rhat)

        sqr_ylm_r = []
        sqr_ylm_i = []
        for l in range(0, self.lmax + 1):
            for m in range(-self.lmax, self.lmax + 1):
                ph = tf.where(
                    tf.equal(tf.abs(m) % 2, 0),
                    tf.constant(1, dtype=self.float_dtype),
                    tf.constant(-1, dtype=self.float_dtype),
                )
                ph_r = ph
                if m < 0 and abs(m) > l:
                    sqr_ylm_r += [0 * ylm_r[0]]
                    sqr_ylm_i += [0 * ylm_i[0]]
                elif m < 0 and abs(m) <= l:
                    ind = self.lmsh(
                        l, m
                    )  # l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2
                    sqr_ylm_r += [ylm_r[ind] * ph_r]
                    sqr_ylm_i += [tf.negative(ylm_i[ind]) * ph_r]
                elif m >= 0 and abs(m) <= l:
                    ind = self.lmsh(
                        l, m
                    )  # l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2
                    sqr_ylm_r += [ylm_r[ind]]
                    sqr_ylm_i += [ylm_i[ind]]
                else:
                    sqr_ylm_r += [0 * ylm_r[0]]
                    sqr_ylm_i += [0 * ylm_i[0]]

        ylm_r = tf.transpose(
            tf.stack(sqr_ylm_r), [1, 0, 2]
        )  ##[nYlm, None, 1] -> [None, nYlm, 1]
        ylm_i = tf.transpose(
            tf.stack(sqr_ylm_i), [1, 0, 2]
        )  ##[nYlm, None, 1] -> [None, nYlm, 1]
        if not self.norm:
            ylm_r *= self.factor4pi
            ylm_i *= self.factor4pi

        return tf.reshape(ylm_r, [-1, self.lmax + 1, 2 * self.lmax + 1]), tf.reshape(
            ylm_i, [-1, self.lmax + 1, 2 * self.lmax + 1]
        )  ##[None, nYlm, 1] -> [None, l, m]

    def _compute_sph_harm(self, rhat):
        rhat = tf.convert_to_tensor(rhat, dtype=self.float_dtype)

        rx = rhat[:, 0]
        ry = rhat[:, 1]
        rz = rhat[:, 2]

        phase_r = rx
        phase_i = ry

        ylm_r = []
        ylm_i = []
        plm = self.legendre(rz)

        m = 0
        for l in range(0, self.lmax + 1):
            ylm_r += [plm[self.lm1d(l, m)]]
            # ylm_i += [self.float_tensor(tf.zeros_like(plm[self.lm1d(l, m)]))]
            ylm_i += [tf.zeros_like(plm[self.lm1d(l, m)], dtype=self.float_dtype)]

        m = 1
        for l in range(1, self.lmax + 1):
            ylm_r += [phase_r * plm[self.lm1d(l, m)]]
            ylm_i += [phase_i * plm[self.lm1d(l, m)]]

        phasem_r = phase_r
        phasem_i = phase_i
        for m in range(2, self.lmax + 1):
            pr_tmp = phasem_r
            phasem_r = phasem_r * phase_r - phasem_i * phase_i
            phasem_i = pr_tmp * phase_i + phasem_i * phase_r

            for l in range(m, self.lmax + 1):
                ylm_r += [phasem_r * plm[self.lm1d(l, m)]]
                ylm_i += [phasem_i * plm[self.lm1d(l, m)]]

        return ylm_r, ylm_i

    def _compute_ylm(self, rhat):
        ylm_r, ylm_i = self._compute_sph_harm(rhat)

        sqr_ylm_r = []
        sqr_ylm_i = []
        for l in range(0, self.lmax + 1):
            for m in range(-self.lmax, self.lmax + 1):
                ph = tf.where(
                    tf.equal(tf.abs(m) % 2, 0),
                    tf.constant(1, dtype=self.float_dtype),
                    tf.constant(-1, dtype=self.float_dtype),
                )
                ph_r = ph
                if m < 0 and abs(m) > l:
                    pass
                elif m < 0 and abs(m) <= l:
                    ind = self.lmsh(
                        l, m
                    )  # l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2
                    sqr_ylm_r += [ylm_r[ind] * ph_r]
                    sqr_ylm_i += [tf.negative(ylm_i[ind]) * ph_r]
                elif m >= 0 and abs(m) <= l:
                    ind = self.lmsh(
                        l, m
                    )  # l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2
                    sqr_ylm_r += [ylm_r[ind]]
                    sqr_ylm_i += [ylm_i[ind]]
                else:
                    pass
        ylm_r = tf.transpose(
            tf.stack(sqr_ylm_r), [1, 0, 2]
        )  ##[nYlm, None, 1] -> [None, nYlm, 1]
        ylm_i = tf.transpose(
            tf.stack(sqr_ylm_i), [1, 0, 2]
        )  ##[nYlm, None, 1] -> [None, nYlm, 1]
        if not self.norm:
            ylm_r *= self.factor4pi
            ylm_i *= self.factor4pi

        return ylm_r[:, :, 0], ylm_i[:, :, 0]  # [None, nYlm, 1] -> [None, nYlm]

    def _compute_rsh(self, rhat):
        ylm_r, ylm_i = self._compute_sph_harm(rhat)

        zlm = []
        for l in range(0, self.lmax + 1):
            for m in range(-l, l + 1):
                ph = tf.where(
                    tf.equal(tf.abs(m) % 2, 0),
                    tf.constant(1, dtype=self.float_dtype),
                    tf.constant(-1, dtype=self.float_dtype),
                )

                sqrt2 = tf.math.sqrt(tf.constant(2.0, dtype=self.float_dtype))
                if m < 0:
                    ind = self.lmsh(l, m)
                    zlm += [sqrt2 * ylm_i[ind] * ph]
                elif m > 0:
                    ind = self.lmsh(l, m)
                    zlm += [sqrt2 * ylm_r[ind] * ph]
                elif m == 0:
                    ind = self.lmsh(l, m)
                    zlm += [ylm_r[ind]]
                else:
                    pass
        zlm = tf.transpose(tf.stack(zlm), [1, 0])

        if not self.norm:
            zlm *= self.factor4pi

        return zlm

    def __call__(self, rhat):
        # if not self.is_built:
        #     self._build(rhat.dtype)

        if self.type == "real":
            return self._compute_rsh(rhat)
        elif self.type == "complex":
            return self._compute_ylm(rhat)
        else:
            raise ValueError(f"Unknown type {self.type}")
