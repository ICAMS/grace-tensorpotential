import tensorflow as tf
from numpy import pi as np_pi


class SphericalHarmonics:
    def __init__(self, lmax, prec="DOUBLE"):
        self._lmax = lmax
        # self.rhat = rhat
        self.prec = Precision(prec)
        self.alm, self.blm = self.pre_compute()
        self.pi = self.float_tensor(np_pi)
        self.l_tile = []

    def lm1d(self, l, m):
        return m + l * (l + 1) // 2

    def lmsh(self, l, m):
        return l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2

    def lm_m(self, l, m):
        return l * (l + 1) + m

    def float_tensor(self, x):
        return tf.convert_to_tensor(x, dtype=self.prec.float)

    def int_tensor(self, x):
        return tf.convert_to_tensor(x, dtype=self.prec.int)

    def pre_compute(self):
        alm = [self.float_tensor(0.0)]
        blm = [self.float_tensor(0.0)]
        lindex = tf.range(0, self._lmax + 1, dtype=self.prec.float)
        for i in range(1, self._lmax + 1):
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
        x = self.float_tensor(x)

        y00 = 1.0 * tf.sqrt(1.0 / (4.0 * self.pi))
        plm = [x * 0 + y00]
        if self._lmax > 0:
            sq3o4pi = tf.sqrt(3.0 / (4.0 * self.pi))
            sq3o8pi = tf.sqrt(3.0 / (8.0 * self.pi))

            plm += [sq3o4pi * x]  # (1,0)
            plm += [x * 0 - sq3o8pi]  # (1,1)

            for l in range(2, self._lmax + 1):
                for m in range(0, l + 1):
                    if m == l - 1:
                        dl = tf.sqrt(2.0 * m + self.float_tensor(3.0))
                        plm += [x * dl * plm[self.lm1d(l - 1, l - 1)]]
                    elif m == l:
                        plm += [
                            self.alm[self.lm1d(l, l)] * plm[self.lm1d(l - 1, l - 1)]
                        ]
                    else:
                        # plm += [0.]
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
        rhat = self.float_tensor(rhat)

        e_x = self.float_tensor([[1.0], [0.0], [0.0]])
        e_y = self.float_tensor([[0.0], [1.0], [0.0]])
        e_z = self.float_tensor([[0.0], [0.0], [1.0]])

        rx = tf.matmul(rhat, e_x)
        ry = tf.matmul(rhat, e_y)
        rz = tf.matmul(rhat, e_z)

        phase_r = rx
        phase_i = ry

        ylm_r = []
        ylm_i = []
        plm = self.legendre(rz)

        m = 0
        for l in range(0, self._lmax + 1):
            ylm_r += [plm[self.lm1d(l, m)]]
            ylm_i += [self.float_tensor(tf.zeros_like(plm[self.lm1d(l, m)]))]

        m = 1
        for l in range(1, self._lmax + 1):
            ylm_r += [phase_r * plm[self.lm1d(l, m)]]
            ylm_i += [phase_i * plm[self.lm1d(l, m)]]

        phasem_r = phase_r
        phasem_i = phase_i
        for m in range(2, self._lmax + 1):
            pr_tmp = phasem_r
            phasem_r = phasem_r * phase_r - phasem_i * phase_i
            phasem_i = pr_tmp * phase_i + phasem_i * phase_r

            for l in range(m, self._lmax + 1):
                ylm_r += [phasem_r * plm[self.lm1d(l, m)]]
                ylm_i += [phasem_i * plm[self.lm1d(l, m)]]

        sqr_ylm_r = []
        sqr_ylm_i = []
        for l in range(0, self._lmax + 1):
            for m in range(-self._lmax, self._lmax + 1):
                ph = tf.where(
                    tf.equal(tf.abs(m) % 2, 0),
                    tf.constant(1, dtype=self.prec.float),
                    tf.constant(-1, dtype=self.prec.float),
                )

                ph_r = ph
                ph_i = tf.zeros_like(ph, dtype=self.prec.float)
                if m < 0 and abs(m) > l:
                    sqr_ylm_r += [0 * ylm_r[0]]
                    sqr_ylm_i += [0 * ylm_i[0]]
                elif m < 0 and abs(m) <= l:
                    ind = self.lmsh(l, m)
                    sqr_ylm_r += [ylm_r[ind] * ph_r]
                    sqr_ylm_i += [tf.negative(ylm_i[ind]) * ph_r]
                elif m >= 0 and abs(m) <= l:
                    ind = self.lmsh(l, m)
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

        return tf.reshape(ylm_r, [-1, self._lmax + 1, 2 * self._lmax + 1]), tf.reshape(
            ylm_i, [-1, self._lmax + 1, 2 * self._lmax + 1]
        )  ##[None, nYlm, 1] -> [None, l, m]

    def compute_ylm(self, rhat):
        rhat = self.float_tensor(rhat)

        e_x = self.float_tensor([[1.0], [0.0], [0.0]])
        e_y = self.float_tensor([[0.0], [1.0], [0.0]])
        e_z = self.float_tensor([[0.0], [0.0], [1.0]])

        rx = tf.matmul(rhat, e_x)
        ry = tf.matmul(rhat, e_y)
        rz = tf.matmul(rhat, e_z)

        phase_r = rx
        phase_i = ry

        ylm_r = []
        ylm_i = []
        plm = self.legendre(rz)

        m = 0
        for l in range(0, self._lmax + 1):
            ylm_r += [plm[self.lm1d(l, m)]]
            ylm_i += [self.float_tensor(tf.zeros_like(plm[self.lm1d(l, m)]))]

        m = 1
        for l in range(1, self._lmax + 1):
            ylm_r += [phase_r * plm[self.lm1d(l, m)]]
            ylm_i += [phase_i * plm[self.lm1d(l, m)]]

        phasem_r = phase_r
        phasem_i = phase_i
        for m in range(2, self._lmax + 1):
            pr_tmp = phasem_r
            phasem_r = phasem_r * phase_r - phasem_i * phase_i
            phasem_i = pr_tmp * phase_i + phasem_i * phase_r

            for l in range(m, self._lmax + 1):
                ylm_r += [phasem_r * plm[self.lm1d(l, m)]]
                ylm_i += [phasem_i * plm[self.lm1d(l, m)]]

        sqr_ylm_r = []
        sqr_ylm_i = []
        for l in range(0, self._lmax + 1):
            for m in range(-self._lmax, self._lmax + 1):
                ph = tf.where(
                    tf.equal(tf.abs(m) % 2, 0),
                    tf.constant(1, dtype=self.prec.float),
                    tf.constant(-1, dtype=self.prec.float),
                )
                # ph = tf.complex(ph, tf.constant(0, dtype=self.prec.float))
                ph_r = ph
                if m < 0 and abs(m) > l:
                    pass
                elif m < 0 and abs(m) <= l:
                    self.l_tile += [l]
                    ind = self.lmsh(l, m)
                    sqr_ylm_r += [ylm_r[ind] * ph_r]
                    sqr_ylm_i += [tf.negative(ylm_i[ind]) * ph_r]
                elif m >= 0 and abs(m) <= l:
                    self.l_tile += [l]
                    ind = self.lmsh(l, m)
                    sqr_ylm_r += [ylm_r[ind]]
                    sqr_ylm_i += [ylm_i[ind]]
                else:
                    pass
        self.l_tile = tf.stack(self.l_tile)
        ylm_r = tf.transpose(
            tf.stack(sqr_ylm_r), [1, 0, 2]
        )  ##[nYlm, None, 1] -> [None, nYlm, 1]
        ylm_i = tf.transpose(
            tf.stack(sqr_ylm_i), [1, 0, 2]
        )  ##[nYlm, None, 1] -> [None, nYlm, 1]

        return ylm_r[:, :, 0], ylm_i[:, :, 0]  # [None, nYlm, 1] -> [None, nYlm]


class Precision:
    def __init__(self, prec):

        if prec == "DOUBLE":
            self.float = tf.float64
            self.int = tf.int64
        else:
            self.float = tf.float32
            self.int = tf.int32
