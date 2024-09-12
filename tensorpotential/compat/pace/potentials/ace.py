import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import product, combinations_with_replacement

from tensorpotential.compat.pace.functions import radial_functions, spherical_harmonics
from tensorpotential.compat.pace.utils.symbols import symbol_to_atomic_number


class ACE:
    def __init__(
        self,
        potconfig,
        nelements=1,
        rcut=5,
        lmbda=5.25,
        nradmax=5,
        lmax=4,
        nradbase=12,
        rankmax=2,
        ndensity=2,
        core_pre=0.0,
        core_lmbda=1.0,
        core_cut=100000.0,
        core_dcut=250.0,
        fs_parameters=None,
        compute_smoothness=False,
        compute_orthogonality=False,
    ):
        self.rcut = rcut
        self.nradmax = nradmax
        self.nradbase = nradbase
        self.lmbda = lmbda
        self.lmax = lmax
        self.rankmax = rankmax
        self.ndensity = ndensity
        self.core_pre = core_pre
        self.core_lmbda = core_lmbda
        self.core_cut = core_cut
        self.core_dcut = core_dcut
        self.nelements = nelements
        self.ranks_sizes = []
        self.tmp_coefs = None
        self.index = None
        self.deltaSplineBins = 0.001
        self.embedingtype = "FinnisSinclairShiftedScaled"
        # radbase
        self.radbasetype = "ChebPow"  # "ChebExpCos"
        self.compute_smoothness = compute_smoothness
        self.compute_orthogonality = compute_orthogonality
        if fs_parameters is not None:
            self.fs_parameters = fs_parameters
        else:
            self.fs_parameters = [1.0, 1.0, 1.0, 0.5][: 2 * self.ndensity]

        self.BbasisFuncs = None
        # Basis configuration
        if isinstance(potconfig, str):
            raise NotImplementedError(
                "Initialization from anything but BBasisConf"
                " is not implemented for multicomponent ACE"
            )
        else:
            self._init_basis_configuration_from_bbasisconf(potconfig)

        self.ncoef = 0
        if self.tmp_coefs is not None:
            self.ncoef = len(self.tmp_coefs)

        self.reg_l1 = 0
        self.reg_l2 = 0
        self.dsor = 0

        self.fit_coefs = tf.Variable(
            self.tmp_coefs, dtype=tf.float64, name="adjustable_coefs"
        )

        if self.compute_smoothness or self.compute_orthogonality:
            self.aux = []
        else:
            self.aux = None

        self.factor4pi = tf.sqrt(4 * tf.constant(np.pi, dtype=tf.float64))

    def compute_regularization(self):
        basis_coefs = self.fit_coefs[self.total_num_crad :]
        self.reg_l1 = tf.reduce_sum(tf.abs(basis_coefs))
        self.reg_l2 = tf.reduce_sum(basis_coefs**2)

    def _init_basis_configuration_from_df(self, potconfile):
        if isinstance(potconfile, str):
            df = pd.read_pickle(potconfile)
        else:
            df = potconfile

        self.config = ConfigBasis(df, self.rankmax, self.lmax)

    def _init_basis_configuration_from_bbasisconf(self, bbasisfunconf):
        from pyace.basis import BBasisConfiguration, ACEBBasisSet

        assert isinstance(
            bbasisfunconf, BBasisConfiguration
        ), "provided configuration is not an instance of BBasisConfiguration"

        self.bbasisset = ACEBBasisSet(bbasisfunconf)
        self.nelements = self.bbasisset.nelements
        self.elem_names = self.bbasisset.elements_name
        self.elems_Z = [symbol_to_atomic_number[symbol] for symbol in self.elem_names]
        self.elem_idx = [i for i in range(self.nelements)]

        self.nradmax = self.bbasisset.nradmax
        self.lmax = self.bbasisset.lmax
        self.nradbase = self.bbasisset.nradbase
        self.bond_specs = self.bbasisset.map_bond_specifications
        self.embed_spec = self.bbasisset.map_embedding_specifications

        self.bond_combs = list(self.bond_specs.keys())

        self.ndensity = self.bbasisset.ndensitymax
        self.rcut = self.bbasisset.cutoffmax

        self.tmp_coefs = self.bbasisset.all_coeffs
        self.tmp_coefs[self.tmp_coefs == 0] += 1e-32

        self.total_num_crad = 0
        bond_mus_unique = [
            k for k in combinations_with_replacement(range(self.nelements), 2)
        ]
        unique_bond_to_slice = {}
        for mus in bond_mus_unique:
            if mus not in self.bond_specs:
                continue
            bond = self.bond_specs[mus]
            n_crad = np.prod(np.shape(bond.radcoefficients))
            unique_bond_to_slice[mus] = [
                self.total_num_crad,
                self.total_num_crad + n_crad,
            ]
            self.total_num_crad += n_crad
        self.bond_to_slice = {
            k: unique_bond_to_slice[tuple(sorted(k))] for k in self.bond_specs.keys()
        }

        self.coefs_part = {}
        self.coefs_r1_part = {c: [] for c in self.bond_combs}
        self.coefs_rk_part = {c: [] for c in self.bond_combs}
        count = 0
        self.ranksmax = []
        for ne in range(self.nelements):
            self.coefs_part[ne] = []
            basis = self.bbasisset.basis_rank1[ne] + self.bbasisset.basis[ne]
            rank = 0
            for f in basis:
                r = f.rank
                for _ in f.coeffs:
                    if r == 1:
                        self.coefs_r1_part[tuple([f.mu0, f.mus[0]])] += [count]
                        self.coefs_rk_part[tuple([f.mu0, f.mus[0]])] += [
                            count + self.total_num_crad
                        ]
                        count += 1
                    else:
                        self.coefs_part[ne] += [count]
                        self.coefs_rk_part[tuple([f.mu0, f.mus[0]])] += [
                            count + self.total_num_crad
                        ]
                        count += 1
                rank = max([rank, r])
            self.coefs_part[ne] = tf.convert_to_tensor(
                self.coefs_part[ne], dtype=tf.int32, name="index_coefs_{}".format(ne)
            )
            self.ranksmax += [rank]
        self.coefs_r1_part = {
            c: tf.convert_to_tensor(
                self.coefs_r1_part[c],
                dtype=tf.int32,
                name="index_coefs_r1_{}".format(c),
            )
            for c in self.bond_combs
        }
        self.rankmax = max(self.ranksmax)

        bbasisfuncspecs = [
            [BBasisFunc(f) for f in self.bbasisset.basis[ne]]
            for ne in range(self.nelements)
        ]
        self.config = ConfigBasis(bbasisfuncspecs, self.ranksmax, self.nelements)

    def set_coefs(self, coefs):
        self.fit_coefs.assign(tf.Variable(coefs, dtype=tf.float64))

    def get_coefs(self):
        return self.fit_coefs

    def get_updated_config(self, updating_coefs=None, prefix=None):
        if updating_coefs is not None:
            self.set_coefs(updating_coefs)
            self.bbasisset.all_coeffs = self.fit_coefs.numpy()

            return self.bbasisset.to_BBasisConfiguration()
        else:
            if self.fit_coefs is not None and self.bbasisset is not None:
                self.bbasisset.all_coeffs = self.fit_coefs.numpy()

                return self.bbasisset.to_BBasisConfiguration()
            else:
                ValueError(
                    "Can't update configuration, no coefficients or nothing to update."
                )

    def cheb_exp_cos(self, d_ij, nfuncs, cutoff, lmbda):
        d_ij_ch_domain = 1.0 - 2.0 * (
            (tf.exp(-lmbda * (d_ij / cutoff - 1.0)) - 1.0) / (tf.exp(lmbda) - 1.0)
        )
        chebpol = radial_functions.chebvander(d_ij_ch_domain, nfuncs)
        chebpol = chebpol[:, 0, :]
        cos_cut = radial_functions.cutoff_func_cos(d_ij, cutoff)
        gk = tf.where(
            tf.math.equal(chebpol, 1.0),
            chebpol * cos_cut,
            0.5 * (1 - chebpol) * cos_cut,
            name="where_chebexp",
        )

        # y00 = 1  # / tf.sqrt(4 * tf.constant(np.pi, dtype=var_dtype))
        if nfuncs == 1:
            return gk[:, :-1]
        else:
            return gk  # * y00

    def cheb_pow(self, d_ij, nfuncs, cutoff, lmbda):
        d_ij_ch_domain = 2.0 * (1.0 - tf.abs(1.0 - d_ij / cutoff) ** lmbda) - 1.0

        chebpol = radial_functions.chebvander(d_ij_ch_domain, nfuncs + 1)
        chebpol = chebpol[:, 0, :]
        chebpol = 0.5 - 0.5 * chebpol[:, 1:]
        res = tf.where(
            tf.less(d_ij, cutoff),
            chebpol,
            tf.zeros_like(chebpol, dtype=tf.float64),
            name="where_chebpow",
        )

        return res

    def bessel_s(self, d_ij, nfuncs, cutoff):
        y = radial_functions.simplified_bessel(d_ij, cutoff, nfuncs)
        res = tf.where(
            tf.less(d_ij, cutoff),
            y,
            tf.zeros_like(y, dtype=tf.float64),
            name="where_bessels",
        )

        return res

    def legendre(self, d_ij, nfuncs, cutoff, lmbda):
        d_ij_l_domain = 2.0 * (1.0 - tf.abs(1.0 - d_ij / cutoff) ** lmbda) - 1.0

        legpol = radial_functions.legendre(d_ij_l_domain, nfuncs + 1)
        legpol = (legpol[:, 1:] - 1) / 2
        res = tf.where(
            tf.less(d_ij, cutoff),
            legpol,
            tf.zeros_like(legpol, dtype=tf.float64),
            name="where_legpol",
        )

        return res

    def radial_function(self, d_ij, nfunc=None, ftype=None, cutoff=None, lmbda=None):
        if cutoff is None:
            cutoff = self.rcut

        if ftype is None:
            ftype = self.radbasetype

        if nfunc is None:
            nfunc = self.nradbase

        if lmbda is None:
            lmbda = self.lmbda

        if ftype == "ChebExpCos":
            return self.cheb_exp_cos(d_ij, nfunc, cutoff, lmbda)
        elif ftype == "ChebPow":
            return self.cheb_pow(d_ij, nfunc, cutoff, lmbda)
        elif ftype == "TEST_LegendrePow":
            return self.legendre(d_ij, nfunc, cutoff, lmbda)
        elif ftype == "TEST_SBessel":
            import warnings

            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                'Name "TEST_SBessel" is deprecated, use "SBessel" instead.',
                DeprecationWarning,
            )
            return self.bessel_s(d_ij, nfunc, cutoff)
        elif ftype == "SBessel":
            return self.bessel_s(d_ij, nfunc, cutoff)
        else:
            raise ValueError("Unknown radial function type {}".format(ftype))

    @staticmethod
    def complexmul(r1, im1, r2, im2):
        real_part = r1 * r2 - im1 * im2
        imag_part = im2 * r1 + im1 * r2

        return real_part, imag_part

    def flat_gather_nd(self, params, indices):
        idx_shape = tf.shape(indices, name="shape_idx_shape")
        params_shape = tf.shape(params, name="shape_params_shape")
        idx_dims = idx_shape[-1]
        gather_shape = params_shape[idx_dims:]
        params_flat = tf.reshape(params, tf.concat([[-1], gather_shape], axis=0))
        axis_step = tf.math.cumprod(
            params_shape[:idx_dims], exclusive=True, reverse=True
        )
        indices_flat = tf.reduce_sum(indices * axis_step, axis=-1)
        result_flat = tf.gather(params_flat, indices_flat, name="flat_gather_nd")

        return tf.reshape(
            result_flat, tf.concat([idx_shape[:-1], gather_shape], axis=0)
        )

    def compute_core_repulsion(
        self, d_ij, core_lmbda, core_pre, core_cut, core_dcut, sum_ind
    ):
        phi_core = core_pre * tf.math.exp(-core_lmbda * d_ij**2) / d_ij
        phi_core = tf.math.segment_sum(
            phi_core * (radial_functions.cutoff_func_poly(d_ij, core_cut, core_dcut)),
            self.index.bond_sum_ind[sum_ind],
        )

        return phi_core

    def f_cut_core_rep(self, rho_core, rho_core_cut, drho_core_cut):
        condition1 = tf.less_equal(rho_core_cut, rho_core)
        condition2 = tf.less(rho_core, rho_core_cut - drho_core_cut)

        res = tf.where(
            condition1,
            tf.zeros_like(rho_core, dtype=tf.float64),
            tf.where(
                condition2,
                tf.ones_like(rho_core, dtype=tf.float64),
                (
                    radial_functions.cutoff_func_poly(
                        rho_core, rho_core_cut, drho_core_cut
                    )
                ),
            ),
        )

        return res

    def compute_bbasis_funcs(self, a_munlm_r, a_munlm_i, cntrl_at, rank):
        a_r = tf.transpose(a_munlm_r[cntrl_at], [1, 2, 3, 0])
        a_i = tf.transpose(a_munlm_i[cntrl_at], [1, 2, 3, 0])

        a_1_r = self.flat_gather_nd(
            a_r, self.config.central_atom[cntrl_at][rank - 2].munlm[0]
        )
        a_2_r = self.flat_gather_nd(
            a_r, self.config.central_atom[cntrl_at][rank - 2].munlm[1]
        )
        a_1_i = self.flat_gather_nd(
            a_i, self.config.central_atom[cntrl_at][rank - 2].munlm[0]
        )
        a_2_i = self.flat_gather_nd(
            a_i, self.config.central_atom[cntrl_at][rank - 2].munlm[1]
        )

        prod_r, prod_i = self.complexmul(a_1_r, a_1_i, a_2_r, a_2_i)

        if rank > 2:
            for k in range(2, rank):
                a_k_r = self.flat_gather_nd(
                    a_r, self.config.central_atom[cntrl_at][rank - 2].munlm[k]
                )
                a_k_i = self.flat_gather_nd(
                    a_i, self.config.central_atom[cntrl_at][rank - 2].munlm[k]
                )

                prod_r, prod_i = self.complexmul(prod_r, prod_i, a_k_r, a_k_i)

        b_base = prod_r * tf.convert_to_tensor(
            self.config.central_atom[cntrl_at][rank - 2].genCG, dtype=tf.float64
        )
        b_base = tf.transpose(
            tf.math.segment_sum(
                b_base, self.config.central_atom[cntrl_at][rank - 2].msum
            ),
            [1, 0],
        )

        return b_base

    def integrate(self, func, x, dx, rcut):
        f = tf.reshape(tf.reduce_sum(tf.abs(func), axis=[1, 2]), [-1, 1])
        trapz = tf.reduce_sum(x**2 * f) * dx
        trapz /= rcut**2

        return tf.reshape(trapz, [-1, 1])

    def compute_smoothness_reg(self):
        start = tf.constant(1e-10, dtype=tf.float64)
        stop = tf.constant(self.rcut, dtype=tf.float64)
        d_cont = tf.reshape(tf.linspace(start, stop - 1e-5, 100), [-1, 1])
        delta_d = d_cont[1] - d_cont[0]
        radial_coefs = self.fit_coefs[: self.total_num_crad]
        r_nl = []
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(d_cont)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(d_cont)
                rcuts = []
                for c_i, c in enumerate(self.index.bond_combs):
                    bond_spec = self.bond_specs[c]
                    nradmaxi = bond_spec.nradmax
                    lmaxi = bond_spec.lmax
                    nradbasei = bond_spec.nradbasemax
                    crad = radial_coefs[slice(*self.bond_to_slice[c])]
                    crad = tf.reshape(crad, [nradmaxi, lmaxi + 1, nradbasei])
                    rcuts += [tf.constant(bond_spec.rcut, dtype=tf.float64)]
                    g_cont = self.radial_function(
                        d_cont,
                        nfunc=nradbasei,
                        ftype=bond_spec.radbasename,
                        cutoff=rcuts[c_i],
                        lmbda=tf.constant(bond_spec.radparameters[0], dtype=tf.float64),
                    )  # [None, nradbase]
                    r_nl += [
                        tf.einsum("jk,nlk->jnl", g_cont, crad)
                    ]  # [None, nradmax, lmax+1]
                self.aux += [
                    tf.reshape(
                        tf.reduce_mean(
                            [
                                self.integrate(r_nl_i, d_cont, delta_d, rcuts[i])
                                for i, r_nl_i in enumerate(r_nl)
                            ]
                        ),
                        [-1, 1],
                    )
                ]
            drnl_dr = [
                tf.squeeze(
                    tape2.batch_jacobian(r_nl_i, d_cont, experimental_use_pfor=True),
                    axis=-1,
                )
                for r_nl_i in r_nl
            ]
            self.aux += [
                tf.reshape(
                    tf.reduce_mean(
                        [
                            self.integrate(drnl_dr_i, d_cont, delta_d, rcuts[i])
                            for i, drnl_dr_i in enumerate(drnl_dr)
                        ]
                    ),
                    [-1, 1],
                )
            ]
        d2rnl_dr2 = [
            tf.squeeze(
                tape1.batch_jacobian(drnl_dr_i, d_cont, experimental_use_pfor=True),
                axis=-1,
            )
            for drnl_dr_i in drnl_dr
        ]
        self.aux += [
            tf.reshape(
                tf.reduce_mean(
                    [
                        self.integrate(d2rnl_dr2_i, d_cont, delta_d, rcuts[i])
                        for i, d2rnl_dr2_i in enumerate(d2rnl_dr2)
                    ]
                ),
                [-1, 1],
            )
        ]

    def eval_atomic_energy(self, r_ij):
        d_ij = tf.reshape(tf.linalg.norm(r_ij, axis=1), [-1, 1])
        rhat = r_ij / d_ij  # [None, 3]/[None, 1] -> [None, 3]

        sh = spherical_harmonics.SphericalHarmonics(self.lmax, prec="DOUBLE")
        ylm_r, ylm_i = sh.compute_ylm(rhat)
        ynlm_r = (
            tf.expand_dims(ylm_r, 1) * self.factor4pi
        )  # [None, 1, (lmax+1) * (lmax+1)]
        ynlm_i = (
            tf.expand_dims(ylm_i, 1) * self.factor4pi
        )  # [None, 1, (lmax+1) * (lmax+1)]

        part_d_ij = tf.dynamic_partition(
            d_ij, self.index.bond_partition, self.index.ncombs
        )
        part_ylm_r = tf.dynamic_partition(
            ynlm_r, self.index.bond_partition, self.index.ncombs
        )
        part_ylm_i = tf.dynamic_partition(
            ynlm_i, self.index.bond_partition, self.index.ncombs
        )

        radial_coefs = self.fit_coefs[: self.total_num_crad]
        basis_coefs = self.fit_coefs[self.total_num_crad :]
        a_munlm_r = tf.zeros(
            [self.index.nat, self.nelements, self.nradmax, (self.lmax + 1) ** 2],
            dtype=tf.float64,
        )
        a_munlm_i = tf.zeros(
            [self.index.nat, self.nelements, self.nradmax, (self.lmax + 1) ** 2],
            dtype=tf.float64,
        )
        rho_r1, phi_core, f_cut_core = [], [], []
        dsor = tf.constant(0, dtype=tf.float64)
        for c_i, c in enumerate(self.index.bond_combs):
            ndensity = self.embed_spec[c[0]].ndensity
            bond_spec = self.bond_specs[c]
            nradmaxi = bond_spec.nradmax
            lmaxi = bond_spec.lmax
            nradbasei = bond_spec.nradbasemax
            crad = radial_coefs[slice(*self.bond_to_slice[c])]
            crad = tf.reshape(crad, [nradmaxi, lmaxi + 1, nradbasei])

            imat = tf.eye(nradmaxi, batch_shape=[lmaxi + 1], dtype=tf.float64)
            wwt = tf.matmul(
                tf.transpose(crad, [1, 0, 2]), tf.transpose(crad, [1, 2, 0])
            )
            if self.rankmax > 1:
                dsor += tf.reduce_mean((wwt - imat) ** 2)

            imat = tf.eye(nradbasei, batch_shape=[lmaxi + 1], dtype=tf.float64)
            wtw = tf.matmul(
                tf.transpose(crad, [1, 2, 0]), tf.transpose(crad, [1, 0, 2])
            )
            if self.rankmax > 1:
                dsor += tf.reduce_mean((wtw - imat) ** 2)

            phi_ij = self.compute_core_repulsion(
                part_d_ij[c_i],
                core_lmbda=tf.constant(bond_spec.lambdahc, dtype=tf.float64),
                core_pre=tf.constant(bond_spec.prehc, dtype=tf.float64),
                core_cut=tf.constant(bond_spec.rcut_in, dtype=tf.float64),
                core_dcut=tf.constant(bond_spec.dcut_in, dtype=tf.float64),
                sum_ind=c_i,
            )
            f_cut_core_ij = self.f_cut_core_rep(
                phi_ij,
                rho_core_cut=tf.constant(
                    self.embed_spec[c[0]].rho_core_cutoff, dtype=tf.float64
                ),
                drho_core_cut=tf.constant(
                    self.embed_spec[c[0]].drho_core_cutoff, dtype=tf.float64
                ),
            )
            phi_core += [phi_ij]
            f_cut_core += [f_cut_core_ij]

            comb_gjk = self.radial_function(
                part_d_ij[c_i],
                nfunc=nradbasei,
                ftype=bond_spec.radbasename,
                cutoff=tf.constant(bond_spec.rcut, dtype=tf.float64),
                lmbda=tf.constant(bond_spec.radparameters[0], dtype=tf.float64),
            )  # [None, nradbase]
            comb_gjk = comb_gjk * (
                1
                - radial_functions.cutoff_func_poly(
                    part_d_ij[c_i],
                    tf.constant(bond_spec.rcut_in, dtype=tf.float64),
                    tf.constant(bond_spec.dcut_in, dtype=tf.float64),
                )
            )
            gk_i = tf.math.segment_sum(comb_gjk, self.index.bond_sum_ind[c_i])

            coefs_r1 = tf.reshape(
                tf.gather(basis_coefs, self.coefs_r1_part[c]), [-1, ndensity]
            )
            e = tf.matmul(gk_i, coefs_r1)
            rho_r1 += [tf.pad(e, [[0, 0], [0, self.ndensity - ndensity]])]

            rj_nl = tf.einsum("jk,nlk->jnl", comb_gjk, crad)
            rj_nl = tf.pad(
                rj_nl, [[0, 0], [0, self.nradmax - nradmaxi], [0, self.lmax - lmaxi]]
            )
            rj_nlm = tf.gather(rj_nl, sh.l_tile, axis=2)
            aj_nlm_r = rj_nlm * part_ylm_r[c_i]
            aj_nlm_i = rj_nlm * part_ylm_i[c_i]

            a_nlm_r = tf.math.segment_sum(aj_nlm_r, self.index.bond_sum_ind[c_i])
            a_nlm_i = tf.math.segment_sum(aj_nlm_i, self.index.bond_sum_ind[c_i])

            update_index = tf.zeros_like(
                self.index.atomic_stitching[c_i]
            ) + tf.constant(c[1], dtype=tf.int32)
            update_index = tf.concat(
                [self.index.atomic_stitching[c_i], update_index], axis=1
            )
            a_munlm_r = tf.tensor_scatter_nd_update(a_munlm_r, update_index, a_nlm_r)
            a_munlm_i = tf.tensor_scatter_nd_update(a_munlm_i, update_index, a_nlm_i)

        if self.compute_orthogonality:
            self.aux += [tf.reshape(dsor, [-1, 1])]

        rho_r1 = tf.math.unsorted_segment_sum(
            tf.concat(rho_r1, 0),
            tf.reshape(tf.concat(self.index.atomic_stitching, 0), [-1]),
            num_segments=self.index.nat,
        )

        phi_core = tf.math.unsorted_segment_sum(
            tf.concat(phi_core, 0),
            tf.reshape(tf.concat(self.index.atomic_stitching, 0), [-1]),
            num_segments=self.index.nat,
        )
        inner_cut = tf.math.unsorted_segment_mean(
            tf.concat(f_cut_core, 0),
            tf.reshape(tf.concat(self.index.atomic_stitching, 0), [-1]),
            num_segments=self.index.nat,
        )

        rho_r1_part = tf.dynamic_partition(
            rho_r1, self.index.atomic_mu_i, self.nelements
        )
        if self.rankmax > 1:
            a_munlm_r = tf.dynamic_partition(
                a_munlm_r, self.index.atomic_mu_i, self.nelements
            )
            a_munlm_i = tf.dynamic_partition(
                a_munlm_i, self.index.atomic_mu_i, self.nelements
            )
            collect_basis = [[] for _ in range(self.nelements)]
            for i, cent_at in enumerate(collect_basis):
                for k in range(2, self.ranksmax[i] + 1):
                    cent_at += [self.compute_bbasis_funcs(a_munlm_r, a_munlm_i, i, k)]
            basis = [tf.concat(cent_at, axis=1) for cent_at in collect_basis]

            at_nrgs = []
            for c_i in range(self.nelements):
                ndensity = self.embed_spec[c_i].ndensity
                fs_parameters = self.embed_spec[c_i].FS_parameters
                coefs = tf.reshape(
                    tf.gather(basis_coefs, self.coefs_part[c_i]), [-1, ndensity]
                )
                rho = tf.matmul(basis[c_i], coefs) + rho_r1_part[c_i]
                safe_rho = tf.where(tf.not_equal(rho, 0.0), rho, rho + 1e-32)
                en_sum = 0
                for dens in range(ndensity):
                    en_sum += tf.constant(
                        fs_parameters[2 * dens], dtype=tf.float64
                    ) * self.embedding_function(
                        safe_rho[:, dens],
                        tf.constant(fs_parameters[2 * dens + 1], dtype=tf.float64),
                        ftype=self.embed_spec[c_i].npoti,
                    )
                at_nrgs += [en_sum]
        else:
            at_nrgs = []
            for c_i in range(self.nelements):
                ndensity = self.embed_spec[c_i].ndensity
                fs_parameters = self.embed_spec[c_i].FS_parameters
                rho = rho_r1_part[c_i]
                safe_rho = tf.where(tf.not_equal(rho, 0.0), rho, rho + 1e-32)
                en_sum = 0
                for dens in range(ndensity):
                    en_sum += tf.constant(
                        fs_parameters[2 * dens], dtype=tf.float64
                    ) * self.embedding_function(
                        safe_rho[:, dens],
                        tf.constant(fs_parameters[2 * dens + 1], dtype=tf.float64),
                        ftype=self.embed_spec[c_i].npoti,
                    )
                at_nrgs += [en_sum]

        at_nrgs = tf.dynamic_stitch(
            self.index.stitch_atomic_mu_i, at_nrgs, "StitchAtomicEnergy"
        )
        e_atom = tf.math.add(
            tf.reshape(at_nrgs, [-1, 1]) * inner_cut, phi_core, "atomic_energies"
        )

        return tf.reshape(e_atom, [-1, 1])

    def compute_atomic_energy(self, r_ij, ind_i, mu_i, mu_j, ind_j):
        lmu_i = []
        lmu_j = []
        for i in range(self.nelements):
            lmu_i += [
                tf.where(
                    tf.equal(mu_i, self.elems_Z[i]),
                    tf.zeros_like(mu_i, dtype=tf.int32) + self.elem_idx[i],
                    tf.zeros_like(mu_i, dtype=tf.int32),
                )
            ]
            lmu_j += [
                tf.where(
                    tf.equal(mu_j, self.elems_Z[i]),
                    tf.zeros_like(mu_j, dtype=tf.int32) + self.elem_idx[i],
                    tf.zeros_like(mu_j, dtype=tf.int32),
                )
            ]
        mu_i = tf.reduce_sum(tf.stack(lmu_i), axis=0)
        mu_j = tf.reduce_sum(tf.stack(lmu_j), axis=0)

        self.index = IndexPartitioner(
            self.nelements, self.bond_combs, ind_i, mu_i, mu_j, ind_j
        )

        e_atom = self.eval_atomic_energy(r_ij)

        if self.compute_smoothness:
            self.compute_smoothness_reg()

        return e_atom

    def embedding_function(self, rho, mexp, ftype="FinnisSinclairShiftedScaled"):
        if ftype == "FinnisSinclairShiftedScaled":
            return self.f_exp_shsc(rho, mexp)
        elif ftype == "FinnisSinclair":
            return self.f_exp_old(rho, mexp)

    def f_exp_old(self, rho, mexp):
        return tf.where(
            tf.less(tf.abs(rho), tf.constant(1e-10, dtype=tf.float64)),
            mexp * rho,
            self.en_func_old(rho, mexp),
        )

    def en_func_old(self, rho, mexp):
        w = tf.constant(10.0, dtype=tf.float64)
        y1 = w * rho**2
        g = tf.where(
            tf.less(tf.constant(30.0, dtype=tf.float64), y1),
            0.0 * rho,
            tf.exp(tf.negative(y1)),
        )

        omg = 1.0 - g
        a = tf.abs(rho)
        y3 = tf.pow(omg * a + 1e-20, mexp)
        y2 = mexp * g * a
        f = tf.sign(rho) * (y3 + y2)
        return f

    def f_exp_shsc(self, rho, mexp):
        eps = tf.constant(1e-10, dtype=tf.float64)
        cond = tf.abs(
            tf.ones_like(rho, dtype=tf.float64) * mexp
            - tf.constant(1.0, dtype=tf.float64)
        )
        mask = tf.where(
            tf.less(cond, eps),
            tf.ones_like(rho, dtype=tf.bool),
            tf.zeros_like(rho, dtype=tf.bool),
        )

        arho = tf.abs(rho)
        exprho = tf.exp(-arho)
        nx = 1.0 / mexp
        xoff = tf.pow(nx, (nx / (1.0 - nx))) * exprho
        yoff = tf.pow(nx, (1 / (1.0 - nx))) * exprho
        func = tf.where(mask, rho, tf.sign(rho) * (tf.pow(xoff + arho, mexp) - yoff))

        return func

    def selective_fitting(self, list_of_bonds, basis_factor=1.0, rad_coefs_factor=1.0):
        assert len(list_of_bonds) <= len(self.bond_combs), ValueError(
            "Number of requested bond types ({}) exceeds the number specified in the potential ({})".format(
                len(list_of_bonds), len(self.bond_combs)
            )
        )
        for bond in list_of_bonds:
            assert bond in self.bond_combs, ValueError(
                "Bond type {} is not in the potential's list of bond combinations".format(
                    bond
                )
            )
        assert basis_factor in (0.0, 1.0), ValueError(
            "Only 0 and 1 are allowed values for a factor"
        )
        assert rad_coefs_factor in (0.0, 1.0), ValueError(
            "Only 0 and 1 are allowed values for a factor"
        )

        factor_list = np.zeros_like(self.tmp_coefs).astype(np.float64)
        for bond in list_of_bonds:
            factor_list[self.coefs_rk_part[bond]] = basis_factor
            factor_list[slice(*self.bond_to_slice[bond])] = rad_coefs_factor

        return factor_list


class IndexPartitioner(object):
    """Taking care of the index permutations"""

    def __init__(self, nelem, bond_combs, ind_i, mu_i, mu_j, ind_j):
        self.nelem = nelem
        self.ind_i = ind_i
        self.ind_j = ind_j
        self.mu_i = mu_i
        self.mu_j = mu_j
        self.bond_combs = bond_combs
        self.ncombs = len(self.bond_combs)
        self.nat = tf.reduce_max(self.ind_i, name="compute_nat") + 1

        self.atomic_mu_i = tf.cast(
            tf.math.unsorted_segment_mean(
                self.mu_i,
                self.ind_i,
                num_segments=tf.reduce_max(self.ind_i) + 1,
                name="mu_i_to_atomic_mu_i",
            ),
            tf.int32,
        )
        self.stitch_atomic_mu_i = tf.dynamic_partition(
            tf.range(self.nat), self.atomic_mu_i, self.nelem, name="stitch_atomic_mu_i"
        )
        self.bond_partition, self.bond_stitching = self.compute_bond_partitioning()
        self.atomic_stitching, self.bond_sum_ind = self.compute_atomic_partitioning()
        self.bond_mu_i = tf.dynamic_partition(
            self.mu_i, self.bond_partition, self.ncombs, name="bond_mu_i"
        )
        self.bond_mu_j = tf.dynamic_partition(
            self.mu_j, self.bond_partition, self.ncombs, name="bond_mu_j"
        )

    def compute_bond_partitioning(self):
        part = tf.zeros_like(self.mu_i, name="bond_partition")
        for i, comb in enumerate(self.bond_combs):
            cond_el1 = tf.equal(self.mu_i, tf.constant(comb[0], dtype=tf.int32))
            cond_el2 = tf.equal(self.mu_j, tf.constant(comb[1], dtype=tf.int32))
            cond_el12 = tf.math.logical_and(cond_el1, cond_el2)
            part = tf.where(cond_el12, part + i, part)
        stitch = tf.dynamic_partition(
            tf.range(tf.shape(self.mu_i)[0]), part, self.ncombs, name="bond_stitch"
        )

        return part, stitch

    def compute_atomic_partitioning(self):
        bond_ind_i = tf.dynamic_partition(
            self.ind_i, self.bond_partition, self.ncombs, name="atomic_partition"
        )
        atomic_stitching = []
        bond_sum_ind = []
        for b_i in range(self.ncombs):
            u, i = tf.unique(bond_ind_i[b_i])
            atomic_stitching += [
                tf.reshape(u, [-1, 1], name="atomic_partition_{}".format(b_i))
            ]
            bond_sum_ind += [i]

        return atomic_stitching, bond_sum_ind


class BBasisFunc:
    def __init__(self, bbasisfunc, lmax=None):
        self.rank = bbasisfunc.rank
        self.ns = bbasisfunc.ns
        self.ls = bbasisfunc.ls
        self.mu0 = bbasisfunc.mu0
        self.mus = bbasisfunc.mus
        self.genCG = np.reshape(bbasisfunc.gen_cgs, [-1, 1])
        self.ms = bbasisfunc.ms_combs
        self.coefs = bbasisfunc.coeffs

        self.munlm = self.get_munlm()
        self.msum = np.zeros(len(self.ms)).astype(np.int32)

        if lmax is not None:
            self.adjust_m_index(lmax)

    def get_munlm(self):
        munlm = [np.zeros((len(self.ms), 4)).astype(np.int32) for _ in range(self.rank)]
        for c in range(len(self.ms)):
            for r in range(self.rank):
                munlm[r][c] = np.array(
                    [self.mus[r], self.ns[r] - 1, self.ls[r], self.ms[c][r]]
                )

        return munlm

    def adjust_m_index(self, lmax):
        for r in range(self.rank):
            self.munlm[r][:, -1] += lmax


class BBasisFuncSet:
    def __init__(
        self,
        list_of_bbasisfunc,
        rank,
    ):
        self.rank = rank
        self.munlm = self.get_munlm(list_of_bbasisfunc)
        self.msum = self.get_msum(list_of_bbasisfunc)
        self.genCG = self.get_gen_cg(list_of_bbasisfunc)
        self.coefs = self.get_coefs(list_of_bbasisfunc)

        for r in range(self.rank):
            self.munlm[r][:, -2] = merge_lm(self.munlm[r][:, -2], self.munlm[r][:, -1])
            self.munlm[r] = self.munlm[r][:, :-1]

    def get_munlm(self, list_of_bbasisfunc):
        total_munlm = []
        for i in range(self.rank):
            total_munlm.append(
                np.vstack([bbasisfunc.munlm[i] for bbasisfunc in list_of_bbasisfunc])
            )

        return total_munlm

    def get_gen_cg(self, list_of_bbasisfunc):
        return np.vstack([bbasisfunc.genCG for bbasisfunc in list_of_bbasisfunc])

    def get_msum(self, list_of_bbasisfunc):
        return np.hstack(
            [bbasisfunc.msum + i for i, bbasisfunc in enumerate(list_of_bbasisfunc)]
        )

    def get_coefs(self, list_of_bbasisfunc):
        return np.vstack([bbasisfunc.coefs for bbasisfunc in list_of_bbasisfunc])

    def rearrange_msum(self, msum):
        new_msum = []
        j = 0
        for i in range(len(msum)):
            if i == 0:
                new_msum += [j]
            else:
                if msum[i] == msum[i - 1]:
                    new_msum += [j]
                else:
                    j += 1
                    new_msum += [j]

        return np.array(new_msum).astype(np.int32)

    def apply_constrains(self, nmax=2, lmax=0):
        mask = np.ones_like(self.munlm[0][:, 0])
        for i in range(self.rank):
            cond = (self.munlm[i][:, 0] < nmax) & (self.munlm[i][:, 1] <= lmax)
            mask = np.logical_and(mask, cond)

        for i in range(self.rank):
            self.munlm[i] = self.munlm[i][mask]

        self.msum = self.rearrange_msum(self.msum[mask] - np.min(self.msum[mask]))
        self.genCG = self.genCG[mask]


class ConfigBasis:
    def __init__(self, bbasisfunc, ranksmax, nelem):
        self.central_atom = []

        if isinstance(bbasisfunc, pd.DataFrame):
            self.set_basis_from_df(bbasisfunc, ranksmax, nelem)
        elif isinstance(bbasisfunc, list):
            self.set_basis_from_list(bbasisfunc, ranksmax, nelem)

    def set_basis_from_df(self, bbasisfunc, rankmax, nelem):
        for r in range(rankmax):
            rank_data = bbasisfunc.loc[bbasisfunc["rank"] == r + 1]
            rank_data = rank_data["func"]
            # basisfuncs = rank_data.apply(BBasisFunc, lmax=lmax)
            basisfuncs = rank_data.tolist()
            basisset = BBasisFuncSet(basisfuncs, rank=r + 1)
            self.central_atom.append(basisset)

    def set_basis_from_list_(self, bbasisfunc, ranksmax, nelem):
        for r in range(1, rankmax):
            total_rank = [[] for _ in range(nelem)]
            for elem in range(nelem):
                funcs_of_rank = [f for f in bbasisfunc[elem] if f.rank == r + 1]
                total_rank[elem] = BBasisFuncSet(funcs_of_rank, rank=r + 1)
            self.central_atom.append(total_rank)

    def set_basis_from_list(self, bbasisfunc, ranksmax, nelem):
        for elem in range(nelem):
            # total_rank = [[] for _ in range(2, ranksmax[elem] + 1)]
            total_rank = []
            for r in range(
                2, ranksmax[elem] + 1
            ):  # We only collect starting from rank2
                funcs_of_rank = [f for f in bbasisfunc[elem] if f.rank == r]
                total_rank += [BBasisFuncSet(funcs_of_rank, rank=r)]
            self.central_atom.append(total_rank)


class ConfigFromBBasisConf:
    def __init__(self, df, rankmax, lmax):
        self.ranks = []
        for r in range(rankmax):
            rank_data = df.loc[df["rank"] == r + 1]
            rank_data = rank_data["func"]
            # basisfuncs = rank_data.apply(BBasisFunc, lmax=lmax)
            basisfuncs = rank_data.tolist()
            basisset = BBasisFuncSet(basisfuncs, rank=r + 1, lmax=lmax)
            self.ranks.append(basisset)


def merge_lm(l, m):
    return l * (l + 1) + m
