import tensorflow as tf
from tensorpotential.compat.pace.utils.utilities import input2evaloss, input2eval
from tensorpotential.compat.pace.constants import *


class TensorPotential(tf.Module):

    def __init__(
        self,
        potential,
        loss_specs=None,
        compile_vector_loss=False,
        jit_op_compile=False,
    ):
        super(TensorPotential, self).__init__()
        self.potential = potential

        if loss_specs is None:
            self.loss_specs = self._set_default_loss_specs()
        else:
            self.loss_specs = self._set_loss_specs(loss_specs)

        self.compile_vector_loss = compile_vector_loss

        self.pos = None
        self.fit_coefs = None

        self.reg_components = []

        self.list_of_int_placeholders = [
            "ind_i",
            "ind_j",
            "mu_i",
            "mu_j",
            "pair_map",
            "cell_map",
            "e_map",
        ]
        self.list_of_float_placeholders = [
            "positions",
            "offsets",
            "cell",
            "energy",
            "forces",
            "eweights",
            "fweights",
        ]

        if isinstance(jit_op_compile, list):
            self.compute_loss_compile = jit_op_compile[0]
            self.op_compile = jit_op_compile[1]
        elif isinstance(jit_op_compile, bool):
            self.compute_loss_compile = self.op_compile = jit_op_compile
        else:
            raise ValueError("jit_op_compile input is not understood")

        if self.compile_vector_loss:
            self.decorate_vector_loss()

        self.decorate_evaluate_loss()

    def _set_default_loss_specs(self):
        spec_dict = {
            LOSS_TYPE: "per-atom",
            LOSS_FORCE_FACTOR: tf.constant(0.0, tf.float64),
            LOSS_ENERGY_FACTOR: tf.constant(1.0, tf.float64),
            L1_REG: tf.constant(0.0, tf.float64),
            L2_REG: tf.constant(0.0, tf.float64),
            AUX_LOSS_FACTOR: tf.constant([0], tf.float64),
        }

        return spec_dict

    def _set_loss_specs(self, specs):
        valid_loss_types = ["per-atom", "per-structure", "sqrt"]
        assert (
            specs[LOSS_TYPE] in valid_loss_types
        ), "{0} is not a supported loss type".format(specs[LOSS_TYPE])
        loss_type = specs[LOSS_TYPE]
        spec_dict = {
            LOSS_TYPE: loss_type,
            LOSS_FORCE_FACTOR: tf.constant(
                specs.get(LOSS_FORCE_FACTOR, 0.0), tf.float64
            ),
            LOSS_ENERGY_FACTOR: tf.constant(
                specs.get(LOSS_ENERGY_FACTOR, 1.0), tf.float64
            ),
            L1_REG: tf.constant(specs.get(L1_REG, 0.0), tf.float64),
            L2_REG: tf.constant(specs.get(L2_REG, 0.0), tf.float64),
            AUX_LOSS_FACTOR: tf.constant(specs.get(AUX_LOSS_FACTOR, [0.0]), tf.float64),
        }

        return spec_dict

    def init_placeholders(self, **kwargs):
        self.__dict__.update(
            (k, tf.convert_to_tensor(v, dtype=tf.float64))
            for k, v in kwargs.items()
            if k in self.list_of_float_placeholders
        )
        self.__dict__.update(
            (k, tf.convert_to_tensor(v, dtype=tf.int32))
            for k, v in kwargs.items()
            if k in self.list_of_int_placeholders
        )

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, 3], tf.float64, name="positions"),
            tf.TensorSpec([None, 3], tf.float64, name="offsets"),
            tf.TensorSpec([None, 3, 3], tf.float64, name="cell"),
            tf.TensorSpec([None], tf.int32, name="ind_i"),
            tf.TensorSpec([None], tf.int32, name="ind_j"),
            tf.TensorSpec([None], tf.int32, name="mu_i"),
            tf.TensorSpec([None], tf.int32, name="mu_j"),
            tf.TensorSpec([None], tf.int32, name="pair_map"),
            tf.TensorSpec([None], tf.int32, name="cell_map"),
            tf.TensorSpec([None], tf.int32, name="e_map"),
        ]
    )
    def _evaluate(
        self,
        positions,
        offsets,
        cell,
        ind_i,
        ind_j,
        mu_i,
        mu_j,
        pair_map,
        cell_map,
        e_map,
    ):

        self.init_placeholders(
            positions=positions,
            offsets=offsets,
            cell=cell,
            ind_i=ind_i,
            ind_j=ind_j,
            mu_i=mu_i,
            mu_j=mu_j,
            pair_map=pair_map,
            cell_map=cell_map,
            e_map=e_map,
        )
        e, f, s = self.compute_energy_forces_stress()

        return e, f, s

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, 3], tf.float64, name="positions"),
            tf.TensorSpec([None, 3], tf.float64, name="offsets"),
            tf.TensorSpec([None, 3, 3], tf.float64, name="cell"),
            tf.TensorSpec([None], tf.int32, name="ind_i"),
            tf.TensorSpec([None], tf.int32, name="ind_j"),
            tf.TensorSpec([None], tf.int32, name="mu_i"),
            tf.TensorSpec([None], tf.int32, name="mu_j"),
            tf.TensorSpec([None], tf.int32, name="pair_map"),
            tf.TensorSpec([None], tf.int32, name="cell_map"),
            tf.TensorSpec([None], tf.int32, name="e_map"),
        ]
    )
    def _evaluate_hess(
        self,
        positions,
        offsets,
        cell,
        ind_i,
        ind_j,
        mu_i,
        mu_j,
        pair_map,
        cell_map,
        e_map,
    ):

        self.init_placeholders(
            positions=positions,
            offsets=offsets,
            cell=cell,
            ind_i=ind_i,
            ind_j=ind_j,
            mu_i=mu_i,
            mu_j=mu_j,
            pair_map=pair_map,
            cell_map=cell_map,
            e_map=e_map,
        )
        e, hess = self.compute_hessian()

        return e, hess

    def _eager_evaluate(
        self,
        positions,
        offsets,
        cell,
        ind_i,
        ind_j,
        mu_i,
        mu_j,
        pair_map,
        cell_map,
        e_map,
    ):

        self.init_placeholders(
            positions=positions,
            offsets=offsets,
            cell=cell,
            ind_i=ind_i,
            ind_j=ind_j,
            mu_i=mu_i,
            mu_j=mu_j,
            pair_map=pair_map,
            cell_map=cell_map,
            e_map=e_map,
        )
        e, f, s = self.compute_energy_forces_stress()

        return e, f, s

    def decorate_evaluate_loss(self):
        self._evaluate_loss = tf.function(
            func=self._evaluate_loss,
            input_signature=[
                tf.TensorSpec([None, 3], tf.float64),
                tf.TensorSpec([None, 3], tf.float64),
                tf.TensorSpec([None, 3, 3], tf.float64),
                tf.TensorSpec([None, 1], tf.float64),
                tf.TensorSpec([None, 3], tf.float64),
                tf.TensorSpec([None, 1], tf.float64),
                tf.TensorSpec([None, 1], tf.float64),
                tf.TensorSpec([None], tf.int32),
                tf.TensorSpec([None], tf.int32),
                tf.TensorSpec([None], tf.int32),
                tf.TensorSpec([None], tf.int32),
                tf.TensorSpec([None], tf.int32),
                tf.TensorSpec([None], tf.int32),
                tf.TensorSpec([None], tf.int32),
            ],
            jit_compile=self.compute_loss_compile,
        )

    def _evaluate_loss(
        self,
        positions,
        offsets,
        cell,
        energy,
        forces,
        eweights,
        fweights,
        ind_i,
        ind_j,
        mu_i,
        mu_j,
        pair_map,
        cell_map,
        e_map,
    ):

        self.init_placeholders(
            positions=positions,
            offsets=offsets,
            cell=cell,
            energy=energy,
            forces=forces,
            eweights=eweights,
            fweights=fweights,
            ind_i=ind_i,
            ind_j=ind_j,
            mu_i=mu_i,
            mu_j=mu_j,
            pair_map=pair_map,
            cell_map=cell_map,
            e_map=e_map,
        )
        if self.op_compile:
            with tf.xla.experimental.jit_scope():
                loss, grad_loss, e, f = self.compute_grad()
        else:
            loss, grad_loss, e, f = self.compute_grad()

        return loss, grad_loss, e, f, self.reg_components

    def _evaluate_vector_loss(
        self,
        positions,
        offsets,
        cell,
        energy,
        forces,
        eweights,
        fweights,
        ind_i,
        ind_j,
        mu_i,
        mu_j,
        pair_map,
        cell_map,
        e_map,
    ):

        self.init_placeholders(
            positions=positions,
            offsets=offsets,
            cell=cell,
            energy=energy,
            forces=forces,
            eweights=eweights,
            fweights=fweights,
            ind_i=ind_i,
            ind_j=ind_j,
            mu_i=mu_i,
            mu_j=mu_j,
            pair_map=pair_map,
            cell_map=cell_map,
            e_map=e_map,
        )
        loss, grad_loss, e, f = self.compute_vector_grad()

        return loss, grad_loss, e, f, self.reg_components

    def decorate_vector_loss(self):
        self._evaluate_vector_loss = tf.function(
            func=self._evaluate_vector_loss,
            input_signature=[
                tf.TensorSpec([None, 3], tf.float64),
                tf.TensorSpec([None, 3], tf.float64),
                tf.TensorSpec([None, 3, 3], tf.float64),
                tf.TensorSpec([None, 1], tf.float64),
                tf.TensorSpec([None, 3], tf.float64),
                tf.TensorSpec([None, 1], tf.float64),
                tf.TensorSpec([None, 1], tf.float64),
                tf.TensorSpec([None], tf.int32),
                tf.TensorSpec([None], tf.int32),
                tf.TensorSpec([None], tf.int32),
                tf.TensorSpec([None], tf.int32),
                tf.TensorSpec([None], tf.int32),
                tf.TensorSpec([None], tf.int32),
                tf.TensorSpec([None], tf.int32),
            ],
        )

    def _eager_evaluate_loss(
        self,
        positions,
        offsets,
        cell,
        energy,
        forces,
        eweights,
        fweights,
        ind_i,
        ind_j,
        mu_i,
        mu_j,
        pair_map,
        cell_map,
        e_map,
    ):

        self.init_placeholders(
            positions=positions,
            offsets=offsets,
            cell=cell,
            energy=energy,
            forces=forces,
            eweights=eweights,
            fweights=fweights,
            ind_i=ind_i,
            ind_j=ind_j,
            mu_i=mu_i,
            mu_j=mu_j,
            pair_map=pair_map,
            cell_map=cell_map,
            e_map=e_map,
        )
        loss, grad_loss, e, f = self.compute_grad()

        return loss, grad_loss, e, f, self.reg_components

    def compute_positions(self):
        cells_pos = tf.gather(self.cell, self.cell_map)
        pos = tf.reshape(self.positions, [-1, 1, 3])
        pos = tf.reshape(tf.matmul(pos, cells_pos), [-1, 3])
        self.pos = pos

        return self.pos

    def compute_pair_distance(self, pos):
        cells = tf.gather(self.cell, self.pair_map)
        r_i = tf.gather(pos, self.ind_i)
        j_ofst = tf.reshape(self.offsets, [-1, 1, 3])
        j_ofst = tf.reshape(tf.matmul(j_ofst, cells), [-1, 3])
        r_j = tf.gather(pos, self.ind_j) + j_ofst
        r_ij = r_j - r_i

        return r_ij

    def forward_pass(self, pos):
        r_ij = self.compute_pair_distance(pos)

        e_atomic = self.potential.compute_atomic_energy(
            r_ij,
            (self.ind_i - tf.reduce_min(self.ind_i)),
            self.mu_i,
            self.mu_j,
            self.ind_j,
        )
        e = tf.math.unsorted_segment_sum(
            e_atomic,
            self.e_map,
            num_segments=tf.reduce_max(self.e_map) + 1,
            name="predict_energy",
        )

        return e

    def compute_vector_loss(self, e):
        assert (
            self.loss_specs[LOSS_TYPE] == "per-atom"
        ), "Only per-atom loss type is compatible with vector loss"

        total_residuals = []
        counter = tf.ones_like(self.e_map, dtype=tf.float64)
        natoms = tf.math.unsorted_segment_sum(
            counter, self.e_map, num_segments=tf.reduce_max(self.e_map) + 1
        )
        natoms = tf.reshape(tf.cast(natoms, tf.float64), [-1, 1])
        e_res = tf.reshape(
            self.loss_specs[LOSS_ENERGY_FACTOR]
            * (self.eweights * ((e - self.energy) / natoms)),
            [-1],
        )
        total_residuals += [e_res]

        reg_components = []
        self.potential.compute_regularization()
        total_residuals += [
            tf.reshape(self.potential.reg_l1 * self.loss_specs[L1_REG], [-1])
        ]
        total_residuals += [
            tf.reshape(self.potential.reg_l2 * self.loss_specs[L2_REG], [-1])
        ]
        reg_components += [
            tf.reshape(self.potential.reg_l1, [1, 1]),
            tf.reshape(self.potential.reg_l2, [1, 1]),
        ]
        if self.potential.aux is not None:
            for i in range(self.loss_specs[AUX_LOSS_FACTOR].shape[0]):
                total_residuals += [
                    tf.reshape(
                        tf.squeeze(
                            self.potential.aux[i] * self.loss_specs[AUX_LOSS_FACTOR][i]
                        ),
                        [-1],
                    )
                ]
                reg_components += [self.potential.aux[i]]
        self.reg_components = tf.stack(reg_components)

        return tf.concat(total_residuals, axis=0)

    def compute_loss(self, e, f):
        total_loss = 0
        if self.loss_specs[LOSS_TYPE] == "per-structure":
            total_loss += self.loss_specs[LOSS_ENERGY_FACTOR] * tf.reduce_sum(
                self.eweights * (e - self.energy) ** 2
            )
        elif self.loss_specs[LOSS_TYPE] == "per-atom":
            counter = tf.ones_like(self.e_map, dtype=tf.float64)
            natoms = tf.math.unsorted_segment_sum(
                counter, self.e_map, num_segments=tf.reduce_max(self.e_map) + 1
            )
            natoms = tf.reshape(tf.cast(natoms, tf.float64), [-1, 1])
            total_loss += self.loss_specs[LOSS_ENERGY_FACTOR] * tf.reduce_sum(
                self.eweights * ((e - self.energy) / natoms) ** 2
            )

        loss_f = tf.reduce_sum(self.fweights * (f - self.forces) ** 2)
        total_loss += self.loss_specs[LOSS_FORCE_FACTOR] * loss_f

        reg_components = []
        self.potential.compute_regularization()
        total_loss += self.potential.reg_l1 * self.loss_specs[L1_REG]
        total_loss += self.potential.reg_l2 * self.loss_specs[L2_REG]
        reg_components += [
            tf.reshape(self.potential.reg_l1, [1, 1]),
            tf.reshape(self.potential.reg_l2, [1, 1]),
        ]
        if self.potential.aux is not None:
            for i in range(self.loss_specs[AUX_LOSS_FACTOR].shape[0]):
                total_loss += tf.squeeze(
                    self.potential.aux[i] * self.loss_specs[AUX_LOSS_FACTOR][i]
                )
                reg_components += [self.potential.aux[i]]
        self.reg_components = tf.stack(reg_components)

        return total_loss

    def compute_grad(self):
        pos = self.compute_positions()
        self.fit_coefs = self.potential.fit_coefs
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(pos)
            e = self.forward_pass(pos)
            f = tf.negative(tape.gradient(e, pos))
            loss = self.compute_loss(e, f)
        grad_loss = tape.gradient(loss, self.fit_coefs)

        f = tf.convert_to_tensor(f)

        return loss, grad_loss, e, f

    def compute_vector_grad(self):
        pos = self.compute_positions()
        self.fit_coefs = self.potential.fit_coefs
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(pos)
            e = self.forward_pass(pos)
            loss = self.compute_vector_loss(e)
        f = tf.negative(tape.gradient(e, pos))
        grad_loss = tape.jacobian(loss, self.fit_coefs, experimental_use_pfor=True)

        f = tf.convert_to_tensor(f)

        return loss, grad_loss, e, f

    def compute_energy_forces_stress(self):
        pos = self.compute_positions()
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(pos)
            tape.watch(self.cell)
            e = self.forward_pass(pos)
        f = tape.gradient(e, pos)
        f = tf.negative(tf.convert_to_tensor(f))
        s = tape.gradient(e, self.cell)
        s = tf.convert_to_tensor(s)

        return e, f, s

    def compute_hessian(self):
        pos = self.compute_positions()
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(pos)
            with tf.GradientTape() as tape1:
                tape1.watch(pos)
                e = self.forward_pass(pos)
            f = tape1.gradient(e, pos)
        hess = tape.jacobian(f, pos)
        shp = tf.shape(hess)
        hess = tf.reshape(hess, [shp[0] * shp[1], shp[2] * shp[3]])

        return e, hess

    def compute_ace_basis(self, ref_basis=None):
        pos = self.compute_positions()
        with tf.GradientTape(persistent=False) as tape:
            tape.watch(pos)
            e = self.forward_pass(pos)
            basis = tf.concat(self.potential.BbasisFuncs, axis=1)
            cost = tf.reduce_sum((ref_basis - basis) ** 2)
        dBdp = tape.gradient(cost, pos)

        return cost, basis, tf.convert_to_tensor(dBdp)

    def external_fit(self, coefs, data, eager=False):
        self.potential.set_coefs(coefs)
        if eager:
            loss, grad_loss, e, f, self.reg_components = self._eager_evaluate_loss(
                *input2evaloss(data)
            )
        else:
            loss, grad_loss, e, f, self.reg_components = self._evaluate_loss(
                *input2evaloss(data)
            )

        return loss, grad_loss, e, f

    def native_fit(self, data, eager=False):
        if eager:
            loss, grad_loss, e, f, self.reg_components = self._eager_evaluate_loss(
                *input2evaloss(data)
            )
        else:
            loss, grad_loss, e, f, self.reg_components = self._evaluate_loss(
                *input2evaloss(data)
            )

        return loss, grad_loss, e, f

    def external_vector_fit(self, coefs, data):
        self.potential.set_coefs(coefs)
        loss, grad_loss, e, f, self.reg_components = self._evaluate_vector_loss(
            *input2evaloss(data)
        )

        return loss, grad_loss, e, f

    def save_model(self, path):
        tf.saved_model.save(self, path)

    @staticmethod
    def load_model(path):
        imported = tf.saved_model.load(path)

        return imported

    def evaluate(self, data):
        e, f, s = self._evaluate(*input2eval(data))

        return e, f, s

    def evaluate_hessian(self, data):
        e, hess = self._evaluate_hess(*input2eval(data))

        return e, hess

    def eager_evaluate(self, data):
        e, f, s = self._eager_evaluate(*input2eval(data))

        return e, f, s

    def evaluate_loss(self, data):
        loss, grad_loss, e, f, self.reg_components = self._evaluate_loss(
            *input2evaloss(data)
        )

        return loss, grad_loss, e, f

    def eager_evaluate_loss(self, data):
        loss, grad_loss, e, f, self.reg_components = self._eager_evaluate_loss(
            *input2evaloss(data)
        )

        return loss, grad_loss, e, f
