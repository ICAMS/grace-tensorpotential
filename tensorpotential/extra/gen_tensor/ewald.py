import tensorflow as tf
import numpy as np

from tensorpotential import constants


def generate_k_vectors(k_max, cell_vectors):
    """
    Generates reciprocal lattice vectors up to a specified cutoff.

    Args:
        k_max (int): maximum wave vector

        cell_vectors (tf.Tensor): A (None, 3, 3) tensor where rows are the
                                  lattice vectors (a1, a2, a3).
                                  None is a batch dimension

    Returns:
        k_vectors (tf.Tensor): A (None, M, 3) tensor of k-vectors (k != 0).
        k_squared (tf.Tensor): A (None, M) tensor of the squared magnitudes |k|^2.

    """

    # b = 2 * pi * (A^T)^-1
    recip_basis = 2.0 * np.pi * tf.transpose(tf.linalg.pinv(cell_vectors), [0, 2, 1])

    grid_size = tf.range(
        -k_max, k_max + 1, dtype=cell_vectors.dtype
    )  # Shape ( (2*k_max+1)^3, 3 )
    hkl_grid = tf.stack(
        tf.meshgrid(grid_size, grid_size, grid_size, indexing="ij"), axis=-1
    )
    hkl_flat = tf.reshape(hkl_grid, (-1, 3))

    # k = h*b1 + k*b2 + l*b3
    k_vectors_all = tf.einsum("cij,ki->ckj", recip_basis, hkl_flat)
    k_squared = tf.reduce_sum(tf.square(k_vectors_all), axis=-1)

    return k_vectors_all, k_squared


def det_comp(x, idx):
    comp = tf.gather(x, idx, axis=1)
    return tf.reduce_prod(comp, axis=1, keepdims=True)


def det(x):
    x = tf.reshape(x, [-1, 9])
    a = det_comp(x, [0, 4, 8])
    b = det_comp(x, [1, 5, 6])
    c = det_comp(x, [2, 3, 7])
    d = det_comp(x, [2, 4, 6])
    e = det_comp(x, [1, 3, 8])
    f = det_comp(x, [0, 5, 7])
    return a + b + c - d - e - f


# def reciprocal_ewald_energy(positions, charges, dipoles, cell_vectors, sigma, k_max):
def reciprocal_ewald_energy(
    input_data: dict, dipoles: tf.Tensor, sigma: float, k_max: int
):
    """
    Calculates the reciprocal space part of the Ewald summation.

    Args:
        positions (tf.Tensor): Particle positions, shape (None, 3).
        charges (tf.Tensor): Particle charges, shape (None,).
        dipoles (tf.Tensor): particle dipole, shape (None, 3).
        cell_vectors (tf.Tensor): Cell lattice vectors, shape (None, 3, 3).
        sigma (float): smearing.

    Returns:
        tf.Tensor: A scalar tensor representing the reciprocal energy.
    """

    k_vectors, k_squared = generate_k_vectors(
        k_max, input_data[constants.CELL_VECTORS]
    )  # (None, G, 3), (None,  G)

    atom_k_vec = tf.gather(
        k_vectors, input_data[constants.ATOMS_TO_STRUCTURE_MAP], axis=0
    )

    k_dot_r = tf.einsum(
        "akj,aj->ak", atom_k_vec, input_data[constants.ATOMIC_POS]
    )  # (None, G),
    if atom_k_vec.dtype == tf.float32:
        c_dtype = tf.complex64
    else:
        c_dtype = tf.complex128

    envelope = tf.cast(tf.exp(-k_squared * sigma**2 / 2), dtype=c_dtype)
    # envelope = tf.gather(envelope, input_data[constants.ATOMS_TO_STRUCTURE_MAP], axis=0)
    # exp(-i * k.r)
    exp_term = tf.exp(tf.cast(1j, c_dtype) * tf.cast(k_dot_r, c_dtype))

    # charges_row = tf.cast(charges[:, tf.newaxis], CDTYPE)

    # This has to be unsorted_segment_sum with atoms_to_structure as index
    # monopole_density = tf.reduce_sum(exp_term * charges_row, axis=0, keepdims=True)
    # monopole_density *= envelope

    k_dot_p = tf.einsum("akj,aj->ak", atom_k_vec, dipoles)  # (None, G),
    # Weighted sum over atoms: i ∑_a (G · p_a) * exp(i G · R_a)
    sum_dip = tf.math.unsorted_segment_sum(
        tf.cast(k_dot_p, dtype=c_dtype) * exp_term,
        input_data[constants.ATOMS_TO_STRUCTURE_MAP],
        num_segments=input_data[constants.N_STRUCTURES_BATCH_TOTAL],
    )  # shape: (1, G)
    dipole_density = tf.cast(1j, c_dtype) * sum_dip  # shape: (None, G)
    dipole_density *= envelope

    # return monopole_density, dipole_density
    # total_density = monopole_density + dipole_density
    total_density = dipole_density
    total_density_sqr = tf.square(tf.abs(total_density))
    total_density_sqr = tf.where(
        k_squared == 0, tf.zeros_like(total_density_sqr), total_density_sqr
    )
    unit_factor = 14.39964546868
    volume = det(input_data[constants.CELL_VECTORS])
    # scale = 0.5 * 4 * np.pi / volume * unit_factor
    scale = 2 * np.pi * unit_factor / volume
    scale /= tf.where(k_squared == 0, tf.ones_like(k_squared), k_squared)

    # Sum over G
    energy = tf.reduce_sum(scale * total_density_sqr, axis=1, keepdims=True)
    # TODO: batching will also involve fake atoms, contributions from which must be removed

    # compute norm of dipoles with well defined derivatives

    norm_sqr = tf.reduce_sum(dipoles**2, axis=-1, keepdims=True)
    # self energy contribution for each dipole
    # self_energy_per_d = norm_sqr
    # sum over all dipoles per structure
    self_energy = tf.math.unsorted_segment_sum(
        norm_sqr,
        input_data[constants.ATOMS_TO_STRUCTURE_MAP],
        num_segments=input_data[constants.N_STRUCTURES_BATCH_TOTAL],
    )
    self_energy = unit_factor / (12 * sigma**3 * np.pi**0.5) * self_energy
    # subtract self energy
    energy = tf.subtract(energy, self_energy)

    return energy
