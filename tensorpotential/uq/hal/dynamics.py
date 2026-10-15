"""Dynamics helpers for HAL walkers."""

from __future__ import annotations

import numpy as np
from ase import units
from ase.md.langevin import Langevin

from tensorpotential.uq.hal.calculator import HALStop

LIGHT_ELEMENTS = frozenset({1, 2, 3, 4})  # H, He, Li, Be


def default_timestep_fs(numbers) -> float:
    """2 fs, or 0.5 fs when H, He, Li or Be is present."""
    return 0.5 if LIGHT_ELEMENTS & set(int(z) for z in numbers) else 2.0


class RampedLangevin(Langevin):
    """Langevin dynamics with a linear volume ramp and one force call per step.

    At the start of step n the cell (with the atoms) is rescaled to
    V = V0 * (v_start + (v_end - v_start) * min(n, n_steps) / n_steps). The first
    velocity half-kick uses the forces of the previous step, fetched from the
    calculator cache before the rescale; the error is of the order of one ramp
    increment. Rescaling in an observer instead would invalidate those forces and
    cost a second call.
    """

    def __init__(
        self,
        atoms,
        timestep,
        *,
        temperature_K,
        friction,
        n_steps,
        v_start=1.0,
        v_end=1.4,
        **kwargs,
    ):
        super().__init__(
            atoms, timestep, temperature_K=temperature_K, friction=friction, **kwargs
        )
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.cell0 = atoms.get_cell().copy()
        self.v_start, self.v_end, self.ramp_steps = (
            float(v_start),
            float(v_end),
            int(n_steps),
        )

    def volume_factor(self, n: int) -> float:
        frac = min(n, self.ramp_steps) / self.ramp_steps
        return self.v_start + (self.v_end - self.v_start) * frac

    def step(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces(md=True)  # cache hit after the previous step
        scale = self.volume_factor(self.nsteps + 1) ** (1.0 / 3.0)
        self.atoms.set_cell(self.cell0 * scale, scale_atoms=True)
        return super().step(forces=forces)


class VolumeGuard:
    """Observer: stop the walker when V/V0 leaves [lo, hi]."""

    def __init__(self, atoms, lo: float = 0.75, hi: float = 1.30):
        self.atoms, self.lo, self.hi = atoms, float(lo), float(hi)
        self.v0 = atoms.get_volume()

    def __call__(self):
        ratio = self.atoms.get_volume() / self.v0
        if not self.lo <= ratio <= self.hi:
            raise HALStop(f"volume_guard (V/V0={ratio:.3f})")


def internal_temperature(atoms) -> float:
    """Temperature (K) over the 3N-3 internal degrees of freedom: 2 E_kin / ((3N-3) k_B).

    Every HAL ensemble has zero centre-of-mass momentum (Langevin: COM reset every step;
    MTK conserves momentum), so 3 of the 3N dof carry no kinetic energy. ASE's
    ``get_temperature`` divides by 3N (HAL sets no constraints). 0.0 for a single atom.
    """
    dof = 3 * len(atoms) - 3
    return 2.0 * atoms.get_kinetic_energy() / (dof * units.kB) if dof > 0 else 0.0


def radius_of_gyration(atoms) -> float:
    """Mass-weighted radius of gyration (Angstrom)."""
    m = atoms.get_masses()
    rel = atoms.positions - atoms.get_center_of_mass()
    return float(np.sqrt(np.sum(m * np.sum(rel**2, axis=1)) / m.sum()))


FRAGMENT_MULT = 1.25  # bond if d < FRAGMENT_MULT * (r_cov,i + r_cov,j)


def count_fragments(atoms, mult: float = FRAGMENT_MULT) -> int:
    """Connected components of the bond graph (pbc-aware).

    Two atoms are bonded when their distance is below ``mult`` * (r_cov,i + r_cov,j), with
    ASE covalent radii. In a periodic cell this catches isolated atoms, molecules and
    clusters, but not a planar crack, which stays connected through the periodic images.
    Caveat for bulk: absolute covalent cutoffs sit close to metallic nearest-neighbour
    distances (Al-Al cutoff 3.03 A vs 2.86 A in fcc Al-Li), so an expanded or hot cell can
    read as several fragments without anything breaking; read it with V/V0, d_min and T.
    """
    from ase.data import covalent_radii
    from ase.neighborlist import neighbor_list
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    zs = sorted({int(z) for z in atoms.numbers})
    table = {(a, b): mult * float(covalent_radii[a] + covalent_radii[b])
             for k, a in enumerate(zs) for b in zs[k:]}
    n = len(atoms)
    i, j = neighbor_list("ij", atoms, table)
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    n_comp, _ = connected_components(graph, directed=False)
    return int(n_comp)


def cluster_wall(atoms, margin: float = 2.0, k: float = 1.0):
    """Containment wall around the initial centre of mass: (center, r_wall, k)."""
    center = atoms.get_center_of_mass()
    r_max = float(np.max(np.linalg.norm(atoms.positions - center, axis=1)))
    return center, r_max + float(margin), float(k)


def estimate_bulk_modulus(
    atoms, calc, factors=(0.97, 0.985, 1.0, 1.015, 1.03)
) -> float:
    """B = V0 * d2E/dV2 from a quadratic fit of E(V) (eV/Angstrom^3); 5 single points."""
    v0 = atoms.get_volume()
    vols, energies = [], []
    for f in factors:
        a = atoms.copy()
        a.set_cell(atoms.cell * f ** (1.0 / 3.0), scale_atoms=True)
        a.calc = calc
        energies.append(a.get_potential_energy())
        vols.append(a.get_volume())
    c2 = np.polyfit(vols, energies, 2)[0]
    b = 2.0 * c2 * v0
    if not np.isfinite(b) or b <= 0:
        raise ValueError(
            f"bulk modulus estimate is not positive ({b:.4g} eV/A^3); "
            "pass bulk_modulus= / --bulk-modulus explicitly"
        )
    return float(b)
