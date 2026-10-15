"""ASE calculator that adds the HAL bias to an updatable CALM UQ calculator."""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

from tensorpotential import constants
from tensorpotential.uq import constants as uqc
from tensorpotential.uq.hal.bias import (
    ForceBiasScaler,
    PairTable,
    StressBiasDirection,
    containment_wall,
    model_virial_to_ase,
    stress_bias,
    zbl_switched,
)


class HALStop(Exception):
    """Raised to end a HAL walker; ``reason`` goes into the run summary."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class HALCalculator(Calculator):
    """Biased forces F + tau_r * grad(sum gamma) plus switched ZBL and stress bias.

    One base-model call per ``calculate``. The reported energy is the physical
    E_model + E_ZBL. Between CALM updates (and at fixed tau_r) the bias force derives from
    U_b = -tau_r * sum(gamma), which is not added to the reported energy; a CALM update
    changes U_b (see the update-work removal in ``HALCollector``). The base
    calculator must return ``gamma`` and ``dsigma_dr`` (a UQ TPCalculator built with
    ``load_uq_artifacts(..., compute_dsigma_dr=True)`` or ``get_gmm_uq_calculator``, or
    a UQ SavedModel calculator after ``enable_uq(mode="full")``).
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        base_calc,
        tau: float = 0.15,
        tau_ema: int = 10,
        zbl: str = "auto",
        zbl_switch=(0.55, 0.75),
        reject_bond_factor: float = 0.65,
        pair_overrides: dict | None = None,
        stress_tau: float | None = None,
        bulk_modulus: float | None = None,
        wall=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if zbl not in ("auto", "off"):
            raise ValueError("zbl must be 'auto' or 'off'")
        if stress_tau and bulk_modulus is None:
            raise ValueError("stress_tau needs bulk_modulus")
        self.base = base_calc
        self.scaler = ForceBiasScaler(tau, tau_ema)
        self.zbl_on = zbl == "auto"
        self.pairs = PairTable(zbl_switch, reject_bond_factor, pair_overrides)
        self.stress_tau = stress_tau
        self.bulk_modulus = bulk_modulus
        self.stress_dir = StressBiasDirection(tau_ema) if stress_tau else None
        self.wall = wall
        self.n_base_calls = 0
        self._cutoff_key = None
        self._cutoff = None
        self._base_covers = None

    def _base_pairs(self, atoms):
        """Pairs closer than the ZBL/reject cutoff, taken from the base model's own bond list.

        Returns (i, j, d, D) in the ``neighbor_list("ijdD")`` convention (full list, D = r_j - r_i),
        or None when the base list cannot be trusted to contain every such pair (no segment-sum
        bond data, or a model cutoff, global or per element pair, shorter than ours).
        """
        base = self.base
        data = getattr(base, "data", None)
        if getattr(base, "dense_reshape", True) or not hasattr(data, "get"):
            return None
        if any(data.get(k) is None for k in (constants.BOND_IND_I, constants.BOND_IND_J, constants.BOND_VECTOR)):
            return None
        if not self._covers(atoms):
            return None
        n = len(atoms)
        i = np.asarray(data[constants.BOND_IND_I])
        j = np.asarray(data[constants.BOND_IND_J])
        v = np.asarray(data[constants.BOND_VECTOR], dtype=np.float64)
        d = np.linalg.norm(v, axis=1)
        keep = (d < self._cutoff) & (i < n) & (j < n)  # padded bonds are far; fake atoms have index >= n
        return i[keep], j[keep], d[keep], v[keep]

    def _covers(self, atoms) -> bool:
        """True if the base model's bond list contains every pair inside our cutoff (cached per species set)."""
        if self._base_covers is None or self._base_covers[0] != self._cutoff_key:
            self._base_covers = (self._cutoff_key, self._compute_covers(atoms))
        return self._base_covers[1]

    def _compute_covers(self, atoms) -> bool:
        base = self.base
        if getattr(base, "cutoff", 0.0) < self._cutoff:
            return False
        cutoff_dict = getattr(base, "cutoff_dict", None)
        if not cutoff_dict:
            return True
        symbols = sorted(set(atoms.get_chemical_symbols()))
        for a in symbols:
            for b in symbols:
                c = cutoff_dict.get((a, b), cutoff_dict.get((b, a)))
                if c is None or c < self._cutoff:
                    return False
        return True

    def _short_range(self, atoms):
        key = tuple(sorted(set(atoms.numbers.tolist())))
        if key != self._cutoff_key:
            self._cutoff_key, self._cutoff = key, self.pairs.max_cutoff(atoms.numbers)
        n = len(atoms)
        # The base model already built a neighbour list this step: if its shortest bond (and
        # its cutoff covers ours) is beyond the ZBL/reject range, no pair can be inside it.
        d_min = getattr(self.base, "current_min_dist", np.nan)
        if d_min > self._cutoff and self._covers(atoms):
            return 0.0, np.zeros((n, 3)), np.zeros(6), np.inf
        pairs = self._base_pairs(atoms)
        if pairs is None:
            pairs = neighbor_list("ijdD", atoms, self._cutoff)
        i, j, d, D = pairs
        if len(d) == 0:
            return 0.0, np.zeros((n, 3)), np.zeros(6), np.inf
        z = atoms.numbers
        r_in, r_out, r_rej = self.pairs.radii(z[i], z[j])
        ratio = float(np.min(d / r_rej))
        if not self.zbl_on:
            return 0.0, np.zeros((n, 3)), np.zeros(6), ratio
        volume = atoms.get_volume() if atoms.cell.rank == 3 else None
        e, f, s = zbl_switched(i, j, d, D, z[i], z[j], r_in, r_out, n, volume)
        return e, f, s, ratio

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        n = len(atoms)
        # The base calculator pads non-periodic structures in place (adds a cell, pbc and a
        # shift); give it a copy so our own state matches the live atoms and the cache hits.
        self.base.calculate(atoms.copy(), ["energy", "forces", "stress"], system_changes)
        self.n_base_calls += 1
        r = self.base.results
        for key in (uqc.ATOMIC_GAMMA, uqc.DSIGMA_DR):
            if key not in r:
                raise RuntimeError(
                    f"HALCalculator needs '{key}' from the base calculator: build it with "
                    "get_gmm_uq_calculator(...) or load_uq_artifacts(..., "
                    "compute_dsigma_dr=True) and an artifact with calibrated thresholds; "
                    "for a UQ SavedModel call calc.enable_uq(mode='full')"
                )
        forces = np.asarray(r["forces"], dtype=np.float64)[:n]
        grad = np.asarray(r[uqc.DSIGMA_DR], dtype=np.float64)[:n]
        gamma = np.asarray(r[uqc.ATOMIC_GAMMA], dtype=np.float64).reshape(-1)[:n]
        if not (np.all(np.isfinite(forces)) and np.all(np.isfinite(grad)) and np.all(np.isfinite(gamma))):
            raise HALStop("non_finite")
        grace_energy = float(r["energy"])
        grace_stress = np.asarray(r["stress"], dtype=np.float64).reshape(-1)[:6].copy()

        tau_r = self.scaler(forces, grad)
        f_out = forces + tau_r * grad
        e_zbl, f_zbl, s_zbl, ratio = self._short_range(atoms)
        e_wall, n_out = 0.0, 0
        if self.wall is not None:
            e_wall, f_wall, n_out = containment_wall(atoms.positions, *self.wall)
            f_zbl = f_zbl + f_wall
        stress = grace_stress + s_zbl
        if self.stress_dir is not None:
            w = self.stress_dir(model_virial_to_ase(r[uqc.VIRIAL_SIGMA]))
            stress = stress + stress_bias(w, self.stress_tau, self.bulk_modulus)

        self.results = {
            "energy": grace_energy + e_zbl,
            "free_energy": grace_energy + e_zbl,
            "forces": f_out + f_zbl,
            "stress": stress,
            "gamma": gamma,
            "features": np.asarray(r[uqc.FEATURES])[:n] if uqc.FEATURES in r else None,
            "tau_r": tau_r,
            "f_norm_ema": self.scaler.f_norm,
            "g_norm_ema": self.scaler.g_norm,
            "d_min": float(getattr(self.base, "current_min_dist", np.nan)),
            "bond_ratio_min": ratio,
            "wall_energy": e_wall,
            "n_outside_wall": n_out,
            "grace_energy": grace_energy,
            "grace_forces": forces.copy(),
            "grace_stress": grace_stress,
        }
