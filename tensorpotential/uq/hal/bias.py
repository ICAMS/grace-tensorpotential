"""Array-level building blocks of the HAL bias.

Everything here takes and returns NumPy arrays (no ASE Atoms / calculators), so a
future batched multi-walker driver can reuse it unchanged.
"""

from __future__ import annotations

import numpy as np
from ase.data import atomic_numbers, covalent_radii

# e^2 / (4 pi eps0) in eV * Angstrom
_COULOMB = 14.399645
_ZBL_C = np.array([0.18175, 0.50986, 0.28022, 0.02817])
_ZBL_D = np.array([3.19980, 0.94229, 0.40290, 0.20162])


class ForceBiasScaler:
    """tau_r = tau * EMA(sum_i |F_i|) / EMA(sum_i |grad_i U|).

    The bias is zero for the first ``ema`` calls while the averages fill.
    """

    def __init__(self, tau: float, ema: int = 10, eps: float = 1e-12):
        if ema < 1:
            raise ValueError("ema must be >= 1")
        self.tau = float(tau)
        self.ema = int(ema)
        self.eps = eps
        self.n_calls = 0
        self.f_norm: float | None = None
        self.g_norm: float | None = None

    def __call__(self, forces, grad_u) -> float:
        f = float(np.sum(np.linalg.norm(forces, axis=1)))
        g = float(np.sum(np.linalg.norm(grad_u, axis=1)))
        if self.f_norm is None:
            self.f_norm, self.g_norm = f, g
        else:
            beta = 1.0 / self.ema
            self.f_norm += beta * (f - self.f_norm)
            self.g_norm += beta * (g - self.g_norm)
        self.n_calls += 1
        if self.tau == 0.0 or self.n_calls <= self.ema:
            return 0.0
        return self.tau * self.f_norm / max(self.g_norm, self.eps)


class PairTable:
    """Per element pair: ZBL switching radii and rejection distance.

    Defaults scale with the covalent-radius sum s_ij = r_cov,i + r_cov,j:
    r_in = zbl_switch[0] * s_ij, r_out = zbl_switch[1] * s_ij,
    r_reject = reject_factor * s_ij. ``overrides`` maps "A-B" to
    {"zbl": [r_in, r_out], "reject": r} (either key optional, Angstrom).
    """

    def __init__(self, zbl_switch=(0.55, 0.75), reject_factor=0.65, overrides=None):
        self.zbl_switch = (float(zbl_switch[0]), float(zbl_switch[1]))
        if not 0 < self.zbl_switch[0] < self.zbl_switch[1]:
            raise ValueError("zbl_switch must satisfy 0 < r_in < r_out")
        self.reject_factor = float(reject_factor)
        self.overrides = {}
        for key, val in (overrides or {}).items():
            a, b = self._pair_key(key)
            self._check_override(key, val)
            self.overrides[(min(a, b), max(a, b))] = val

    @staticmethod
    def _pair_key(key):
        parts = [s.strip() for s in str(key).split("-")]
        if len(parts) != 2:
            raise ValueError(f"pair override key must look like 'A-B', got {key!r}")
        unknown = [s for s in parts if s not in atomic_numbers]
        if unknown:
            raise ValueError(f"pair override {key!r}: unknown element(s) {unknown}")
        return atomic_numbers[parts[0]], atomic_numbers[parts[1]]

    @staticmethod
    def _check_override(key, val):
        if not isinstance(val, dict):
            raise ValueError(f"pair override {key!r} must be a dict with 'zbl' and/or 'reject'")
        extra = sorted(set(val) - {"zbl", "reject"})
        if extra:
            raise ValueError(f"pair override {key!r}: unknown key(s) {extra} (allowed: zbl, reject)")
        if "zbl" in val:
            z = val["zbl"]
            if not (np.ndim(z) == 1 and len(z) == 2 and 0 < float(z[0]) < float(z[1])):
                raise ValueError(f"pair override {key!r}: 'zbl' must be [r_in, r_out] with 0 < r_in < r_out")
        if "reject" in val and not float(val["reject"]) > 0:
            raise ValueError(f"pair override {key!r}: 'reject' must be > 0")

    def radii(self, zi, zj):
        zi = np.asarray(zi)
        zj = np.asarray(zj)
        s = covalent_radii[zi] + covalent_radii[zj]
        r_in = self.zbl_switch[0] * s
        r_out = self.zbl_switch[1] * s
        r_rej = self.reject_factor * s
        for (a, b), val in self.overrides.items():
            m = ((zi == a) & (zj == b)) | ((zi == b) & (zj == a))
            if not m.any():
                continue
            if "zbl" in val:
                r_in[m] = val["zbl"][0]
                r_out[m] = val["zbl"][1]
            if "reject" in val:
                r_rej[m] = val["reject"]
        return r_in, r_out, r_rej

    def max_cutoff(self, numbers) -> float:
        z = np.array(sorted(set(int(x) for x in numbers)))
        zi, zj = (x.ravel() for x in np.meshgrid(z, z))
        _, r_out, r_rej = self.radii(zi, zj)
        return float(max(r_out.max(), r_rej.max()))


def _switch(r, r_in, r_out):
    """C2 switch S(r): 1 below r_in, 0 above r_out. Returns (S, dS/dr)."""
    width = r_out - r_in
    t = np.clip((r - r_in) / width, 0.0, 1.0)
    s = 1 - 10 * t**3 + 15 * t**4 - 6 * t**5
    ds = (-30 * t**2 + 60 * t**3 - 30 * t**4) / width
    return s, ds


def zbl_pair_energy(r, zi, zj):
    """Universal ZBL pair energy (eV) and dE/dr (eV/Angstrom)."""
    zi = np.asarray(zi, dtype=np.float64)
    zj = np.asarray(zj, dtype=np.float64)
    a = 0.46850 / (zi**0.23 + zj**0.23)
    x = r / a
    e = np.exp(-_ZBL_D[None, :] * x[:, None])
    phi = (_ZBL_C * e).sum(axis=1)
    dphi = -(_ZBL_C * _ZBL_D * e).sum(axis=1) / a
    pref = _COULOMB * zi * zj
    return pref * phi / r, pref * (dphi / r - phi / r**2)


def zbl_switched(i, j, d, D, zi, zj, r_in, r_out, n_atoms, volume=None):
    """Switched ZBL over a full neighbour list (both (i, j) and (j, i) present).

    D[k] = r_j - r_i (+ periodic shift), as returned by ase.neighborlist "D".
    Returns energy, forces [n_atoms, 3] and stress (ASE Voigt xx, yy, zz, yz, xz, xy,
    sign convention sigma = (1/V) dE/d(strain)); zero stress when volume is None.
    """
    forces = np.zeros((n_atoms, 3))
    m = d < r_out
    if not np.any(m):
        return 0.0, forces, np.zeros(6)
    i, d, D = i[m], d[m], D[m]
    e_pair, de_pair = zbl_pair_energy(d, zi[m], zj[m])
    s, ds = _switch(d, r_in[m], r_out[m])
    e = e_pair * s
    de = de_pair * s + e_pair * ds
    energy = 0.5 * float(e.sum())
    np.add.at(forces, i, (de / d)[:, None] * D)
    if volume is None:
        return energy, forces, np.zeros(6)
    w = 0.5 * np.einsum("k,ka,kb->ab", de / d, D, D) / volume
    stress = np.array([w[0, 0], w[1, 1], w[2, 2], w[1, 2], w[0, 2], w[0, 1]])
    return energy, forces, stress


def model_virial_to_ase(v) -> np.ndarray:
    """Model virial order (xx, yy, zz, xy, xz, yz) -> ASE Voigt (xx, yy, zz, yz, xz, xy)."""
    return np.asarray(v, dtype=np.float64).reshape(-1)[:6][[0, 1, 2, 5, 4, 3]]


class StressBiasDirection:
    """EMA-smoothed unit direction of the uncertainty virial (ASE Voigt order)."""

    def __init__(self, ema: int = 10):
        self.ema = int(ema)
        self.w: np.ndarray | None = None

    def __call__(self, virial_ase) -> np.ndarray:
        v = np.asarray(virial_ase, dtype=np.float64)
        n = np.linalg.norm(v)
        if np.isfinite(n) and n > 0:
            u = v / n
            self.w = u if self.w is None else self.w + (u - self.w) / self.ema
        if self.w is None:
            return np.zeros(6)
        nw = np.linalg.norm(self.w)
        return self.w / nw if nw > 0 else np.zeros(6)


def stress_bias(direction, tau_s: float, bulk_modulus: float) -> np.ndarray:
    """Stress that pushes the cell toward higher sum(gamma): -tau_s * B * W_hat."""
    return -float(tau_s) * float(bulk_modulus) * np.asarray(direction, dtype=np.float64)


def containment_wall(positions, center, r_wall: float, k: float):
    """Flat-bottom harmonic wall: E = 0.5 * k * sum_i max(0, |r_i - c| - R)^2.

    Returns energy, forces [N, 3] and the number of atoms outside R.
    """
    rel = np.asarray(positions, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    r = np.linalg.norm(rel, axis=1)
    excess = np.maximum(r - r_wall, 0.0)
    out = excess > 0
    forces = np.zeros_like(rel)
    forces[out] = -(k * excess[out] / r[out])[:, None] * rel[out]
    return 0.5 * k * float(np.sum(excess**2)), forces, int(out.sum())
