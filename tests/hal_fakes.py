"""Model-free stand-ins for HAL tests."""

import types

import numpy as np

from tensorpotential.uq import constants as uqc
from tensorpotential.uq.artifact_builder import atomic_savez


class FakeUQCalculator:
    """Mimics the result keys of an updatable UQ TPCalculator."""

    def __init__(self, gamma=None, grad=None, forces=None, features_dim=4):
        self.gamma, self.grad, self.forces = gamma, grad, forces
        self.features_dim = features_dim
        self.gmm_uq_model = None
        self.element_map = {}
        self.n_calls = 0
        self.results = {}
        self.current_min_dist = np.inf

    def calculate(self, atoms, properties=None, system_changes=None):
        self.n_calls += 1
        n = len(atoms)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((n, 3)) if self.forces is None else np.array(self.forces),
            "stress": np.zeros(6),
            uqc.ATOMIC_GAMMA: np.ones(n) if self.gamma is None else np.array(self.gamma),
            uqc.DSIGMA_DR: np.ones((n, 3)) if self.grad is None else np.array(self.grad),
            uqc.VIRIAL_SIGMA: np.ones(6),
            uqc.FEATURES: np.zeros((n, self.features_dim)),
        }


class FakeLive:
    """Live tower stand-in: fixed trial gammas, records updates."""

    def __init__(self, trial, inflate=(2, (4.0, 2.0), 1.1), gamma_after=0.0):
        self.trial = np.asarray(trial, dtype=float)
        self.gamma_after = gamma_after  # per-atom gamma reported after an update
        self.n_gamma_calls = 0
        self.updates = []
        self.modes = []  # update mode per trial_gamma / update_one call
        self.targets = []
        self.inflations = []  # (target, max_rounds, max_factor) per inflate() call
        self._inflate = inflate  # (n_rounds, factors, gamma_max_after)

    def trial_gamma(self, features, element_indices, alphas, target=None, mode="convex"):
        self.targets.append(target)
        self.modes.append(("trial", mode))
        out = self.trial[: len(alphas)].copy()
        if target is not None:  # mimic the real early exit: NaN after the first hit
            hit = np.nonzero(out < target)[0]
            if len(hit):
                out[hit[0] + 1 :] = np.nan
        return out

    def update_one(self, features, element_indices, alpha=None, mode="convex"):
        self.updates.append(alpha)
        self.modes.append(("update", mode))

    def alpha_search_update(self, features, element_indices, alphas, target, mode="additive"):
        """As GMMUQModel: trial_gamma, then update_one with the chosen alpha."""
        trial = self.trial_gamma(features, element_indices, alphas, target=target, mode=mode)
        ok = np.nonzero(trial < target)[0]
        alpha = alphas[ok[0]] if len(ok) else alphas[-1]
        self.update_one(features, element_indices, alpha=alpha, mode=mode)
        return float(alpha), bool(len(ok))

    def gamma_from_features(self, features, element_indices):
        self.n_gamma_calls += 1
        return np.broadcast_to(np.asarray(self.gamma_after, dtype=float), (len(features),)).copy(), None

    def inflate(self, features, element_indices, target, max_rounds=3, max_factor=4.0):
        self.inflations.append((target, max_rounds, max_factor))
        n, factors, after = self._inflate
        return types.SimpleNamespace(n_rounds=n, factors=list(factors), gamma_max_after=after)

    def save(self, path):
        atomic_savez(path, updates=np.array(self.updates, dtype=float))


# a sidecar hash for collectors whose pristine tower is a fake (no source file)
FAKE_META = {"artifact_sha256": "fake-sha256"}


class FakePristine:
    """Constant gamma, or a per-atom vector."""

    def __init__(self, gamma=0.5):
        self.gamma = gamma

    def gamma_from_features(self, features, element_indices):
        return np.broadcast_to(np.asarray(self.gamma, dtype=float), (len(features),)).copy(), None

    def reinvert_all(self, elements=None):
        """run_hal re-inverts the towers at set-up; nothing to do for a constant gamma."""


class _Base:
    def __init__(self, live):
        self.gmm_uq_model = live


class FakeHALCalc:
    """Results as HALCalculator produces them; set .gamma / .ratio between steps."""

    def __init__(self, n, live=None):
        self.n = n
        self.base = _Base(live)
        self.gamma = np.ones(n)
        self.ratio = np.inf
        self.tau_r = 0.1
        self.features = None  # per-atom features; zeros (n, 4) when None

    @property
    def results(self):
        n = self.n
        return {
            "energy": 0.0, "forces": np.zeros((n, 3)), "stress": np.zeros(6),
            "gamma": self.gamma,
            "features": np.zeros((n, 4)) if self.features is None else self.features,
            "tau_r": self.tau_r,
            "f_norm_ema": 1.0, "g_norm_ema": 1.0, "d_min": 2.5,
            "bond_ratio_min": self.ratio, "wall_energy": 0.0, "grace_energy": -1.0,
            "grace_forces": np.zeros((n, 3)), "grace_stress": np.zeros(6),
        }


def make_walker_dir(path, atoms, element_indices, features, gamma_pristine, *,
                    feature_gamma_min=1.0, feature_meta=None, finalize=True):
    """A real walker directory (frames + sidecar) written by HALCollector from given data.

    ``features[k]`` ([n, D]) and ``gamma_pristine[k]`` ([n]) are frame k's collected
    features and pristine gammas; positions are shifted a little per frame.
    """
    from tensorpotential.uq.hal.collector import HALCollector

    atoms = atoms.copy()
    calc = FakeHALCalc(len(atoms), None)
    calc.gamma = np.full(len(atoms), 2.0)
    pristine = FakePristine()
    col = HALCollector(atoms, calc, pristine=pristine, element_indices=element_indices,
                       output_dir=str(path), equil_steps=0, trigger_gap=1, update="none",
                       remove_update_work=False, history_every=0,
                       feature_gamma_min=feature_gamma_min,
                       feature_meta=FAKE_META if feature_meta is None else feature_meta)
    for k, (f, g) in enumerate(zip(features, gamma_pristine)):
        calc.features, pristine.gamma = np.asarray(f), np.asarray(g, dtype=float)
        atoms.positions[0, 0] += 0.01 * (k + 1)
        col()
    if finalize:
        col.finalize("max_steps")
    return col
