"""Time one HAL step vs one plain compute_uq call. Not part of the test suite.

Usage:
  CUDA_VISIBLE_DEVICES=0  $PY tests/benchmarks/bench_hal_step.py --model-dir ~/.cache/grace/checkpoints/GRACE-1L-SMAX-large
  CUDA_VISIBLE_DEVICES=-1 $PY tests/benchmarks/bench_hal_step.py --model-dir ... --sizes 64 256
"""

import argparse
import os
import time

import numpy as np
from ase import units
from ase.build import bulk, molecule
from ase.md.langevin import Langevin

from tensorpotential.uq.factories import get_gmm_uq_calculator
from tensorpotential.uq.hal import HALCalculator, RampedLangevin

REPEAT = {64: (2, 2, 4), 256: (4, 4, 4), 1024: (8, 8, 4)}


def make_atoms(case, n):
    """'al': pure Al (no pair inside the ZBL range). 'al+ch4': Al with a CH4 molecule in a void,
    so C-H pairs (1.09 A) sit inside the ZBL/reject cutoff every step."""
    atoms = bulk("Al", "fcc", a=4.05, cubic=True) * REPEAT[n]
    if case == "al+ch4":
        centre = atoms.cell.diagonal() / 2
        del atoms[np.linalg.norm(atoms.positions - centre, axis=1) < 3.0]
        ch4 = molecule("CH4")
        ch4.positions += centre - ch4.positions.mean(axis=0)
        atoms += ch4
    atoms.rattle(0.05, seed=0)
    return atoms


def median_time(fn, n=30, warmup=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 256, 1024])
    p.add_argument("--cases", nargs="+", default=["al", "al+ch4"], choices=["al", "al+ch4"])
    p.add_argument("--mode", default=None, help="TPCalculator mode (uniform/diverse)")
    a = p.parse_args()
    d = os.path.expanduser(a.model_dir)
    kw = {"mode": a.mode} if a.mode else {}
    base = get_gmm_uq_calculator(model_yaml=f"{d}/model.yaml", checkpoint=f"{d}/checkpoint",
                                 gmm_artifact_path=f"{d}/gmm_artifacts.npz", **kw)
    print(f"{'case':>7} {'N':>6} {'compute_uq ms':>14} {'HAL nvt ms':>11} {'HAL ramp ms':>12} "
          f"{'overhead nvt':>13} {'overhead ramp':>14}")
    n_warm_total = 5 + 30
    for case in a.cases:
        for n in a.sizes:
            atoms = make_atoms(case, n)
            rng = np.random.default_rng(0)

            def plain():
                atoms.positions += rng.normal(0, 1e-4, atoms.positions.shape)
                base.calculate(atoms, ["energy", "forces", "stress"], ["positions"])

            t_plain = median_time(plain)
            timings = []
            for ramp in (False, True):
                at = atoms.copy()
                at.calc = hal = HALCalculator(base)
                if ramp:
                    dyn = RampedLangevin(at, 2 * units.fs, temperature_K=600, friction=0.01 / units.fs,
                                         n_steps=10_000, rng=np.random.default_rng(1))
                else:
                    dyn = Langevin(at, 2 * units.fs, temperature_K=600, friction=0.01 / units.fs,
                                   rng=np.random.default_rng(1))
                timings.append(median_time(lambda: dyn.run(1)))
                extra = hal.n_base_calls - n_warm_total
                print(f"# {case} N={len(atoms)} ramp={ramp}: base calls = steps + {extra}")
                if extra not in (1, 2):
                    print("# WARNING: expected one base call per step plus one or two at the start")
            t_nvt, t_ramp = timings
            print(f"{case:>7} {len(atoms):>6} {1e3 * t_plain:>14.2f} {1e3 * t_nvt:>11.2f} {1e3 * t_ramp:>12.2f} "
                  f"{100 * (t_nvt / t_plain - 1):>12.1f}% {100 * (t_ramp / t_plain - 1):>13.1f}%")


if __name__ == "__main__":
    main()
