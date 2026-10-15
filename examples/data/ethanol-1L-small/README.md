# ethanol-1L-small: GRACE-1L-small for an isolated ethanol molecule with CALM uncertainty artifact

Tutorial model for `grace_uq` and HAL (`grace_uq hal --ensemble cluster`). Small, fast to fit, stable in molecular dynamics at 300 and 600 K.
It is a demonstration model, not a production potential.

## Files

| file | content |
|---|---|
| `model.yaml` | GRACE_1LAYER_latest, elements C, H, O, cutoff 5.0 A, float32 |
| `checkpoint.index`, `checkpoint.data-00000-of-00001` | best-validation-loss weights, optimizer state stripped (`grace_utils resave_checkpoint`) |
| `gmm_artifacts.npz` | CALM uncertainty artifact (built with `grace_uq build`) |
| `input.yaml` | the gracemaker input used for the fit (data path made relative) |
| `train.pkl.gz` | the 800 training molecules, trimmed to the columns `ase_atoms`, `energy`, `forces`, `energy_corrected` |

Checkpoint prefix for the tools is `checkpoint` (no suffix).

## Data

MD17-style ethanol: 1000 C2H6O configurations of an isolated molecule (non-periodic, 9 atoms each, no stress labels), split 800 / 100 / 100 (train / test / validation) with `numpy.random.default_rng(42)`. Only the 800-molecule training split is shipped.
`energy` is the total energy (about -4209 eV); `energy_corrected` is the energy relative to single-atom references. The fit used `reference_energy: auto` (per-element least-squares E0 on the training set: C -205.344, H -616.033, O -102.672 eV, which gracemaker recomputes from `energy`).
The earlier PACE ladder fits of the same configurations used all 1000 structures with no holdout, so a PACE number on this validation split would be in-sample for PACE. The DFT settings are not recorded in the data files.
Other columns of the original pickle (`NUMBER_OF_ATOMS`, `pbc`, `tp_atoms`, ...) are not needed and were dropped.

## Fit

Gracemaker, preset `GRACE_1LAYER_latest` with `lmax 4, n_rad_max 32, prod_func_n_max 32, max_order 4, n_mlp_dens 10, n_rad_base 8`, float32, cutoff 5.0 A.
Adam, learning rate 0.008 cosine-decayed to 1e-4, EMA 0.99, batch 16, Huber(0.01) loss, energy weight 16, forces weight 32, 12500 updates (250 epochs), about 25 minutes on one RTX 3060. The 100-molecule validation split was used for checkpoint selection (`input.yaml` lists it as the held-out set; it is not shipped). Full details in `input.yaml`.

Accuracy:

| split | energy RMSE | force-component RMSE |
|---|---|---|
| test (100) | 0.31 meV/atom (2.8 meV/molecule) | 17.7 meV/A |
| validation (100) | 0.32 meV/atom | 18.7 meV/A |

## CALM artifact

```
grace_uq build --model-yaml model.yaml --checkpoint checkpoint --train-data train.pkl.gz \
               --rp-dim 32 --n-clusters 1 --artifact-path gmm_artifacts.npz
```

Feature dimension D = 34 (`--rp-dim 32` random-projection features plus two density channels: the log norms of the full basis and of its single invariant block), one cluster per element, regularisation 1e-6, robust-MAD threshold.
theta is the interpolation threshold (`eff_interp_thresholds`); the fraction gamma > 1 is measured on the training set; theta^2/d_eff compares the threshold to the effective dimension of the training scatter.

| element | atoms | theta | frac gamma > 1 (train) | d_eff | theta^2/d_eff |
|---|---|---|---|---|---|
| C | 1600 | 8.90 | 0.019 | 32.7 | 2.42 |
| H | 4800 | 8.69 | 0.026 | 33.0 | 2.29 |
| O | 800 | 8.84 | 0.020 | 32.7 | 2.39 |

On the test set the fraction gamma > 1 is 0.021.
Inspect with `grace_uq info gmm_artifacts.npz`.
Re-exporting a SavedModel from the files in this folder: `grace_uq build --model-yaml model.yaml --checkpoint checkpoint --artifact-path gmm_artifacts.npz`.

## Usage notes

- Python: `get_gmm_uq_calculator("model.yaml", "checkpoint", "gmm_artifacts.npz")` from `tensorpotential.uq.factories`.
- Stability: 20000-step Langevin runs (0.5 fs) from a training and a test molecule at 300 and 600 K keep the molecule intact; NVE total-energy std over 2 ps is 1.6 meV.
- HAL example: `grace_uq hal --ensemble cluster -T 300 --n-steps 20000` from row 0 of `train.pkl.gz` collects few structures at the default `--tau 0.15` (2 in one run) and 6-9 at `--tau 0.25`, none rejected or fragmented; counts vary between GPU runs. See the "Choosing tau" section of the UQ workflows docs page.
- Thermostat: plain ASE Langevin with `fixcm=True` runs hotter than the target for small molecules with mixed masses (measured mean 378-404 K at a 300 K target); this is a thermostat effect, not a model problem. HAL (`nvt`, `nvt_ramp`, `cluster`) uses `fixcm=False` with a centre-of-mass momentum reset every step; cluster walkers at 300 K then measure 300-315 K (`temperature_mean_K` over the run).

## Licence

Same as the repository.
