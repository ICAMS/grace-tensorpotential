# AlLi-1L-small: GRACE-1L-small for Al-Li with CALM uncertainty artifact

Tutorial model for `grace_uq` and HAL (`grace_uq hal`). Small, fast to fit, in-distribution on the shipped training set, and stable in MD at 300-600 K.
It is a demonstration model, not a production Al-Li potential.

## Files

| file | content |
|---|---|
| `model.yaml` | GRACE_1LAYER_latest, elements Al and Li, cutoff 5.0 A, float32 |
| `checkpoint.index`, `checkpoint.data-00000-of-00001` | best-test-loss weights, optimizer state stripped (`grace_utils resave_checkpoint`) |
| `gmm_artifacts.npz` | CALM uncertainty artifact (built with `grace_uq build`) |
| `input.yaml` | the gracemaker input used for the fit (data path made relative) |
| `train.pkl.gz` | the 2938 training structures, trimmed to the columns `ase_atoms`, `energy`, `forces`, `energy_corrected` |

Checkpoint prefix for the tools is `checkpoint` (no suffix).

## Data

`train.pkl.gz` is the AlLi-TRAIN set: 2938 DFT structures, 34708 Al and 42952 Li atoms, 2 to 192 atoms per structure (mean 26.4), no stress labels.
Single-atom reference energy is 0 (`energy_corrected` equals `energy`; least-squares E0 on the training set would be Al -3.765 eV, Li -1.892 eV).
This is the same pair of files (train 2938, test 200) that was used for a PACE-1.0 reference fit; no validation split exists, so the 200-structure AlLi-TEST set served both as the monitor set and as the reported test set (the best epoch was the last epoch, so there is no selection bias).
The test set is not shipped. The DFT settings are not recorded in the data files.
Other columns of the original pickle (composition, formation energies, `energy_corrected_orig`, ...) are not needed by gracemaker or `grace_uq build` and were dropped.

## Fit

Gracemaker, preset `GRACE_1LAYER_latest` with `lmax 4, n_rad_max 32, prod_func_n_max 32, max_order 4, n_mlp_dens 10, n_rad_base 8`, float32, cutoff 5.0 A.
Adam, learning rate 0.008 cosine-decayed to 1e-4, EMA 0.99, batch 16, Huber(0.01) loss, energy weight 16, forces weight 32, 20056 updates (109 epochs), about 17 minutes on one RTX 3060. Full details in `input.yaml`.

Test accuracy (AlLi-TEST, 200 structures):

| | RMSE | MAE |
|---|---|---|
| energy | 6.09 meV/atom | 2.85 meV/atom |
| forces (components) | 25.3 meV/A | 13.8 meV/A |

## CALM artifact

```
grace_uq build --model-yaml model.yaml --checkpoint checkpoint --train-data train.pkl.gz \
               --rp-dim 32 --n-clusters 1 --artifact-path gmm_artifacts.npz
```

Feature dimension D = 34 (`--rp-dim 32` random-projection features plus two density channels: the log norms of the full basis and of its single invariant block), one cluster per element, regularisation 1e-6, robust-MAD threshold.
theta is the interpolation threshold (`eff_interp_thresholds`); the fraction gamma > 1 is measured on the training set; theta^2/d_eff compares the threshold to the effective dimension of the training scatter.

| element | atoms | theta | frac gamma > 1 (train) | d_eff | theta^2/d_eff |
|---|---|---|---|---|---|
| Al | 34708 | 9.64 | 0.049 | 32.8 | 2.83 |
| Li | 42952 | 8.57 | 0.080 | 32.9 | 2.23 |

On the test set the fraction gamma > 1 is 0.041 (Al) and 0.070 (Li), max gamma 2.95.
Inspect with `grace_uq info gmm_artifacts.npz`.
Re-exporting a SavedModel from the files in this folder: `grace_uq build --model-yaml model.yaml --checkpoint checkpoint --artifact-path gmm_artifacts.npz`.

## Usage notes

- Python: `get_gmm_uq_calculator("model.yaml", "checkpoint", "gmm_artifacts.npz")` from `tensorpotential.uq.factories`.
- HAL start structure used for testing: row 111 of `train.pkl.gz` (Al81Li27, 108 atoms, cubic cell 12.08 A, max gamma 0.52). Because the model is stable and in-distribution, the default bias strength (`--tau 0.15`) collects nothing in 2000 steps at 300 K. `--tau 0.5` collects in every ensemble (one run each: nvt 3, nvt_ramp 5, npt 23, npt_stress 25; one short-bond rejection in npt); alpha stays <= 0.5 and no inflation is needed. Counts vary between GPU runs; see the "Choosing tau" section of the UQ workflows docs page.

## Licence

Same as the repository.
