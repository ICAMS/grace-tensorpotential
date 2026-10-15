## Available presets

A preset is a model architecture with default hyperparameters. Choose it in `input.yaml`
(`potential: preset:`) or with `gracemaker -t`.

| | `FS` | `GRACE_1LAYER_latest` (default) | `GRACE_2LAYER_latest` |
|---|---|---|---|
| Layers (message passing) | 1 | 1 | 2 |
| Basis functions | stars (local) | stars (local) | stars and trees (semi-local) |
| Radial functions | linear | MLP | MLP |
| Readout | Finnis-Sinclair-like | MLP | MLP |
| ASE | TensorFlow or C++ ([GRACE/FS](quickstart.md#ase-gracefs)) | TensorFlow | TensorFlow |
| LAMMPS | `grace/fs`, `grace/fs/kk` (C++, Kokkos) | `grace`, `grace/1l/kk` | `grace/2layer/*`, `grace/2l/kk` |
| Extrapolation grade | D-optimality ([active set](quickstart.md#build-active-set-for-gracefs-only)) | [CALM](uq.md) | [CALM](uq.md) |

All presets use a chemical embedding, so the number of elements is not limited.

* **FS** is closest to the classical PACE model. It can be [exported](quickstart.md#gracefs) to a
  YAML file for the C++ implementation, which needs no TensorFlow and runs with MPI in LAMMPS.
* **GRACE_1LAYER_latest** adds MLP radial functions and an MLP readout. The interactions stay within
  one cutoff, so the model parallelizes with MPI in LAMMPS without extra cost.
* **GRACE_2LAYER_latest** adds a second layer of message passing: more accurate, more expensive.
* **GRACE-3L:** there is no 3-layer preset yet. For a 3-layer model,
  [fine-tune](foundation.md#fine-tuning-foundation-models) a GRACE-3L
  [foundation model](foundation.md#three-layer-3l-models) on your data, e.g. `GRACE-3L-OMAT-large`
  (`gracemaker -t` offers it).

!!! note "`_latest` presets change; older versions keep a fixed name"
    `GRACE_1LAYER_latest` and `GRACE_2LAYER_latest` always point to the currently recommended
    architecture, which is updated from time to time. A new version can give a different model
    for the same `input.yaml`. Older versions stay available under a fixed name
    `<NAME>_v<version>_<year>`: `GRACE_1LAYER_v1_24` and `GRACE_2LAYER_v1_24` are the 2024
    architectures. To keep exactly the same architecture across updates, use such a fixed name.
    A fitted model is not affected: its architecture is stored in `model.yaml`.

---

## Model hyperparameters {#models-hyperparameters}

Set the preset and change its parameters in `input.yaml`:

```yaml
cutoff: 6   # the cutoff radius (Å) of the model; not a preset parameter

potential:
  preset: GRACE_1LAYER_latest   # FS, GRACE_1LAYER_latest, GRACE_2LAYER_latest
  kwargs: {n_rad_max: 32, n_mlp_dens: 12}   # override preset parameters
```

Parameters left out of `kwargs` take the defaults below. `gracemaker -t` offers each preset in
three sizes, `small`, `medium` and `large` ("model complexity"), and writes the values of the
tables into `kwargs` (and the suggested `cutoff`). The full list of parameters is in
[presets.py](https://github.com/ICAMS/grace-tensorpotential/blob/master/tensorpotential/potentials/presets.py).

The GRACE presets (not FS) also take:

* `zbl_cutoff` - add a ZBL core repulsion, e.g. `kwargs: {zbl_cutoff: {Mo: 1, MoNb: 2, W: 1}}`
* bond-specific cutoffs: `cutoff_dict` in `input.yaml`
  ([FAQ](faq.md#can-i-have-different-cutoffs-for-different-bond-types))

### FS

| Parameter | default | small | medium | large |
|---|---|---|---|---|
| `cutoff` (suggested) | | 7 | 7 | 7 |
| `max_order` | 4 | 3 | 4 | 4 |
| `lmax` | [5, 5, 4, 3] | [5, 5, 4] | [5, 5, 4, 3] | [5, 5, 4, 3] |
| `n_rad_max` | [20, 15, 10, 5] | [20, 15, 10] | [20, 15, 10, 5] | [20, 20, 15, 10] |
| `embedding_size` | 64 | 32 | 64 | 72 |
| `fs_parameters` | 4 terms | 2 terms | 4 terms | 5 terms |

* `basis_type` - radial basis: `"SBessel"` (default) or `"Cheb"`
* `n_rad_base` - number of radial basis functions (default 20)
* `embedding_size` - number of chemical embedding channels
* `max_order` - maximum product order of the B-functions
* `n_rad_max` - number of radial functions for each product order of the _output_ B-functions
* `lmax` - maximum l for each product order of the _input_ B-functions
* `Lmax` - maximum l for each product order of the _output_ B-functions (default `[None, 3, 0, 0]`)
* `max_sum_l` - limits the number of B-functions (default `[None, None, 6, 4]`)
* `fs_parameters` - the FS embedding $c_1 \phi_1^{m_1} + c_2 \phi_2^{m_2} + \dots$ as
  `[[c1, m1], [c2, m2], ...]` (default `[[1.0, 1.0], [1.0, 0.5], [1.0, 2], [1.0, 0.75]]`)

### GRACE_1LAYER_latest

| Parameter | default | small | medium | large |
|---|---|---|---|---|
| `cutoff` (suggested) | | 6 | 6 | 6 |
| `lmax` | 4 | 4 | 4 | 4 |
| `max_order` | 4 | 4 | 4 | 4 |
| `n_rad_base` | 10 | 8 | 8 | 10 |
| `n_rad_max` | 32 | 32 | 32 | 42 |
| `prod_func_n_max` | 64 | 32 | 64 | 64 |
| `n_mlp_dens` | 16 | 10 | 12 | 16 |

* `basis_type` - radial basis: `"Cheb"` (default) or `"SBessel"`
* `n_rad_base` - number of radial basis functions
* `cutoff_function_order` - order of the cutoff function (default 16)
* `embedding_size` - size of the chemical embedding (default 128)
* `n_rad_max` - number of radial functions
* `lmax` - maximum l of the product functions
* `max_order` - maximum product order of the B-functions
* `prod_func_n_max` - number of channels of the product functions
* `n_mlp_dens` - number of non-linear readout densities (one linear density is added)

### GRACE_2LAYER_latest

| Parameter | default | small | medium | large |
|---|---|---|---|---|
| `cutoff` (suggested) | | 6 | 6 | 6 |
| `lmax` | [4, 3] | [4, 3] | [4, 3] | [4, 3] |
| `max_order` | 4 | 4 | 4 | 4 |
| `indicator_lmax` | 3 | 1 | 1 | 3 |
| `n_rad_base` | 10 | 8 | 10 | 10 |
| `n_rad_max` | [42, 32] | [32, 32] | [42, 32] | [42, 32] |
| `prod_func_n_max` | [42, 64] | [32, 32] | [42, 64] | [42, 64] |
| `n_mlp_dens` | 16 | 12 | 16 | 16 |

The parameters of GRACE_1LAYER_latest, with per-layer values `[first layer, second layer]` for
`lmax`, `n_rad_max` and `prod_func_n_max`, plus:

* `indicator_lmax` - maximum l of the first-layer message passed to the second layer

The first values of `n_rad_max` and `prod_func_n_max` must be equal.
