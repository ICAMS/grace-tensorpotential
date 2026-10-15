# Pretrained GRACE foundation models {#pretrained-grace-foundation-models}

GRACE foundation models are pre-trained on large datasets that cover most of the periodic table.
You can use them directly in [ASE](quickstart.md#usage-in-ase) and [LAMMPS](quickstart.md#usage-in-lammps),
or [fine-tune](#fine-tuning-foundation-models) them on your own data.

!!! note "Use the full model name"
    The tables list the **full names**, for example `GRACE-2L-OMAT-large-ft-E`. Use them
    in `grace_fm(...)`, in `grace_models` and in LAMMPS (the model folder is
    `~/.cache/grace/<full name>/`). Old short names are listed in
    [Short and old names](#short-and-old-names).

**Which model?** For general use we recommend the **SMAX-OMAT** models. More layers give higher
accuracy but cost more time: 1L models are the fastest, 3L models the most accurate.

## What the models offer {#model-features}

| Models | Precision | UQ (γ) | Kokkos | Cutoff |
| :--- | :--- | :---: | :---: | :--- |
| medium and large 1L/2L models (SMAX, OMAT, OAM) | fp32; `-fp64` variant | ✓ | ✓ | 6 Å (SMAX: 5.0–7.5 Å) |
| 3L models | fp32 only | ✓ | ✓ | 6 Å |
| small models (**deprecated**): `GRACE-1L-OMAT`, `GRACE-2L-OMAT`, `GRACE-1L-OAM`, `GRACE-2L-OAM`, `GRACE-FS-OMAT`, `GRACE-FS-OAM` | fp64 only | — | — | 6 Å |

!!! warning "Small models are deprecated"
    The six small models will be retired. Use the medium or large models for new work: they are
    more accurate and also give UQ (γ) and Kokkos support.

* **UQ (γ).** The model computes the [CALM extrapolation grade](uq.md#calm) γ for every atom.
  γ ≤ 1 means that the atomic environment is similar to the training data. `grace_fm` switches
  UQ **off** by default, because it makes the calculator slower. Switch it on with
  `grace_fm(name, enable_uq=True)` or later with `calc.enable_uq()`; γ is then in
  `calc.results["gamma"]`. The CALM artifact (`gmm_artifacts.npz`) comes with the checkpoint.
  See [UQ workflows](uq_workflows.md) for γ in ASE and LAMMPS.
* **Kokkos.** A `kokkos.npz` file for the TensorFlow-free LAMMPS-Kokkos pair styles
  (`grace/1l/kk`, `grace/2l/kk`, `grace/3l/kk`) is available. It is a **separate download**:
  `grace_models download <name> --kokkos`. One file serves both precisions; the pair style
  chooses the compute precision (see
  [LAMMPS Kokkos](quickstart.md#lammps-grace-1l-grace-2l-grace-3l-kokkos-no-tensorflow)).
  For 3L models use `grace/3l/kk` (fp32 network, fp64 geometry) or `grace/3l/kk/fp32`.
* **Precision.** The plain name (e.g. `GRACE-2L-OMAT-large-ft-E`) is the **fp32** model: about
  2× faster and half the memory of fp64, with a negligible difference (γ changes by less than
  5×10⁻⁵, relative). For tight finite differences or bit-reproducible work, use the fp64
  variant: add `-fp64` to the name (e.g. `GRACE-2L-OMAT-large-ft-E-fp64`). 3L models are fp32
  only.
* **Parallel runs.** `GRACE-2L-OMAT` and `GRACE-2L-OAM` (small, deprecated) cannot be split over several MPI
  ranks with `pair_style grace`. All other models can.

`grace_models list` shows the same information for every model, and `grace_models info <name>`
shows it for one model.

## SMAX models {#smax-models}

Reference: [arXiv:2602.23489](https://arxiv.org/abs/2602.23489)

**SMAX** (**maximum entropy**) is a dataset made with a structure generator that maximizes the
information entropy over many chemical elements. Most datasets focus on low-energy, near-equilibrium
structures. SMAX samples a much broader range of structures on purpose. This gives a robust model
for large strains and phase transformations, defects in complex alloys and reaction barriers in
catalysis.

SMAX models use an **element-dependent cutoff** between **5.0 Å and 7.5 Å**.

The **SMAX-OMAT** models are trained on SMAX and OMat24. They combine the robustness of SMAX with
the accuracy of OMat24 for energies and forces. They are our general-purpose recommendation.

| Full name | Layers | Size | κSRME | UQ | Kokkos | Training data |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **GRACE-2L-SMAX-OMAT-medium** | 2L | medium | 0.197 | ✓ | ✓ | SMAX + OMat24 |
| **GRACE-2L-SMAX-OMAT-large** | 2L | large | 0.191 | ✓ | ✓ | SMAX + OMat24 |
| GRACE-2L-SMAX-medium | 2L | medium | 0.469 | ✓ | ✓ | SMAX |
| GRACE-2L-SMAX-large | 2L | large | 0.444 | ✓ | ✓ | SMAX |
| **GRACE-1L-SMAX-OMAT-large** | 1L | large | 0.338 | ✓ | ✓ | SMAX + OMat24 |
| GRACE-1L-SMAX-large | 1L | large | 0.696 | ✓ | ✓ | SMAX |

All SMAX models also have an `-fp64` variant.

## OMAT models {#omat-models}

Reference: [npj Comput. Mater.](https://www.nature.com/articles/s41524-026-01979-1) (open access)

These models are trained on the [OMat24](https://huggingface.co/datasets/fairchem/OMAT24#omat24-dataset)
dataset. The `-ft-E` models are the `-base` models, fine-tuned with more weight on energies. All
use a fixed **6 Å cutoff**.

| Full name | Layers | Size | κSRME | UQ | Kokkos | Notes |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **GRACE-3L-OMAT-large** | 3L | large | 0.141 | ✓ | ✓ | fp32 only |
| GRACE-2L-OMAT-large-base | 2L | large | **0.165** | ✓ | ✓ | base |
| **GRACE-2L-OMAT-large-ft-E** | 2L | large | 0.186 | ✓ | ✓ | fine-tuned on energies |
| GRACE-2L-OMAT-medium-base | 2L | medium | 0.212 | ✓ | ✓ | base |
| **GRACE-2L-OMAT-medium-ft-E** | 2L | medium | 0.217 | ✓ | ✓ | fine-tuned on energies |
| GRACE-2L-OMAT | 2L | small | 0.288 | — | — | **deprecated**; fp64 only; no MPI split |
| GRACE-1L-OMAT-large-base | 1L | large | **0.354** | ✓ | ✓ | base |
| **GRACE-1L-OMAT-large-ft-E** | 1L | large | 0.383 | ✓ | ✓ | fine-tuned on energies |
| GRACE-1L-OMAT-medium-base | 1L | medium | 0.380 | ✓ | ✓ | base |
| **GRACE-1L-OMAT-medium-ft-E** | 1L | medium | 0.417 | ✓ | ✓ | fine-tuned on energies |
| GRACE-1L-OMAT | 1L | small | 0.398 | — | — | **deprecated**; fp64 only |
| GRACE-FS-OMAT | FS | small | — | — | — | **deprecated**; Finnis-Sinclair-like local model; fp64 only |

## OAM models {#oam-models}

Reference: [npj Comput. Mater.](https://www.nature.com/articles/s41524-026-01979-1) (open access)

These models are first trained on **OMat24** and then fine-tuned on the
[sAlex](https://huggingface.co/datasets/fairchem/OMAT24#salex-dataset) dataset (10.4M structures)
together with the
[MPtrj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842?file=41619375)
dataset (1.58M structures). All use a fixed **6 Å cutoff**. The larger models have `-ft-AM` in
their full name.

| Full name | Layers | Size | F1 | κSRME | UQ | Kokkos | Notes |
| :--- | :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **GRACE-3L-OMAT-large-ft-AM** | 3L | large | 0.925 | **0.121** | ✓ | ✓ | fp32 only |
| **GRACE-2L-OMAT-large-ft-AM** | 2L | large | 0.889 | **0.168** | ✓ | ✓ | |
| GRACE-2L-OMAT-medium-ft-AM | 2L | medium | 0.881 | 0.200 | ✓ | ✓ | |
| GRACE-2L-OAM | 2L | small | 0.880 | 0.294 | — | — | **deprecated**; fp64 only; no MPI split |
| **GRACE-1L-OMAT-large-ft-AM** | 1L | large | 0.815 | **0.377** | ✓ | ✓ | |
| GRACE-1L-OMAT-medium-ft-AM | 1L | medium | 0.800 | 0.411 | ✓ | ✓ | |
| GRACE-1L-OAM | 1L | small | 0.824 | 0.516 | — | — | **deprecated**; fp64 only |
| GRACE-FS-OAM | FS | small | — | — | — | — | **deprecated**; Finnis-Sinclair-like local model; fp64 only |

## Short and old names {#short-and-old-names}

??? info "Short aliases and renamed models"
    Old input files may use these short aliases:

    | Alias | Full name |
    | :--- | :--- |
    | `GRACE-2L-OMAT-L` | `GRACE-2L-OMAT-large-ft-E` |
    | `GRACE-2L-OMAT-M` | `GRACE-2L-OMAT-medium-ft-E` |
    | `GRACE-1L-OMAT-L` | `GRACE-1L-OMAT-large-ft-E` |
    | `GRACE-1L-OMAT-M` | `GRACE-1L-OMAT-medium-ft-E` |
    | `GRACE-2L-OAM-L` | `GRACE-2L-OMAT-large-ft-AM` |
    | `GRACE-2L-OAM-M` | `GRACE-2L-OMAT-medium-ft-AM` |
    | `GRACE-1L-OAM-L` | `GRACE-1L-OMAT-large-ft-AM` |
    | `GRACE-1L-OAM-M` | `GRACE-1L-OMAT-medium-ft-AM` |

    The aliases work in `finetune_foundation_model`, `grace_models info` and
    `grace_models update`. `grace_fm(...)`, `grace_models download` and
    `grace_models checkpoint` need the full name.

    `GRACE-2L-UEA-OMAT-medium` is the old name of `GRACE-2L-SMAX-OMAT-medium`. It is not
    in the registry any more; use the new name.

---

## Downloading foundation models {#downloading-foundation-models}

Models are downloaded automatically the first time you use them. You can also manage them with
the `grace_models` tool:

| Command | What it does |
| :--- | :--- |
| `grace_models list` | Table of all models and their capabilities; `-v` for the full description of each model |
| `grace_models info <name>` | Capabilities of one model; if it is downloaded, also what the files contain |
| `grace_models download <name>` | Download the model (SavedModel); add `--kokkos` to also get `kokkos.npz` |
| `grace_models checkpoint <name>` | Download only the checkpoint (needed for fine-tuning) |
| `grace_models update <name>` or `--all` | Download again the cached copies that are out of date |

Models are stored in `$HOME/.cache/grace`. Set the `GRACE_CACHE` environment variable to use
another folder.

Upgrading `tensorpotential` does not refresh models that are already in the cache. Use
`grace_models update`, see [Updating cached models](utilities.md#updating-cached-models).

---

## Fine-tuning foundation models {#fine-tuning-foundation-models}

Fine-tuning starts from the model's **checkpoint**, not from the SavedModel. `gracemaker`
downloads the checkpoint when it needs it (or run `grace_models checkpoint <name>` before).

**3-layer models.** There is no 3-layer [preset](presets.md) yet. To get a 3L model for your
system, fine-tune `GRACE-3L-OMAT-large` or `GRACE-3L-OMAT-large-ft-AM`.
{#three-layer-3l-models}

### Write `input.yaml` with the wizard {#automatic-input-file-generation}

The quickest way is the interactive wizard:

```bash
gracemaker -t
```

Choose **finetune foundation model**. The wizard asks for the number of layers (FS, 1L, 2L or
3L), the training dataset of the model (OAM, OMAT or SMAX-OMAT), the size, the variant (e.g.
`base` or `ft-E`) and, if there is one, the fp64 variant. Pure SMAX models are not in the
wizard; set them [by hand](#manual-setup). For fine-tuning it writes:

* `finetune_foundation_model: <name>`, `reduce_elements: True` and `shift: auto`
  (see [Manual setup](#manual-setup));
* `trainable_variable_names` for [frozen-weights fine-tuning](#frozen-weights-fine-tuning),
  which is the default mode (or nothing for "naive" fine-tuning, where all weights change);
* `learning_rate: 0.001`, `eval_init_stats: True` and a learning-rate schedule
  (see [Learning rate](#learning-rate-reduction)).

Frozen-weights mode in the wizard works only for 1L, 2L and 3L models. For an FS model, choose
naive fine-tuning.

### Manual setup {#manual-setup}

Add the foundation model to the `potential` section of `input.yaml`:

```yaml
potential:
  finetune_foundation_model: GRACE-2L-SMAX-OMAT-medium   # full name (or alias)
  reduce_elements: True   # keep only the elements of your dataset
  shift: auto             # align the model energies with your DFT data
```

| Key | Meaning |
| :--- | :--- |
| `finetune_foundation_model` | Model to start from. `gracemaker` loads its checkpoint and `model.yaml`. |
| `reduce_elements` | `True`: the new model has only the elements of your dataset (smaller and faster). `False` (default): it keeps all elements of the foundation model. |
| `shift: auto` | Fits per-element energy shifts between the foundation model and your data before training, see [below](#automatic-energy-shift-correction-shift-auto). |

The rest of `input.yaml` (data, loss, optimizer) is the same as for a normal fit, see
[Input file](inputfile.md).

### Align energies with your DFT data: `shift: auto` {#automatic-energy-shift-correction-shift-auto}

Different DFT settings shift the energy by an almost constant amount per element. With
`potential: shift: auto`, `gracemaker` compares the foundation-model energies with your DFT
energies on about 50 training structures, fits one energy shift per element, and adds it to the
model before training. The fit then starts with a small energy error. Keep
`data: reference_energy` as usual.

### Learning rate {#learning-rate-reduction}

The foundation model is already a good starting point, so use a **small learning rate**
(for example `0.001` or `0.0001`). Set `eval_init_stats: True` to see the errors of the
foundation model on your data before the first step. This is what the wizard writes:

```yaml
fit:
  eval_init_stats: True       # metrics before training
  optimizer: Adam
  opt_params: {learning_rate: 0.001, use_ema: True, ema_momentum: 0.99,
               weight_decay: null, clipnorm: 1.0}
  scheduler: cosine_decay
  scheduler_params: {minimum_learning_rate: 8.3e-05}   # about learning_rate / 12
```

!!! note "`minimum_learning_rate`"
    The key is `minimum_learning_rate`. Its default is `1.0e-4`. If you lower `learning_rate`
    to `1.0e-4` or below, also set `minimum_learning_rate` below it; otherwise the schedule
    does not decrease. A misspelled key such as `minimal_learning_rate` is ignored, with a
    warning in the log. All schedulers are listed in
    [Learning-rate schedulers](inputfile.md#learning-rate-schedulers).

Without `maxiter` or `target_total_updates`, a fine-tuning run with Adam makes 10 000 updates.

### Frozen-weights fine-tuning {#frozen-weights-fine-tuning}

By default, fine-tuning updates all weights ("naive" fine-tuning). On a small dataset this can
make the model forget what it learned before (catastrophic forgetting). To avoid this, train only
some of the weights and keep the rest frozen.

List the weights to train in `fit: trainable_variable_names`. A weight is trained if its name
contains one of the strings; all other weights stay frozen. The recommended lists train only the
ACE expansion coefficients (`reducing_` weights). These depend on the element of the central
atom, so the elements that are not in your data keep their pre-trained behavior.

| Model | `trainable_variable_names` |
| :--- | :--- |
| 1L | `["rho/reducing_"]` |
| 2L | `["I2/reducing_", "rho/reducing_", "I1/reducing_"]` |
| 3L | `["rho1/reducing_", "rho2/reducing_", "rho3/reducing_", "eq1/reducing_", "eq2/reducing_"]` |

For example, for a 2L model:

```yaml
fit:
  trainable_variable_names: ["I2/reducing_", "rho/reducing_", "I1/reducing_"]
```

??? info "What the 2L names mean"
    * `rho/reducing_`: ACE expansion coefficients of the first layer.
    * `I1/reducing_`: ACE expansion coefficients of the message from the first to the second
      layer.
    * `I2/reducing_`: ACE expansion coefficients of the second layer.
    * `LinMLPOut2ScalarTarget_`: the linear + MLP energy readout (frozen in the lists above).

To see all weight names of a model, print its summary:

```bash
grace_utils -p ~/.cache/grace/checkpoints/<MODEL-NAME>/model.yaml summary -v 1
```

### Uncertainty of the fine-tuned model {#uq-after-fine-tuning}

Fine-tuning changes the weights, so the CALM artifact of the foundation model does not apply
to the new model. Build a new one in the fit folder with `grace_uq build`, see
[UQ workflows, step 0](uq_workflows.md#0-build-the-calm-artifact-for-a-fitted-model).

---

## User-defined models {#user-defined-models}

You can add your own models to the local registry. Create the file
`$HOME/.cache/grace/models_registry.yaml` (or `$GRACE_CACHE/models_registry.yaml`):

```yaml
FULL-MODEL-NAME:
  path: /path/to/saved_model
  checkpoint_path: /path/to/checkpoint/  # folder with model.yaml and checkpoint.index
  description: some description
  license: some license
```

Then `grace_fm("FULL-MODEL-NAME")`, `finetune_foundation_model: FULL-MODEL-NAME` and
`grace_models` work with this model. `grace_models list` shows it under "Extra models". An entry
cannot replace a built-in model with the same name.
