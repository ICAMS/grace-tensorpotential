# **Pretrained GRACE Foundation Models**

Several GRACE models are pre-trained on large datasets.

**NOTE\!** You should use the "Full Name" column to refer to the model in LAMMPS and ASE.

## **SMAX models**

Reference: [arXiv](https://arxiv.org/abs/2602.23489)

The **SMAX** (**Maximum Entropy**) models are trained on a chemistry-agnostic dataset generated via a multicomponent
maximum information entropy structure generation protocol.

Unlike traditional datasets that focus on low-energy equilibrium structures, SMAX is constructed to deliberately sample
broad and diverse regions of configurational space. This provides a robust physical prior for atomic interactions across the entire periodic table, enabling accurate modeling of large-strain phase transformations, defects in complex alloys, and reaction barriers in catalytic systems.

**Custom Cutoffs & Interaction Ranges:** SMAX models utilize a **custom element-dependent cutoff radius** ranging from
**5.0 Å to 7.5 Å**.

**Recommendation:** For most general-purpose applications, we recommend using the **SMAX-OMAT** models.
They offer the best balance of structural robustness (from SMAX) and high-precision energy/force accuracy (from OMat24).

#### **Single-layer, local models**

| Model Name | Full Name | Size | κSRME​ | Description |
| :---- | :---- | :---- | :---- | :---- |
| GRACE-1L-SMAX-L | GRACE-1L-SMAX-large | large | 0.696 | Single-layer local (SMAX) |
| **GRACE-1L-SMAX-OMAT-L** | **GRACE-1L-SMAX-OMAT-large** | **large** | 0.338 | **Single-layer local (SMAX \+ OMat24)** |

#### **Two-layer, semilocal models**

| Model Name | Full Name | Size | κSRME​ | Description |
| :---- | :---- | :---- | :---- | :---- |
| GRACE-2L-SMAX-M | GRACE-2L-SMAX-medium | medium | 0.469 | Two-layer semi-local (SMAX) |
| GRACE-2L-SMAX-L | GRACE-2L-SMAX-large | large | 0.444 | Two-layer semi-local (SMAX) |
| **GRACE-2L-SMAX-OMAT-M** | **GRACE-2L-SMAX-OMAT-medium** | **medium** | 0.197 | **Two-layer semi-local (SMAX \+ OMat24)** |
| **GRACE-2L-SMAX-OMAT-L** | **GRACE-2L-SMAX-OMAT-large** | **large** | 0.191 | **Two-layer semi-local (SMAX \+ OMat24)** |

## OMAT models

Reference: [npj Comp. Mat.](https://www.nature.com/articles/s41524-026-01979-1), [arXiv](https://arxiv.org/abs/2508.17936)

The base models (**-OMAT**) are trained on the [OMat24](https://huggingface.co/datasets/fairchem/OMAT24#omat24-dataset) dataset.
The fine-tuned versions (**-OMAT-ft-E**) are derived from these base models by fine-tuning with more emphasis on energies.
 All models listed use a fixed **6 Å cutoff**.

#### Single-layer, local models

| Model Name             | Full Name | Size | $\kappa_\mathrm{SRME}$ | Description                    |
|:-----------------------| :--- | :--- |:-----------------------|:-------------------------------|
| **GRACE-1L-OMAT**        | GRACE-1L-OMAT | small | 0.398                  | Single-layer local             |
| GRACE-1L-OMAT-M-base | GRACE-1L-OMAT-medium-base | medium | 0.380                  | Single-layer local (base)  |
| **GRACE-1L-OMAT-M**      | GRACE-1L-OMAT-medium-ft-E | medium | 0.417                  | Single-layer local (finetuned on energy)  |
| GRACE-1L-OMAT-L-base | GRACE-1L-OMAT-large-base | large | **0.354**                 | Single-layer local (base)             |
| **GRACE-1L-OMAT-L**      | GRACE-1L-OMAT-large-ft-E | large | 0.383                  | Single-layer local (finetuned on energy)            |

#### Two-layer, semilocal models

| Model Name              | Full Name | Size | $\kappa_\mathrm{SRME}$ | Description |
|:------------------------| :--- | :--- |:-----------------------| :--- |
| **GRACE-2L-OMAT**         | GRACE-2L-OMAT | small | 0.288                  | Two-layer semi-local |
| GRACE-2L-OMAT-M-base   | GRACE-2L-OMAT-medium-base | medium | 0.212                  | Two-layer semi-local (base) |
| **GRACE-2L-OMAT-M**     | GRACE-2L-OMAT-medium-ft-E | medium | 0.217                  | Two-layer semi-local (finetuned on energy) |
| GRACE-2L-OMAT-L-base    | GRACE-2L-OMAT-large-base | large | **0.165**                  | Two-layer semi-local (base)|
| **GRACE-2L-OMAT-L**     | GRACE-2L-OMAT-large-ft-E | large | 0.186                 | Two-layer semi-local (finetuned on energy) |

***

## OAM models

Reference: [npj Comp. Mat.](https://www.nature.com/articles/s41524-026-01979-1), [arXiv](https://arxiv.org/abs/2508.17936)

These models are first pre-trained on **OMat24** and then fine-tuned on a combination of the [sAlex](https://huggingface.co/datasets/fairchem/OMAT24#salex-dataset) dataset (10.4M structures) and the [MPtraj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842?file=41619375) dataset (1.58M structures).

#### Single-layer, local models

| Model Name | Full Name                  | Size | F1    | $\kappa_\mathrm{SRME}$ | Description |
| :--- |:---------------------------| :--- |:------|:-----------------------| :--- |
| GRACE-1L-OAM | GRACE-1L-OAM               | small | 0.824 | 0.516                  | Single-layer local |
| GRACE-1L-OAM-M| GRACE-1L-OMAT-medium-ft-AM | medium | 0.800 | 0.411         | Single-layer local |
| GRACE-1L-OAM-L| GRACE-1L-OMAT-large-ft-AM  | large | 0.815 | **0.377**           | Single-layer local |

#### Two-layer, semilocal models

| Model Name | Full Name | Size | F1    | $\kappa_\mathrm{SRME}$ | Description |
| :--- |:--- | :--- |:------| :--- | :--- |
| GRACE-2L-OAM | GRACE-2L-OAM | small | 0.880 | 0.294 | Two-layer semi-local |
| GRACE-2L-OAM-M| GRACE-2L-OMAT-medium-ft-AM | medium | 0.881 | 0.200 | Two-layer semi-local |
| GRACE-2L-OAM-L| GRACE-2L-OMAT-large-ft-AM | large | 0.889 | **0.168** | Two-layer semi-local |

---

## Downloading Foundation Models

You can use the `grace_models` utility to download pre-trained GRACE models:  

- Run `grace_models list` to view the list of available models.  
- Run `grace_models download <model_name>` to download a specific model.  

These models can be used for simulations within [ASE](../quickstart/#usage-in-ase) and [LAMMPS](../quickstart/#usage-in-lammps).  

By default, all foundational models are stored into `$HOME/.cache/grace`,
but you can overwrite it with `GRACE_CACHE` environment variable.

---

## Fine-tuning foundation models

### Automatic input file generation

You can generate an `input.yaml` file for fine-tuning a foundation model by running `gracemaker -t`.

### Manual setup

Fine-tuning foundation GRACE models can only be performed using checkpoints, not saved models.
Run `grace_models list` to view the available models that include a `CHECKPOINT:` field.
You can either download this checkpoint manually using `grace_models checkpoint <MODEL-NAME>`, or it will be downloaded automatically when needed.

To fine-tune a foundation model, add the following to the `potential` section of your `input.yaml`:

```yaml
potential:
  finetune_foundation_model: GRACE-1L-OAM
  shift: auto  # automatically align FM energies with your DFT reference data
# reduce_elements: True # if True - select from original models only elements present in the CURRENT dataset

```

### Automatic energy shift correction (`shift: auto`)

When `shift: auto` is set alongside `finetune_foundation_model`, GRACEmaker automatically computes optimal per-element energy shifts that minimize the difference between the foundation model's predictions and your reference DFT data. These shifts are injected directly into the model (via `ConstantScaleShiftTarget`) before training begins.

**How it works:**

1. Selects ~50 diverse structures from the training set covering all element combinations.
2. Predicts their energies with the foundation model (prior to training).
3. Solves a least-squares problem: `A @ x = (E_DFT - E_FM)`, where `A` is the composition matrix and `x` represents per-element shifts.
4. Injects the computed shifts into the model's `ConstantScaleShiftTarget` instruction.

This is particularly useful when your DFT reference data uses different pseudopotentials or settings than the foundation model's training data, which usually results in a systematic energy offset. The correction minimizes the initial energy mismatch and leads to faster convergence.

**Note:** `shift: auto` is independent of `data::reference_energy`. Here is a typical usage example:

* `reference_energy: 0` (or dict) — set your DFT reference frame
* `shift: auto` — align the FM output to your DFT reference frame

### Learning rate reduction

Additionally, it is recommended to set a small `learning_rate` (e.g., 1e-3 or 1e-4) and evaluate the initial metrics by setting `eval_init_stats: True`:

```yaml
fit:

  # set a small learning rate
  opt_params: {learning_rate: 0.001,  ... }
  
  # evaluate initial metrics
  eval_init_stats: True  


```

### Frozen-weights fine-tuning

By default, a naive fine-tuning approach is used, where all model parameters are updated. In some cases, this can lead to catastrophic forgetting of pre-trained knowledge. To mitigate this issue, we recommend using a frozen-weights fine-tuning approach.

For frozen-weights fine-tuning, you must specify the `trainable_variable_names` parameter in the `fit` section of your `input.yaml`.

Here are a few name pattern examples (for GRACE-2L models):

* `LinMLPOut2ScalarTarget_`: Typical name for a linear+MLP energy readout function.
* `I2/reducing_`: ACE expansion coefficients in the second layer.
* `rho/reducing_`: ACE expansion coefficients in the first layer.
* `I1/reducing_`: ACE expansion coefficients for the message being sent from the first layer to the second layer.

Here are the recommended trainable variables to preserve the pre-trained knowledge of unaffected chemical elements:

* For GRACE-2L models:

```yaml
fit:
  trainable_variable_names: ["I2/reducing_" ,"rho/reducing_","I1/reducing_"]

```

* For GRACE-1L models:

```yaml
fit:
  trainable_variable_names: ["rho/reducing_"]

```

You can find more variable names by running a model summary:

```bash
grace_utils -p ~/.cache/grace/checkpoints/SOME_FOUNDATIONAL_MODEL_NAME/model.yaml  summary -v 1

```

---

## User-defined models

You can add user-defined models to the local registry of foundation models.
You need to create a `$HOME/.cache/grace/models_registry.yaml` file with a content:

```yaml
FULL-MODEL-NAME:
  path: /path/to/saved_mode
  checkpoint_path: /path/to/checkpoint/  # model.yaml and checkpoint.index should be in this folder
  description: some description
  license: some license
```
