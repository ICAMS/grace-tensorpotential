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

Fine-tuning of foundation GRACE models can only be performed using checkpoints and not saved models.
Run `grace_models list` to view the list of available models with a `CHECKPOINT:` field.
You can either download this checkpoint by `grace_models checkpoint <MODEL-NAME>` or it will be done automatically when needed.

In order to fine-tune foundation model, write in `input.yaml::potential` section:
```yaml
potential:
  finetune_foundation_model: GRACE-1L-OAM
#  reduce_elements: True # if True - select from original models only elements presented in the CURRENT dataset
```

Also, it is recommended to set small `learning_rate` (i.e. 1e-3 or 1e-4) and evaluate initial metrics `eval_init_stats: True`:
```yaml
fit:

  # set small learning rate
  opt_params: {learning_rate: 0.001,  ... }
  
  # evaluate initial metrics
  eval_init_stats: True  

  ### specify trainable variables name pattern (depends on the model config)
  # trainable_variable_names: ["rho/reducing_", "Z/ChemicalEmbedding"] 

```

Also, you can generate `input.yaml` for fine-tuning foundation model by running `gracemaker -t` 

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
