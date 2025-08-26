# Pretrained GRACE Foundation Models

Several GRACE models are pre-trained on large datasets. All models listed use a fixed **6 Å cutoff**.

**NOTE!** You should use the "Full Name" column to refer to the model in LAMMPS and ASE.

***

## OMAT models

The base models (**-OMAT**) are trained on the [OMat24](https://huggingface.co/datasets/fairchem/OMAT24#omat24-dataset) dataset. The fine-tuned versions (**-OMAT-ft-E**) are derived from these base models by fine-tuning with more emphasis on energies.

#### Single-layer, local models

| Model Name             | Full Name | Size | $\kappa_\mathrm{SRME}$ | Description                    |
|:-----------------------| :--- | :--- |:-----------------------|:-------------------------------|
| **GRACE-1L-OMAT**        | GRACE-1L-OMAT | small | 0.398                  | Single-layer local             |
| GRACE-1L-OMAT-M-base | GRACE-1L-OMAT-medium-base | medium | 0.380                  | Single-layer local (base fit)  |
| **GRACE-1L-OMAT-M**      | GRACE-1L-OMAT-medium-ft-E | medium | 0.417                  | Single-layer local (finetuned  |
| GRACE-1L-OMAT-L-base | GRACE-1L-OMAT-large-base | large | **0.354**                 | Single-layer local             |
| **GRACE-1L-OMAT-L**      | GRACE-1L-OMAT-large-ft-E | large | 0.383                  | Single-layer local             |

#### Two-layer, semilocal models

| Model Name              | Full Name | Size | $\kappa_\mathrm{SRME}$ | Description |
|:------------------------| :--- | :--- |:-----------------------| :--- |
| **GRACE-2L-OMAT**         | GRACE-2L-OMAT | small | 0.288                  | Two-layer semi-local |
| GRACE-2L-OMAT-M-base   | GRACE-2L-OMAT-medium-base | medium | 0.212                  | Two-layer semi-local |
| **GRACE-2L-OMAT-M**     | GRACE-2L-OMAT-medium-ft-E | medium | 0.217                  | Two-layer semi-local |
| GRACE-2L-OMAT-L-base    | GRACE-2L-OMAT-large-base | large | **0.165**                  | Two-layer semi-local |
| **GRACE-2L-OMAT-L**     | GRACE-2L-OMAT-large-ft-E | large | 0.186                 | Two-layer semi-local |

***

## OAM models

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

## MPtraj (deprecated)

Dataset: [MPtraj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842?file=41619375)  (1.58M structures for 146k materials)

  - `GRACE-1L-MP-r6`:  (old `MP_GRACE_1L_r6_4Nov2024`) single-layer GRACE with a cutoff radius of 6 Å.  
  - `GRACE-2L-MP-r5`: (old `MP_GRACE_2L_r5_4Nov2024`): two-layer GRACE with a cutoff radius of 5 Å.  
  - `GRACE-2L-MP-r6`: (old `MP_GRACE_2L_r6_11Nov2024`) two-layer GRACE with a cutoff radius of 6 Å. This model is currently featured on [Matbench Discovery](https://matbench-discovery.materialsproject.org/) and demonstrates high accuracy in predicting thermal conductivity.
 


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
