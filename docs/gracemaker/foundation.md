# Pretrained GRACE Foundation Models

Several GRACE models are pre-trained on large datasets:  

- **[MPtraj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842?file=41619375)** :  
  A dataset containing 1.5 million DFT calculations of approximately 150,000 materials, covering 89 elements.
  Available Models:

  - `GRACE-1L-MP-r6`:  (old `MP_GRACE_1L_r6_4Nov2024`) single-layer GRACE with a cutoff radius of 6 Å.  
  - `GRACE-2L-MP-r5`: (old `MP_GRACE_2L_r5_4Nov2024`): two-layer GRACE with a cutoff radius of 5 Å.  
  - `GRACE-2L-MP-r6`: (old `MP_GRACE_2L_r6_11Nov2024`) two-layer GRACE with a cutoff radius of 6 Å. This model is currently featured on [Matbench Discovery](https://matbench-discovery.materialsproject.org/) and demonstrates high accuracy in predicting thermal conductivity.

- **[OMat24](https://huggingface.co/datasets/fairchem/OMAT24#omat24-dataset)**:
OMAT (101M structures for 3.23M materials)

  - `GRACE-1L-OAM_2Feb25`: 	A single-layer local GRACE model, pre-fitted on the OMat24 and fine-tuned on sAlex+MPTraj datasets, with fixed 6 Å cutoff.
  - `GRACE_2L_OAM_28Jan25`: A two-layer semi-local GRACE model, pre-fitted on the OMat24 and fine-tuned on sAlex+MPTraj datasets, with fixed 6 Å cutoff.


- **[OMat24](https://huggingface.co/datasets/fairchem/OMAT24#omat24-dataset)/[sAlex](https://huggingface.co/datasets/fairchem/OMAT24#salex-dataset)+[MPtraj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842?file=41619375)**:
OAM = OMat (101M structures for 3.23M materials) + sAlex (subsampled Alexandria: 10.4M structures for 3.23M materials) + MPtrj (1.58M structures for 146k materials).

  - `GRACE-1L-OAM`: (old `GRACE-1L-OAM_2Feb25`)	A single-layer local GRACE model, pre-fitted on the OMat24 and fine-tuned on sAlex+MPTraj datasets, with fixed 6 Å cutoff.
  - `GRACE-2L-OAM`: (old `GRACE_2L_OAM_28Jan25`) A two-layer semi-local GRACE model, pre-fitted on the OMat24 and fine-tuned on sAlex+MPTraj datasets, with fixed 6 Å cutoff.
  

- **SPICE v1** [[1](https://github.com/openmm/spice-dataset), [2](https://doi.org/10.17863/CAM.107498)]:  
  A dataset with about 1 million DFT calculations of drug-like small molecules. It includes the following elements: Br, C, Cl, F, H, I, N, O, P, and S. *(Not yet publicly available.)*  

By default, all foundational models are stored into `$HOME/.cache/grace`, 
but you can overwrite it with `GRACE_CACHE` environment variable. 

Run `grace_models list` to view the list of available models.  

---

## Downloading Foundation Models  

You can use the `grace_models` utility to download pre-trained GRACE models:  

- Run `grace_models list` to view the list of available models.  
- Run `grace_models download <model_name>` to download a specific model.  

These models can be used for simulations within [ASE](../quickstart/#usage-in-ase) and [LAMMPS](../quickstart/#usage-in-lammps).  

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
```

Also, you can generate `input.yaml` for fine-tuning foundation model by running `gracemaker -t` 

---

## User-defined models

You can add user-defined models to the local registry of foundation models.
You need to create a `$HOME/.cache/grace/models_registry.yaml` file with a content:
```yaml
MODEL-NAME:
  path: /path/to/saved_mode
  checkpoint_path: /path/to/checkpoint/  # model.yaml and checkpoint.index should be in this folder
  description: some description
  license: some license
```