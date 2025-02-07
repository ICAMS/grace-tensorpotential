# Pretrained GRACE Foundation Models

Several GRACE models are pre-trained on large datasets:  

- **[MPtraj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842?file=41619375)** :  
  A dataset containing 1.5 million DFT calculations of approximately 150,000 materials, covering 89 elements.
  Available Models:

  - `MP_GRACE_1L_r6_4Nov2024`: Single-layer GRACE with a cutoff radius of 6 Å.  
  - `MP_GRACE_2L_r5_4Nov2024`: Two-layer GRACE with a cutoff radius of 5 Å.  
  - `MP_GRACE_2L_r6_11Nov2024`: Two-layer GRACE with a cutoff radius of 6 Å. This model is currently featured on [Matbench Discovery](https://matbench-discovery.materialsproject.org/) and demonstrates high accuracy in predicting thermal conductivity.

- **[OMat24](https://huggingface.co/datasets/fairchem/OMAT24#omat24-dataset)/[sAlex](https://huggingface.co/datasets/fairchem/OMAT24#salex-dataset)+[MPtraj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842?file=41619375)**:
OMat24: 101M structures for 3.23M materials, subsampled Alexandria: 10.4M structures for 3.23M materials, MPtrj: 1.58M structures for 146k materials.

  - `GRACE-1L-OAM_2Feb25`: 	A single-layer local GRACE model, pre-fitted on the OMat24 and fine-tuned on sAlex+MPTraj datasets, with fixed 6 Å cutoff.
  - `GRACE_2L_OAM_28Jan25`: 	A two-layer semi-local GRACE model, pre-fitted on the OMat24 and fine-tuned on sAlex+MPTraj datasets, with fixed 6 Å cutoff.
  
- **SPICE v1** [[1](https://github.com/openmm/spice-dataset), [2](https://doi.org/10.17863/CAM.107498)]:  
  A dataset with about 1 million DFT calculations of drug-like small molecules. It includes the following elements: Br, C, Cl, F, H, I, N, O, P, and S. *(Not yet publicly available.)*  

---

## Downloading Foundation Models  

You can use the `grace_models` utility to download pre-trained GRACE models:  

- Run `grace_models list` to view the list of available models.  
- Run `grace_models download <model_name>` to download a specific model.  

These models can be used for simulations within [ASE](../quickstart/#usage-in-ase) and [LAMMPS](../quickstart/#usage-in-lammps).  
