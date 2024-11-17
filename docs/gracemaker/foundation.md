# Pretrained GRACE foundation models

There are several GRACE models pretrained on available large datasets:

* MPtraj [[1](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842?file=41619375)]: dataset of 1.5M DFT calculations of ~150k materials and 89 elements. 
* SPICE v1[[1](https://github.com/openmm/spice-dataset),[2](https://doi.org/10.17863/CAM.107498)]: dataset of approximately 1M DFT calculations of drug-like small
molecules. Contains the following elements: Br, C, Cl, F, H, I, N, O, P, S.

## Download foundation models

Use `grace_modles` utility for downloading pretrained GRACE models

* Run `grace_models list` to see a list of available models
* Run `grace_models dowload <model_name>` for downloading a specific model

These models can be used for simulation within [ASE](../quickstart/#usage-in-ase) 
and [LAMMPS](../quickstart/#usage-in-lammps).