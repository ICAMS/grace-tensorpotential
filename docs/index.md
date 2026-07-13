# gracemaker

`gracemaker` is a tool for fitting interatomic potentials in a general non-linear Graph Atomic Cluster Expansion (GRACE) form.

Project GRACEmaker is a heavily modified and in large parts rewritten version of the [PACEmaker](https://pacemaker.readthedocs.io/)
software geared towards support for multi-component materials and graph architectures.

## What's new


### 07 July 2026: [0.6.0 Release](https://github.com/ICAMS/grace-tensorpotential/releases/tag/0.6.0)

- **[Uncertainty estimates](https://gracemaker.readthedocs.io/en/latest/gracemaker/uq/).** Models can now report a per-atom uncertainty score (gamma) that flags when a prediction is an extrapolation — useful for spotting unreliable regions and for active learning. Comes with a new `grace_uq` command line tool.
- **[LAMMPS Kokkos support](https://gracemaker.readthedocs.io/en/latest/gracemaker/quickstart/#lammps-kokkos-npz-grace-1l-grace-2l-grace-3l).** You can now export GRACE models for the fast Kokkos pair styles in LAMMPS. The uncertainty score can be carried along, so it's available during LAMMPS runs too.
- **[Foundation models on HuggingFace](https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models)**. The released models now include uncertainty support out of the box,
-  **[GRACE-3L-OMAT/OAM](https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/#three-layer-3l-models).** There are new, larger **3-layer models** (`GRACE-3L-OMAT-large` and `GRACE-3L-OMAT-large-ft-AM`).
- **[Faster and lighter by default](https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/).** Foundation models now use fp32 precision by default, which roughly halves memory use and about doubles speed. (fp64 versions are still available under the `-fp64` name.) 
- **Better tooling.** A new `grace_dashboard` shows training curves and metrics in the browser; the `grace_models` command was reworked to list, inspect, and download models (including Kokkos weights); and training can now save and resume in the middle of an epoch.


## Series of ACE models

![Series of ACE models](models.png)


## Features

* Support for multi-component material systems with unlimited number of interacting elements.
* Extension of the local ACE models to also include semi-local interactions (a.k.a. message passing).
* Line of models targeting various performance/complexity regimes and hardware:
    * **GRACE/FS**: Fast, mildly non-linear local model with standalone C++ implementation enabling
    efficient multi-CPU parallelization using MPI in LAMMPS. Intended for large-scale simulations of systems
    containing up to several million atoms.
   
    * **GRACE-1L**: Local, non-linear model with improved accuracy intended to run on GPU.
    Utilizes TensorFlow library to run simulations in LAMMPS or in python and supports multi-GPU parallelization.
    Suitable for modeling both small systems and large-scale simulations of hundreds of thousands of atoms.
  
    * **GRACE-2L**: Semi-local, non-linear model offering state-of-the-art accuracy. 
    Utilizes TensorFlow library to run simulations in LAMMPS or in python.
    Best applied for simulating molecular systems and materials with up to tens of thousand of atoms.

## What's next ?
* !!NEW!! [Video tutorial](https://www.youtube.com/watch?v=rndnkiu9LGE)
* [Installation](gracemaker/install)
* [Quick start](gracemaker/quickstart)
* [Tutorials](gracemaker/tutorials)
* [FAQ](gracemaker/faq)

## Documentation

Please use the navigation bar on the left to explore documentation

## License  

This code and the foundation models are distributed under the [Academic Software License](https://github.com/ICAMS/grace-tensorpotential/blob/master/LICENSE.md).  


## Citation

Please cite following papers if you use GRACEmkaer in your work:

- [Bochkarev, A., Lysogorskiy, Y. and Drautz, R. Graph Atomic Cluster Expansion for Semilocal Interactions beyond Equivariant Message Passing. Phys. Rev. X 14, 021036 (2024)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.14.021036)


<details>
<summary>BibTeX (click to expand)</summary>

```bibtex
@article{PhysRevX.14.021036,
  title = {Graph Atomic Cluster Expansion for Semilocal Interactions beyond Equivariant Message Passing},
  author = {Bochkarev, Anton and Lysogorskiy, Yury and Drautz, Ralf},
  journal = {Phys. Rev. X},
  volume = {14},
  issue = {2},
  pages = {021036},
  numpages = {28},
  year = {2024},
  month = {Jun},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevX.14.021036},
  url = {https://link.aps.org/doi/10.1103/PhysRevX.14.021036}
}

```
</details>
