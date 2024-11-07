# gracemaker

`gracemaker` is a tool for fitting of interatomic potentials in a general nonlinear Graph Atomic Cluster Expansion (GRACE) form.

Project GRACEmaker is a heavily modified and in large parts rewritten version of the PACEmaker software geared towards support for multi-component materials and graph architectures.

# Features

* Support from multi-component material systems
* Support for one- and two-layer message passing architectures
* Custom C++ implementation of GRACE/FS  model, that enable fast, multi-CPU parallelization using MPI in LAMMPS,
without GPU and extra dependencies (i.e. TensorFlow)

# Documentation

Please use the top navigation bar to explore:

* [Installation](gracemaker/install.md)
* [Quick start](gracemaker/quickstart.md)

# Citation

Please cite following papers:

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