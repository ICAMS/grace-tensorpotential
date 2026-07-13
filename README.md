## What's new


### 07 July 2026: [0.6.0 Release](https://github.com/ICAMS/grace-tensorpotential/releases/tag/0.6.0)

- **Uncertainty estimates.** Models can now report a per-atom uncertainty score (gamma) that flags when a prediction is an extrapolation — useful for spotting unreliable regions and for active learning. Comes with a new `grace_uq` command line tool.
- **LAMMPS Kokkos support.** You can now export GRACE models for the fast Kokkos pair styles in LAMMPS. The uncertainty score can be carried along, so it's available during LAMMPS runs too.
- **Foundation models on HuggingFace** ([AMS-ICAMS-RUB/grace-foundation-models](https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models)). The released models now include uncertainty support out of the box,
-  **GRACE-3L-OMAT/OAM**. There are new, larger **3-layer models** (`GRACE-3L-OMAT-large` and `GRACE-3L-OMAT-large-ft-AM`).
- **Faster and lighter by default.** Foundation models now use fp32 precision by default, which roughly halves memory use and about doubles speed. (fp64 versions are still available under the `-fp64` name.) 
- **Better tooling.** A new `grace_dashboard` shows training curves and metrics in the browser; the `grace_models` command was reworked to list, inspect, and download models (including Kokkos weights); and training can now save and resume in the middle of an epoch.

# Important Note  

If a model was fitted with `gracemaker` version < 0.5.1, it will not be compatible with newer versions due to a format change.  
You can convert it to the new format using the following command:  

```bash
grace_utils -p seed/1/model.yaml -c seed/1/checkpoint/checkpoint.best_test_loss.index update_model
```  

This will generate new files with the "-converted" suffix, which you can replace the old files (`model.yaml` and checkpoints) with.

# GRACE - GRaph Atomic Cluster Expansion

Project GRACEmaker is a heavily modified and in large parts rewritten version of the PACEmaker software geared towards support for multi-component materials and graph architectures.

# Documentation

Please see [documentation](https://gracemaker.readthedocs.io/) for installation instructions and examples. 

# Tutorial

You can find tutorial materials [here](https://gracemaker.readthedocs.io/en/latest/gracemaker/tutorials/#tutorial-materials)

Also in a video [format](https://www.youtube.com/watch?v=rndnkiu9LGE)

# Support

Also, you may join ACE support Zulip channel for additional resources:
https://acesupport.zulipchat.com/join/xtwxu2grjbtg64m3vnhypi6p/

# Reference
Please see 
* [Y.Lysogorskiy, A. Bochkarev, R.Drautz, Graph atomic cluster expansion for foundational machine learning interatomic potentials, arXiv:2508.17936](https://arxiv.org/abs/2508.17936)

```bibtex
@article{lysogorskiy2025graph,
  title={Graph atomic cluster expansion for foundational machine learning interatomic potentials},
  author={Lysogorskiy, Yury and Bochkarev, Anton and Drautz, Ralf},
  journal={arXiv preprint arXiv:2508.17936},
  year={2025}
}
```

* [Anton Bochkarev, Yury Lysogorskiy, and Ralf Drautz Graph Atomic Cluster Expansion for Semilocal Interactions beyond Equivariant Message Passing. Phys. Rev. X 14, 021036 (2024)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.14.021036)

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
