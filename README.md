## What's new

### [0.6.2](https://github.com/ICAMS/grace-tensorpotential/releases/tag/0.6.2) (October 2026)

- **[Uncertainty-driven exploration (HAL)](https://gracemaker.readthedocs.io/en/latest/gracemaker/uq_workflows/).** `grace_uq hal` runs molecular dynamics biased towards high uncertainty and collects the structures where the extrapolation grade γ is high, as new training data. Step by step: the [UQ workflows](https://gracemaker.readthedocs.io/en/latest/gracemaker/uq_workflows/) tutorial.

### [0.6.1](https://github.com/ICAMS/grace-tensorpotential/releases/tag/0.6.1) (3 September 2026)

- **[Foundation models re-exported](https://gracemaker.readthedocs.io/en/latest/gracemaker/utilities/#updating-cached-models).** The distributed models were rebuilt to fix a NaN in `virial_sigma` (the uncertainty contribution to the virial). Cached models now record where they were downloaded from, and the new `grace_models update` re-downloads only the copies that are out of date.
- **[Tunable uncertainty threshold](https://gracemaker.readthedocs.io/en/latest/gracemaker/uq/#calibrating-the-threshold).** `grace_uq build --threshold-percentile Q` puts γ = 1 at the Q-th percentile of the training distribution instead of the default robust outlier fence, for training sets where many atoms read γ > 1. `grace_uq info` reports which calibration was used.
- **[Bulk prediction](https://gracemaker.readthedocs.io/en/latest/gracemaker/utilities/#python-api-predict_structures).** A new `predict_structures` helper evaluates a whole list of structures with a single calculator instance; `grace_predict` and `grace_uq predict` now both use it.

### [0.6.0](https://github.com/ICAMS/grace-tensorpotential/releases/tag/0.6.0) (7 July 2026)

- **[Uncertainty estimates](https://gracemaker.readthedocs.io/en/latest/gracemaker/uq/).** Models can report a per-atom uncertainty score γ that flags extrapolation, for spotting unreliable predictions and for active learning. New command-line tool: `grace_uq`.
- **[LAMMPS Kokkos support](https://gracemaker.readthedocs.io/en/latest/gracemaker/quickstart/#lammps-kokkos-npz-grace-1l-grace-2l-grace-3l).** GRACE models can be exported for the fast Kokkos pair styles in LAMMPS, with the uncertainty score γ.
- **[Foundation models on HuggingFace](https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models).** The released models include uncertainty support out of the box.
- **[GRACE-3L-OMAT/OAM](https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/#three-layer-3l-models).** There are new, larger **3-layer models** (`GRACE-3L-OMAT-large` and `GRACE-3L-OMAT-large-ft-AM`).
- **[Faster and lighter by default](https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/).** Foundation models use fp32 by default: about half the memory and twice the speed. The fp64 versions have the suffix `-fp64`.
- **Better tooling.** `grace_dashboard` shows training curves and metrics in the browser; `grace_models` lists, inspects and downloads models (including Kokkos weights); a fit can save and resume in the middle of an epoch.

# Important Note  

If a model was fitted with `gracemaker` version < 0.5.1, it will not be compatible with newer versions due to a format change.  
You can convert it to the new format using the following command:  

```bash
grace_utils -p seed/1/model.yaml -c seed/1/checkpoints/checkpoint.best_test_loss.index update_model
```  

This will generate new files with the "-converted" suffix, which you can replace the old files (`model.yaml` and checkpoints) with.

# GRACE - GRaph Atomic Cluster Expansion

Project GRACEmaker is a heavily modified and in large parts rewritten version of the PACEmaker software geared towards support for multi-component materials and graph architectures.

# Documentation

Please see [documentation](https://gracemaker.readthedocs.io/) for installation instructions and examples. 

# Tutorial

* Fitting tutorials and their materials: [here](https://gracemaker.readthedocs.io/en/latest/gracemaker/tutorials/#tutorial-materials).
* Uncertainty and active learning (CALM, HAL, selection for DFT): [UQ workflows](https://gracemaker.readthedocs.io/en/latest/gracemaker/uq_workflows/).

Also in a video [format](https://www.youtube.com/watch?v=rndnkiu9LGE)

# Support

Also, you may join ACE support Zulip channel for additional resources:
https://acesupport.zulipchat.com/join/xtwxu2grjbtg64m3vnhypi6p/

# Reference
Please see 
* [Y.Lysogorskiy, A. Bochkarev, R.Drautz, Graph atomic cluster expansion for foundational machine learning interatomic potentials. npj Computational Materials 12(1), 114 (2026)](https://www.nature.com/articles/s41524-026-01979-1)

```bibtex
@article{lysogorskiy2026graph,
  title={Graph atomic cluster expansion for foundational machine learning interatomic potentials},
  author={Lysogorskiy, Yury and Bochkarev, Anton and Drautz, Ralf},
  journal={npj Computational Materials},
  volume={12},
  number={1},
  pages={114},
  year={2026},
  doi={10.1038/s41524-026-01979-1},
  url={https://www.nature.com/articles/s41524-026-01979-1}
}
```

* For the CALM uncertainty indication (extrapolation grade `gamma`): [Y. Lysogorskiy, A. Bochkarev, R. Drautz, A latent-space extrapolation grade built into graph atomic cluster expansion foundation potentials. arXiv:2609.40060 (2026)](https://arxiv.org/abs/2609.40060)

```bibtex
@article{lysogorskiy2026calm,
  title={A latent-space extrapolation grade built into graph atomic cluster expansion foundation potentials},
  author={Lysogorskiy, Yury and Bochkarev, Anton and Drautz, Ralf},
  journal={arXiv preprint arXiv:2609.40060},
  year={2026},
  url={https://arxiv.org/abs/2609.40060}
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
