# News
* 25 August 2025: Release of new [foundational GRACE potentials](https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/), parameterized on OMat24 + sAlex+MPTraj

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
