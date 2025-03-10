# Important Note  

If a model was fitted with `gracemaker` version < 0.5.1, it will not be compatible with newer versions due to a format change.  
You can convert it to the new format using the following command:  

```bash
grace_utils -p seed/1/model.yaml -c seed/1/checkpoint/checkpoint.best_test_loss.index update_model
```  

This will generate new files with the "-converted" suffix, which you can replace the old files (`model.yaml` and checkpoints) with.

# GRACE - GRaph Atomic Cluster Expansion

Project GRACEmaker is a heavily modified and in large parts rewritten version of the PACEmaker software geared towards support for multi-component materials and graph architectures.

Please see [documentation](https://gracemaker.readthedocs.io/) for installation instructions and examples. 

# Reference
Please see 
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
