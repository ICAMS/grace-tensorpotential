## Available Presets

Several presets are available, each with adjustable parameters:

* **FS**: Closest to the classical PACE model, this preset uses linear radial functions and a Finnis-Sinclair-like
embedding (read-out) function, similar to PACE, along with chemical embedding, which allows for an unlimited number
of elements to be parameterized by the model. It can be [exported](../quickstart/#gracefs) and executed with a custom C++ implementation and
supports MPI parallelization with domain decomposition in LAMMPS. Extrapolation grade based on D-optimality criteria 
can be used, after construction of [active set](../quickstart/#build-active-set-for-gracefs-only).

  
* **GRACE_1LAYER**: A one-layer model (i.e., only star graphs basis functions) extending the FS preset with MLP for radial functions and readouts,
and additional improvements. This model supports only TensorFlow-based execution; GPU usage is highly recommended but not mandatory. 
Due to the local interactions (within a single cutoff distance), no extra overhead is required for domain decomposition, and the model can be parallelized with MPI in LAMMPS on multiple GPUs.

* **GRACE_2LAYER**: A two-layer model (i.e., star and tree graph basis functions) that builds upon GRACE_1LAYER by adding 
a second layer of message passing. It also supports only TensorFlow-based execution, making GPU usage highly recommended.

* **_Custom Model_**: Users can construct custom models by combining building blocks from the previous presets. 
More details on customization will be provided in future documentation.

---
## Models hyperparameters
Presets can be selected in `input.yaml` input file, in section `potential`:
```yaml
potential:
  # LINEAR, FS, GRACE_1LAYER, GRACE_2LAYER
  preset: "FS" 
  
  # kw-arguments that will be passed to preset or custom function
  kwargs: {n_rad_base: 16, embedding_size: 64}  
```

Latest and complete list of arguments can be found in definition of the models in the [presets.py](https://github.com/ICAMS/grace-tensorpotential/blob/master/tensorpotential/potentials/presets.py) file.

### FS

Following parameters can be provided to tune FS model:

* `basis_type: "SBessel"` - type of radial base functions: SBessel, Cheb
* `n_rad_base: 20` - number of radial base functions
* `embedding_size: 64` - number of chemical embedding channels
* `max_order: 4` - maximum product order for B-functions
* `n_rad_max: [20, 15, 10, 5]` - max number of radial functions for each product order of _output_ B-functions
* `lmax: [5, 5, 4, 3]` - max l-character for each product order of _input_ B-functions used to construct B-functions for current body order  
* `Lmax: [None, 3, 0, 0]` -  max l-character for each product order of resulted _output_ B-functions
* `max_sum_l: [None, None, 6, 4]` - used to limit number of B-functions 
* `fs_parameters: [[1.0, 1.0], [1.0, 0.5], [1.0, 2], [1.0, 0.75]]` - parameters of FS embedding in a form `[[c1,m1], [c2,m2], ...]`
that corresponds to $c_1 \phi_1^{m_1} + c_2 \phi_2^{m_2} + ...$

### GRACE_1LAYER

* `basis_type: "Cheb"` - type of the radial basis functions: SBessel, Cheb
* `n_rad_base: 8`  - number of radial basis functions
* `mlp_radial: True` - use radial function as an MLP (True) or linear (False) transformation of the radial basis
* `embedding_size: 128` -  size of the chemical embedding
* `max_order: 4` - maximum product order for B-functions
* `n_rad_max: 42` - max number of radial functions 
* `lmax: 4` -  maximum l-character for constructing product functions  
* `n_mlp_dens: 16` - number of non-linear readout densities (+1 extra for linear readout automatically added)


### GRACE_2LAYER

* `basis_type: "Cheb"` - type of radial basis functions: SBessel, Cheb
* `n_rad_base: 8`  - number of radial basis functions
* `mlp_radial: True` - use radial function as an MLP (True) or linear (False) transformation of the radial basis
* `embedding_size: 128` -  size of the chemical embedding
* `max_order: 4` - maximum product order for B-functions
* `n_mlp_dens: 10` - number of non-linear readout densities (+1 extra for linear readout automatically added)
* `n_rad_max: [32, 48]` -  per-layer max number of radial functions 
* `lmax: [3, 3]` - maximum per-layer l-character for constructing product functions 
* `indicator_lmax: 1` - maximum l-character of the first layer expansion (aka message) passed to the second layer