# Fitting Generic Tensor Properties

GRACEmaker can fit first- and second-rank tensor properties, together with energy and forces or
alone. Some examples:

- **Electric field gradients (EFG)**: symmetric, traceless, per atom
- **Born effective charges (BEC)**: general (not symmetric), per atom
- **Stress tensor**: symmetric, per structure
- **Forces as a tensor**: first rank, per atom (useful for testing)

The quickest start is the wizard: run `gracemaker -t` and choose the fit type
**Generic Tensors (L<=2)**. It asks for the tensor type, per-atom or per-structure, the model and
the loss, and writes `input.yaml`. It can also check your dataset (columns, tensor shape and
symmetry) and build the `tensor_property` column from another column.

---

## Theory

A general second-rank tensor $T$ is split into three irreducible parts:

$$
T = a_0\,I + A_1 + A_2 ,
$$

where $a_0 I$ is the isotropic part ($l=0$, one number), $A_1$ is the antisymmetric part ($l=1$,
three numbers) and $A_2$ is the symmetric traceless part ($l=2$, five numbers). The model predicts
the parts you list in `tensor_components` and adds them up:

| `tensor_components` | Tensor | `tensor_rank` | Output per atom or structure |
|---|---|---|---|
| `[1]` | Vector (first rank) | `1` | 3 values `[X, Y, Z]` |
| `[2]` | Symmetric, traceless | `2` | 9 values |
| `[0, 2]` | Symmetric | `2` | 9 values |
| `[1, 2]` | Traceless (not symmetric) | `2` | 9 values |
| `[0, 1, 2]` | General | `2` | 9 values |

Other combinations (`[0]`, `[0, 1]`) are not supported.

!!! note "Pure scalars are not supported"
    The $l=0$ part exists only as part of a second-rank tensor. To fit a per-atom or
    per-structure scalar, use the normal energy output (`compute_energy: True`) with a regular
    `energy` loss term.

---

## Data format

The training data must be a pickled pandas DataFrame (`.pkl.gz`) with the columns `ase_atoms` and
**`tensor_property`**. Extended XYZ files cannot be used, because they have no
`tensor_property` column. Each row of `tensor_property` is a NumPy array:

- **First rank** (`tensor_rank: 1`): shape `(N, 3)`, components `[X, Y, Z]`.
- **Second rank** (`tensor_rank: 2`): shape `(N, 9)`, components
  `[XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ]`. Give all nine, also for symmetric tensors.

**N** is the number of atoms for per-atom properties and **N = 1** for per-structure properties.
If energy or forces are in the loss, the usual `energy` and `forces` columns are needed too.

**Weights (optional).** A column `ref_tensor_weight` gives a weight to each atom (shape `(N,)`)
or to each structure (one number). Without it, all weights are 1. The weights are normalized to
sum to 1 over the whole dataset (`ReferenceTensorDataBuilder: {normalize_weights: False}` turns
this off).

### Per-structure properties are extensive {#per-structure-properties-and-extensivity}

The model predicts a tensor for each atom. For a per-structure property it **sums** these atomic
tensors, so the prediction is **extensive** (it grows with the number of atoms). The reference
data must then be extensive too. For example, to fit the stress tensor $\sigma$:

- store $\sigma \cdot V$ (stress times cell volume) in `tensor_property`;
- after prediction, divide `predict_tensor` by the cell volume $V$ to get the stress back.

---

## Configuration (`input.yaml`)

Compared to a normal energy and force fit, four sections change. They use the general keys for
custom training, described in [Input file: custom training](inputfile.md#custom-training).

| Section | What to set |
|---|---|
| `data.extra_components` | `ReferenceTensorDataBuilder`: reads `tensor_property` |
| `potential` | preset `TENSOR_1L` or `TENSOR_2L` with `tensor_components` and `compute_energy` |
| `fit.train_function`, `fit.compute_function`, `fit.compute_function_config` | `ComputeBatchEFTensor`: computes the tensor (and energy, forces) |
| `fit.loss.extra_components` | `WeightedTensorLoss`: the tensor loss term |

Some settings appear in two places and **must agree**:

- `tensor_components`: in `potential.kwargs` and in `fit.compute_function_config`.
- `per_structure`: in `ReferenceTensorDataBuilder` and in `fit.compute_function_config`.
- `compute_energy`: in `potential.kwargs` and in `fit.compute_function_config`. The defaults
  differ (`False` in the presets, `True` in `ComputeBatchEFTensor`), so always set both.
- `tensor_rank: 1` goes with `tensor_components: [1]`; `tensor_rank: 2` with all others.

### Complete example {#complete-examples}

This file fits Born effective charges (general tensor, per atom) together with energy and forces.
Comments mark the lines that change for the other cases in the [table below](#other-cases).

```yaml
cutoff: 6
seed: 1

data:
  filename: ./bec_as_tensor_prop.pckl.gz
  test_size: 0.05
  reference_energy: 0.0                 # used only by the energy loss
  extra_components: {
    ReferenceTensorDataBuilder: {
      tensor_rank: 2,                   # 1: vector, 2: second-rank tensor
      per_structure: False,             # True: one tensor per structure
    },
  }

potential:
  preset: TENSOR_2L                     # or TENSOR_1L
  kwargs: {
    compute_energy: True,               # False: tensor only
    tensor_components: [0, 1, 2],
  }
  scale: False
  shift: False

fit:
  train_function: ComputeBatchEFTensor
  compute_function: ComputeBatchEFTensor
  compute_function_config: {
    tensor_components: [0, 1, 2],       # same as potential.kwargs
    per_structure: False,               # same as ReferenceTensorDataBuilder
    compute_energy: True,               # same as potential.kwargs
    compute_forces: True,               # needs compute_energy: True
  }
  loss: {
    energy: {weight: 1., type: huber, delta: 0.1},   # remove if compute_energy: False
    forces: {weight: 1., type: huber, delta: 0.1},   # remove if compute_forces: False
    extra_components: {
      WeightedTensorLoss: {weight: 10., type: huber, delta: 0.1},
    },
  }
  maxiter: 500
  optimizer: Adam
  opt_params: {learning_rate: 0.008, use_ema: True, ema_momentum: 0.99,
               weight_decay: 5.e-9, clipnorm: 1.0}
  scheduler: cosine_decay
  scheduler_params: {minimum_learning_rate: 0.0005}
  batch_size: 5
  jit_compile: True
```

Run the fit as usual:

```bash
gracemaker input.yaml
```

The log shows the tensor loss as `loss_component/WeightedTensorLoss` and the tensor errors as
`mae/d_tensor` and `rmse/d_tensor`: the mean absolute and root-mean-square error per tensor
component, over all atoms (per-atom tensors) or structures (per-structure tensors).

### Other cases {#other-cases}

Change only these lines of the example above:

| Case | `tensor_rank` | `per_structure` | `tensor_components` | `preset` | `compute_energy` | `compute_forces` | `loss` |
|---|---|---|---|---|---|---|---|
| EFG (symmetric, traceless, per atom) | `2` | `False` | `[2]` | `TENSOR_1L` | `True` | `False` | `energy` + tensor |
| Stress × volume (symmetric, per structure) | `2` | `True` | `[0, 2]` | `TENSOR_2L` | `True` | `True` | `energy` + `forces` + tensor |
| BEC (general, per atom), the example above | `2` | `False` | `[0, 1, 2]` | `TENSOR_2L` | `True` | `True` | `energy` + `forces` + tensor |
| Forces as a vector (tensor only) | `1` | `False` | `[1]` | `TENSOR_1L` | `False` | `False` | tensor only |

For a tensor-only fit, `fit.loss` keeps just the tensor term:

```yaml
  loss: {
    extra_components: {
      WeightedTensorLoss: {weight: 10., type: square},
    },
  }
```

---

## Settings {#parameter-reference}

### `ReferenceTensorDataBuilder` (in `data.extra_components`)

| Key | Default | Meaning |
|---|---|---|
| `tensor_rank` | `2` | `1` (vector) or `2` (second-rank tensor) |
| `per_structure` | `False` | `True`: one tensor per structure; `False`: one tensor per atom |
| `normalize_weights` | `True` | normalize `ref_tensor_weight` to sum 1 over the dataset |

### Presets `TENSOR_1L` and `TENSOR_2L` (in `potential`)

`TENSOR_1L` has one message-passing layer (`lmax: 4`, `embedding_size: 32`). `TENSOR_2L` has two
layers and is more expressive (`lmax: 4`, `embedding_size: 128`). Keys in `potential.kwargs`:

| Key | Default | Meaning |
|---|---|---|
| `tensor_components` | `[0, 1, 2]` | parts of the tensor, see [Theory](#theory) |
| `compute_energy` | `False` | also build the energy output |
| `lmax`, `n_rad_base`, `n_rad_max`, `embedding_size`, `n_mlp_dens`, `max_order` | preset values | model size, as in the GRACE presets |

`compute_forces` is not a model key. Forces come from the energy and are switched on only in
`fit.compute_function_config`.

### `ComputeBatchEFTensor` (in `fit.compute_function_config`)

| Key | Default | Meaning |
|---|---|---|
| `tensor_components` | `[0, 1, 2]` | must match `potential.kwargs` |
| `per_structure` | `False` | `True`: sum the atomic tensors over each structure |
| `compute_energy` | `True` | compute the energy; must match `potential.kwargs` |
| `compute_forces` | `True` | compute forces and virial from the energy; used only if `compute_energy: True` |

### `WeightedTensorLoss` (in `fit.loss.extra_components`)

| Key | Default | Meaning |
|---|---|---|
| `weight` | required | weight of the tensor loss relative to energy and forces |
| `type` | `huber` | `huber` or `square` |
| `delta` | `0.01` | Huber delta; ignored for `square` |
| `normalize_by_samples` | `True` | divide by the sum of weights in the batch |

The loss averages the error over the 3 or 9 components of each tensor $i$ and weights it with
$w_i$ (`ref_tensor_weight`):

$$
L = \frac{\sum_i w_i \,\frac{1}{C}\sum_{c=1}^{C} \ell\left(T_{ic} - \hat T_{ic}\right)}{\sum_i w_i},
$$

where $C$ is 3 or 9 and $\ell$ is the Huber or square function.

---

## Prediction

After training, load the model in the ASE calculator and ask for the extra output
`predict_tensor`:

```python
from tensorpotential.calculator import TPCalculator

calc = TPCalculator("./seed/1/final_model/", extra_properties=["predict_tensor"])
atoms.calc = calc
atoms.get_potential_energy()  # runs the model; the energy is 0 if compute_energy: False

tensor = atoms.calc.results["predict_tensor"]
```

The shape of the result:

```python
# per atom: the calculator adds padding atoms, so crop to the real atoms
# shape (N_atoms, 3) for rank 1, (N_atoms, 9) for rank 2
tensor = atoms.calc.results["predict_tensor"][: len(atoms)]

# per structure: shape (1, 9); for stress, divide by the volume
stress_tensor = atoms.calc.results["predict_tensor"][0].reshape(3, 3) / atoms.get_volume()
```

For per-atom properties, `TPCalculator(..., truncate_extras_by_natoms=True)` does the cropping
for you.

!!! note "`get_stress()` is not the fitted tensor"
    `atoms.get_forces()` and `atoms.get_stress()` come from the energy (by automatic
    differentiation). They do not use the `tensor_property` fit. The fitted tensor is only in
    `calc.results["predict_tensor"]`.
