# Fitting Generic Tensor Properties

GRACEmaker supports fitting of general first- and second-rank tensor properties alongside (or instead of) energy and forces.
Some examples:

- **Electric field gradients (EFG)** — symmetric, traceless, per-atom
- **Born effective charges (BEC)** — general (non-symmetric), per-atom
- **Stress tensor** — symmetric, per-structure
- **Forces as a tensor** — first-rank, per-atom (useful for testing)

---

## Theory

A general second-rank tensor is decomposed into three irreducible spherical components:

| `tensor_components` | Symmetry | Rank |
|---|---|---|
| `[1]` | — | First-rank (vector) |
| `[2]` | Symmetric, traceless | Second-rank |
| `[0, 2]` | Symmetric, non-traceless | Second-rank |
| `[1, 2]` | Antisymmetric, traceless | Second-rank |
| `[0, 1, 2]` | General (non-symmetric, non-traceless) | Second-rank |

> **Note: pure scalars are not supported.** 
> The `l=0` component exists only as part of a second-rank decomposition
> To fit a per-atom or per-structure scalar property, use the standard energy output (`potential.kwargs.compute_energy: True`) together with a regular `energy` loss term.

---

## Data Format

The training DataFrame (`.pkl.gz`) must contain a column named **`tensor_property`**:

- **First-rank** (`tensor_rank: 1`): NumPy array of shape `(N, 3)` with components `[X, Y, Z]`
- **Second-rank** (`tensor_rank: 2`): NumPy array of shape `(N, 9)` with components `[XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ]` — all nine components must be present even for symmetric tensors

where **N** is the number of atoms for per-atom properties, or **N=1** for per-structure properties.

### Per-structure properties and extensivity

The model predicts a tensor by summing atomic contributions, making it an **extensive** quantity.
For per-structure properties the reference data must therefore also be extensive.
For example, to fit the stress tensor:

- Store `stress_tensor × cell_volume` in the `tensor_property` column
- After prediction, divide `predict_tensor` by the cell volume to recover the stress

---

## Configuration (`input.yaml`)

Four sections require modification relative to a standard energy/force fit:

### 1. `data.extra_components`

Register the tensor data builder:

```yaml
data:
  filename: ./data.pckl.gz
  reference_energy: 0.0
  extra_components: {
    ReferenceTensorDataBuilder: {
      tensor_rank: 2,       # 1 for vectors, 2 for second-rank tensors
      per_structure: False, # True for per-structure, False (default) for per-atom
    },
  }
```

### 2. `potential`

Use one of the tensor-specific presets and specify the components:

```yaml
potential:
  preset: TENSOR_2L  # TENSOR_1L or TENSOR_2L
  kwargs: {
    compute_energy: True,        # False to fit tensor only
    compute_forces: True,        # requires compute_energy: True
    tensor_components: [0, 2],   # see table above
  }
  scale: False
  shift: False
```

Available presets: `TENSOR_1L` (one message-passing layer) and `TENSOR_2L` (two layers, more expressive).

### 3. `fit` — compute functions

```yaml
fit:
  compute_function: ComputeBatchEFTensor
  train_function: ComputeBatchEFTensor
  compute_function_config: {
    tensor_components: [0, 2],  # must match potential.kwargs.tensor_components
    per_structure: False,       # must match data.extra_components setting
    compute_energy: True,
    compute_forces: True,       # only relevant when compute_energy: True
  }
```

### 4. `fit.loss` — tensor loss term

Add `WeightedTensorLoss` under `extra_components`:

```yaml
fit:
  loss: {
    energy: {weight: 1., type: huber, delta: 0.01},  # omit if compute_energy: False
    forces: {weight: 5., type: huber, delta: 0.01},  # omit if compute_forces: False
    extra_components: {
      WeightedTensorLoss: {
        weight: 10.,      # relative weight of tensor loss
        type: huber,      # "huber" or "square"
        delta: 0.1,       # Huber delta (ignored for "square")
      },
    },
  }
```

---

## Complete Examples

### Example 1: Electric field gradient (EFG, symmetric traceless, per-atom, energy only)

EFG is a symmetric, traceless second-rank tensor → `tensor_components: [2]`.

```yaml
cutoff: 6
seed: 1

data:
  filename: ./efg_as_tensor_prop.pckl.gz
  test_size: 0.5
  reference_energy: 0.0
  extra_components: {
    ReferenceTensorDataBuilder: {tensor_rank: 2, per_structure: False},
  }

potential:
  preset: TENSOR_1L
  kwargs: {
    compute_energy: True,
    tensor_components: [2],
  }
  scale: False
  shift: False

fit:
  compute_function: ComputeBatchEFTensor
  train_function: ComputeBatchEFTensor
  compute_function_config: {
    tensor_components: [2],
    per_structure: False,
    compute_energy: True,
    compute_forces: False,
  }
  loss: {
    energy: {weight: 1., type: square},
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
  batch_size: 10
  jit_compile: True
```

### Example 2: Stress tensor (symmetric, per-structure, with energy and forces)

Stress is a symmetric, non-traceless second-rank tensor → `tensor_components: [0, 2]`.
The `tensor_property` column must store `stress × volume` (extensive form).

```yaml
cutoff: 6
seed: 1

data:
  filename: ./stress_as_tensor_prop.pckl.gz
  test_size: 0.05
  reference_energy: 0.0
  extra_components: {
    ReferenceTensorDataBuilder: {tensor_rank: 2, per_structure: True},
  }

potential:
  preset: TENSOR_2L
  kwargs: {
    compute_energy: True,
    tensor_components: [0, 2],
  }
  scale: False
  shift: False

fit:
  compute_function: ComputeBatchEFTensor
  train_function: ComputeBatchEFTensor
  compute_function_config: {
    tensor_components: [0, 2],
    per_structure: True,
    compute_energy: True,
    compute_forces: True,
  }
  loss: {
    energy: {weight: 1., type: huber, delta: 0.01},
    forces: {weight: 5., type: huber, delta: 0.01},
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
  batch_size: 10
  jit_compile: True
```

### Example 3: Born effective charges (BEC, general tensor, per-atom, with energy and forces)

BEC is a general (non-symmetric, non-traceless) second-rank tensor → `tensor_components: [0, 1, 2]`.

```yaml
cutoff: 6
seed: 1

data:
  filename: ./bec_as_tensor_prop.pckl.gz
  test_size: 0.5
  reference_energy: 0.0
  extra_components: {
    ReferenceTensorDataBuilder: {tensor_rank: 2, per_structure: False},
  }

potential:
  preset: TENSOR_2L
  kwargs: {
    compute_energy: True,
    compute_forces: True,
    tensor_components: [0, 1, 2],
  }
  scale: False
  shift: False

fit:
  compute_function: ComputeBatchEFTensor
  train_function: ComputeBatchEFTensor
  compute_function_config: {
    tensor_components: [0, 1, 2],
    per_structure: False,
    compute_energy: True,
    compute_forces: True,
  }
  loss: {
    energy: {weight: 1., type: huber, delta: 0.1},
    forces: {weight: 1., type: huber, delta: 0.1},
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

### Example 4: Force vector as a first-rank tensor (tensor only, no energy)

Forces are a first-rank tensor (vector) → `tensor_components: [1]`, `tensor_rank: 1`.

```yaml
cutoff: 6
seed: 1

data:
  filename: ./force_as_tensor_prop.pckl.gz
  test_size: 0.5
  reference_energy: 0.0
  extra_components: {
    ReferenceTensorDataBuilder: {tensor_rank: 1, per_structure: False},
  }

potential:
  preset: TENSOR_1L
  kwargs: {
    compute_energy: False,
    tensor_components: [1],
  }
  scale: False
  shift: False

fit:
  compute_function: ComputeBatchEFTensor
  train_function: ComputeBatchEFTensor
  compute_function_config: {
    tensor_components: [1],
    per_structure: False,
    compute_energy: False,
    compute_forces: False,
  }
  loss: {
    extra_components: {
      WeightedTensorLoss: {weight: 10., type: square},
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

---

## Running the fit

Once `input.yaml` is configured, run the fit as usual:

```bash
gracemaker input.yaml
```

---

## Prediction

After training, create an ASE calculator with the `predict_tensor` extra property:

```python
from tensorpotential.calculator import TPCalculator

calc = TPCalculator('./seed/1/final_model/', extra_properties=['predict_tensor'])
```

Then compute predictions:

```python
atoms.calc = calc
atoms.get_potential_energy()  # triggers the forward pass

# for per-atom predictions - crop padding atoms
# shape: (N_atoms, 3) for rank-1, or (N_atoms, 9) for rank-2 per-atom properties
tensor = atoms.calc.results['predict_tensor'][:len(atoms)]

# shape: (1, 9) for rank-2 per-structure properties
tensor = atoms.calc.results['predict_tensor'][:1]
```

### Per-structure properties

For per-structure predictions (e.g. stress), divide by cell volume after prediction:

```python
import numpy as np

atoms.calc = calc
atoms.get_potential_energy()

stress_tensor = atoms.calc.results['predict_tensor'].reshape(3, 3) / atoms.get_volume()
```

### Joint energy + forces + tensor prediction

```python
atoms.calc = calc

energy = atoms.get_potential_energy()    # eV
forces = atoms.get_forces()              # eV/Å  (gradient of energy)
stress = atoms.get_stress()              # eV/Å³ (computed from virial, not from tensor fit)

volume = atoms.get_volume()
tensor = atoms.calc.results['predict_tensor'].reshape(1, -1) / volume  # per-structure
```

> **Note**: `atoms.get_stress()` returns the stress computed from the virial (derived from energy via automatic differentiation). This is independent of the `tensor_property` fit. The fitted tensor is accessed only via `calc.results['predict_tensor']`.

---

## Parameter Reference

| Parameter | Location | Description |
|---|---|---|
| `tensor_rank` | `data.extra_components.ReferenceTensorDataBuilder` | `1` (vector) or `2` (matrix) |
| `per_structure` | `data.extra_components.ReferenceTensorDataBuilder` | `True` for per-structure, `False` for per-atom |
| `preset` | `potential` | `TENSOR_1L` or `TENSOR_2L` |
| `compute_energy` | `potential.kwargs`, `fit.compute_function_config` | Whether to also predict energy |
| `compute_forces` | `potential.kwargs`, `fit.compute_function_config` | Whether to compute forces (requires `compute_energy: True`) |
| `tensor_components` | `potential.kwargs`, `fit.compute_function_config` | Irreducible components: `[1]`, `[2]`, `[0,2]`, `[1,2]`, or `[0,1,2]` |
| `WeightedTensorLoss.weight` | `fit.loss.extra_components` | Relative weight of tensor loss vs. energy/forces |
| `WeightedTensorLoss.type` | `fit.loss.extra_components` | `"huber"` or `"square"` |
| `WeightedTensorLoss.delta` | `fit.loss.extra_components` | Huber delta (default `0.01`) |