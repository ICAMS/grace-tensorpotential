## How to Continue a Current Fit?

- Run `gracemaker -r` in the folder of the original fit to restart from the previous best-test-loss checkpoint.  
- Run `gracemaker -rl` in the folder of the original fit to restart from the latest checkpoint.  
- To continue in a new folder, copy `seed/{number}/checkpoints` and `seed/{number}/model.yaml` into the new folder.  

---

## How to Save Regular Checkpoints?

Use `checkpoint_freq` to specify how frequently to save regular checkpoints (only the last state will be saved).  
To keep all regular checkpoints, add the flag `input.yaml::fit::save_all_regular_checkpoints: True`.  

---

## Can I Have Different Cutoffs for Different Bond Types?

Yes, you can specify bond-specific cutoffs using the `input.yaml::cutoff_dict` option. For example:  
```yaml
cutoff_dict: {Mo: 4, MoNb: 3, W: 5, Ta*: 7}
```  
This can be used alongside `input.yaml::cutoff`.  

---

## What Are Buckets (`train_max_n_buckets` and `test_max_n_buckets`)?

GRACE models are JIT-compiled, which requires all batches to have the same size. This is achieved through padding. To improve efficiency, batches are split into buckets that are then padded.  

The parameters `train_max_n_buckets` and `test_max_n_buckets` define the maximum number of buckets for padding training and testing data, respectively. More buckets reduce padding.  

To estimate the optimal number of buckets, refer to the `[TRAIN] dataset stats:` log line, which shows padding information:  
```
[TRAIN] dataset stats: num. batches: 18 | num. real structures: 576 (+2.78%) | num. real atoms: 10942 (+5.25%) | num. real neighbours: 292102 (+1.74%)
```  
Here, the padding for neighbors is only +1.74%. It is recommended to keep this value below 15%. The same applies to `[TEST] dataset stats`.  

---

## How to Run a GRACE-1LAYER Model in Parallel Within LAMMPS?

```bash
mpirun -np 4 --bind-to none bash -c 'CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_RANK % 4)) lmp -in in.lammps'
```

---

## How to Evaluate Uncertainty Indication for GRACE Models?

- **For all GRACE models:** Use naive ensembling (query-by-committee). Run parameterization with different seeds, e.g.,  
  ```bash
  gracemaker ... --seed 1
  gracemaker ... --seed 2
  ```
  This generates multiple models in `seed/{number}/`. Use these models with the ASE calculator:  
  ```python
  from tensorpotential.calculator import TPCalculator

  calc_ens = TPCalculator(model=[
      "fit/seed/1/saved_model/",
      "fit/seed/2/saved_model/",
      "fit/seed/3/saved_model/",
  ])

  at.calc = calc_ens
  at.get_potential_energy()

  calc.results['energy_std']  # Standard deviation of total energy predictions
  calc.results['forces_std']  # Standard deviation of forces predictions
  calc.results['stress_std']  # Standard deviation of stress predictions
  ```
  
- **For GRACE/FS models:** In addition to the ensembling method, use extrapolation grades based on D-optimality in [ASE](../quickstart/#gracefs_1) and [LAMMPS](../quickstart/#lammps-gracefs).  

---

## How to Perform Multi-GPU Fit?

If you have a node with multiple GPUs, use the `gracemaker ... -m` option to enable data-parallel fitting. In this case, increase the batch size (global batch size).  

---

## How to Reduce TensorFlow Verbosity Level?

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

or  

```bash
export TF_CPP_MIN_LOG_LEVEL=3
```  