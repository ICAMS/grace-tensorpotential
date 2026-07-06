## Gracemaker

```
gracemaker [-h] [-l LOG] [-m] [-rl] [-r] [-rs RESTART_SUFFIX] [--no-intra-epoch-redo] [-p POTENTIAL] [-s] [-sf] [-e] [-nj] [--seed SEED] [-cm] [-t] [-cn CHECKPOINT_NAME] [--reset-epoch-and-step] [input]

Fitting utility for (graph) atomic cluster expansion potentials.

positional arguments:
  input                 input YAML file, default: input.yaml

options:
  -h, --help            show this help message and exit
  -l LOG, --log LOG     log filename, default: log.txt
  -m, --multigpu        Single host/multi GPU distributed fit
  -rl, --restart-latest
                        Restart from latest checkpoint (use separately from -r/-rs)
  -r, --restart-best-test
                        Restart from latest best test checkpoint (use separately from -rs/-rl)
  -rs RESTART_SUFFIX, --restart-suffix RESTART_SUFFIX
                        Suffix of checkpoint to restart from, i.e. .epoch_10 (use separately from -r/-rl)
  --no-intra-epoch-redo
                        When resuming from a mid-epoch checkpoint
                        (intra_epoch_save=True), skip the interrupted epoch's
                        remaining batches entirely and bump the step counter
                        to the nominal end of that epoch. This keeps the LR
                        scheduler aligned with the original maxiter budget at
                        the cost of under-training by (nominal_end -
                        saved_step) gradient updates. Default is to
                        fast-forward the iterator and train only the unseen
                        tail.
  -p POTENTIAL, --potential POTENTIAL
                        Potential configuration to load, model.yaml file
  -s, --save-model      Export model as TF saved model
  -sf, --save--fs       Export FS model as yaml to be loaded in CPP
  -e, --eager           Eager graph execution
  -nj, --no-jit         No JIT
  --seed SEED           Random seed (will overwrite value from input.yaml)
  -cm, --check-model    Check model consistency, without performing fit
  -t, --template        Generate a template 'input.yaml' file by dialog
  -cn CHECKPOINT_NAME, --checkpoint-name CHECKPOINT_NAME
                        Explicit name of the checkpoint (omit .index suffix)
  --reset-epoch-and-step
                        Reset epoch and step counters from prev. runs
```
