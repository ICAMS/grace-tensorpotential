## Gracemaker

```
usage: gracemaker [-h] [-l LOG] [-m] [-rl] [-r] [-rs RESTART_SUFFIX] [-p POTENTIAL] [-s] [-sf] [-e] [-nj] [--seed SEED] [-cm] [input]

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
  -p POTENTIAL, --potential POTENTIAL
                        Potential configuration to load
  -s, --save-model      Export model as TF saved model
  -sf, --save--fs       Export FS model as yaml to be loaded in CPP
  -e, --eager           Eager graph execution
  -nj, --no-jit         No JIT
  --seed SEED           Random seed (will overwrite value from input.yaml)
  -cm, --check-model    Check model consistency, without performing fit and building neighbourlist

```
