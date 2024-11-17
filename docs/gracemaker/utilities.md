## `grace_collect`

Utility to collect VASP calculations from a top-level directory and store them
in a _*.pkl.gz_ file that can be used for fitting with _gracemaker_ or _pacemaker_.
The reference energies could be provided for each element (default value is zero) or extracted automatically from the
calculation with single atom and large enough (>500 Ang^3/atom) volume. 

```
usage: grace_collect [-h] [-wd WORKING_DIR] [--output-dataset-filename OUTPUT_DATASET_FILENAME] [--free-atom-energy [FREE_ATOM_ENERGY ...]] [--selection SELECTION]

options:
  -h, --help            show this help message and exit
  -wd WORKING_DIR, --working-dir WORKING_DIR
                        top directory where keep calculations
  --output-dataset-filename OUTPUT_DATASET_FILENAME
                        pickle filename, default is collected.pckl.gzip
  --free-atom-energy [FREE_ATOM_ENERGY ...]
                        dictionary of reference energies (auto for extraction from dataset), i.e. `Al:-0.123 Cu:-0.456 Zn:auto`, default is zero. If option is `auto`, then it will be extracted from dataset
  --selection SELECTION
                        Option to select from multiple configurations of single VASP calculation: first, last, all, first_and_last (default: last)
```

___

## `grace_models`
Utility to download (all) foundation models

```
usage: grace_models [-h] {list,download} ...

Download foundational GRACE models

positional arguments:
  {list,download}  Sub-command help
    list           List available models
    download       Download a model

options:
  -h, --help       show this help message and exit

```

Example:
```bash
grace_models
```
___
