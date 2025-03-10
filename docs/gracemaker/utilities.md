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

## `grace_utils`
Utility to convert, export and summarize GRACE models

```
grace_utils [-h] -p POTENTIAL [-c CHECKPOINT_PATH] [-os OUTPUT_SUFFIX] {update_model,reduce_elements,cast_model,export,summary} ...

CLI tool for model conversions and summarization

positional arguments:
  {update_model,reduce_elements,cast_model,export,summary}
    update_model        Update model (model.yaml) and corresponding checkpoint.
    reduce_elements     Reduce elements from the model.
    cast_model          Change model's floating point precision.
    export              Export model to saved_model or FS/C++ format.
    summary             Show info about the model

options:
  -h, --help            show this help message and exit
  -p POTENTIAL, --potential POTENTIAL
                        Path to model.yaml
  -c CHECKPOINT_PATH, --checkpoint-path CHECKPOINT_PATH
                        Path to checkpoint
  -os OUTPUT_SUFFIX, --output-suffix OUTPUT_SUFFIX
                        Output suffix for converted



-------------------
Optional arguments for different commands:

--------
update_model:
  None
 
--------
reduce_elements: 

  -e ELEMENTS [ELEMENTS ...], --elements ELEMENTS [ELEMENTS ...]
                        Elements to select

--------
cast_model:

  -curr {fp32,fp64}  Current precision type to cast from
  -to {fp32,fp64}    New precision type to cast into
--------
summary:

-v {0,1,2}, --verbose {0,1,2}
                        Verbosity level: 0, 1 or 2
```
#### Update models
If a model was fitted with `gracemaker` version < 0.5, it will break in the newer versions due to the format change.
Conversion to the new format can be easily done via:
 
```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.best_test_loss.index  -os dict update_model
```
one needs to provide path `-p` to the previously fitted `model.yaml`, `-c` path to the `checkpoint` of the corresponding model.
New updated `checkpoint` files and `model.yaml` will be saved with the suffix provided via `-os`.

#### Reduce model's chemical complexity

If you have a model that was fitted for large number of chemical elements, for example one of the [foundation models](../foundation/#pretrained-grace-foundation-models), but
you're interested only in a few specific, you can reduce the large model to the specified chemistry.
For example, selecting only  Mo, Nb, Ta and W from a large model:
```bash
grace_utils -p /path/to/model-dict.yaml -c /path/to/checkpoint/checkpoint-dict.index  -os MoNbTaW reduce_elements -e Mo Nb Ta W
```

#### Change model's floating point precision
GRACE models can be trained in both single and double floating point precision.
Conversion between the two can be done with the  `cast_model` utility, for example to convert from single to double precision:

```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint.index -os "double" cast_model -curr fp32 -to fp64
```


#### Export model to saved_model or GRACE-FS/C++ format

Export model.yaml + checkpoint into saved_model format:
```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.index export -n my_saved_model
```

For GRACE-FS model, one can export to GRACE-FS/C++ format(.yaml) by adding `-sf` flag:
```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.index export -n my_GRACE-FS.yaml -sf
```

#### Model summary
To print summary of the GRACE model with different level of verbosity (0 - least, 1 - moderate and 2 - most verbose):

```bash
grace_utils -p /path/to/model.yaml summary -v 1
```