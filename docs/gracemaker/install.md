## GRACE

To use GRACE models, including fitting and utilizing pre-fitted models, you need to install the `grace-tensorpotential` package by following these steps.

### Setting Up the Environment

#### Micromamba (Recommended)

For [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), install it by running the following command:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Then, create a new environment:

```bash
micromamba create -n grace python=3.11 
micromamba activate grace
```

#### Anaconda

**WARNING!** Anaconda is subject to special licensing conditions. Check the details [here](https://www.datacamp.com/blog/navigating-anaconda-licensing).

To create a new conda or micromamba environment using conda:

```bash
conda create -n grace python=3.11
conda activate grace
```

---

### Installing TensorFlow and Tensorpotential

Install `tensorpotential` from PyPI:

```bash
pip install tensorpotential
```

For the latest developer version, clone the `grace-tensorpotential` repository:

```bash
git clone https://github.com/ICAMS/grace-tensorpotential.git
cd grace-tensorpotential
pip install .
```

TensorFlow should be installed automatically. However, to manually install TensorFlow with GPU support:

```bash
pip install tensorflow[and-cuda]==2.16.2
```

* (Optional) Download foundation models (these will be stored in `$HOME/.cache/grace`):

```bash
grace_models download all
```

Learn more [here](../foundation/#pretrained-grace-foundation-models).

---

## GRACE/FS (CPU)

This is a standalone C++ implementation of the GRACE/FS model that can be executed on a CPU without the TensorFlow library and parallelized using standard MPI.

* Activate the conda environment:
```bash
conda activate grace
```

* Clone the repository:
```bash
git clone -b feature/grace_fs https://github.com/ICAMS/python-ace.git
```

* Install:
```bash
cd python-ace
pip install .
```

You may need to install `cmake`:
```bash
pip install cmake
```

Once installed, you can use the `pace_activeset` utility to [generate](../quickstart/#build-active-set-for-gracefs-only) the ASI for GRACE/FS models fitted with `gracemaker`.

---

## LAMMPS with GRACE

* Activate the conda environment (it should contain TensorFlow):
```bash
conda activate grace
```

* Clone the LAMMPS repository:
```bash
git clone -b grace --depth=1 https://github.com/yury-lysogorskiy/lammps.git
```

* Prepare the compilation folder:
```bash
cd lammps
mkdir build
cd build
```

* Configure with CMake:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -D BUILD_MPI=ON -DPKG_ML-PACE=ON -DPKG_MC=ON ../cmake
```

Ensure that the line `TensorFlow library is FOUND at ...` appears after running the above command.

* Compile:
```bash
cmake --build . -- -j 8
```

**NOTE:** If you do **NOT** want TensorFlow support for LAMMPS but wish to use only exported `grace/fs` models, add the `-DNO_GRACE_TF=ON` flag:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -D BUILD_MPI=ON -DPKG_ML-PACE=ON -DNO_GRACE_TF=ON ../cmake
```
