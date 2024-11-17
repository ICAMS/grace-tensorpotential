## GRACE

For using GRACE models, fitting and using prefitted models, one needs to install a `grace-tensorpotential` package following a few steps
### Setting up environment


#### Micromamba (recommended) 
For [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), install micromamba executing the following command

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```
and create a new environment

```bash
micromamba create -n grace python=3.11 
micromamba activate grace
```

#### Anaconda 
**WARNING!** Special license conditions, check [here](https://www.datacamp.com/blog/navigating-anaconda-licensing)

Create a new conda or micromamba environment. For conda do
```bash
conda create -n grace python=3.11
conda activate grace
```

---
### Installing TensorFlow and Tensorpotential:


Install `tensorpotential` from PyPI:
```bash
pip install tensorpotential
```

or for latest developer version, clone `grace-tensorpotential` repository:
```bash
git clone https://github.com/ICAMS/grace-tensorpotential.git
cd grace-tensorpotential
pip install .
```

TensorFlow should be installed automatically, but you can manually install it with GPU support as follows
```bash
pip install tensorflow[and-cuda]<=2.16
```


* (optional) Download foundation models (it will be stored in $HOME/.cache/grace):

```bash
grace_models download all
```

Learn more [here](../foundation/#pretrained-grace-foundation-models).



## GRACE/FS (CPU)

This is a standalone C++ implementation of GRACE/FS model, that can be executed on CPU without TensorFlow library and parallelized using standard MPI.

* Activate conda environment
```bash
conda activate grace
```

* Clone repo:
```
git clone -b feature/grace_fs https://github.com/ICAMS/python-ace.git
```

* Install
```
cd python-ace
pip install .
```
You may need to install `cmake`: 
```bash
pip install cmake
```

Now, you can use `pace_activeset` utility to [generate](../quickstart/#build-active-set-for-gracefs-only) ASI for GRACE/FS models, fitted with gracemaker.

## LAMMPS with GRACE

* Activate conda environment (it should contain installed TensorFlow)
```
conda activate grace
```

* Clone LAMMPS
```
git clone -b grace --depth=1 https://github.com/yury-lysogorskiy/lammps.git
```

* Prepare compilation folder:
```
cd lammps
mkdir build
cd build
```

* Configure with CMake:
```
cmake -DCMAKE_BUILD_TYPE=Release -D BUILD_MPI=ON -DPKG_ML-PACE=ON -DPKG_MC=ON ../cmake
```

Check that you have line `TensorFlow library is FOUND at ...` after running previous command.

* Compile
```
cmake --build . -- -j 8
```

**NOTE**  If you do NOT want TensorFlow support for LAMMPS, but use only exported `grace/fs` models, add `-DNO_GRACE_TF=ON` flag :
```
cmake -DCMAKE_BUILD_TYPE=Release -D BUILD_MPI=ON -DPKG_ML-PACE=ON -DNO_GRACE_TF=ON  ../cmake
```
