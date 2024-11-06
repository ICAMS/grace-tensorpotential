# Installation

## GRACE-TENSORPOTENTIAL

Main package, that contains gracemaker and other utilities

1. Create new conda or mamba environment (i.e. grace), Python >=3.9
```
conda create -n grace python<=3.11
conda activate grace
```

2. Install TensorFlow:
```
pip install tensorflow[and-cuda]  # or <=2.16
```

3. Install `tensorpotential`
```
pip install tensorpotential
```

3 (optional). For latest developer version, clone `grace-tensorpotential` repository
```
git clone https://github.com/ICAMS/grace-tensorpotential.git
cd grace-tensorpotential
pip install .
```

5 (optional). Download foundational models (it will be stored in $HOME/.cache/grace)
```
grace_download
```


## GRACE/FS

This is a custom C++ implementation of GRACE/FS model, taht can be executed on CPU without TensorFlow library and parallelized using usual MPI.

1. Activate conda environment
```
conda activate grace
```

2. Clone repo:
```
git clone -b feature/grace_fs https://github.com/ICAMS/python-ace.git
```

3. Install
```
cd pyace
pip install .
```

Now, you can use `pace_activeset` utility to generate ASI for GRACE/FS models, fitted with gracemaker

## LAMMPS+GRACE

1. Activate conda environment (it should contain TF)
```
conda activate grace
```

2. Clone LAMMPS
```
git clone -b grace https://github.com/yury-lysogorskiy/lammps.git
```

3. Prepapre compilation folder:
```
cd lammps-ace
mkdir build
cd build
```

4. Configure with CMake:
```
cmake -DCMAKE_BUILD_TYPE=Release -D BUILD_MPI=ON -DPKG_ML-PACE=ON ../cmake
```

Check that you have line `TensorFlow library is FOUND at ...` after previous command

5. Compile
```
cmake --build . -- -j 16
```

**NOTE**  If you do NOT want TF support for LAMMPS, add `-DNO_GRACE_TF=ON` flag :
```
cmake -DCMAKE_BUILD_TYPE=Release -D BUILD_MPI=ON -DPKG_ML-PACE=ON -DNO_GRACE_TF=ON  ../cmake
```
