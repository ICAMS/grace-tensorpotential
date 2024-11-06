**GRACE-TENSORPOTENTIAL**


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
