## GRACE

To fit GRACE models and to use fitted or foundation models, install the `tensorpotential`
package (repository: [grace-tensorpotential](https://github.com/ICAMS/grace-tensorpotential)).

### 1. Environment

We recommend [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html):

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
micromamba create -n grace python=3.11
micromamba activate grace
```

`conda`/`mamba` or a plain `python -m venv` environment work as well.

### 2. Install `tensorpotential`

```bash
pip install tensorpotential
```

On Linux this also installs TensorFlow with GPU support (`tensorflow[and-cuda]<2.22`); on macOS
and Windows, TensorFlow for CPU. If TensorFlow does not find your GPU, install it by hand:

```bash
pip install "tensorflow[and-cuda]<2.22"
```

??? info "Developer version from source"
    ```bash
    git clone https://github.com/ICAMS/grace-tensorpotential.git
    cd grace-tensorpotential
    pip install .
    ```

### 3. Check the installation

```bash
python -c "import tensorpotential, tensorflow as tf; print(tensorpotential.__version__, tf.config.list_physical_devices('GPU'))"
```

This prints the version and the GPUs TensorFlow sees (an empty list `[]` means CPU only).

### 4. Foundation models (optional)

```bash
grace_models list                   # available models
grace_models download <model_name>  # one model, stored in $HOME/.cache/grace
```

`grace_models download all` downloads every model (many GB). See
[foundation models](foundation.md#pretrained-grace-foundation-models).

??? info "TensorFlow notes: Keras, GPU memory leaks"
    **Keras.** GRACE needs the legacy Keras API (`tf_keras`, installed as a dependency).
    `import tensorpotential` sets `TF_USE_LEGACY_KERAS=1` itself. Set it by hand only if your
    script imports TensorFlow *before* tensorpotential:

    ```bash
    export TF_USE_LEGACY_KERAS=1
    ```

    **GPU memory leaks.** First update the NVIDIA driver. If the leak stays, downgrade TensorFlow:

    ```bash
    pip install "tensorflow[and-cuda]==2.16.2"
    ```

---

## GRACE/FS (CPU)

A standalone C++ implementation of GRACE/FS models. It runs on CPU without TensorFlow and in
parallel with MPI. In the same environment:

```bash
pip install python-ace==0.4.0rc1
```

It also provides `pace_activeset`, which [builds the active set](quickstart.md#build-active-set-for-gracefs-only)
(ASI) of GRACE/FS models fitted with `gracemaker`.

??? info "Install from source"
    ```bash
    git clone -b feature/grace_fs https://github.com/ICAMS/python-ace.git
    cd python-ace
    pip install .
    ```

    A source install may need `cmake` (`pip install cmake`).

---

## LAMMPS with GRACE

Build LAMMPS in the activated `grace` environment (it provides TensorFlow):

```bash
git clone -b grace --depth=1 https://github.com/yury-lysogorskiy/lammps.git
cd lammps
mkdir build
cd build
```

### TensorFlow pair styles (`grace`, `grace/fs`)

```bash
cmake -DCMAKE_BUILD_TYPE=Release -D BUILD_MPI=ON -DPKG_ML-PACE=ON -DPKG_MC=ON ../cmake
cmake --build . -- -j 8
```

The `cmake` output must contain `TensorFlow library is FOUND at ...`. To use a given TensorFlow
library file (`libtensorflow_cc.so`, `.dylib` or `.dll`) instead of the automatic search and
download, add `-D TF_LIB_FILE=/path/to/library`.

Only GRACE/FS models (`grace/fs`), no TensorFlow: add `-DNO_GRACE_TF=ON`.

### Kokkos build (GPU or OpenMP) {#kokkos-build}

Kokkos is available for `grace/fs` and for `grace/1l/kk`, `grace/2l/kk`, `grace/3l/kk`. The
`grace/{1l,2l,3l}/kk` styles read a `.npz` weights file from
[`grace_utils export_kokkos`](utilities.md#export-to-npz-for-lammps-kokkos-pair-style) and do not
need TensorFlow at run time. The same build also has the TensorFlow styles (`grace`), unless you
add `-DNO_GRACE_TF=ON`, so one binary runs all GRACE pair styles, e.g. the whole
[UQ tutorial](uq_workflows.md).

GPU (CUDA):

```bash
cmake ../cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_MPI=ON \
      -DPKG_ML-PACE=ON \
      -DPKG_MC=ON \
      -DPKG_KOKKOS=ON \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ENABLE_OPENMP=ON \
      -DKokkos_ARCH_NATIVE=ON
cmake --build . -- -j 8
```

`-DKokkos_ARCH_NATIVE=ON` tunes the code for the CPU of the build machine. The GPU is detected
separately: `cmake` asks the GPU of the build machine for its architecture.

??? info "Building for another GPU (e.g. on a login node without a GPU)"
    The binary runs only on the GPU architecture it was built for. When the build machine has no
    GPU, or a different one, set the flag of the target GPU:

    | GPU | Flag |
    |---|---|
    | H100, H200 | `-DKokkos_ARCH_HOPPER90=ON` |
    | A100, A30 | `-DKokkos_ARCH_AMPERE80=ON` |
    | RTX 30xx, A40, A6000 | `-DKokkos_ARCH_AMPERE86=ON` |
    | RTX 40xx, L40S | `-DKokkos_ARCH_ADA89=ON` |

CPU only (OpenMP): leave out `-DKokkos_ENABLE_CUDA=ON`. A CUDA build cannot run on a machine
without a GPU.

!!! warning "Use CUDA >= 12.8 for the Kokkos build"
    With CUDA 12.2-12.6 the build can hang for many hours (without any error
    message) while compiling `pair_grace_2l_kokkos.cpp` and
    `pair_grace_2l_cpu_kokkos.cpp`. This is an `nvcc` bug that is fixed in
    CUDA 12.8; lowering the optimization level with `-Xptxas` does not help.
    See the FAQ entry
    [LAMMPS KOKKOS build hangs for hours on `pair_grace_2l` files](faq.md#lammps-kokkos-build-hangs-for-hours-on-pair_grace_2l-files)
    for the diagnosis and for how to get a newer `nvcc` without touching your
    system modules.

Run with Kokkos (the binary is `lmp` in `build/`):

```bash
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lammps   # one GPU
lmp -k on t 8 -sf kk -pk kokkos newton on neigh half -in in.lammps   # 8 OpenMP threads
```
