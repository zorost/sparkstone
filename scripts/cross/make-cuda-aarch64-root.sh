#!/usr/bin/env bash
# Assemble a CUDA 13.0 root for cross-compiling Basalt to aarch64 + sm_121 on an x86-64 Linux machine, from PyPI
# wheels only (no NVIDIA apt repository needed): x86-64 nvcc/ptxas/headers and AArch64 libcudart/libcublas.
#   scripts/cross/make-cuda-aarch64-root.sh [DEST]     (default: ~/.cuda-aarch64)
# Then: cmake -S engine -B build-gb10 --toolchain cmake/aarch64-gb10-cross.cmake -DCUDAToolkit_ROOT=DEST/root \
#             -DCMAKE_CUDA_ARCHITECTURES=121 -DBASALT_MODELS=inkling2
# Needs: python3 -m venv, pip, and g++-13-aarch64-linux-gnu (apt).
set -euo pipefail
dest=${1:-$HOME/.cuda-aarch64}
mkdir -p "$dest"
python3 -m venv "$dest/venv-x86"
# nvcc, nvvm, crt and cccl must be the same release, or cicc emits PTX ptxas cannot read
"$dest/venv-x86/bin/pip" install -q "nvidia-cuda-nvcc==13.0.88" "nvidia-nvvm==13.0.88" "nvidia-cuda-crt==13.0.88" \
    "nvidia-cuda-cccl==13.0.85" "nvidia-cuda-runtime==13.0.96" "nvidia-cublas==13.0.2.14"
x86=$("$dest/venv-x86/bin/python" -c 'import nvidia, os; print(os.path.join(nvidia.__path__[0], "cu13"))')
mkdir -p "$dest/wheels" "$dest/aarch64"
pip download -q --no-deps --only-binary=:all: --platform manylinux2014_aarch64 --python-version 3.12 \
    -d "$dest/wheels" "nvidia-cuda-runtime==13.0.96" "nvidia-cublas==13.10.1.6"   # same libcublas.so.13 soname
for w in "$dest"/wheels/*aarch64*.whl; do python3 -m zipfile -e "$w" "$dest/aarch64"; done
lib=$dest/aarch64/nvidia/cu13/lib
for l in cudart cublas cublasLt; do ln -sf "lib$l.so.13" "$lib/lib$l.so"; done
root=$dest/root
rm -rf "$root" && mkdir -p "$root"
cp -r "$x86/bin" "$root/bin"
ln -s "$x86/include" "$root/include"
ln -s "$x86/nvvm" "$root/nvvm"
ln -s "$lib" "$root/lib"
ln -s lib "$root/lib64"
echo "CUDA aarch64 cross root: $root"
