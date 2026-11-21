# Cross-compile Basalt for DGX Spark (GB10: aarch64 host, sm_121 GPU) from an x86-64 Linux machine.
#
#   cmake -S engine -B build-gb10 --toolchain cmake/aarch64-gb10-cross.cmake \
#         -DCUDAToolkit_ROOT=/path/to/cuda-aarch64-root -DCMAKE_CUDA_ARCHITECTURES=121
#
# CUDA_AARCH64_ROOT (or CUDAToolkit_ROOT) must hold an x86-64 nvcc (bin/, include/, nvvm/) with AArch64 libraries in
# lib/ (libcudart, libcublas): scripts/cross/make-cuda-aarch64-root.sh assembles one from the PyPI wheels.
# The result is used for build and link verification in CI; build natively on the Spark for real use.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(_gcc_ver 13)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc-${_gcc_ver})
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++-${_gcc_ver})
set(CMAKE_CUDA_HOST_COMPILER aarch64-linux-gnu-g++-${_gcc_ver})
if(DEFINED ENV{CUDA_AARCH64_ROOT} AND NOT CUDAToolkit_ROOT)
  set(CUDAToolkit_ROOT $ENV{CUDA_AARCH64_ROOT})
endif()
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_ROOT}/bin/nvcc)
# the toolkit's libraries are the target's; programs (nvcc) are the build machine's
set(CMAKE_FIND_ROOT_PATH ${CUDAToolkit_ROOT} /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
