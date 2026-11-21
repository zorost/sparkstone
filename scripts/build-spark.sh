#!/usr/bin/env bash
# Build Basalt natively on a DGX Spark (GB10: aarch64 host, sm_121 GPU, DGX OS 7 with CUDA 13).
#   scripts/build-spark.sh                 # Inkling 2 Mini only (the supported model on GB10 today)
#   MODELS=all scripts/build-spark.sh      # every model (Flash-Next and DeepSeek still need the x86 CPU tier: see docs)
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
engine=${ENGINE_DIR:-$root/engine}
[ -d "$engine" ] || "$root/scripts/fetch-engine.sh"
models=${MODELS:-inkling2}
jobs=${JOBS:-$(nproc)}
arch=$(uname -m)
[ "$arch" = aarch64 ] || echo "warning: host is $arch, not aarch64; for a cross build use cmake/aarch64-gb10-cross.cmake" >&2
command -v nvcc >/dev/null || export PATH=/usr/local/cuda/bin:$PATH
nvcc --version | tail -1
cmake -S "$engine" -B "$engine/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DBASALT_MODELS="$models" ${EXTRA_CMAKE:-}
cmake --build "$engine/build" -j "$jobs"
if [ ! -d "$engine/.venv" ]; then
  python3 -m venv "$engine/.venv"
  "$engine/.venv/bin/pip" install -q -r "$engine/requirements.txt"
fi
echo "built: $engine/bin/basalt ($(file -b "$engine/bin/basalt" | cut -d, -f1-2))"
