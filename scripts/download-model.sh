#!/usr/bin/env bash
# Download the Inkling 2 Mini Basalt pack (19.7 GB) into models/.
#   scripts/download-model.sh [DEST]
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
dest=${1:-$root/models}
# DGX OS (Ubuntu 24.04) refuses pip installs into the system Python, so the CLI gets its own venv
hf=$(command -v hf || true)
if [ -z "$hf" ]; then
  [ -x "$root/.venv-tools/bin/hf" ] || { python3 -m venv "$root/.venv-tools" &&
    "$root/.venv-tools/bin/pip" install -q -U huggingface_hub; }
  hf=$root/.venv-tools/bin/hf
fi
"$hf" download jesdga/Inkling-2-Mini-Basalt Inkling-2-Mini-NVFP4-q8.basalt --local-dir "$dest"
ls -la "$dest"
