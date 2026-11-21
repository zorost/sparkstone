#!/usr/bin/env bash
# Download the Inkling 2 Mini Basalt pack (19.7 GB) into models/.
#   scripts/download-model.sh [DEST]
set -euo pipefail
dest=${1:-$(cd "$(dirname "$0")/.." && pwd)/models}
command -v hf >/dev/null || pip install -q -U "huggingface_hub[cli]"
hf download jesdga/Inkling-2-Mini-Basalt Inkling-2-Mini-NVFP4-q8.basalt --local-dir "$dest"
ls -la "$dest"
