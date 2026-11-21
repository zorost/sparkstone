#!/usr/bin/env bash
# Start Basalt on this Spark with configs/basalt_config.spark.json (or CONFIG=...).
#   scripts/serve-spark.sh                       # 127.0.0.1:8080
#   HOST=0.0.0.0 API_KEY_FILE=~/.basalt-key scripts/serve-spark.sh   # serve the link / LAN (always with a key)
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
engine=${ENGINE_DIR:-$root/engine}
config=${CONFIG:-$root/configs/basalt_config.spark.json}
args=(serve --config "$config")
[ -n "${HOST:-}" ] && args+=(--host "$HOST")
if [ -n "${API_KEY_FILE:-}" ]; then args+=(--api-key-file "$API_KEY_FILE")
elif [ "${HOST:-127.0.0.1}" != 127.0.0.1 ]; then echo "refusing to serve $HOST without API_KEY_FILE" >&2; exit 1; fi
if [ "${DROP_CACHES:-0}" = 1 ]; then sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; fi
exec "$engine/bin/basalt" "${args[@]}" "$@"
