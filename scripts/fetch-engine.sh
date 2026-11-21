#!/usr/bin/env bash
# Clone upstream Basalt at the pinned commit (UPSTREAM) into engine/ and apply the GB10 patch series.
#   scripts/fetch-engine.sh            # engine/ at the pinned commit + patches/*.patch, on branch gb10
#   ENGINE_DIR=/path scripts/fetch-engine.sh
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
repo=$(sed -n 's/^repo=//p' "$root/UPSTREAM")
commit=$(sed -n 's/^commit=//p' "$root/UPSTREAM")
dir=${ENGINE_DIR:-$root/engine}
if [ ! -d "$dir/.git" ]; then
  git clone --filter=blob:none "$repo" "$dir"
fi
cd "$dir"
git fetch --quiet origin "$commit" 2>/dev/null || git fetch --quiet origin
git checkout --quiet -B gb10 "$commit"
git -c user.name=sparkstone -c user.email=sparkstone@localhost am --quiet --3way "$root"/patches/*.patch
echo "engine ready: $dir (branch gb10, $(git rev-list --count "$commit"..HEAD) patches on $repo@${commit:0:12})"
