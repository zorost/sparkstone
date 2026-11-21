#!/usr/bin/env bash
# Check a DGX Spark before building or benchmarking: GPU, driver, CUDA, memory, clocks, page cache, CPU features.
# Prints one line per check; exit status 1 when a hard requirement fails.
set -uo pipefail
fail=0
ok()   { printf '  ok    %s\n' "$*"; }
warn() { printf '  warn  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=1; }
echo "sparkstone preflight on $(hostname) ($(uname -m), kernel $(uname -r))"
[ "$(uname -m)" = aarch64 ] && ok "aarch64 host" || bad "host is $(uname -m): GB10 builds need aarch64 (or the cross toolchain)"
if command -v nvidia-smi >/dev/null; then
  q=$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null | head -1)
  ok "GPU: $q"
  case "$q" in *12.1*) ok "compute capability 12.1 (sm_121)";; *) bad "expected compute capability 12.1 (GB10)";; esac
  drv=$(echo "$q" | awk -F', ' '{print $3}')
  [ "${drv%%.*}" -ge 580 ] 2>/dev/null && ok "driver $drv (R580+)" || bad "driver $drv: R580 or newer needed for CUDA 13"
  clk=$(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader 2>/dev/null | head -1)
  [ -n "$clk" ] && ok "SM clock now / max: $clk (a max near 650-720 MHz means a power/thermal fault: see docs/03)"
else
  bad "nvidia-smi not found"
fi
if command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ]; then
  v=$( (command -v nvcc >/dev/null && nvcc --version || /usr/local/cuda/bin/nvcc --version) | grep -oE 'release [0-9.]+' )
  case "$v" in *" 13."*) ok "CUDA $v";; *) bad "CUDA $v: Basalt needs CUDA 13";; esac
else
  bad "nvcc not found (install CUDA 13 or add /usr/local/cuda/bin to PATH)"
fi
mem_total=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo)
mem_avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
cached=$(awk '/^Cached/ {print int($2/1048576)}' /proc/meminfo)
ok "unified memory: ${mem_total} GiB total, ${mem_avail} GiB available, ${cached} GiB page cache"
[ "$mem_avail" -ge 40 ] || warn "under 40 GiB available: the 19.7 GB pack plus KV may not fit"
[ "$cached" -lt 20 ] || warn "large page cache: free it before loading (sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches')"
flags=$(grep -m1 -i '^Features' /proc/cpuinfo | cut -d: -f2)
for f in asimd sve2 bf16 i8mm; do
  echo "$flags" | grep -qw "$f" && ok "CPU feature $f" || warn "CPU feature $f not reported"
done
ok "page size $(getconf PAGESIZE) bytes; $(nproc) CPUs"
python3 -c 'import sys; assert sys.version_info >= (3,10)' 2>/dev/null && ok "python $(python3 -V | cut -d' ' -f2)" || bad "python 3.10+ needed"
command -v cmake >/dev/null && ok "cmake $(cmake --version | head -1 | awk '{print $3}')" || bad "cmake 3.24+ needed"
exit $fail
