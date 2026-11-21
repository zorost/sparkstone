# 02 · Porting guide: Basalt → GB10

The port is five small patches on Basalt 0.5.6 (`97e2e71`): **14 files, +310 / −58 lines**. They are written to
leave x86/sm_120 behaviour unchanged, so they can go upstream as-is.

```
0001 build: target sm_121 (GB10 / DGX Spark) as well as sm_120
0002 platform: build the host code on AArch64
0003 kernels_cpu: leave the x86 expert kernels out of non-x86 builds
0004 vision: portable CPU GEMM tile for non-x86 hosts
0005 platform: a RAM-guard pause waits out a judgement in flight
```

## Why the port is small

| | RTX 5090 | DGX Spark (GB10) | Consequence |
|---|---|---|---|
| Compute capability | 12.0 | **12.1** | Same consumer-Blackwell ISA. An `sm_120a` cubin will not load on 12.1, so rebuild with `121a` |
| Tensor cores | `mma.sync` block-scaled FP4/FP8 | same | Basalt's NVFP4 kernels carry over. Neither chip has tcgen05/TMEM |
| SMs / L2 | 170 / 96 MB | **48 / ~24 MB** | Correct as is. Tuning is phase 2 |
| Memory | 32 GB GDDR7, 1,792 GB/s + PCIe to host RAM | **128 GB LPDDR5x unified, 273 GB/s** | No "spill to RAM over the bus": everything is GPU-addressable |
| Host CPU | x86-64 (AVX2/AVX-512) | **AArch64** (Cortex-X925/A725, NEON/SVE2) | The x86 SIMD CPU tier is not needed for Inkling. It is left out, not ported |

## Patch by patch

### 0001: build for sm_121

- `CMakeLists.txt`: `CMAKE_CUDA_ARCHITECTURES` may be `120` or `121` (one per build; mixing them is refused,
  because the arch-specific targets must name one). New variables `BASALT_SM` and
  `BASALT_SM_ARCH_SPECIFIC` (`121a-real`) replace five hard-coded `120a-real`. `BASALT_BUILT_SM` is passed to the code.
- `src/program/gpu_select.cpp`, `src/models/deepseek41/model.cpp`: the GPU gate accepts the compute capability the
  binary was built for. A 12.x card of the other minor version gets a message that says which
  `-DCMAKE_CUDA_ARCHITECTURES` to rebuild with. An sm_120 build's messages are unchanged.
- Tests: the GPU-selection fixtures' "12.0" cards follow the build's architecture.

*Why not a fat binary (`120a;121a`)?* A fat binary would be possible, but it doubles the CUDA build time (already
~35 min for Inkling alone on 4 cores), and every Spark builds from source anyway. A family target (`120f`) would
cover both chips with one cubin, but needs CMake ≥ 3.31.8 to spell. It is a possible follow-up.

### 0002: AArch64 host

`include/basalt/platform/compiler.hpp` held all of the x86 assumptions outside the kernels. Now:

| Helper | x86 | AArch64 |
|---|---|---|
| `cpuid`, `xgetbv0` | instruction | read zeros, so every x86 feature probe answers "absent" |
| `cpu_relax` | `pause` | `yield` |
| `prefetch_l2` (new) | `prefetcht1` | `__builtin_prefetch(p, 0, 2)` |
| `cpu_has_avx512_bf16` | probe | `false` |

`graph.cpp` and the CPU worker pool call these instead of raw intrinsics. Before this patch, 19 of 119 host files
failed to compile for AArch64. After the series, all 107 files in an AArch64 build compile.

### 0003: CPU expert tier left out off x86

Upstream can hold some MoE experts in system RAM and compute them on the CPU (AVX2/AVX-512/VNNI). On GB10 that tier
has no purpose: the GPU reads all 128 GB directly, and the CPU shares the same 273 GB/s memory anyway. So instead of
porting about 6,000 lines of SIMD:

- CMake sets `BASALT_HOST_X86`. Off x86, the 11 SIMD kernel files, the two ggml dot-product builds and their 7
  tests/benches are not built.
- `src/kernels/cpu/portable/x86_tier_absent.cpp` provides the 30 symbols the rest of the engine links against.
  Probes return "not supported". Row kernels stop the process with a clear message if they are ever called; they
  can only be reached through a planner bug, so failing loudly is safer than computing wrong values.

Scope: **Inkling 2 Mini needs no CPU kernel.** Flash-Next (`qfn_run.cpp` requires AVX2) and DeepSeek's CPU tier
need NEON ports before they run on AArch64 (see the [roadmap](06-roadmap.md)).

### 0004: vision CPU GEMM

The vision encoder's CPU GEMM had only an AVX2 tile. A plain C++ tile with the same 8-lane partial-sum
structure (GCC vectorizes it to NEON) is used off x86. `gemm_cpu_portable.cpp` stands in for the AVX-512 BF16 file.

### 0005: an upstream race, found by the port

Under `qemu-aarch64`, `platform_memory_test` failed about half the time. The cause is not ARM memory ordering; all
the atomics involved are sequentially consistent. It is a time-of-check/time-of-use race. The RAM guard's thread
checked "paused?", read `/proc/meminfo`, then judged the reading, so a `ram_guard_pause(true)` that landed in between
let one stale judgement through after the pause had returned. Native x86 rarely hits the window; emulation's slow
`/proc` read hits it often. Now the pause takes the judgement lock, and the guard drops a reading taken across a
pause: 0 failures in 12 runs. This one is worth sending upstream on its own.

## Building

### On a DGX Spark (native)

```bash
scripts/build-spark.sh            # = fetch-engine.sh + cmake -DCMAKE_CUDA_ARCHITECTURES=121 -DBASALT_MODELS=inkling2
```

Needs DGX OS 7 (Ubuntu 24.04 aarch64), CUDA 13.0+, driver R580+, CMake ≥ 3.24, GCC 13. Check with
`scripts/spark-preflight.sh`.

### Cross-compiling on x86-64 (CI, or no Spark at hand)

```bash
sudo apt install g++-13-aarch64-linux-gnu qemu-user
```

```bash
scripts/cross/make-cuda-aarch64-root.sh ~/.cuda-aarch64          # CUDA 13.0 from PyPI wheels, no NVIDIA apt repo
```

```bash
cmake -S engine -B build-gb10 --toolchain cmake/aarch64-gb10-cross.cmake \
      -DCUDAToolkit_ROOT=$HOME/.cuda-aarch64/root -DCMAKE_CUDA_ARCHITECTURES=121 -DBASALT_MODELS=inkling2 \
      -DBUILD_TESTING=ON -DCMAKE_CROSSCOMPILING_EMULATOR=/usr/bin/qemu-aarch64
```

```bash
cmake --build build-gb10 -j
```

```bash
cd build-gb10 && QEMU_LD_PREFIX=/usr/aarch64-linux-gnu LD_LIBRARY_PATH=$HOME/.cuda-aarch64/aarch64/nvidia/cu13/lib ctest -L cpu
```

Tests that start child processes need `qemu-aarch64` registered with binfmt_misc (`qemu-user-binfmt`).

Two pitfalls hit while building this:
- The PyPI CUDA wheels must all be the **same release**. A newer `nvidia-nvvm` beside nvcc 13.0 emits PTX 9.5, which
  ptxas 13.0 rejects ("Unsupported .version 9.5").
- `-arch=sm_121a` on the nvcc command line still produced `compute_121` PTX in our probe. Use
  `-gencode arch=compute_121a,code=sm_121a`, or CMake's `121a-real`, as the patches do.

## Verification tools

| Tool | Checks |
|---|---|
| [`tools/cubin_archs.py`](../tools/cubin_archs.py) `bin/basalt` | every embedded cubin is `sm_121` (`{'sm_121': 45}`) |
| [`tools/check_aarch64_host.py`](../tools/check_aarch64_host.py) `BUILD` | every host source compiles with `aarch64-linux-gnu-g++ -march=armv9-a` |
| `ctest -L cpu` | Basalt's CPU suite: **78/78 on AArch64** (Inkling build), **106/106 on x86-64** (all five models, regression) |
| `ctest -L gpu` | kernels vs CPU references **on the GPU**: run on the Spark (`scripts/selftest.sh` stage 4) |

## Still open, and needing the hardware

1. `ctest -L gpu` on GB10: the real correctness gate for every kernel, Inkling's DSpark and attention parity
   included.
2. Greedy-identity and perplexity/KLD (`tools/greedy_identity.py`, `tools/ppl_gate.py`) against the 5090 output.
3. Unified memory: how `cudaMemGetInfo` reads with a large page cache, and whether the planner keeps every expert
   resident (it should: about 20 GB of about 119 GiB usable). Drop caches before loading if not.
4. NVML on GB10: memory size and bus width may read "not supported". That only matters for multi-GPU ordering.
5. Tuning for 48 SMs: grid sizes, split-K and the decode window's block counts. Profile with Nsight Systems first.
