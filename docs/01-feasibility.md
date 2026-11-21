# 01 · Feasibility: Basalt on DGX Spark (one and two)

**Verdict: yes, it can be done, and most of the hard part is already done in this repository.** Basalt's GPU code
is consumer-Blackwell code, and GB10 is a consumer-Blackwell GPU (sm_121). The engine needed a retarget, not a
rewrite. With the four patches in [`patches/`](../patches), an Inkling 2 Mini build of Basalt **compiles, links,
and starts on AArch64 with sm_121 GPU code**. Everything that needs the real machine (correctness, speed, memory
behaviour) is still open and listed below.

This page sums up the evidence. The detail is in [02 · porting guide](02-porting-guide.md),
[03 · dual Spark](03-dual-spark.md), [04 · performance model](04-performance-model.md) and
[research/](research/).

## The question

The trigger was a post by @crazyMLguy on X (2026-11-21): *"Basalt: Running Inkling 2 Mini at 613 Tok/s on a Single RTX 5090"*.
Basalt 0.5.5 and later run Thinking Machines' Inkling 2 Mini (26.4B total / 3.4B active, a looped MoE, native
NVFP4) at up to 628 tok/s on structured output, 14,100 tok/s prefill and 1M context on one RTX 5090. Can the same
engine run on NVIDIA DGX Spark, and on two of them, in a form worth publishing?

## What Basalt is (and why it did not run on a Spark)

- A C++/CUDA engine (MIT, Jesús Gulfo; a fork of Strata). Python server with OpenAI and Anthropic APIs.
  Hand-written kernels for NVFP4 MoE decode, speculative decoding with each model's own drafter (DSpark for
  Inkling), and prefill on block-scaled FP4 tensor cores. Details: [research/basalt-engine.md](research/basalt-engine.md).
- Upstream supports **x86-64 hosts and sm_120 GPUs only**. The CMake configure stops on any other architecture, the
  runtime refuses any GPU other than 12.0, and the CPU expert tier (experts spilled to RAM and computed on the
  CPU) is AVX2/AVX-512 code.
- A DGX Spark is **GB10: an sm_121 GPU and an AArch64 CPU (10× Cortex-X925 + 10× Cortex-A725) sharing 128 GB of
  LPDDR5x at 273 GB/s**. So upstream fails at configure, at the GPU check, and when compiling the CPU tier.

## Evidence

| Question | Result | How it was checked |
|---|---|---|
| Do the kernels' special instructions exist on sm_121? | **Yes.** Block-scaled FP4/FP8 `mma.sync` (`kind::mxf4nvf4`, `kind::mxf8f6f4`), `cvt.*e2m1x2`, TMA, mbarrier and `setmaxnreg` all assemble for `sm_121a`, and for `compute_120f → sm_121` | Probe kernel compiled with CUDA 13.0.88 `nvcc`/`ptxas` (see [02 · porting guide](02-porting-guide.md)) |
| Is there any tcgen05/TMEM code (datacenter Blackwell only, absent on GB10)? | **No.** Basalt uses consumer `mma.sync` paths | grep over all 300+ CUDA sources |
| Are there arch-feature guards that would silently compile to empty code on 12.1? | **None found** (no `__CUDA_ARCH_FEAT_SM120_ALL` / `== 1200` checks) | grep |
| Does the whole Inkling engine build for sm_121? | **Yes, with 0 errors.** All 45 cubins in `bin/basalt` are sm_121 (ELF arch byte `0x79`) | Full x86-host build with `-DCMAKE_CUDA_ARCHITECTURES=121`, ELF headers inspected with [`tools/cubin_archs.py`](../tools/cubin_archs.py) |
| Does the host code build for AArch64? | **Yes.** 107/107 host C/C++ files compile. Before the patches, 19 failed, all on x86 SIMD | [`tools/check_aarch64_host.py`](../tools/check_aarch64_host.py) with `aarch64-linux-gnu-g++-13 -march=armv9-a` |
| Does it link into an AArch64 binary? | **Yes.** `bin/basalt`, `basalt-bench`, `basalt-probe`, `basalt-vision` are AArch64 ELF linking `libcudart.so.13` / `libcublas.so.13`, with no unresolved symbols | Cross build: [`cmake/aarch64-gb10-cross.cmake`](../cmake/aarch64-gb10-cross.cmake) + [`scripts/cross/make-cuda-aarch64-root.sh`](../scripts/cross/make-cuda-aarch64-root.sh) |
| Does the AArch64 binary run? | **Its host side starts and parses its CLI** under QEMU AArch64 emulation (no GPU there) | `qemu-aarch64 bin/basalt --help` |
| Does the AArch64 host code behave correctly? | **Yes: 78/78 of Basalt's CPU tests pass** (tokenizers, sampler, server protocol, platform layer, GPU planning, drafting policy). An intermittent failure turned out to be a latent upstream race, fixed in patch 0005 | `ctest -L cpu` on the cross build via `CMAKE_CROSSCOMPILING_EMULATOR=qemu-aarch64` |
| Does the x86 / RTX 5090 build still work with the patches? | **Yes.** All 5 models build with 0 errors; `ctest -L cpu` passes **106/106**; the binary holds 123 sm_120 cubins, as before | Full 5-model sm_120 build with `-DBUILD_TESTING=ON` on x86-64 |
| Does it produce correct tokens on a Spark? How fast? | **Not yet known: needs hardware** | [05 · benchmarking](05-benchmarking.md) is the procedure |

## What it is likely to be worth

Decode at batch 1 is limited by memory bandwidth. One Spark has about 6.6× less bandwidth than an RTX 5090
(273 vs 1,792 GB/s), so the 5090's headline numbers will not carry over. The question is how Basalt compares with
what Spark owners can run today. The model in [04](04-performance-model.md) is calibrated on Basalt's own 5090
measurements. Every Spark number below is a projection, not a measurement:

| Inkling 2 Mini, batch 1, tok/s | plain | conversation | struct | prose |
|---|---:|---:|---:|---:|
| vLLM / SGLang on one Spark today (community, with DSpark) | ~38 | 40–52 | n/a | n/a |
| **Basalt on one Spark (projection, ρ 0.3–0.7)** | **~36** | **37–59** | **91–145** | **25–40** |
| Basalt on one RTX 5090 (measured upstream) | 264 | 347 | 628 | 296 |

So the realistic pitch is **"the fastest way to run Inkling 2 Mini on a DGX Spark for structured and agentic
output, competitive for chat"**, *if* the hardware run confirms it. It is not "613 tok/s on a Spark". Claiming
that would be wrong.

## Two Sparks

Inkling 2 Mini (19.7 GB pack) fits on one Spark many times over, even with a 1M-token context, so a second Spark
does not add capacity for this model. What it can add:

1. **Data parallel (shipped here):** both Sparks run the whole model and [`router/spark_router.py`](../router)
   balances conversations across them with prefix-cache affinity. This gives 2× aggregate throughput and 2× the KV
   pool, works today, and needs no engine changes.
2. **Expert parallel over ConnectX-7 (research):** split each layer's 128 experts 64/64 across the nodes. The
   model says about 1.6× per stream on drafted windows, **but only if** a small-message hop costs ≤ 25 µs. Measure
   that first with [`scripts/dual/check-link.sh`](../scripts/dual/check-link.sh). Forum reports range from
   ~10 µs to 1–1.5 ms at default power settings, and community tensor-parallel results for small-active MoE models
   on two Sparks are only 1.17–1.56×.
3. **Prefill/decode split (research):** one Spark reads long prompts and ships the KV cache (~16 KiB/token with
   `--kv fp8`) to the other over the 200 GbE link. This is useful for agent workloads with 100K+ token prompts.
4. **Bigger models (later):** Hy4-flash's NVFP4 pack (158.7 GB) does not fit one Spark but fits two. This needs
   Basalt's layer split carried over the network (a host-staged transport, since GB10 has no GPUDirect RDMA).

Details and decision gates: [03 · dual Spark](03-dual-spark.md).

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Kernels tuned for 170 SMs / 96 MB L2 run poorly on 48 SMs / ~24 MB L2 | Slower than projected | Profile on hardware, re-tune grid sizes and split-K. The 5090 code paths are a correct starting point |
| Basalt's "fits on card vs spill to pinned RAM" planner misreads unified memory | Wrong spill decisions, or OOM | Inkling needs ~20 GB of 119 GiB usable. Drop the page cache before loading; validate `cudaMemGetInfo` on hardware |
| NVML reports no memory size/bandwidth on GB10 | Multi-GPU ordering only | Irrelevant with one GPU. Fallback patch if needed |
| Numerics differ on AArch64 (CPU-side sampling, tokenizer) | Output drift | Run upstream's quality gate (`tools/ppl_gate.py`) and greedy-identity tool on the Spark |
| Upstream moves fast (0.5.5 → 0.5.6 in a day) | Patches rot | Pin `UPSTREAM`; offer the patches upstream (they are small, behaviour-neutral on x86, and the author asks for Blackwell test help) |
| Flash-Next / DeepSeek need the CPU expert tier | Only Inkling runs on AArch64 for now | NEON ports of the CPU kernels are on the [roadmap](06-roadmap.md) |

## Bottom line

- **Build: done.** A reproducible patch series, native and cross builds, and verification tools.
- **Run on one Spark: next step**, which needs one DGX Spark for an afternoon ([05](05-benchmarking.md)).
- **Two Sparks: useful now as data parallel.** Faster single streams depend on a latency measurement.
- **Publishing:** the materials are drafted in [`publish/`](../publish). Gate every public number on a hardware run,
  and coordinate with Basalt's author.
