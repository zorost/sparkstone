<!-- Research notes compiled 2026-11-21 by an automated research pass. Source tags: [R] = primary file read
(cloned repo), [S] = search snippet only, [M] = background knowledge, UNVERIFIED = could not confirm.
Re-verify every [S]/[M] item on hardware or a machine with full web access before quoting it publicly. -->

# DGX Spark (GB10) research for porting an sm_120 engine, and for running it on two Sparks

**Access limits.** The proxy blocked direct fetches of docs.nvidia.com, developer.nvidia.com, forums.developer.nvidia.com, build.nvidia.com, huggingface.co, thinkingmachines.ai, vllm.ai and nvidia.com. Only `git clone` from github.com worked.

What that means for each source tag below:
- **[R]** – I read the primary file myself. These are cloned repos: `NVIDIA/dgx-spark-playbooks` (which is the source of the build.nvidia.com/spark pages), CUTLASS, llama.cpp, vLLM, SGLang, and two community Inkling-on-Spark repos.
- **[S]** – I only saw search-engine snippets of the page, not the page.
- **[M]** – my own background knowledge, not checked.
- **UNVERIFIED** – the claim could not be confirmed.

Before quoting anything marked [S], re-check it on a machine that can reach those hosts.

Note: the local repo `upstream Basalt` is the engine being ported. Its `CMakeLists.txt` (lines 3–49) stops with a hard error for any CUDA arch other than 12.0. Its CPU expert kernels are x86-only (AVX2, AVX-512 and VNNI, in `src/kernels/cpu/*`).

---

## 1. Facts table

### GPU
| Item | Value | Source / confidence |
|---|---|---|
| Compute capability | **12.1 (sm_121)** | [R] playbooks `nvidia/playbook-cutile-kernels/README.md` ("sm_121 (12.1)"); [S] https://developer.nvidia.com/cuda/gpus. High |
| SM count | **48** | [R] same playbook file, spec table "SMs: 48"; [S] forum deviceQuery-style posts. High |
| CUDA cores | 6,144 (48 SMs × 128) | [S] https://docs.nvidia.com/dgx/dgx-spark/hardware.html. High |
| Tensor cores | 5th-gen Blackwell, **consumer SM12x style**: `mma.sync` with block-scaled FP4/FP6/FP8 (`kind::mxf8f6f4`, `kind::mxf4nvf4` with 2X ue8m0 and 4X ue4m3 scales). The 4X ue8m0 variant needs CUDA 13.1 or later. | [R] CUTLASS `include/cute/arch/config.hpp`: SM121/121A enable `CUTE_ARCH_MMA_SM120_ENABLED`, `MXF4NVF4_*`, `F8F6F4`. High |
| tcgen05 / TMEM | **Not available.** CUTLASS turns on `TCGEN05_*` and `TCGEN05_TMEM` only for SM100, 101, 103 and 107. There is also no multicast and no 2-SM MMA. | [R] CUTLASS config.hpp; [S] forum thread "Dearest CUTLASS TEAM…tcgen05 FP4 support for GB10" https://forums.developer.nvidia.com/t/dearest-cutlass-team-when-the-hell-are-you-going-to-properly-fix-tcgen05-fp4-support-for-dgx-spark-gb10-sm121/359598. High |
| TMA | **Available.** CUTLASS defines `CUTE_ARCH_TMA_SM90_ENABLED` and `TMA_SM120_ENABLED` for SM121A and SM121F. `LDSM/STSM_SM100A`, `LOAD256/STORE256` and CLC are also enabled for 121a/121f. | [R] CUTLASS config.hpp and config.h. High |
| NVFP4 / MXFP8 | Native in hardware: `mma.sync` block-scale with e2m1 values and ue4m3/ue8m0 scales | [R] CUTLASS; [R] vLLM `vllm/models/inkling2/nvidia/ops/moe_sm120.py` ("SM120/121 expert GEMMs with native NVFP4 MMA and dequantized MXFP8") |
| FP4 peak | Marketing figure: 1 PFLOP NVFP4 with 2:4 sparsity, so about 500 TFLOPS dense. A forum user reproduced about 1 PFLOP sparse. Best community dense CUTLASS NVFP4 GEMM: **356 TFLOPS**. | [S] https://forums.developer.nvidia.com/t/gb10-really-does-hit-1-pflop-nvfp4-2-4-sparse-measured-with-an-open-source-tool-to-reproduce-it/373618 ; [S] https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960. Medium |
| BF16 GEMM (measured) | **90.1 TFLOP/s** at 4096³ | [R] `outsmartchad/inkling-2-mini-single-dgx-spark-vllm/results/tuning/hardware.json`. Medium (one sample) |
| L2 cache | **24 MB** (user-reported device query; no official spec found) | [S] forum benchmark output. Medium |
| Shared memory per SM | Probably 100 KB per SM and 99 KB per block, as for CC 12.0 | [M] UNVERIFIED for 12.1. Query `cudaDevAttrMaxSharedMemoryPerMultiprocessor` on the device |
| Clocks | Users report a max SM clock of 3003 MHz and an application clock of 2418 MHz. There are also reports of clocks stuck at 650–721 MHz under thermal or power faults. | [S] https://forums.developer.nvidia.com/t/dgx-spark-gb10-gpu-clock-pinned-at-721-mhz-under-full-load-no-throttling-not-liftable-via-nvidia-smi/376039 and the "15W / 650MHz" thread. Low–medium |

### Memory
| Item | Value | Source / confidence |
|---|---|---|
| Capacity / type | **128 GB LPDDR5x**, unified between CPU and GPU. NVIDIA's NCCL playbook also lists a **64 GB** Spark variant, and its vLLM-cluster playbook has recipes for "two 64GB Sparks". | [R] playbooks (`playbook-nccl/README.md`: "64GB or 128GB Unified Memory"; `playbook-vllm-cluster`). High |
| Bandwidth | **273 GB/s** peak. Measured device copy: **219 GB/s** read+write combined. | [R] playbook-cutile-kernels spec table; [R] hardware.json above |
| CPU–GPU coherence | NVLink-C2C with ATS. On ATS systems, HMM is disabled and managed memory is CPU-accessible without migration. | [S] CUDA Programming Guide §2.6 https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html. Medium |
| cudaMalloc | The porting guide says memory "returned by the pinned device memory allocators (e.g. cudaMalloc) cannot be coherently accessed by the CPU complex nor by I/O peripherals like PCI Express devices (e.g. the Mellanox NIC)". **So GPUDirect RDMA is not supported.** Its advice: use `cudaHostAlloc` buffers registered with `ib_reg_mr`, and check `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED` / `DMA_BUF_SUPPORT` at runtime. | [S] https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html ; [S] https://forums.developer.nvidia.com/t/dgx-spark-gpudirect-rdma/348787 . High (official wording, snippet only) |
| cudaMemGetInfo | The porting guide says it may under-report how much can actually be allocated, because the OS can reclaim DRAM pages. Many NVIDIA playbooks tell users to flush the buffer cache with `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` when memory runs short, e.g. txt2kg: "Linux buffer cache consuming GPU memory". | [S] porting guide PDF https://docs.nvidia.com/dgx/dgx-spark-porting-guide/1.0.0/dgx-spark-porting-guide.pdf ; [R] playbooks `txt2kg/README.md:162`, `trt-llm`, `playbook-vllm`, `llama-cpp` and others |
| cudaMallocManaged | The **RTX Spark** porting guide (a sibling product) says it is "technically supported" but uses a compatibility path that can degrade performance, and recommends `cudaHostRegister` only for buffers both sides touch. | [S] https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/latest/uma/umacuda.html. Medium; not confirmed for DGX Spark specifically |
| Other OOM behaviour | One report says the system hangs ("zombie") instead of returning a CUDA OOM. Another says unified-memory allocations bypass cgroup memory limits. | [S] https://forums.developer.nvidia.com/t/dgx-spark-becomes-unresponsive-zombie-instead-of-throwing-cuda-oom/353752 |
| Page size, `cudaHostRegister` on an mmap'd file, `pageableMemoryAccess` value | **UNVERIFIED.** Check `getconf PAGESIZE` and the `cudaDevAttrPageableMemoryAccess` / `…UsesHostPageTables` attributes on the device. | — |

### CPU and OS
| Item | Value | Source / confidence |
|---|---|---|
| CPU | 20 cores: 10 Cortex-X925 + 10 Cortex-A725, in two clusters of 5+5. L3 is 16 MiB on cluster 0 and 8 MiB on cluster 1, plus a 16 MB SLC. X925 runs up to 3.9–4.0 GHz; A725 at 2.8 GHz. | [S] GB10 whitepaper https://www.nvidia.com/content/dam/en-zz/Solutions/dgx-spark/nvidia-gb10-superchip-whitepaper.pdf ; [S] Chips and Cheese https://old.chipsandcheese.com/?p=37687 |
| ISA | Armv9.2 with NEON, SVE and SVE2 (whitepaper). BF16 and I8MM are mandatory from Armv8.6, so they should be present. SVE vector length is 128-bit. | SVE2: [S]; BF16, I8MM and VL=128: [M] UNVERIFIED (check `/proc/cpuinfo` flags `bf16 i8mm sve2`) |
| OS | DGX OS 7, based on Ubuntu 24.04, aarch64. Launch kernel was 6.11; DGX OS 7.4.0 moved to 6.17; later updates ship kernel 7.0.0-10xx-nvidia. | [S] https://docs.nvidia.com/dgx/dgx-spark-porting-guide/latest/porting/software-requirements.html ; [S] DGX OS release notes |
| CUDA / driver | CUDA **13.0** (13.0.2 in OS 7.4.0) with driver **580.126.09**. Machines seen in Nov 2026 run 580.178.04 (from a benchmark log), and one forum report mentions 595.91.07. | [S] release notes; [R] `thefullctx/inkling-2-mini-vllm-dgx-spark/README.md` ("driver 580.178.04, 121.6 GiB usable unified RAM") |

### Two-Spark networking
| Item | Value | Source / confidence |
|---|---|---|
| NIC | ConnectX-7 with 2× QSFP ports, 200 GbE. **Each physical port appears as two netdevs/HCAs** (e.g. `enp1s0f1np1`/`rocep1s0f1` plus `enP2p1s0f1np1`/`roceP2p1s0f1`). This is because the SoC connects the NIC over two PCIe Gen5 x4 links, each worth about 100 Gb/s. | [R] `connect-two-sparks/README.md` ("each physical port has two logical interface"); [S] ServeTheHome https://www.servethehome.com/the-nvidia-gb10-connectx-7-200gbe-networking-is-really-different/ |
| Getting 200G | One cable is enough, but **both logical interfaces need IPs**. The official example uses 192.168.100.x/24 and 192.168.101.x/24. "When two QSFP cables are connected, all four interfaces must be assigned IP addresses to obtain full bandwidth." | [R] connect-two-sparks playbook |
| MTU | Use **9000 for two Sparks on a direct link** and 1500 for ring or switch layouts. This matches what NVIDIA Sync Cluster Assistant sets. | [R] `playbook-connect-multiple-sparks/README.md:401-406` |
| Raw RDMA bandwidth | `ib_write_bw`, one QP per half-link: 92.57 + 97.28 = **189.85 Gb/s** | [R] `playbook-connect-multiple-sparks/assets/performance_benchmarking_guide.md` |
| NCCL pass threshold | NVIDIA's cluster-setup script requires NCCL average busbw ≥ **21.875 GB/s (175 Gb/s)** for a direct link and ≥ 10 GB/s for a ring | [R] `multi-sparks-through-switch/assets/spark_cluster_setup/spark_cluster_setup.py:53-54` |
| Measured NCCL all_reduce | About 179.5 Gb/s (≈22.4 GB/s) busbw at 2 GiB using both HCAs; 22–24 GB/s is commonly reported. Many threads report 2.8–15 GB/s caused by misconfiguration (one HCA only, socket fallback, firmware). | [S] https://forums.developer.nvidia.com/t/dual-dgx-spark-roce-bandwidth-expectations/369673 and others. Medium |
| Small-message latency | **No nccl-tests 8-byte numbers found.** One forum anecdote reports **1–1.5 ms** RDMA latency by default, attributed to CPU C-states. The playbook describes how to measure with `ib_write_lat` / `ib_read_lat` but gives no figures. | UNVERIFIED and important; measure it yourself |
| NCCL build | NCCL v2.30.7-1 (one playbook says v2.31.2-1), built with `NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"`, OpenMPI from `/usr/lib/aarch64-linux-gnu/openmpi` | [R] `nccl/README.md`, `playbook-nccl/README.md` |
| NCCL tuning | llama.cpp sets `config.maxCTAs = 8` on GB10 because "NCCL picks 64 channels by default, which is slower and uses more memory than 8". vLLM cluster recipes set `NCCL_MAX_CTAS=8`, `NCCL_BUFFSIZE=1048576`, and `VLLM_ENABLE_ROCE_ALLREDUCE=1` with `VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB`. That RoCE allreduce is in the community image `eugr/spark-vllm-b12x`, **not in upstream vLLM** (I grepped and found nothing). | [R] llama.cpp `ggml/src/ggml-cuda/ggml-cuda.cu:~1721`; [R] `playbook-vllm-cluster/README.md:484-560` |
| Official playbooks | connect-two-sparks, connect-three-sparks, multi-sparks-through-switch, playbook-connect-multiple-sparks (includes a `setup-cluster.sh --topology direct --mtu 9000` script), nccl, vllm and playbook-vllm-cluster (TP=2 over `--nnodes 2`), trt-llm, sglang, llama-cpp | [R] https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia |

---

## 2. CUDA compile compatibility (sm_120 → sm_121)

- **An `sm_120a` cubin will NOT run on 12.1.** Architecture-specific (`a`) code runs only on the exact compute capability it was built for, and has no forward compatibility for either PTX or cubin. [S] CUDA Programming Guide §5.1 https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html ; [S] https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/
- **A plain `sm_120` cubin does run on 12.1**, under normal binary compatibility (same major version, equal or higher minor). This is why stock PyTorch sm_120 wheels run on GB10. But a plain sm_120 build cannot use block-scaled FP4 MMA, which needs the `a` or `f` target. [S] https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744
- **Family targets (`f`)** arrived in CUDA 12.9. `compute_120f` code runs on the same major version at equal or higher minor, so 12.0 and 12.1. NVIDIA's own example is that 100f runs on sm_100 and sm_103. I could not read the Table 28 rows that list the exceptions. CUTLASS defines separate `SM120F` and `SM121F` feature macros and turns on the FP4 MMA atoms for `SM121`/`SM121A`. Confidence: medium–high. To be safe, **build `121a-real` alongside `120a-real`** — this is exactly what llama.cpp does with CUDA ≥ 12.9 (`ggml/src/ggml-cuda/CMakeLists.txt`: "the architectures 120a-real and 121a-real should work with basically any CMake version"). llama.cpp also notes that `120f-virtual` needs CMake ≥ 3.31.8 (below 4.0) or ≥ 4.0.2, because older CMake rejects the `f` suffix.
- **Toolkit versions:** CUTLASS gates SM121 behind `nvcc` ≥ 12.9 (`include/cutlass/arch/config.h:227-246`). DGX OS ships CUDA 13.0. The `MXF4NVF4_4X_UE8M0` MMA needs CUDA 13.1 or later.
- **Ecosystem examples:**
  - llama.cpp has `GGML_CUDA_CC_GB10 = 1210` with GB10-specific tuning tables (mmvq, mmq-few, fattn) and forces FP32 accumulation on GB10.
  - vLLM's Inkling2 path uses `is_device_capability_family(120)` for its SM12x BF16 GEMV and NVFP4 MoE kernels (Triton `tl.dot_scaled` e2m1). Its custom Inkling allreduce only supports CC 9 and 10 (`ops/allreduce.py:86`), so it falls back to NCCL on Spark.
  - vLLM Spark recipes set `CUTE_DSL_ARCH=sm_121a`.
  - Forum pain points: TransformerEngine lacked sm_121 MXFP8/MXFP4 support; cuBLAS/cuDNN lack tuned sm_121 kernels; for 1-node CUTLASS MoE, a "K=64 TMA descriptor" fix was needed for SM120/SM121 block-scaled MoE.

---

## 3. LLM decode benchmarks on DGX Spark

Single Spark:

| Model / engine | Decode | Source |
|---|---|---|
| gpt-oss-120b MXFP4, llama.cpp | tg32 42 → 52.9 t/s (Oct 2025), about 59 t/s later with the native MXFP4 PR; 38.8 t/s at 32k depth | [S] https://github.com/ggml-org/llama.cpp/discussions/16578 , https://jetsonhacks.com/wp-content/uploads/2025/10/spark-llamacpp-bench.html |
| gpt-oss-120b, vLLM TP1 | 51.8–60 t/s | [S] forum thread below |
| Qwen3.8-Flash-Next NVFP4, vLLM TP1 | 41.7 t/s | [S] https://ai-muninn.com/en/blog/qwen38-flash-next-tp2-two-dgx-sparks |
| Qwen2-7B BF16, cuTile (NVIDIA playbook) | 18.5 t/s | [R] |

**Inkling 2 Mini on one Spark:**
- vLLM, DSpark ×7, single stream: **40–43 tok/s**; 5 streams: about 108–112 tok/s aggregate. Without the drafter: 37.8 tok/s. Prefill: 3,149 tok/s at 32K and 2,464 tok/s at 128K. Peak memory about 78–82 GiB at `gpu-mem-util` 0.70. KV capacity was about 599k tokens with auto dtype and 836k with FP8 KV. [R] https://github.com/outsmartchad/inkling-2-mini-single-dgx-spark-vllm
- vLLM vs SGLang head-to-head:
  - Short-prompt decode: vLLM 52.3 tok/s (peak 62.8), SGLang 49.6.
  - At 15.6K context: decode vLLM 38.4 vs SGLang 45.1; prefill SGLang 3,480 tok/s vs vLLM 1,690.
  - 4 streams: vLLM 91.2 tok/s aggregate.
  - Memory used: SGLang 65 GiB vs vLLM 105 GiB.
  - [R] https://github.com/thefullctx/inkling-2-mini-vllm-dgx-spark
- Other reports: Classmethod 37.8–39.4 tok/s; Tech2Wild about 47 tok/s at full 1M context with FP8 KV (3.3M-token KV pool). [S] https://dev.classmethod.jp/en/articles/dgx-spark-inkling-2-mini-first-touch/ , https://x.com/Tech2Wild/status/2123954941630750733
- **The "~75 tok/s by Soumith Chintala" figure is only relayed second-hand** (X summaries, daily.dev). I could not find the original post or its setup. It is about 1.5–1.8× every independent measurement. UNVERIFIED; it is plausible only with code-heavy output and high draft acceptance.

**Two Sparks (TP=2), all [S] and community-reported:**

| Model | 1 Spark → 2 Sparks | Speedup | Source |
|---|---|---|---|
| gpt-oss-120b (patched vLLM 0.17) | 51.8 → 80.9 t/s | +56% | https://forums.developer.nvidia.com/t/vllm-0-17-0-mxfp4-patches-for-dgx-spark-qwen3-5-35b-a3b-70-tok-s-gpt-oss-120b-80-tok-s-tp-2/362824 |
| gpt-oss-120b (another repo) | 57–60 → 72 t/s | about +22% | https://github.com/mark-ramsey-ri/vllm-dgx-spark |
| Qwen3.8-Flash-Next NVFP4 | 41.7 → 51.9 t/s | +24.5% | ai-muninn link above |
| Qwen3-30B-A3B NVFP4 | — | only **1.17×** | https://gpuwerk.com/dgx-spark/guides/best-models-2026/ |

The pattern: MoE models with few active parameters gain little from TP=2. **No dual-Spark Inkling 2 Mini numbers were found.**

---

## 4. Inkling 2 Mini architecture

Mostly from [R] the vLLM recipe `recipes/models/thinkingmachines/Inkling-2-Mini.yaml` and the [R] SGLang cookbook `docs/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini.mdx`.

- **Repo:** `thinkingmachines/Inkling-2-Mini` (revision `ed40e7da` was used in the benchmarks). Released 2026-11-20.
- **License:** Apache 2.0. [S] https://artificialanalysis.ai/articles/inkling-2-mini , daily.dev. I could not read the model card itself.
- **Size:** 26.4B total parameters.
  - Active-parameter figures differ by source: 3.4B (OpenRouter), 3.5B (vLLM), 3.1B unique (Artificial Analysis).
  - About 5.4B "effective" counting the three loop passes (SGLang and Artificial Analysis).
  - Checkpoint about 18.8–20 GB, including a 0.64 GB DSpark draft (`mtp.safetensors`).
- **Layout:** 24 layers.
  - Prelude: layers 0–5.
  - **Loop core: layers 6–17, run three passes per token.** Thinking Machines' chart numbers these 7–18 (1-indexed).
  - Coda: layers 18–23.
  - That gives **48 layer visits per token**.
  - From the second pass on, each core sublayer input reads a gated message (a rank-64 controller) from the previous pass's attention-residual blocks.
  - **Every core layer keeps its own KV cache and conv-state per pass** (48 of each). SGLang has an option to share KV across passes, which roughly doubles the KV pool, but DSpark refuses a non-default number of passes.
- **Width and MoE:**
  - Hidden size 2304; expert intermediate size 1152.
  - **128 routed experts, top-8, no shared expert.**
  - Vocabulary 201,024 (vLLM config default; not confirmed for Mini specifically).
- **Attention:**
  - Relative-bias attention with a gate and a sink, and no RoPE ([S] basalt issue #650).
  - **20 sliding-window layers + 4 full-attention layers.** The default sliding window is 1024 (config default, not confirmed for Mini).
  - 24 query heads, 8 KV heads; head_dim is probably 128 (config default, not confirmed).
  - Four short convolutions per layer; block attention-residuals.
  - **KV cache ≈ 69 KB per token** (vLLM recipe: "~69 KB of cache per token").
- **Precision:**
  - Routed experts are **NVFP4 in layers 0–19 and MXFP8 in layers 20–23**, with activations quantized to the same format.
  - Attention, routers, attention-residual queries, loop controllers and the LM head are BF16/FP32.
  - Per Thinking Machines, NVFP4 was used for the expert layers from pretraining through RL.
  - SGLang: "the NVFP4 experts cannot be TP-sharded, because their activation scales must cover each expert's full input width" → **use EP = TP**.
- **Context:** 1,048,576 tokens. vLLM and SGLang Spark recipes cap at 262,144, although Tech2Wild ran the full 1M on one Spark with FP8 KV.
- **Vision:** a 23-layer ViT image encoder; text and image in, text out.
- **DSpark drafter:** a 3-layer parallel block drafter plus a recurrent head. It reads target hidden states at 4 layer visits, 2 of them inside the loop. vLLM proposes 7 draft tokens per step (probabilistic); SGLang verifies 8.
- **Engine recipes:**
  - vLLM needs ≥ 0.35 or nightly, image `vllm/vllm-openai:inkling-2-mini`, and lists `dgx_spark_gb10: verified`.
  - SGLang uses image `lmsysorg/sglang:inkling-2-mini` (PR #51926). On Spark it runs TP1/EP1 with `--attention-backend fa4`, `--mem-fraction-static 0.45` and `MAX_JOBS=4`, and scores gsm8k 93.6–93.9%.

---

## 5. Porting implications (sm_120 / RTX 5090 x86 → GB10 aarch64)

**Build**
1. Allow arch 121 in `CMakeLists.txt` (it currently hard-fails at line 49). Build `120a-real;121a-real` with CUDA ≥ 12.9 (13.0 on DGX OS), or use `120f`. The existing `120a` cubins will not load on GB10.
2. Where code checks `__CUDA_ARCH__ == 1200`, extend it to 1210, or use `__CUDA_ARCH_FAMILY_SPECIFIC__`.
3. The PTX instruction set is the same as consumer Blackwell: `mma.sync` block-scaled e2m1/e4m3 with ue4m3/ue8m0 scales, TMA, `ldmatrix`/`stmatrix`, 256-bit loads, and CLC. There is still no tcgen05/TMEM, so engine code with no sm_100 dependence should port without changes.

**Tuning**
4. GB10 has **48 SMs** against the 5090's 170 ([M]), and about **24 MB L2** against 96 MB ([M] for the 5090).
5. Re-tune for the smaller GPU: grid sizes, split-K, persistent-kernel CTA counts and occupancy.
6. Re-check any kernel that assumes a large L2 for KV or weight reuse.

**Bandwidth**
7. Peak memory bandwidth is 273 GB/s against 1,792 GB/s ([M]), about 6.6× lower; measured copy bandwidth is about 219 GB/s.
8. Decode is fully bandwidth-bound. Measured Inkling weight traffic is about **5.5 GB per token** (active experts plus BF16 attention and dense weights across 48 visits), so the plain-decode ceiling is about 40 tok/s at measured bandwidth and about 50 at peak.
9. The levers are speculative decoding (DSpark verifies about 8 tokens per weight read) and quantizing the BF16 attention/dense weights that are re-read three times inside the loop.

**Memory model**
10. Drop the "experts in system RAM, CPU computes experts" tier: all 128 GB is GPU-addressable, and CPU compute over the same LPDDR5x is not a win.
11. The x86 AVX2/AVX-512/VNNI kernels will not compile on aarch64. Exclude them from the build, or port them to NEON/SVE2 (I8MM/BF16-dot assumed available, UNVERIFIED) only if a CPU fallback is wanted.
12. Allocate the weights with `cudaMalloc` (the fastest GPU path).
13. Loading safetensors through the page cache can double the resident footprint (page cache plus the `cudaMalloc` copy), and `cudaMemGetInfo` under-reports free memory. Options: read with O_DIRECT, call `posix_fadvise(DONTNEED)` after upload, or document `drop_caches`.
14. Zero-copy `cudaHostRegister` of an mmap'd file is UNVERIFIED on Spark; test it before relying on it.
15. Use `cudaHostAlloc` buffers for anything the NIC touches.

**Networking**
16. The NIC cannot DMA `cudaMalloc` memory: no GPUDirect, no `nvidia-peermem`. NCCL handles this internally by staging through host memory.
17. A custom transport would have to be RoCE verbs over `cudaHostAlloc` memory.

**Capacity**
18. Inkling 2 Mini on one Spark: about 20 GB weights plus 69 KB/token of KV. 262K context ≈ 18 GB; 1M ≈ 69 GB (about half with FP8 KV). One Spark holds the full 1M context.

## 6. Using two Sparks (back-of-envelope)

Assumptions: 5.5 GB of weight traffic per token, 219 GB/s effective bandwidth per Spark, about 22 GB/s NCCL busbw, and a hidden vector of 2304 × 2 bytes = 4.6 KB per token per allreduce.

- **Data parallel (two independent replicas) — recommended default.**
  - The model fits easily on one Spark, so DP needs zero interconnect.
  - It gives about 2× aggregate throughput (about 2 × 110 tok/s at 5 streams in vLLM) and doubles the KV/context pool.
  - Single-stream speed does not change. A simple router over the 10 GbE management link is enough.
- **TP/EP = 2 (attention split by heads 12q/4kv per node; MoE as EP with 64 experts per node; NVFP4 experts cannot be TP-split).**
  - Per-token weight time drops from about 25 ms to roughly 12.5–16 ms. It is not a clean halving: at batch 1, top-8 routing splits unevenly, so the busier node holds about 5 of the 8 experts.
  - The cost is about 96 or more latency-bound allreduce/all-to-all calls per token (48 visits × attention + MoE, plus loop-message and DSpark traffic).
  - At 10–20 µs per collective ([M] typical for CX-7 RoCE) that adds about 1–2 ms, giving roughly 60–75 tok/s plain decode, about 1.5–1.8× faster.
  - At the forum-reported **1–1.5 ms RDMA latency** (C-states), TP becomes far slower than one Spark.
  - Real dual-Spark results for small-active MoE models are only 1.17–1.25×.
  - **Measure 8-byte to 8-KB all_reduce latency first** (`all_reduce_perf -b 8 -e 64K` with both HCAs, MTU 9000, and C-states/power management tuned) before investing in TP.
  - Use CUDA-graph-friendly NCCL (`maxCTAs=8`) or a custom low-latency RoCE allreduce (the approach of the community vLLM image).
- **Pipeline parallel.** The model does not need the capacity, and at batch 1 the stages run serially, so latency gets no better. It only helps throughput with micro-batching, which DP already provides more simply. Splitting prelude+core and coda across nodes needs one 4.6 KB transfer per token per boundary, which is cheap but brings no speed gain.
- **Speculative split** (drafter on Spark A, target on Spark B): the drafter is 0.64 GB and reads target hidden states at 4 visits, so the cross-node latency per step would cancel any gain. Not worth doing.
- **Prefill/decode disaggregation** (prefill on one Spark, decode on the other, KV shipped at about 69 KB/token): a 128K prompt is about 9 GB of KV, roughly 0.4 s at 22 GB/s, against about 50 s of prefill. This is viable for long-context agent workloads. SGLang validated 1P1D for this model on GB300, not on Spark.

## 7. Could not verify
- Shared memory per SM and the official L2 size for CC 12.1.
- The official GPU boost clock.
- The exact rows of the CUDA family table for 120f → 121.
- The page size and whether pageable memory access works / `cudaHostRegister` works on mmap'd files on Spark.
- Inter-Spark NCCL small-message latency.
- CPU `bf16`/`i8mm` flags and SVE vector length.
- Soumith Chintala's 75 tok/s setup.
- Inkling 2 Mini's head_dim, vocabulary size, sliding-window size and exact license text, since huggingface.co was unreachable.
- GPUDirect behaviour beyond NVIDIA's "not supported" statement. One community repo claims a DMA-BUF mode reaching 22–23 GB/s, which contradicts it.
