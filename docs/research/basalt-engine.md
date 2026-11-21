# Basalt internals: what Sparkstone builds on

Notes from reading Basalt 0.5.6 (`97e2e71`, 2026-11-21) end to end for the port. Upstream docs are excellent;
this page summarizes what matters for GB10.

## Shape of the code

| | |
|---|---|
| Size | ~1,300 files, 28 MB; C++20 + CUDA 13; Python server |
| Licence | MIT (Jesús Gulfo), a fork of Strata (MIT); kernels adapted from Infernix (Apache-2.0); ggml extracted (MIT) |
| Engine | `src/` + `include/`: one library per model under `src/models/<arch>` (`qwen4exp` = Flash-Next, `deepseek41`, `qwen35` = 27B, `hy4flash`, `inkling2`), selectable with `-DBASALT_MODELS` |
| Server | `serve/`: standard-library Python; OpenAI and Anthropic APIs, web UI, tools, images, documents, video |
| Platform seam | All OS code in `src/platform/` (Linux + Windows), enforced by `tools/check_platform_boundary.sh` |
| Tests | ~190 CTest entries labelled `cpu` (no GPU needed; CI runs these) or `gpu` (kernel parity vs CPU references); quality gates in `tools/ppl_gate.py`, `tools/greedy_identity.py` |
| Bench | `basalt-bench` with seeded fixtures (`conv`, `struct`, `prose`, long prompts); `--community` writes a row for the public table |
| Release cadence | Fast: 0.5.5 (Inkling support) and 0.5.6 (Inkling latency) both landed around 2026-11-20/21 |

## The Inkling 2 Mini path (`src/models/inkling2`)

- **Pack** (`Inkling-2-Mini-NVFP4-q8.basalt`, 19.7 GB): the release's NVFP4 experts bit for bit (L0–19), MXFP8
  experts as Q8_0 (L20–23), other projections Q8_0, head BF16→Q8_0 at load, plus the vision encoder and the DSpark
  drafter.
- **Loop:** prelude L0–5, core L6–17 ×3, coda L18–23 = 48 visits, each with its own KV and conv state; passes 2–3
  mix messages from the previous pass through learned controllers.
- **Attention:** 24 Q / 8 KV heads of 128; 4 full-attention layers (8 visits) and 20 sliding-window (1,024)
  layers; no RoPE, a learned relative bias, per-head sinks, an output gate. KV ~32 KiB/position at fp16, 16 KiB at fp8.
- **MoE:** 128 experts, top-8 sigmoid, 1,152 wide, softcapped SwiGLU, no shared expert.
- **Decode:** windows of up to 8 rows (anchor + up to 7 drafts). DSpark drafts from the target's visits 5, 23 and 41;
  a draft policy picks the window by expected tokens per ms; prompt-lookup drafts for repeats (`--suffix-draft`).
- **Prefill:** chunks of up to 4,096 rows; NVFP4 experts on the block-scaled FP4 tensor cores (`mma.sync
  kind::mxf4nvf4`), the rest through MMQ (ggml-cuda's quantized matmul).
- **Memory planner:** if context does not fit beside every expert, the least-routed experts (shipped routing
  profile) go to page-locked RAM and are read over the bus. On a Spark, 128 GB unified memory means this should never
  trigger for Inkling.

## Multi-GPU (relevant for Twin)

- **Expert tier:** extra cards hold and compute experts every decode window (hottest first, by bandwidth).
- **Layer split:** each card runs a block of layers; the window is handed on once per cut; optional overlapped
  (pipelined) windows with rollback.
- **Tensor parallel:** priced but deliberately not built (host-staged exchanges at ~5.4 µs per 10 KiB cost as much
  as they save).
- The start-up planner prices all three in ms per window and picks one, a design that extends naturally to
  networked nodes with a different exchange cost.
- Inkling today: **one card only** (no tier/split).

## What was x86/sm_120-specific (and what the patches did)

| Assumption | Where | Patch |
|---|---|---|
| CUDA arch must be 120 | `CMakeLists.txt` configure check; five `120a-real` targets | 0001 |
| GPU must be 12.0 | `gpu_select.cpp`, `deepseek41/model.cpp` | 0001 |
| `immintrin.h` / `cpuid` / `xgetbv` | `platform/compiler.hpp`, `graph.cpp`, `pool.cpp` | 0002 |
| AVX2/AVX-512/VNNI CPU expert kernels | `src/kernels/cpu/*`, ggml dots | 0003 (left out off x86) |
| AVX2 vision GEMM | `src/vision/gemm_cpu*.cpp` | 0004 |
| (latent race in the RAM guard) | `platform/memory.cpp` | 0005 |

Not specific (carried over unchanged): every CUDA kernel's instruction set (consumer Blackwell `mma.sync`, TMA,
mbarrier, `setmaxnreg`; no tcgen05), the tokenizers, the server, the sampler, the drafters.
