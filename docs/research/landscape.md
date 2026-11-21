# Landscape: running Inkling 2 Mini (and friends) on DGX Spark

As of 2026-11-21. Items marked *(reported)* come from community posts and repos, not our own runs.

## Engines that run on a Spark today

| Engine | Inkling 2 Mini on 1 Spark | Strengths | Gaps Sparkstone targets |
|---|---|---|---|
| **vLLM** (≥ 0.35 / `vllm-openai:inkling-2-mini`) | 40–52 tok/s single stream with DSpark ×7; ~91–112 tok/s aggregate at 4–5 streams *(reported)* | Batching, ecosystem, official Spark playbook, TP=2 recipes | Python/Triton overheads at batch 1; ~105 GiB memory use reported |
| **SGLang** (`sglang:inkling-2-mini`) | ~50 tok/s short prompt; faster than vLLM at 15.6K context (45 vs 38); 3.5K tok/s prefill *(reported)* | Prefill, memory (65 GiB), FA4 backend | Same batch-1 overheads |
| **llama.cpp / Ollama** (GGUF) | Supported (Unsloth GGUF, 16 GB); no published Spark Inkling number found | Easy install, low memory | On a 5090 Basalt is 3.7× faster than llama.cpp for conversation (347 vs 94.5 tok/s) |
| **TensorRT-LLM** | No Inkling recipe found | NVIDIA-tuned | Model support lag |
| **DGPP** (forum, "a GB10-optimized C++/CUDA inference engine") | Unknown | GB10-first design | Track it: closest in spirit to Sparkstone |
| **Basalt upstream** | ❌ (x86 + sm_120 only) | Fastest known batch-1 Inkling on a 5090 | — that's the port |

## Two Sparks today

| Model | 1 → 2 Sparks (TP=2) | Note *(reported)* |
|---|---|---|
| gpt-oss-120b (patched vLLM) | 51.8 → 80.9 tok/s | +56 % |
| gpt-oss-120b (another repo) | 57–60 → 72 | +22 % |
| Qwen3.8-Flash-Next NVFP4 | 41.7 → 51.9 | +24.5 % |
| Qwen3-30B-A3B NVFP4 | — | 1.17× |

Small-active MoE models barely scale with TP on two Sparks. That is why Sparkstone Twin ships data parallel first.

## Positioning

- **For Spark owners:** the lowest-latency single-stream Inkling 2 Mini, especially for structured/agentic output
  (to be proven on hardware), plus a turnkey two-Spark setup.
- **For the Basalt community:** a second hardware platform (and an upstream contribution) with a test harness.
- **For Zoros:** a credible, verifiable performance story on NVIDIA's developer desktop, a natural lead into
  support and deployment work.

## Sources

- NVIDIA `dgx-spark-playbooks` (connect-two-sparks, nccl, vllm, sglang, setup scripts): https://github.com/NVIDIA/dgx-spark-playbooks
- vLLM Inkling on one Spark: https://github.com/outsmartchad/inkling-2-mini-single-dgx-spark-vllm
- vLLM vs SGLang on Spark: https://github.com/thefullctx/inkling-2-mini-vllm-dgx-spark
- Two-Spark results: NVIDIA developer forums (vLLM 0.17 MXFP4 patches thread), https://github.com/mark-ramsey-ri/vllm-dgx-spark,
  https://ai-muninn.com/en/blog/qwen38-flash-next-tp2-two-dgx-sparks, https://gpuwerk.com/dgx-spark/guides/best-models-2026/
- DGPP: https://forums.developer.nvidia.com/t/dgpp-a-gb10-optimized-c-cuda-inference-engine/383406
- Hardware facts and their sources: [dgx-spark-hardware.md](dgx-spark-hardware.md)
