# X / Twitter launch thread

Post **only after** `scripts/selftest.sh` is green on a Spark and the numbers below are replaced by measured ones
(`{{...}}` placeholders). Attach `assets/brand/png/social-card-1200x630.png` to post 1.
Tag @jesdga95's project only after telling them (see upstream-pr.md).

---

**1/**
Thinking Machines' Inkling 2 Mini now runs on the Basalt engine on @NVIDIA DGX Spark.

Meet ⚡🪨 **Sparkstone**: Basalt ported to GB10 (sm_121 + Arm), plus a one-endpoint setup for two Sparks.

{{conv}} tok/s chat · {{struct}} tok/s structured · 1M context · open source
🧵

**2/**
Why it wasn't possible before: Basalt (the engine behind the "613 tok/s on one RTX 5090" posts) targets x86 + sm_120.
A Spark is an Arm CPU + sm_121 GPU sharing 128 GB of memory.

Same tensor cores, though. The port is 5 small patches. No kernel rewrites.

**3/**
What's in the patches:
• sm_121a builds of every NVFP4 kernel (block-scaled mma.sync, TMA)
• AArch64 host build (no x86 SIMD needed: the GPU reads all 128 GB)
• 78/78 CPU tests passing on Arm
• one upstream race fix the port exposed

**4/**
Honest numbers: a Spark has 273 GB/s vs the 5090's 1,792 GB/s. Decode is bandwidth-bound, so you won't see 613 tok/s on a Spark.

What you do get: {{conv}} tok/s chat (vs {{vllm}} for vLLM on the same box), measured with Basalt's own seeded benchmark.

**5/**
Two Sparks? Sparkstone Twin puts both behind one OpenAI/Anthropic endpoint: conversation-aware routing, prefix-cache affinity, failover.
2× throughput, 2× KV pool, zero engine changes.

Expert-parallel over the 200 GbE link is next, gated on a latency measurement we publish too.

**6/**
All credit for the engine goes to Basalt by Jesús Gulfo, built on Strata, Infernix and ggml. Please sponsor them.
Sparkstone is the GB10 port + tooling. Patches are MIT and offered upstream.

**7/**
Try it:
github.com/zoros-ai/sparkstone
`FULL=1 scripts/selftest.sh` gives you a full report from your own Spark.

Built by @{{zoros_handle}}. #DGXSpark #LocalAI #LLM #OpenSource
