# 04 · Performance model

All Spark figures in this repository come from [`tools/perf_model.py`](../tools/perf_model.py). They are
**projections, until a hardware run replaces them**. Run the model yourself:

```bash
python3 tools/perf_model.py
```

## Method

Batch-1 decode is memory-bound: the time per window is the bytes read divided by bandwidth.

**Bytes per token** (Basalt pack: NVFP4 experts in layers 0–19, Q8_0 experts in 20–23, Q8_0 dense):

| | |
|---|---|
| Experts | 8 routed × (44 NVFP4 visits × 4.48 MB + 4 Q8_0 visits × 8.46 MB) = **1.85 GB** |
| Dense + head | 1.10 GiB on the card; the layers' share is read on each of 48 visits (the core runs 3×), the head once = **1.87 GB** |
| **Total** | **3.72 GB per token** |

**Drafting.** An 8-row verification window reads the dense weights once and the *union* of the rows' experts. With
independent routing that is 51.6 of 128 experts per layer; real consecutive tokens overlap, which the model
captures with ρ (0 = independent, 1 = identical; 0.5 gives 29.8). Tokens per window come from the upstream
acceptance rates: conversation 68.0 %, struct 97.5 %, prose 50.7 %.

**Calibration.** Basalt's measured plain decode on the RTX 5090 (264 tok/s) is 61 % of the 5090's streaming
bandwidth. The same efficiency is applied to GB10's *measured* 219 GB/s (spec 273 GB/s).

## Results

| tok/s, batch 1 | plain | conversation | struct | prose |
|---|---:|---:|---:|---:|
| RTX 5090, measured (Basalt 0.5.6) | 264 | 347 | 628 | 296 |
| RTX 5090, model ρ=0.3 / 0.5 / 0.7 | 264 | 272 / 334 / 434 | 668 / 822 / 1068 | 184 / 226 / 294 |
| **1× Spark, model ρ=0.3 / 0.5 / 0.7** | **36** | **37 / 45 / 59** | **91 / 112 / 145** | **25 / 31 / 40** |
| 2× Spark data parallel, aggregate (ρ=0.5) | 72 | 91 | 223 | 61 |
| 2× Spark expert parallel, 10 µs hops | 42 | 73 | 180 | 49 |

The model reproduces the 5090's measured conversation figure at ρ≈0.4 and its prose figure at ρ≈0.7. Struct
is over-predicted at ρ=0.5, because the measured 628 includes per-window overheads that the pure byte count
leaves out. **So read the Spark struct range as optimistic and the conversation range as central.**

## Sanity check against the field

| Inkling 2 Mini on one DGX Spark | Decode tok/s | Source |
|---|---:|---|
| vLLM, DSpark ×7, single stream | 40–43 | outsmartchad/inkling-2-mini-single-dgx-spark-vllm |
| vLLM vs SGLang, short prompt | 52.3 / 49.6 | thefullctx/inkling-2-mini-vllm-dgx-spark |
| vLLM, no drafter | 37.8 | same |
| "~75 tok/s" (relayed on X, attributed to S. Chintala) | 75 | unverified, setup unknown |
| **Sparkstone (model)** | **37–59 conv, 91–145 struct** | this page |

The model's plain-decode ceiling (36 tok/s) matches vLLM's no-drafter measurement (37.8), which supports the byte
count. Where Basalt should pull ahead is the drafted, structured case: its DSpark path verifies 8 rows per weight
read with kernels built for exactly that.

## What can make it wrong

- **Fixed costs on 48 SMs.** Launch latency and small kernels are a bigger share of a Spark window than of a 5090
  window, which pushes results down.
- **L2 reuse.** The core's weights are read 3 times per token. A 24 MB L2 holds little of them; the 5090's 96 MB
  holds more. That could flatter the 5090 calibration.
- **Unified memory contention.** CPU traffic (server, tokenizer) shares the same 273 GB/s.
- **Clocks.** Forum reports describe GB10 units stuck at 650–720 MHz after power or thermal faults. The self-test
  records clocks for this reason.
