<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/sparkstone-lockup-dark.svg">
    <img src="assets/brand/sparkstone-lockup-light.svg" alt="Sparkstone" width="560">
  </picture>
</p>

<p align="center">
  <b>Native NVFP4 MoE inference on NVIDIA DGX Spark: the Basalt engine, ported to GB10 (sm_121 + AArch64).</b><br>
  One Spark, or two as <i>Sparkstone Twin</i>.
</p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-alpha%20%C2%B7%20builds%20verified%2C%20hardware%20runs%20pending-orange">
  <img alt="GPU" src="https://img.shields.io/badge/GPU-GB10%20sm__121-444">
  <img alt="host" src="https://img.shields.io/badge/host-AArch64-444">
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0%20%2B%20MIT%20patches-blue">
</p>

---

**Sparkstone** brings [Basalt](https://github.com/jesdga95/basalt), the engine that runs Thinking Machines'
*Inkling 2 Mini* at up to **628 tok/s on one RTX 5090**, to the **NVIDIA DGX Spark**: a 128 GB unified-memory
Grace Blackwell desktop. Upstream Basalt is x86 + sm_120 only. Sparkstone adds a small, upstreamable patch series
(sm_121 kernels, an AArch64 host, unified memory), plus what a Spark owner needs around it: a hardware self-test,
benchmark tooling, and a router that turns **two Sparks into one endpoint**.

## Status

| | |
|---|---|
| ✅ **GPU code builds for GB10** | Every kernel instruction (block-scaled FP4/FP8 `mma.sync`, FP4 converts, TMA, `setmaxnreg`) assembles for `sm_121a`. The Inkling engine links with 45 sm_121 cubins and no sm_120 leftovers |
| ✅ **Host builds for AArch64** | 107/107 host sources compile, and `basalt`, `basalt-bench`, `basalt-probe` and `basalt-vision` link as AArch64 executables |
| ✅ **Unit tests pass on AArch64** | 78/78 of Basalt's CPU tests pass (under `qemu-aarch64`). One upstream race the port exposed is fixed (patch 0005) |
| ✅ **Twin-Spark router** | OpenAI/Anthropic-compatible, conversation-affine, SSE streaming, failover. Standard library only, with 7 tests |
| ⏳ **Runs on a real Spark** | Next step: `scripts/selftest.sh` runs Basalt's GPU parity suite, a smoke test and the community benchmark on the hardware |
| 🔬 **Faster single streams on two Sparks** | Expert parallel over ConnectX-7, gated on a latency measurement ([docs/03](docs/03-dual-spark.md)) |

**What to expect on one Spark.** These are *projections*, not measurements
([model](docs/04-performance-model.md)): about **37–59 tok/s** for chat and **91–145 tok/s** for structured/JSON
output from Inkling 2 Mini at batch 1. vLLM/SGLang today manage about 40–52 tok/s. A Spark has 6.6× less memory
bandwidth than an RTX 5090, so the 5090's 613 tok/s will **not** carry over, and we won't claim it does.

## Quick start (on a DGX Spark)

```bash
git clone https://github.com/zoros-ai/sparkstone && cd sparkstone
```

```bash
scripts/spark-preflight.sh          # GPU 12.1, driver R580+, CUDA 13, memory, CPU features
```

```bash
scripts/build-spark.sh              # fetches Basalt at the pinned commit, applies patches/, builds for sm_121
```

```bash
scripts/download-model.sh           # Inkling-2-Mini-NVFP4-q8.basalt (19.7 GB)
```

```bash
scripts/serve-spark.sh              # OpenAI + Anthropic API on 127.0.0.1:8080
```

Test everything and collect one report:

```bash
FULL=1 scripts/selftest.sh
```

### Two Sparks (Sparkstone Twin)

```bash
sudo NODE=1 scripts/dual/setup-link.sh      # on Spark A; NODE=2 on Spark B: ConnectX-7 direct link, MTU 9000
```

```bash
SPARK_A=spark-a SPARK_B=spark-b API_KEY_FILE=~/.sparkstone-key scripts/dual/start-dual.sh
```

Both Sparks run the full model. [`router/spark_router.py`](router/spark_router.py) keeps each conversation on the
Spark that holds its prefix cache and spills to the other under load. That gives **2× throughput and 2× KV pool**
behind one endpoint. Why it is data-parallel first, and when expert-parallel pays off: [docs/03](docs/03-dual-spark.md).

## Repository map

| Path | What |
|---|---|
| [`patches/`](patches) | The engine changes: five commits on Basalt 0.5.6 (MIT, meant to go upstream) |
| [`UPSTREAM`](UPSTREAM) | The pinned Basalt commit |
| [`scripts/`](scripts) | Preflight, build, serve, self-test, model download; `dual/` for two Sparks, `cross/` for the CI cross toolchain |
| [`router/`](router) | The Twin router and its tests |
| [`tools/`](tools) | `perf_model.py` (roofline projections), `check_aarch64_host.py`, `cubin_archs.py` |
| [`configs/`](configs) | Basalt configs for a Spark (256K and 1M context, FP8 KV) |
| [`cmake/`](cmake) | The AArch64 + sm_121 cross-compile toolchain |
| [`docs/`](docs) | Feasibility, porting guide, dual-Spark design, performance model, benchmarking, roadmap, research |
| [`publish/`](publish) | Launch kit: GitHub release, Hugging Face card, X thread, website page, upstream PR, checklist |
| [`assets/brand/`](assets/brand) | Logo, Twin mark, lockups, social card (trademarked, see below) |

## Documentation

1. [Feasibility](docs/01-feasibility.md): the verdict and the evidence behind it
2. [Porting guide](docs/02-porting-guide.md): each patch explained; native and cross builds
3. [Two Sparks](docs/03-dual-spark.md): data parallel, expert parallel, prefill/decode split, decision gates
4. [Performance model](docs/04-performance-model.md): where the projections come from
5. [Benchmarking](docs/05-benchmarking.md): how to produce numbers worth publishing
6. [Roadmap](docs/06-roadmap.md)
7. [Legal, licensing and credits](docs/07-legal-and-licensing.md)
8. Research: [DGX Spark hardware](docs/research/dgx-spark-hardware.md) ·
   [Basalt internals](docs/research/basalt-engine.md) · [landscape](docs/research/landscape.md)

## Credits

Sparkstone stands on **[Basalt](https://github.com/jesdga95/basalt) by Jesús Gulfo** (MIT), which builds on
Strata, Infernix and ggml/llama.cpp. All of the engine's speed is their work; Sparkstone's part is the GB10 port and
what surrounds it. If Sparkstone is useful to you, please also [sponsor Basalt](https://github.com/sponsors/jesdga95).
*Inkling 2 Mini* is by Thinking Machines Lab (Apache-2.0). Thanks to the DGX Spark community whose public
measurements are cited in [docs/research](docs/research).

## License

- Sparkstone's own code and docs: **Apache-2.0** ([LICENSE](LICENSE)). Redistributions must keep [NOTICE](NOTICE).
- The engine patches in [`patches/`](patches): **MIT**, same as Basalt ([patches/LICENSE](patches/LICENSE)).
- Name and logos: trademarks of Zoros, not open-licensed ([TRADEMARKS.md](TRADEMARKS.md)).
- Model weights are not included; each model has its own licence.

Sparkstone is independent and is not affiliated with or endorsed by NVIDIA, the Basalt project or Thinking Machines
Lab. NVIDIA, DGX Spark and GB10 are trademarks of NVIDIA Corporation.

<p align="center"><sub>Made by <a href="https://zoros.com">Zoros</a>. Cite it with <a href="CITATION.cff">CITATION.cff</a>.</sub></p>
