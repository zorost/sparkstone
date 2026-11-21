# 05 · Benchmarking: numbers worth publishing

A number goes into a README, a post or a slide only if it was produced the way described here, on real hardware,
with the raw report kept in `bench/results/`.

## The one command

```bash
FULL=1 scripts/selftest.sh
```

| Stage | What | Why |
|---|---|---|
| 1 preflight | GPU 12.1, driver, CUDA, memory, page cache, clocks, CPU features | Rules out a misconfigured box (e.g. clocks stuck at 650 MHz) |
| 2 build | Patched engine, `-DBUILD_TESTING=ON`, cubins checked with `cubin_archs.py` | The binary under test is the one described |
| 3 cpu tests | `ctest -L cpu` | The AArch64 host code |
| 4 **gpu tests** | `ctest -L gpu`: every kernel against a CPU reference **on GB10** | **The correctness gate.** Publish nothing if this fails |
| 5 probe + smoke | `basalt-probe` (machine rates) and one JSON chat request through the server | End-to-end sanity |
| 6 bench | `basalt-bench --community` | Upstream's fixed preset (`conv`, `struct`, `prose`, a 16K prompt), seeded, same settings as every published Basalt row, so Spark results compare directly with the RTX 5090's |
| 7 link (Twin) | `check-link.sh`: RDMA bandwidth and **small-message latency** | The expert-parallel go/no-go |

The archive `selftest-<host>-<date>.tar.gz` holds every log, `nvidia-smi -q`, `/proc/cpuinfo` and `meminfo`, the
versions, and the community benchmark row.

## Rules for publishable numbers

1. **Stage 4 passed** on the same build.
2. **Quality checked.** Greedy output matches the 5090 reference within upstream's tolerance
   (`tools/greedy_identity.py`), and the perplexity/KLD gate passes (`tools/ppl_gate.py`,
   [method](https://github.com/jesdga95/basalt/blob/main/docs/precision/quality.md)). Speed with wrong tokens is not speed.
3. **Same preset** as upstream (`--community`). Report drafting acceptance next to tok/s: struct output drafts at
   97.5 % and is not representative of chat.
4. **State the conditions:** DGX OS and driver version, power mode, clocks, context length, KV format, single
   stream or batched.
5. **Compare fairly.** vLLM and SGLang baselines run on the same Spark, same prompts, same lengths, both with
   drafting on and off. Use the recipes in [research/landscape.md](research/landscape.md).
6. **Keep raw data.** Add the archive to `bench/results/` and link it from any public claim.

## Results table (fill from hardware)

| Setup | conv tok/s | struct tok/s | prose tok/s | 16K prefill tok/s | accept (conv/struct/prose) | report |
|---|---:|---:|---:|---:|---|---|
| 1× DGX Spark, Sparkstone | — | — | — | — | — | — |
| 1× DGX Spark, vLLM (same prompts) | — | — | — | — | — | — |
| 1× DGX Spark, SGLang (same prompts) | — | — | — | — | — | — |
| 2× DGX Spark, Sparkstone Twin (aggregate, 8 streams) | — | — | — | — | — | — |
| RTX 5090, Basalt 0.5.6 (upstream) | 347 | 628 | 296 | 12,310 | 68.0 / 97.5 / 50.7 % | upstream docs |

## Submitting upstream

`basalt-bench --community` writes a row in Basalt's own format. A GB10 row in upstream's
[community table](https://basalt.jgulfo.com/community/) is the most credible public proof the port works, and it is
good for both projects.
