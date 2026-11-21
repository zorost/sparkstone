# Sparkstone v0.1.0: "It builds"

The Basalt inference engine, ported to NVIDIA DGX Spark (GB10: sm_121 GPU, AArch64 host).

## Highlights
- **GB10 builds:** five patches on Basalt 0.5.6 (MIT, offered upstream). Every NVFP4/FP8 kernel builds for `sm_121a`;
  the Inkling 2 Mini engine links with 45 sm_121 cubins.
- **AArch64:** the host builds and links natively or cross-compiled; Basalt's CPU test suite passes 78/78 on Arm.
- **Sparkstone Twin:** two Sparks behind one OpenAI/Anthropic endpoint (`router/spark_router.py`), with link setup
  and latency measurement scripts for the ConnectX-7 direct cable.
- **Self-test:** `FULL=1 scripts/selftest.sh` runs the GPU parity suite, a smoke test and the community benchmark,
  and packs a report.
- **Docs:** feasibility, porting guide, dual-Spark design, performance model, benchmarking rules, legal.

## Status
Alpha. Builds and CPU tests are verified; **GPU correctness and speed on real GB10 hardware are pending** (v0.2).
Performance figures in the docs are projections from a calibrated model and are labelled as such.

## Credits
Engine: [Basalt](https://github.com/jesdga95/basalt) by Jesús Gulfo (MIT). Model: Inkling 2 Mini by Thinking Machines
Lab (Apache-2.0). Not affiliated with or endorsed by NVIDIA, the Basalt project or Thinking Machines Lab.
