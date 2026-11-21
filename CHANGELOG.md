# Changelog

## 0.1.0 (2026-11-21): "It builds"

- Patch series on Basalt 0.5.6 (`97e2e71`): sm_121 targets, AArch64 host, x86 CPU tier gated, portable vision GEMM,
  RAM-guard race fix.
- Verified: Inkling engine builds with sm_121 cubins only; AArch64 cross build links; `ctest -L cpu` 78/78 on AArch64
  (qemu-aarch64);
  x86 regression: all five models build for sm_120, `ctest -L cpu` 106/106.
- Twin router (data parallel) with tests; dual-Spark link setup and latency measurement scripts.
- Hardware self-test, preflight, build, serve and model-download scripts; cross toolchain from PyPI wheels.
- Docs: feasibility, porting guide, dual Spark, performance model, benchmarking, roadmap, legal; research notes.
- Brand: Sparkstone and Sparkstone Twin marks, lockups, social card; launch kit; website page.
