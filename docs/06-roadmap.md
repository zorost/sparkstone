# 06 · Roadmap

## v0.1: "It builds" ✅ (this release)

- [x] Patch series: sm_121 + AArch64 + upstream race fix (5 patches, MIT)
- [x] Native build script; cross toolchain from PyPI wheels; `cubin_archs.py` and `check_aarch64_host.py`
- [x] 78/78 CPU tests on AArch64 (emulated)
- [x] Twin router (data parallel), link setup and measurement scripts
- [x] Performance model, research notes, launch kit, brand

## v0.2: "It runs" (needs one Spark for a day)

- [ ] `scripts/selftest.sh` green on GB10: **GPU parity suite**, smoke test
- [ ] Greedy-identity and PPL/KLD gate vs the 5090 reference
- [ ] First benchmark row; vLLM/SGLang baselines on the same box
- [ ] Fix whatever the hardware finds: unified-memory planning, NVML fallbacks, page-cache handling
- [ ] Offer patches 0001–0005 upstream ([publish/upstream-pr.md](../publish/upstream-pr.md))

## v0.3: "It's fast" (GB10 tuning)

- [ ] Nsight Systems profile of a decode window: launch gaps, occupancy on 48 SMs
- [ ] Re-tune decode-window kernels (block counts, split-K) for 48 SMs / ~24 MB L2
- [ ] CUDA-graph the whole window if it is not already; measure fixed cost per window
- [ ] Unified memory: test zero-copy (`cudaHostRegister` of the mmap'd pack) to skip the load copy and halve the
      load-time footprint
- [ ] A `120f` family build (one binary for RTX 50 and GB10) once CMake ≥ 3.31.8 is common

## v0.4: "Twin" (two Sparks)

- [ ] Link latency report from real hardware (decides the order below)
- [ ] Prefill/decode split: ship KV over RoCE
- [ ] Layer split across nodes (Hy4-flash 158.7 GB on two Sparks)
- [ ] Expert parallel over RoCE verbs, if latency ≤ 25 µs

## v0.5: more models

- [ ] NEON ports of the CPU kernels Flash-Next requires (`qfn_run` AVX2 path), then Qwen3.8-Flash-Next on one Spark
- [ ] DeepSeek-V4.1-Flash: a smaller pack that fits two Sparks

## Commercial track (Zoros)

- [ ] Prebuilt, signed AArch64 releases (archive + container) for DGX OS
- [ ] "Sparkstone Twin" appliance guide: cabling, power, router as a systemd service, monitoring
- [ ] Support and tuning engagements; private model packs; air-gapped installs
