# Contributing the port back to Basalt

Getting patches 0001–0005 merged upstream is the best possible outcome for reputation. "Zoros added DGX Spark
support to Basalt" is a permanent, public credit in a fast-growing project, and it removes the maintenance cost of
carrying patches. Basalt's README asks for help with Blackwell testing hardware, so this is a welcome contribution.

## Order of operations

1. **Before any public launch,** open a GitHub Discussion or issue on jesdga95/basalt (text below). Don't surprise
   the author with an announcement.
2. Run `scripts/selftest.sh` on a Spark and attach its SUMMARY.md: maintainers merge what is shown to work.
3. Send **0005 (the race fix) as its own small PR first.** It helps every user, it's easy to review, and it builds trust.
4. Then 0001–0004 as one PR "AArch64 + sm_121 (DGX Spark / GB10) support".
5. Submit the GB10 `basalt-bench --community` row as a separate benchmark PR.

## Issue / discussion text

> **Title:** DGX Spark (GB10, sm_121 + AArch64) support: patches ready, offering upstream
>
> Hi! We've ported Basalt to NVIDIA DGX Spark (GB10: sm_121 GPU, AArch64 host) and would like to contribute it
> back. It's five small commits on 0.5.6 (14 files, +310/−58), and x86/sm_120 builds are unchanged:
>
> 1. build: `CMAKE_CUDA_ARCHITECTURES` 120 or 121; the five `120a-real` targets become `${BASALT_SM}a-real`;
>    the GPU gate accepts the built arch (all instructions in use assemble for sm_121a with CUDA 13.0)
> 2. platform: `compiler.hpp` x86-only includes; `cpu_relax`/`prefetch_l2` with Arm forms
> 3. kernels_cpu: x86 SIMD tier and its tests built only on x86; a stub answers "not supported" elsewhere
>    (GB10's GPU reads every expert in unified memory, so the tier has nothing to add there)
> 4. vision: portable CPU GEMM tile
> 5. platform: fixes a TOCTOU race between `ram_guard_pause(true)` and the guard thread's judgement
>    (`platform_memory_test` failed ~50 % under qemu-aarch64; 0/12 after)
>
> Verified so far: Inkling build for sm_121 links with only sm_121 cubins; the AArch64 cross build links; `ctest -L cpu`
> 78/78 on AArch64 (qemu); on x86 the full 5-model sm_120 build is clean and `ctest -L cpu` passes 106/106. GPU parity and a community bench row
> from real Spark hardware will follow. Happy to restructure however you prefer: one PR, separate PRs, or a
> different CMake approach (e.g. `120f` family targets).
>
> We'd also like to publish a Spark-focused distribution called Sparkstone (Apache-2.0 tooling around Basalt,
> with full credit to Basalt and a sponsor link). Glad to hear any concerns about naming or presentation.

## Credit line to use everywhere

"Engine: Basalt by Jesús Gulfo (MIT). GB10 port and Sparkstone by Zoros."
