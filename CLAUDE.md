# Sparkstone: notes for Claude sessions

Sparkstone ports the Basalt inference engine (upstream: github.com/jesdga95/basalt, MIT) to NVIDIA DGX Spark
(GB10: sm_121 GPU, AArch64 host). The engine is NOT vendored: `scripts/fetch-engine.sh` clones Basalt at the
commit in `UPSTREAM` into `engine/` (gitignored) and applies `patches/*.patch` on branch `gb10`.

## If you are running on a DGX Spark

The owner wants real hardware numbers. Follow `HARDWARE_RUNBOOK.md` step by step, from step 0. Write results to
`bench/results/` and push them on a `hardware-results-YYYYMMDD` branch. Never invent or extrapolate a number. A
failing step is a result: record it, then continue where the runbook allows. Ask before changing system settings
beyond what the runbook lists.

## Working on the code

- Engine changes: edit `engine/` (branch `gb10`), then `git -C engine format-patch -o ../patches <upstream>..gb10`
  to regenerate `patches/` (MIT; keep x86/sm_120 behaviour unchanged so they can go upstream).
- Tests: `python3 -m unittest discover -s router`, `python3 -m unittest discover -s bench`,
  `shellcheck -S warning scripts/*.sh scripts/*/*.sh`, and `ctest -L cpu` in the engine build.
- Docs state projections as projections (`tools/perf_model.py`); measured numbers need a `bench/results/` file.
- Licences: Sparkstone code Apache-2.0 (keep NOTICE), patches MIT, name/logo trademarked (TRADEMARKS.md).
