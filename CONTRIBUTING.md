# Contributing to Sparkstone

Thanks for helping. Three kinds of contributions matter most:

1. **Hardware reports.** Run `FULL=1 scripts/selftest.sh` on a DGX Spark (or an OEM GB10 system) and open an issue
   with the archive. Reports from different units, BIOS/OS versions and power modes are all valuable.
2. **Engine changes.** Engine code lives upstream in [Basalt](https://github.com/jesdga95/basalt). Make the change
   on `engine/` (branch `gb10`), then export it with `git format-patch` into `patches/`. Engine patches are MIT, so
   they can go upstream. Keep x86/sm_120 behaviour unchanged, and run both `ctest -L cpu` builds (x86 and the
   AArch64 cross build).
3. **Tooling and docs** (router, scripts, docs): Apache-2.0. `python3 -m unittest discover -s router` must pass.

By submitting a contribution you agree it is licensed under the licence of the directory it touches (MIT for
`patches/`, Apache-2.0 elsewhere). Sign off your commits (`git commit -s`, Developer Certificate of Origin).

Numbers in docs need a report behind them ([docs/05](docs/05-benchmarking.md)). Label projections as projections.
