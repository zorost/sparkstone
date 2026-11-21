#!/usr/bin/env python3
"""Syntax-check every host C/C++ source of a Basalt build tree with an AArch64 cross compiler.

Usage: check_aarch64_host.py BUILD_DIR [--cxx aarch64-linux-gnu-g++-13] [--cc aarch64-linux-gnu-gcc-13]

BUILD_DIR must be configured with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON. x86 `-m*` flags are dropped and
`-march=armv9-a` (GB10's Cortex-X925/A725 baseline) is added. CUDA (.cu) files are skipped: nvcc checks those.
Exit status 1 when any file fails.
"""
import argparse, concurrent.futures as cf, json, os, shlex, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("build_dir")
ap.add_argument("--cxx", default="aarch64-linux-gnu-g++-13")
ap.add_argument("--cc", default="aarch64-linux-gnu-gcc-13")
ap.add_argument("-j", type=int, default=os.cpu_count() or 4)
a = ap.parse_args()

seen, jobs = set(), []
for e in json.load(open(os.path.join(a.build_dir, "compile_commands.json"))):
    f = e["file"]
    if f.endswith(".cu") or f in seen:
        continue
    seen.add(f)
    args = shlex.split(e["command"])
    cmd = [a.cc if f.endswith(".c") else a.cxx, "-fsyntax-only", "-march=armv9-a"]
    skip = False
    for x in args[1:]:
        if skip:
            skip = False
            continue
        if x == "-o":
            skip = True
            continue
        if x == "-c" or x.startswith("-m") or x == f:
            continue
        cmd.append(x)
    jobs.append((f, cmd + [f], e["directory"]))

def run(j):
    f, cmd, d = j
    r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
    return f, r.returncode, [l for l in r.stderr.splitlines() if "error" in l]

res = list(cf.ThreadPoolExecutor(a.j).map(run, jobs))
bad = sorted(r for r in res if r[1])
print(f"host C/C++ files: {len(res)}; fail on aarch64: {len(bad)}")
for f, _, err in bad:
    print(f"  {os.path.relpath(f)}: {err[0][:160] if err else '?'}")
sys.exit(1 if bad else 0)
