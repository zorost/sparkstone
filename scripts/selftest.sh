#!/usr/bin/env bash
# Sparkstone hardware self-test: run on a DGX Spark, get one report archive back.
#
#   scripts/selftest.sh                  # stages 1-4: preflight, build, CPU + GPU tests  (~20-40 min, no model)
#   FULL=1 scripts/selftest.sh           # + stage 5-6: download Inkling 2 Mini (19.7 GB), smoke test, benchmark
#   PEER=192.168.100.2 FULL=1 scripts/selftest.sh   # + stage 7: twin-Spark link check (peer runs check-link.sh server)
#   SUBMITTER=name    the name on the community benchmark row (one word, default "sparkstone")
#
# Writes selftest-<host>-<date>/ and a .tar.gz of it. Every stage logs to its own file; a failing stage is recorded
# and the next one still runs, so one archive tells the whole story. Nothing leaves the machine unless you send it.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
engine=${ENGINE_DIR:-$root/engine}
out=${OUT:-$root/selftest-$(hostname)-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$out"
summary=$out/SUMMARY.md
echo "# Sparkstone self-test: $(hostname), $(date -Is)" > "$summary"
echo "" >> "$summary"
echo "| stage | result | log |" >> "$summary"
echo "|---|---|---|" >> "$summary"

stage() {   # stage NAME CMD...: run, log, record
  local name=$1; shift
  local log="$out/$name.log"
  echo "== $name"
  local t0=$SECONDS
  ( "$@" ) > "$log" 2>&1
  local rc=$?
  local res
  res=$([ $rc -eq 0 ] && echo "pass" || echo "**FAIL ($rc)**")
  echo "| $name | $res ($((SECONDS - t0)) s) | $name.log |" >> "$summary"
  echo "   $res  ($out/$name.log)"
  return $rc
}

stage 1-preflight "$root/scripts/spark-preflight.sh"
{ nvidia-smi -q 2>/dev/null || true; } > "$out/nvidia-smi-q.txt"
{ cat /proc/cpuinfo; echo; cat /proc/meminfo; echo; uname -a; cat /etc/os-release; } > "$out/system.txt" 2>&1

stage 2-build env EXTRA_CMAKE="-DBUILD_TESTING=ON ${EXTRA_CMAKE:-}" "$root/scripts/build-spark.sh"
[ -x "$engine/bin/basalt" ] && python3 "$root/tools/cubin_archs.py" "$engine/bin/basalt" >> "$out/2-build.log"

# Basalt's own suites: `cpu` runs anywhere; `gpu` compares every kernel with a CPU reference on this GB10.
stage 3-tests-cpu bash -c "cd '$engine/build' && ctest -L cpu -j$(nproc) --output-on-failure --timeout 900"
stage 4-tests-gpu bash -c "cd '$engine/build' && ctest -L gpu --output-on-failure --timeout 1800"

if [ "${FULL:-0}" = 1 ]; then
  pack=${PACK:-$root/models/Inkling-2-Mini-NVFP4-q8.basalt}
  [ -f "$pack" ] || stage 5a-download "$root/scripts/download-model.sh" "$(dirname "$pack")"
  # free the page cache the download left, so the engine sees the unified memory as available
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "(drop_caches skipped: no passwordless sudo)" >> "$out/notes.txt"
  stage 5b-probe "$engine/bin/basalt-probe" --model "$pack"
  stage 5c-smoke bash -c "
    set -e
    '$engine/bin/basalt' serve '$pack' --port 18080 > '$out/5c-server.log' 2>&1 &
    pid=\$!; trap 'kill \$pid 2>/dev/null' EXIT
    for i in \$(seq 1 180); do curl -fs localhost:18080/health >/dev/null && break; sleep 2; done
    curl -fs localhost:18080/v1/chat/completions -H 'Content-Type: application/json' -d '{
      \"model\": \"inkling-2-mini\", \"temperature\": 0, \"max_tokens\": 256, \"reasoning_effort\": \"none\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Reply with a JSON object listing the first five prime numbers under key primes.\"}]}'
    echo; curl -fs localhost:18080/metrics | head -50 || true"
  # the upstream community preset: same fixtures and settings as every published Basalt row
  stage 6-bench bash -c "cd '$engine' && bin/basalt-bench --community --submitter '${SUBMITTER:-sparkstone}' --model '$pack'"
  cp "$engine/docs/community/benchmarks.json" "$out/benchmarks.json" 2>/dev/null || true
fi

if [ -n "${PEER:-}" ]; then
  stage 7-link env PEER="$PEER" OUT="$out/link-report.txt" "$root/scripts/dual/check-link.sh" client
fi

python3 "$root/tools/perf_model.py" --json > "$out/perf-model.json" 2>/dev/null
(cd "$root" && git rev-parse HEAD 2>/dev/null; cd "$engine" && git log --oneline -8) > "$out/versions.txt" 2>&1
tar czf "$out.tar.gz" -C "$(dirname "$out")" "$(basename "$out")"
echo ""
cat "$summary"
echo ""
echo "report: $out.tar.gz"
