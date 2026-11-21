# Hardware runbook: real numbers from two DGX Sparks

Written to be followed **step by step by a person or by a Claude Code session running on Spark A**. Every step
names its command, what "done" looks like, and what to record. Results land in `bench/results/<date>-<label>/`.

**Ground rules**
- Record what happens, not what should happen. A failed step is a result: save its log, note it, and continue
  with the next step that does not depend on it.
- No number goes into README, docs, the website, the X thread or the HF card unless it has a file in
  `bench/results/` behind it. Replace projections with measurements, and keep the word "projection" on anything
  that was not measured.
- Ask the owner before anything the steps below do not already cover that changes the system: driver or OS
  updates, firewall changes, BIOS or power settings. The runbook's own `sudo` uses are: `apt install perftest`,
  `drop_caches`, and `scripts/dual/setup-link.sh`.
- Commit results to a branch (`hardware-results-YYYYMMDD`), not `main`, and push it for review.

## 0 · Prerequisites (owner, once)

| | |
|---|---|
| Both Sparks | DGX OS 7, CUDA 13 (`nvcc --version`), driver R580+, internet access to github.com and huggingface.co |
| Spark A | this repository cloned (`git clone https://github.com/zorost/sparkstone`), GitHub access to push a branch |
| A → B | passwordless SSH from A to B (`ssh spark-b true` works); the same user can `sudo` on both |
| Cable | one QSFP cable between the two ConnectX-7 ports (any port pair) |
| Names | set `SPARK_B=<ssh host of B>` in the shell; `SUBMITTER=<one word>` for the benchmark row |

```bash
export SPARK_B=spark-b SUBMITTER=zorost RESULTS=bench/results/$(date +%Y%m%d)-dgx-spark
mkdir -p "$RESULTS"
```

## 1 · Spark A: preflight, build, test suites (≈ 30–60 min, no model needed)

```bash
scripts/selftest.sh 2>&1 | tee "$RESULTS/selftest-a-stage1-4.txt"
```

Done when the SUMMARY shows **1-preflight, 2-build, 3-tests-cpu, 4-tests-gpu = pass**.
- `2-build.log` must end with `{'sm_121': N}` (only sm_121 cubins).
- **4-tests-gpu is the correctness gate.** If it fails, stop the performance work. Collect `ctest -L gpu
  --output-on-failure` for the failing tests, look at the failing kernel, and fix it as a new patch: edit
  `engine/` on branch `gb10`, then `git format-patch` into `patches/` and run `ctest` again. Record each failure,
  even if it gets fixed.

Copy the archive `selftest-*.tar.gz` into `$RESULTS/`.

## 2 · Spark A: model, smoke test, upstream benchmark (≈ 30 min + download)

```bash
FULL=1 SUBMITTER=$SUBMITTER scripts/selftest.sh 2>&1 | tee "$RESULTS/selftest-a-full.txt"
```

Done when **5b-probe, 5c-smoke, 6-bench = pass**. From `6-bench.log`, record: `decode_conv_tps`,
`decode_struct_tps`, `decode_prose_tps`, prefill tok/s, and `accept_*_pct`. This is the number comparable to the RTX 5090
row (347 / 628 / 296 tok/s).

If the engine refuses to start or spills experts:
- check `free -g` and drop the page cache (`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`);
- read the engine log's memory lines (`~/.cache/basalt/<model>/engine.log`);
- `nvidia-smi -q -d MEMORY` on GB10 may show "N/A": note it in the results. It is a known GB10 trait, and only
  multi-GPU ordering depends on it.

## 3 · Spark A: correctness and load (≈ 20 min)

Start the server and keep it running for this step:

```bash
scripts/serve-spark.sh > "$RESULTS/server-a.log" 2>&1 &
until curl -fs localhost:8080/health >/dev/null; do sleep 2; done
```

```bash
python3 bench/sanity_eval.py --url http://127.0.0.1:8080 --out "$RESULTS/sanity-sparkstone.json"
```

Expect 24/24 (or very close). Several failures, garbage or looping output mean numerics are wrong: go back to step 1's gate.

```bash
python3 bench/load_test.py --url http://127.0.0.1:8080 --label "1x Spark, Sparkstone" \
  --concurrency 1,2,4,8 --out "$RESULTS/load-sparkstone-1x.json" | tee "$RESULTS/load-sparkstone-1x.txt"
```

Stop the server afterwards (`kill %1`).

## 4 · Spark A: vLLM baseline, same box, same prompts (≈ 30 min, optional but needed for comparisons)

Only if Docker and the NVIDIA container toolkit are present (they are on DGX OS). Stop Sparkstone first; the two
must not share the GPU.

```bash
docker run --rm -d --name vllm-baseline --gpus all --ipc=host -p 8090:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface vllm/vllm-openai:inkling-2-mini \
  --model thinkingmachines/Inkling-2-Mini --served-model-name inkling-2-mini --max-model-len 262144 \
  --gpu-memory-utilization 0.7
until curl -fs localhost:8090/health >/dev/null; do sleep 5; done
```

```bash
python3 bench/sanity_eval.py --url http://127.0.0.1:8090 --out "$RESULTS/sanity-vllm.json"
python3 bench/load_test.py --url http://127.0.0.1:8090 --label "1x Spark, vLLM" --concurrency 1,2,4,8 \
  --out "$RESULTS/load-vllm-1x.json" | tee "$RESULTS/load-vllm-1x.txt"
docker stop vllm-baseline
```

If the image tag or flags differ from the model's current vLLM recipe, use the recipe's command and note the
exact command and image digest in the results. If drafting (DSpark) is a separate option there, run both with
and without it and say which is which.

## 5 · Spark B: build and model (≈ 30–60 min, mostly unattended)

```bash
ssh "$SPARK_B" 'test -d sparkstone || git clone https://github.com/zorost/sparkstone'
ssh "$SPARK_B" 'cd sparkstone && git pull --ff-only && scripts/selftest.sh' 2>&1 | tee "$RESULTS/selftest-b-stage1-4.txt"
```

Copy the model over the cable rather than downloading it twice (after step 6 sets up the link, or over the LAN):

```bash
ssh "$SPARK_B" 'mkdir -p sparkstone/models'
rsync -a --info=progress2 models/ "$SPARK_B":sparkstone/models/
```

## 6 · The cable: setup and measurement (≈ 15 min)

```bash
sudo apt-get install -y perftest
ssh "$SPARK_B" 'sudo apt-get install -y perftest'
ibdev2netdev; ip -br link | grep -i enp
```

Check that the interface names match the defaults in `scripts/dual/setup-link.sh` (`enp1s0f1np1`,
`enP2p1s0f1np1`) for the port the cable is in. Set `IF_A` / `IF_B` (and `HCA_A` / `HCA_B` for check-link) if not.

```bash
sudo NODE=1 scripts/dual/setup-link.sh
ssh "$SPARK_B" 'cd sparkstone && sudo NODE=2 scripts/dual/setup-link.sh'
ssh "$SPARK_B" 'cd sparkstone && nohup scripts/dual/check-link.sh server > link-server.log 2>&1 &'
PEER=192.168.100.2 OUT="$RESULTS/link-report.txt" scripts/dual/check-link.sh client
ssh "$SPARK_B" 'pkill -f "check-link.sh server"; pkill ib_write' || true
```

Record: RDMA bandwidth (both halves; expect ~190 Gb/s total) and **write latency at 8 B, 1 KiB, 4 KiB, 8 KiB**.
That latency decides whether expert parallel is worth building (≤ 25 µs; see docs/03).

## 7 · Sparkstone Twin: two Sparks, one endpoint (≈ 20 min)

```bash
umask 077; openssl rand -hex 24 > ~/.sparkstone-key; scp ~/.sparkstone-key "$SPARK_B":~/.sparkstone-key
SPARK_A=localhost SPARK_B=$SPARK_B API_KEY_FILE=$HOME/.sparkstone-key REMOTE_ROOT=sparkstone \
  scripts/dual/start-dual.sh > "$RESULTS/router.log" 2>&1 &
until curl -fs localhost:8000/router/health >/dev/null; do sleep 3; done
```

`start-dual.sh` uses `ssh` to start both servers. If `ssh localhost` is not set up on A, start A's server by hand
(`HOST=192.168.100.1 API_KEY_FILE=~/.sparkstone-key scripts/serve-spark.sh &`) and B's over ssh. Then run
`router/spark_router.py` with `--backend http://192.168.100.1:8080 --backend http://192.168.100.2:8080`.

```bash
python3 bench/sanity_eval.py --url http://127.0.0.1:8000 --api-key-file ~/.sparkstone-key --out "$RESULTS/sanity-twin.json"
python3 bench/load_test.py --url http://127.0.0.1:8000 --api-key-file ~/.sparkstone-key --label "2x Spark, Sparkstone Twin" \
  --concurrency 1,2,4,8,16 --out "$RESULTS/load-twin.json" | tee "$RESULTS/load-twin.txt"
curl -s localhost:8000/router/stats > "$RESULTS/router-stats.json"
```

Check the `backends` column: both Sparks should serve requests at concurrency ≥ 2.

## 8 · Write up (≈ 15 min)

1. `$RESULTS/README.md`: machine details (DGX OS, driver, CUDA, clocks from `nvidia-smi -q`), what passed or failed
   at each step, and a results table:

   | Setup | conc | stream tok/s (chat / struct / prose) | aggregate tok/s | TTFT | sanity |
   |---|---|---|---|---|---|

   Add the upstream-bench row (step 2) and the link numbers (step 6).
2. Update `docs/05-benchmarking.md`'s results table and `README.md`'s status section with **measured** values and
   links to `$RESULTS`. Compare with the projections in docs/04 and say which way they missed and why.
3. Strip anything that should not be public (hostnames, internal IPs other than 192.168.100.x/101.x).
4. Commit on `hardware-results-YYYYMMDD` and push:
   ```bash
   git switch -c hardware-results-$(date +%Y%m%d)
   git add bench/results docs README.md
   git commit -m "Hardware results: DGX Spark and Sparkstone Twin"
   git push -u origin HEAD
   ```

## Time and space

About 3–4 hours of mostly unattended runs. Disk per Spark: ~20 GB model, ~10 GB engine and build, plus ~60 GB
for the vLLM image and its weights if step 4 runs.
