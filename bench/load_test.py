#!/usr/bin/env python3
"""Concurrent load test for any OpenAI-compatible endpoint: Sparkstone (one Spark), the Twin router (two Sparks),
or a vLLM/SGLang baseline, measured the same way.

    python3 bench/load_test.py --url http://127.0.0.1:8080 --concurrency 1,2,4,8 --out results.json
    python3 bench/load_test.py --url http://127.0.0.1:8000 --api-key-file ~/.sparkstone-key ...   # through the router

For each concurrency level it runs `--requests-per-level` streamed chat requests (at least one per slot), each with a
distinct conversation (so a prefix cache cannot flatter the numbers and the Twin router spreads them), and reports:

  ttft_s        time to the first streamed token (content or reasoning)
  stream_tps    per request: (completion tokens - 1) / (end - first token), the speed one user sees
  aggregate_tps all completion tokens of the level / the level's wall time, the machine's throughput

Workloads (`--workload`): chat (conversational), struct (JSON, temperature 0: drafting accepts most tokens),
prose (an essay, temperature 0.6). Token counts come from the final chunk's `usage` (Basalt always sends it; vLLM and
SGLang send it with stream_options.include_usage, which this requests); without usage it counts content chunks.

Standard library only.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import http.client
import json
import os
import statistics
import sys
import time
import urllib.parse
import uuid

WORKLOADS = {
    "chat": (0.6, [
        "I'm planning a weekend trip to {topic}. What should I see, eat and avoid? Keep it conversational.",
        "Explain {topic} to a curious teenager, with one analogy and one question back to them.",
    ]),
    "struct": (0.0, [
        "Return a JSON array of 40 objects describing fictional products related to {topic}. Each object has the keys "
        "id (integer), name (string), category (string), price_usd (number), in_stock (boolean) and tags (array of 3 "
        "strings). Output only the JSON.",
    ]),
    "prose": (0.6, [
        "Write an essay of about 600 words on {topic}: its history, why it matters today, and one open question.",
    ]),
}
TOPICS = ["Lisbon", "volcanic rock", "the printing press", "honeybees", "tidal power", "jazz improvisation",
          "the Silk Road", "glass making", "black holes", "sourdough", "monsoons", "chess openings", "Kyoto",
          "lighthouses", "coral reefs", "the telegraph", "origami", "desert ecology", "Roman roads", "aurora borealis"]


def one_request(url: str, headers: dict, model: str, workload: str, i: int, max_tokens: int,
                reasoning_effort: str | None, timeout: float) -> dict:
    temp, prompts = WORKLOADS[workload]
    topic = TOPICS[i % len(TOPICS)]
    prompt = prompts[i % len(prompts)].format(topic=topic)
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
            "temperature": temp, "seed": 1000 + i,
            "messages": [{"role": "system", "content": f"Session {uuid.uuid4().hex[:12]}."},
                         {"role": "user", "content": prompt}]}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    u = urllib.parse.urlsplit(url)
    conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout)
    t0 = time.perf_counter()
    first = None
    chunks = 0
    usage = None
    timings = None
    backend = None
    err = None
    try:
        conn.request("POST", (u.path.rstrip("/") or "") + "/v1/chat/completions", body=json.dumps(body),
                     headers={**headers, "Content-Type": "application/json", "Accept": "text/event-stream"})
        r = conn.getresponse()
        backend = r.getheader("X-Spark-Backend")
        if r.status != 200:
            err = f"HTTP {r.status}: {r.read(300).decode(errors='replace')}"
        else:
            buf = b""
            while True:
                data = r.read1(65536) if hasattr(r, "read1") else r.read(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        continue
                    try:
                        ev = json.loads(payload)
                    except ValueError:
                        continue
                    if ev.get("usage"):
                        usage = ev["usage"]
                    if ev.get("timings"):
                        timings = ev["timings"]
                    for ch in ev.get("choices") or []:
                        d = ch.get("delta") or {}
                        if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
                            chunks += 1
                            if first is None:
                                first = time.perf_counter()
    except (OSError, http.client.HTTPException) as e:
        err = f"{e.__class__.__name__}: {e}"
    finally:
        conn.close()
    end = time.perf_counter()
    tokens = (usage or {}).get("completion_tokens") or chunks
    res = {"i": i, "workload": workload, "backend": backend, "error": err, "tokens": tokens,
           "tokens_from": "usage" if usage else "chunks", "start": t0, "end": end,
           "ttft_s": (first - t0) if first else None,
           "stream_tps": ((tokens - 1) / (end - first)) if first and tokens > 1 and end > first else None}
    if timings:
        res["server_timings"] = timings
    return res


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def run_level(args, headers, workload: str, conc: int) -> dict:
    n = max(conc, args.requests_per_level)
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(conc) as ex:
        futs = [ex.submit(one_request, args.url, headers, args.model, workload, i, args.max_tokens,
                          args.reasoning_effort or None, args.timeout) for i in range(n)]
        rows = [f.result() for f in futs]
    wall = time.perf_counter() - t0
    ok = [r for r in rows if not r["error"]]
    tps = [r["stream_tps"] for r in ok if r["stream_tps"]]
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    backends = {}
    for r in ok:
        backends[r["backend"] or "-"] = backends.get(r["backend"] or "-", 0) + 1
    return {"workload": workload, "concurrency": conc, "requests": n, "errors": len(rows) - len(ok),
            "wall_s": wall, "completion_tokens": sum(r["tokens"] for r in ok),
            "aggregate_tps": sum(r["tokens"] for r in ok) / wall if wall > 0 else None,
            "stream_tps_median": statistics.median(tps) if tps else None, "stream_tps_p10": pct(tps, 0.10),
            "ttft_s_median": statistics.median(ttft) if ttft else None, "ttft_s_p90": pct(ttft, 0.90),
            "tokens_from": sorted({r["tokens_from"] for r in ok}), "backends": backends,
            "first_error": next((r["error"] for r in rows if r["error"]), None), "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080", help="server base URL (no /v1)")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--api-key-file", help="file whose first line is the API key (sent as Bearer)")
    ap.add_argument("--workload", default="chat,struct,prose", help=f"comma list of {', '.join(WORKLOADS)}")
    ap.add_argument("--concurrency", default="1,2,4,8", help="comma list of concurrent request counts")
    ap.add_argument("--requests-per-level", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--reasoning-effort", default="none",
                    help="sent as reasoning_effort (Inkling: none keeps thinking short); '' to omit")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--label", default="", help="a name for this setup, e.g. '1x Spark sparkstone'")
    ap.add_argument("--out", help="write the full results (every request) as JSON here")
    a = ap.parse_args(argv)

    headers = {}
    if a.api_key_file:
        key = open(os.path.expanduser(a.api_key_file)).readline().strip()
        headers["Authorization"] = f"Bearer {key}"
    levels = [int(x) for x in a.concurrency.split(",") if x]
    workloads = [w for w in a.workload.split(",") if w]
    for w in workloads:
        if w not in WORKLOADS:
            ap.error(f"unknown workload {w}")

    # warm-up: one short request so model load / graph capture is not measured
    one_request(a.url, headers, a.model, workloads[0], 999, 32, a.reasoning_effort or None, a.timeout)

    results = []
    print(f"{a.label or a.url}   (max_tokens {a.max_tokens}, reasoning_effort {a.reasoning_effort or '-'})")
    print(f"{'workload':8s} {'conc':>4s} {'req':>4s} {'err':>3s} {'agg tok/s':>10s} {'stream tok/s (med / p10)':>25s}"
          f" {'TTFT s (med / p90)':>20s}  backends")
    for w in workloads:
        for c in levels:
            r = run_level(a, headers, w, c)
            results.append(r)
            f = lambda x, d=1: "-" if x is None else f"{x:.{d}f}"
            print(f"{w:8s} {c:4d} {r['requests']:4d} {r['errors']:3d} {f(r['aggregate_tps']):>10s} "
                  f"{f(r['stream_tps_median']):>12s} / {f(r['stream_tps_p10']):<10s} "
                  f"{f(r['ttft_s_median'], 2):>8s} / {f(r['ttft_s_p90'], 2):<9s}  "
                  f"{', '.join(f'{k}: {v}' for k, v in r['backends'].items())}", flush=True)
            if r["first_error"]:
                print(f"         first error: {r['first_error']}", flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"label": a.label, "url": a.url, "model": a.model, "max_tokens": a.max_tokens,
                       "reasoning_effort": a.reasoning_effort, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "levels": results}, fh, indent=1)
        print(f"wrote {a.out}")
    return 1 if any(r["errors"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
