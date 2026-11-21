#!/usr/bin/env python3
"""Fast correctness check for a served model: questions with one exact answer, JSON output, a tool call.

A model running on wrong kernels fails these at once (garbage, loops, wrong digits), while a healthy Inkling 2 Mini
should score at or near 100 %. Run it against Sparkstone and against a vLLM/SGLang baseline on the same box:
matching scores are the quick evidence that the port computes what the model computes.

    python3 bench/sanity_eval.py --url http://127.0.0.1:8080 [--out sanity.json]

It is not a benchmark of model quality (use upstream's tools/ppl_gate.py for the KLD gate). Standard library only.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import time
import urllib.parse

# (prompt, kind, expected): kind "exact" = expected substring (case-insensitive) in the final answer,
# "number" = the last number in the answer equals expected
CASES = [
    ("What is 17 * 23? Answer with just the number.", "number", 391),
    ("What is 1234 + 5678? Answer with just the number.", "number", 6912),
    ("What is 2 to the power of 10? Answer with just the number.", "number", 1024),
    ("How many days are in a leap year? Answer with just the number.", "number", 366),
    ("What is 144 divided by 12? Answer with just the number.", "number", 12),
    ("A train leaves at 9:40 and arrives at 11:15. How many minutes is the trip? Answer with just the number.",
     "number", 95),
    ("If x + 7 = 19, what is x? Answer with just the number.", "number", 12),
    ("What is the sum of the first 10 positive integers? Answer with just the number.", "number", 55),
    ("What is the capital of Australia? One word.", "exact", "canberra"),
    ("What is the capital of Canada? One word.", "exact", "ottawa"),
    ("What is the chemical symbol for gold? Answer with the symbol only.", "exact", "au"),
    ("Which planet is known as the Red Planet? One word.", "exact", "mars"),
    ("What is the largest ocean on Earth? Answer in at most three words.", "exact", "pacific"),
    ("Who wrote 'Romeo and Juliet'? Answer with the surname only.", "exact", "shakespeare"),
    ("What gas do plants absorb from the air for photosynthesis? Answer with the common name.", "exact",
     "carbon dioxide"),
    ("Translate 'thank you' into Spanish. Answer with the translation only.", "exact", "gracias"),
    ("Spell the word 'basalt' backwards. Answer with the reversed word only.", "exact", "tlasab"),
    ("What is the opposite of 'ancient'? One word.", "exact", "modern"),
    ("In Python, what does len([3, 1, 4, 1, 5]) return? Answer with just the number.", "number", 5),
    ("What is the binary representation of the decimal number 10? Answer with the digits only.", "exact", "1010"),
    ("How many sides does a hexagon have? Answer with just the number.", "number", 6),
    ("What is 15% of 200? Answer with just the number.", "number", 30),
]
JSON_CASE = ("Return a JSON object with keys \"name\" (the string \"Sparkstone\"), \"cores\" (the integer 20) and "
             "\"tags\" (an array of the strings \"gb10\" and \"nvfp4\"). Output only the JSON, no code fences.")
TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
TOOL_CASE = "What's the weather in Paris right now? Use the tool."


def chat(url, headers, model, messages, max_tokens, reasoning_effort, tools=None, timeout=600):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 7}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if tools:
        body["tools"] = tools
    u = urllib.parse.urlsplit(url)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    c = cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout)
    try:
        c.request("POST", (u.path.rstrip("/") or "") + "/v1/chat/completions", body=json.dumps(body),
                  headers={**headers, "Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {data[:300]!r}")
        return json.loads(data)
    finally:
        c.close()


def strip_thinking(text: str) -> str:
    # servers that do not split reasoning out may leave a think block in the content
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--api-key-file")
    ap.add_argument("--reasoning-effort", default="low", help="'' to omit")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    headers = {}
    if a.api_key_file:
        headers["Authorization"] = "Bearer " + open(os.path.expanduser(a.api_key_file)).readline().strip()

    rows, passed = [], 0
    t0 = time.time()
    for prompt, kind, want in CASES:
        try:
            resp = chat(a.url, headers, a.model, [{"role": "user", "content": prompt}], a.max_tokens,
                        a.reasoning_effort)
            ans = strip_thinking(resp["choices"][0]["message"].get("content"))
            if kind == "number":
                nums = re.findall(r"-?\d+(?:\.\d+)?", ans.replace(",", ""))
                ok = bool(nums) and float(nums[-1]) == float(want)
            else:
                ok = str(want).lower() in ans.lower()
        except Exception as e:  # noqa: BLE001 - recorded, the run goes on
            ans, ok = f"ERROR {e}", False
        passed += ok
        rows.append({"prompt": prompt, "expected": want, "answer": ans[:300], "pass": ok})
        print(f"{'PASS' if ok else 'FAIL'}  {prompt[:60]:60s} -> {ans[:60]!r}", flush=True)

    try:   # JSON
        resp = chat(a.url, headers, a.model, [{"role": "user", "content": JSON_CASE}], a.max_tokens, a.reasoning_effort)
        ans = strip_thinking(resp["choices"][0]["message"].get("content"))
        obj = json.loads(ans.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        ok = str(obj.get("name", "")).lower() == "sparkstone" and obj.get("cores") == 20 and sorted(obj.get("tags", [])) == ["gb10", "nvfp4"]
    except Exception as e:  # noqa: BLE001
        ans, ok = f"ERROR {e}", False
    passed += ok
    rows.append({"prompt": JSON_CASE, "expected": "valid JSON object", "answer": str(ans)[:300], "pass": ok})
    print(f"{'PASS' if ok else 'FAIL'}  JSON object                                                   -> {str(ans)[:60]!r}")

    try:   # tool call
        resp = chat(a.url, headers, a.model, [{"role": "user", "content": TOOL_CASE}], a.max_tokens, a.reasoning_effort,
                    tools=[TOOL])
        calls = resp["choices"][0]["message"].get("tool_calls") or []
        args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
        ok = bool(calls) and calls[0]["function"]["name"] == "get_weather" and "paris" in str(args.get("city", "")).lower()
        ans = json.dumps(calls)[:300]
    except Exception as e:  # noqa: BLE001
        ans, ok = f"ERROR {e}", False
    passed += ok
    rows.append({"prompt": TOOL_CASE, "expected": "get_weather(city=Paris)", "answer": ans, "pass": ok})
    print(f"{'PASS' if ok else 'FAIL'}  tool call                                                     -> {ans[:60]!r}")

    total = len(rows)
    print(f"\nsanity: {passed}/{total} passed in {time.time() - t0:.0f} s ({a.url})")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"url": a.url, "passed": passed, "total": total, "rows": rows}, fh, indent=1)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
