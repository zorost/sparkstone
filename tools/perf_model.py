#!/usr/bin/env python3
"""Bandwidth roofline for Inkling 2 Mini decode on Basalt: RTX 5090 vs one DGX Spark vs two DGX Sparks.

Every projection in docs/ comes from this file. It is a model, not a measurement: it counts the bytes a decode
window must read from memory, divides by bandwidth, and calibrates one efficiency factor against Basalt's own
published RTX 5090 numbers. Replace the assumptions with measured values as hardware results arrive.

Usage:  python3 tools/perf_model.py            # tables
        python3 tools/perf_model.py --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict

# ---- the model (upstream docs/models/inkling2.md, Basalt 0.5.6) -----------------------------------------------
HIDDEN = 2304
EXPERT_FF = 1152
EXPERTS = 128
TOP_K = 8
LAYERS = 24
VISITS_NVFP4 = 6 + 12 * 3 + 2      # layers 0-19 visited per token: prelude 0-5, core 6-17 x3, coda 18-19
VISITS_Q8 = 4                      # layers 20-23 (MXFP8 in the release, Q8_0 in the pack)
VISITS = VISITS_NVFP4 + VISITS_Q8  # 48
PARAMS_PER_EXPERT = 3 * HIDDEN * EXPERT_FF          # gate, up, down
B_NVFP4 = 0.5625                                    # 4 bits + one E4M3 scale per 16 values
B_Q8_0 = 1.0625                                     # 8 bits + one fp16 scale per 32 values
DENSE_GIB = 1.10                                    # attention/conv/gate/out projections + Q8_0 head, on the card
VOCAB = 201_024                                     # head rows (vLLM config default; UNVERIFIED for Mini)
HEAD_BYTES = VOCAB * HIDDEN * B_Q8_0

# ---- hardware ------------------------------------------------------------------------------------------------
@dataclass
class Device:
    name: str
    peak_gbs: float        # spec memory bandwidth
    eff_gbs: float         # achievable streaming bandwidth (measured copy where known)

RTX5090 = Device("RTX 5090", 1792.0, 1792.0 * 0.90)
SPARK = Device("DGX Spark (GB10)", 273.0, 219.0)   # 219 GB/s: measured device copy (community, see research doc)

# Basalt 0.5.6 on an RTX 5090 at 400 W (upstream docs/models/inkling2.md)
MEASURED_5090 = {"plain": 264.0, "conversation": 347.0, "struct": 628.0, "prose": 296.0}
ACCEPT = {"conversation": 0.680, "struct": 0.975, "prose": 0.507}   # draft acceptance, same source
SPEC_WINDOW = 8                                                       # --spec 8 default: anchor + 7 drafts


def bytes_per_token_plain() -> dict:
    expert_nvfp4 = PARAMS_PER_EXPERT * B_NVFP4
    expert_q8 = PARAMS_PER_EXPERT * B_Q8_0
    experts = TOP_K * (VISITS_NVFP4 * expert_nvfp4 + VISITS_Q8 * expert_q8)
    dense_total = DENSE_GIB * 2**30
    dense_layers = dense_total - HEAD_BYTES
    dense = dense_layers / LAYERS * VISITS + HEAD_BYTES     # the core's projections are read on each of 3 passes
    return {"experts": experts, "dense": dense, "total": experts + dense}


def distinct_experts(rows: int, rho: float = 0.5) -> float:
    """Expected distinct experts a layer reads for `rows` tokens. rho=0: independent routing (upper bound);
    rho=1: every row routes alike. Consecutive tokens share experts, so a middle value is used and swept."""
    indep = EXPERTS * (1 - (1 - TOP_K / EXPERTS) ** rows)
    return rho * TOP_K + (1 - rho) * indep


def window_bytes(rows: int, rho: float) -> float:
    b = bytes_per_token_plain()
    return b["dense"] + b["experts"] * distinct_experts(rows, rho) / TOP_K


def tokens_per_window(accept: float, window: int = SPEC_WINDOW) -> float:
    """Expected tokens a window yields with i.i.d. per-draft acceptance `accept` (anchor always counts)."""
    drafts = window - 1
    return 1 + sum(accept ** i for i in range(1, drafts + 1))


def efficiency_5090() -> float:
    """Fraction of streaming bandwidth Basalt reaches in plain decode on the 5090 (calibration)."""
    t_ideal = bytes_per_token_plain()["total"] / (RTX5090.eff_gbs * 1e9)   # seconds per token at full bandwidth
    t_measured = 1 / MEASURED_5090["plain"]
    return t_ideal / t_measured


def project(dev: Device, eff: float, rho: float, overhead_ms: float = 0.0) -> dict:
    out = {}
    plain_s = bytes_per_token_plain()["total"] / (dev.eff_gbs * 1e9 * eff) + overhead_ms / 1e3
    out["plain"] = 1 / plain_s
    for wl, acc in ACCEPT.items():
        tpw = tokens_per_window(acc)
        w_s = window_bytes(SPEC_WINDOW, rho) / (dev.eff_gbs * 1e9 * eff) + overhead_ms / 1e3
        out[wl] = tpw / w_s
    return out


def dual_spark_ep(eff: float, rho: float, hop_us: float) -> dict:
    """Two Sparks, experts split 64/64 per layer (expert parallel), dense replicated on both nodes.
    Each layer visit costs two network hops (dispatch + combine) on the critical path."""
    b = bytes_per_token_plain()
    out = {}
    def t(rows):
        exp = b["experts"] * distinct_experts(rows, rho) / TOP_K
        # the busier node holds ~ the larger share of the routed experts at small batch
        share = 0.5 if rows >= 4 else 0.62
        mem = (b["dense"] + exp * share) / (SPARK.eff_gbs * 1e9 * eff)
        net = VISITS * 2 * hop_us / 1e6
        return mem + net
    out["plain"] = 1 / t(1)
    for wl, acc in ACCEPT.items():
        out[wl] = tokens_per_window(acc) / t(SPEC_WINDOW)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    b = bytes_per_token_plain()
    eff = efficiency_5090()
    rows = {}
    for rho in (0.3, 0.5, 0.7):
        rows[f"5090 model (rho={rho})"] = project(RTX5090, eff, rho)
        rows[f"1x Spark (rho={rho})"] = project(SPARK, eff, rho)
    rows["2x Spark data-parallel, per stream (rho=0.5)"] = project(SPARK, eff, 0.5)
    rows["2x Spark data-parallel, aggregate (rho=0.5)"] = {k: 2 * v for k, v in project(SPARK, eff, 0.5).items()}
    for hop in (10, 25, 100, 1000):
        rows[f"2x Spark expert-parallel, hop {hop} us"] = dual_spark_ep(eff, 0.5, hop)

    if a.json:
        print(json.dumps({"bytes_per_token": b, "efficiency": eff, "measured_5090": MEASURED_5090,
                          "devices": [asdict(RTX5090), asdict(SPARK)], "projections": rows}, indent=2))
        return

    gb = lambda x: f"{x / 1e9:.2f} GB"
    print("Inkling 2 Mini, Basalt pack (NVFP4 experts L0-19, Q8_0 L20-23 and dense), batch 1\n")
    print(f"  bytes read per token, plain decode: experts {gb(b['experts'])} + dense/head {gb(b['dense'])}"
          f" = {gb(b['total'])}")
    print(f"  distinct experts per layer in an 8-row window: {distinct_experts(8, 0.0):.1f} (independent routing),"
          f" {distinct_experts(8, 0.5):.1f} (rho=0.5)")
    print(f"  tokens per 8-row window: " + ", ".join(f"{k} {tokens_per_window(v):.2f}" for k, v in ACCEPT.items()))
    print(f"  calibrated efficiency (5090 plain {MEASURED_5090['plain']:.0f} tok/s measured): {eff:.0%} of streaming BW\n")
    cols = ["plain", "conversation", "struct", "prose"]
    print(f"  {'tok/s':52s}" + "".join(f"{c:>13s}" for c in cols))
    print(f"  {'RTX 5090 measured (Basalt 0.5.6)':52s}" + "".join(f"{MEASURED_5090[c]:13.0f}" for c in cols))
    for k, v in rows.items():
        print(f"  {k:52s}" + "".join(f"{v[c]:13.0f}" for c in cols))
    print("\n  Model only. 'rho' = routing overlap between rows of a window (0 = independent, 1 = identical).")
    print("  Spark rows assume Basalt reaches the same fraction of bandwidth on GB10 as on the 5090; launch")
    print("  overheads are larger on 48 SMs, so treat the 1x Spark row as an upper-middle estimate.")


if __name__ == "__main__":
    main()
