"""Aggregate perf blocks per load: PR vs base with a Welch 95% interval on the difference of rep means.

usage: summarize_perf.py <case-dir> [--metrics client|all]
Each block (one server launch) is one sample; requests inside a block are not treated as independent.
"""

import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CASE_DIR = Path(sys.argv[1])
SHOW_SERVER = "--metrics" in sys.argv and sys.argv[sys.argv.index("--metrics") + 1] == "all"

CLIENT = [
    ("mean_ttft_ms", "TTFT mean ms"),
    ("median_ttft_ms", "TTFT p50 ms"),
    ("p99_ttft_ms", "TTFT p99 ms"),
    ("mean_itl_ms", "ITL mean ms"),
    ("p99_itl_ms", "ITL p99 ms"),
    ("output_throughput", "output tok/s"),
]
SERVER = [
    ("prefill:sglang:kv_transfer_latency_ms", "P xfer latency ms"),
    ("prefill:sglang:kv_transfer_bootstrap_ms", "P bootstrap ms"),
    ("decode:sglang:kv_transfer_alloc_ms", "D alloc ms"),
]
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
       11: 2.20, 12: 2.18, 14: 2.14, 16: 2.12, 20: 2.09, 30: 2.04}


def t_crit(df):
    keys = sorted(T95)
    for k in keys:
        if df <= k:
            return T95[k]
    return 1.96


def server_mean(prefix, key):
    before = json.loads(Path(f"{prefix}.metrics_before.json").read_text())
    after = json.loads(Path(f"{prefix}.metrics_after.json").read_text())
    dc = after.get(f"{key}_count", 0) - before.get(f"{key}_count", 0)
    ds = after.get(f"{key}_sum", 0) - before.get(f"{key}_sum", 0)
    return ds / dc if dc else float("nan")


samples = defaultdict(lambda: defaultdict(list))
pattern = re.compile(r"^(base|pr)-r(\d+)-(\w+)\.jsonl$")
for f in sorted(CASE_DIR.glob("*.jsonl")):
    m = pattern.match(f.name)
    if not m:
        continue
    ver, rep, load = m.group(1), int(m.group(2)), m.group(3)
    lines = f.read_text().splitlines()
    if not lines:
        continue
    client = json.loads(lines[-1])
    row = {k: client[k] for k, _ in CLIENT}
    row["completed"] = client["completed"]
    row["rep"] = rep
    prefix = str(f)[: -len(".jsonl")]
    for k, _ in SERVER:
        row[k] = server_mean(prefix, k)
    samples[load][ver].append(row)


def fmt(v):
    if math.isnan(v):
        return "nan"
    return f"{v:.2f}" if abs(v) < 1000 else f"{v:.0f}"


load_order = sorted(samples, key=lambda x: (re.sub(r"\d+$", "", x), int(re.findall(r"\d+$", x)[0]) if re.findall(r"\d+$", x) else 0))
for load in load_order:
    base, pr = samples[load]["base"], samples[load]["pr"]
    if len(base) < 2 or len(pr) < 2:
        continue
    print(f"\n=== {load}  reps base={len(base)} pr={len(pr)}  completed base={sorted({r['completed'] for r in base})} pr={sorted({r['completed'] for r in pr})}")
    print(f"{'metric':18s} {'base mean':>10s} {'base sd':>8s} {'pr mean':>10s} {'pr sd':>8s} {'delta %':>8s} {'95% CI of delta %':>22s}  verdict")
    metrics = CLIENT + (SERVER if SHOW_SERVER else [])
    for key, label in metrics:
        b = [r[key] for r in base if not math.isnan(r[key])]
        p = [r[key] for r in pr if not math.isnan(r[key])]
        if len(b) < 2 or len(p) < 2:
            continue
        bm, pm = statistics.mean(b), statistics.mean(p)
        bs, ps = statistics.stdev(b), statistics.stdev(p)
        se2 = bs**2 / len(b) + ps**2 / len(p)
        se = math.sqrt(se2)
        if se > 0:
            df = se2**2 / ((bs**2 / len(b)) ** 2 / (len(b) - 1) + (ps**2 / len(p)) ** 2 / (len(p) - 1) or 1e-12)
            half = t_crit(int(df)) * se
        else:
            half = 0.0
        d = pm - bm
        lo, hi = (d - half) / bm * 100, (d + half) / bm * 100
        verdict = "no detectable difference" if lo <= 0 <= hi else ("PR higher" if lo > 0 else "PR lower")
        print(f"{label:18s} {fmt(bm):>10s} {fmt(bs):>8s} {fmt(pm):>10s} {fmt(ps):>8s} {d / bm * 100:>+7.1f}% [{lo:>+8.1f}%, {hi:>+8.1f}%]  {verdict}")
