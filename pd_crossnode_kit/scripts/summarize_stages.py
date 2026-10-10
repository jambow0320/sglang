"""Per-stage latency (ms) per load: PR vs base with a Welch 95% interval on the difference of rep means.

usage: summarize_stages.py <case-dir>
Only loads whose before/after snapshots both contain the per-stage series are used.
"""

import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CASE_DIR = Path(sys.argv[1])
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
       12: 2.18, 16: 2.12, 20: 2.09, 30: 2.04}


def t_crit(df):
    for k in sorted(T95):
        if df <= k:
            return T95[k]
    return 1.96


data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
pattern = re.compile(r"^(base|pr)-r(\d+)-(\w+)\.metrics_after\.json$")
for f in sorted(CASE_DIR.glob("*.metrics_after.json")):
    m = pattern.match(f.name)
    if not m:
        continue
    ver, load = m.group(1), m.group(3)
    after = json.loads(f.read_text())
    before = json.loads(Path(str(f).replace("metrics_after", "metrics_before")).read_text())
    for key in after:
        if ":stage:" not in key or not key.endswith("_count") or key not in before:
            continue
        stage = key[: -len("_count")]
        dc = after[key] - before[key]
        ds = after[f"{stage}_sum"] - before[f"{stage}_sum"]
        if dc > 0:
            data[load][stage][ver].append(ds / dc * 1000)

for load in sorted(data):
    print(f"\n=== {load}")
    print(f"{'stage':34s} {'n b/p':>6s} {'base ms':>10s} {'pr ms':>10s} {'delta ms':>9s} {'95% CI of delta ms':>22s}")
    for stage in sorted(data[load]):
        b, p = data[load][stage]["base"], data[load][stage]["pr"]
        if len(b) < 2 or len(p) < 2:
            continue
        bm, pm = statistics.mean(b), statistics.mean(p)
        bs, ps = statistics.stdev(b), statistics.stdev(p)
        se2 = bs**2 / len(b) + ps**2 / len(p)
        if se2 > 0:
            df = se2**2 / ((bs**2 / len(b)) ** 2 / (len(b) - 1) + (ps**2 / len(p)) ** 2 / (len(p) - 1) or 1e-12)
            half = t_crit(int(df)) * math.sqrt(se2)
        else:
            half = 0.0
        d = pm - bm
        print(f"{stage:34s} {len(b)}/{len(p):<4d} {bm:10.3f} {pm:10.3f} {d:+9.3f} [{d - half:+9.3f}, {d + half:+9.3f}]")
