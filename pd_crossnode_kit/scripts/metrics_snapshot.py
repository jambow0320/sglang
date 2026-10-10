"""Sum Prometheus histogram _sum/_count series for KV transfer and per-stage latency metrics.

usage: metrics_snapshot.py <out.json> [--prefill-url URL] [--decode-url URL]
"""

import argparse
import json
import os
import re
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("--prefill-url", default=os.environ.get("PREFILL_URL"))
parser.add_argument("--decode-url", default=os.environ.get("DECODE_URL"))
cli = parser.parse_args()

URLS = {"prefill": cli.prefill_url, "decode": cli.decode_url}
KV_PATTERN = re.compile(r"^(sglang:kv_transfer_[a-z_]+?)_(sum|count)\{[^}]*\} ([0-9.eE+-]+)$")
STAGE_PATTERN = re.compile(
    r'^sglang:per_stage_req_latency_seconds_(sum|count)\{[^}]*stage="([a-z_]+)"[^}]*\} ([0-9.eE+-]+)$'
)

snapshot = {}
for side, url in URLS.items():
    text = urllib.request.urlopen(f"{url}/metrics", timeout=10).read().decode()
    for line in text.splitlines():
        m = KV_PATTERN.match(line)
        if m:
            key = f"{side}:{m.group(1)}_{m.group(2)}"
            snapshot[key] = snapshot.get(key, 0.0) + float(m.group(3))
            continue
        m = STAGE_PATTERN.match(line)
        if m:
            key = f"{side}:stage:{m.group(2)}_{m.group(1)}"
            snapshot[key] = snapshot.get(key, 0.0) + float(m.group(3))
json.dump(snapshot, open(cli.out, "w"), indent=1)
