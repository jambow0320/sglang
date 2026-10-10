"""Greedy-decode fixed random-token prompts one at a time and record output token ids and logprobs.

usage: determinism_check.py --out <json> [--base-url URL]
"""

import argparse
import json
import os
import random

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default=os.environ.get("LB_URL"))
parser.add_argument("--out", required=True)
parser.add_argument("--max-new-tokens", type=int, default=32)
cli = parser.parse_args()

LENGTHS = [128, 1000, 4000, 8192, 8193, 12000, 16384, 24000, 32000]
records = []
for seed in (1, 2):
    for length in LENGTHS:
        rng = random.Random(seed * 100003 + length)
        ids = [rng.randrange(1000, 100000) for _ in range(length)]
        resp = requests.post(
            cli.base_url + "/generate",
            json={
                "input_ids": ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": cli.max_new_tokens, "ignore_eos": True},
                "return_logprob": True,
            },
            timeout=600,
        )
        resp.raise_for_status()
        meta = resp.json()["meta_info"]
        out = meta["output_token_logprobs"]
        records.append(
            {
                "seed": seed,
                "length": length,
                "token_ids": [t[1] for t in out],
                "logprobs": [t[0] for t in out],
            }
        )
json.dump(records, open(cli.out, "w"))
print(f"recorded {len(records)} prompts")
