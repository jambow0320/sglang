"""Run sgl-eval GSM8K (greedy, thinking disabled) against a server and save the metrics.

usage: eval_gsm8k.py --out <json> [--base-url URL] [--num-examples N] [--eval-out-dir DIR]
"""

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from sglang.test.sgl_eval_utils import run_sgl_eval

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default=os.environ.get("LB_URL"))
parser.add_argument("--num-examples", type=int, default=None)
parser.add_argument("--threads", type=int, default=128)
parser.add_argument("--max-tokens", type=int, default=1024)
parser.add_argument("--eval-out-dir", default="results/sgl_eval")
parser.add_argument("--out", required=True)
cli = parser.parse_args()

args = SimpleNamespace(
    base_url=cli.base_url,
    eval_name="gsm8k",
    max_tokens=cli.max_tokens,
    num_examples=cli.num_examples,
    num_threads=cli.threads,
    temperature=0.0,
    chat_template_kwargs={"enable_thinking": False},
    sgl_eval_out_dir=cli.eval_out_dir,
)
start = time.time()
metrics = run_sgl_eval(args)
elapsed = time.time() - start
record = {"metrics": metrics, "elapsed_s": round(elapsed, 1), "args": vars(cli)}
Path(cli.out).write_text(json.dumps(record, indent=2, default=str))
print(json.dumps(record, default=str))
