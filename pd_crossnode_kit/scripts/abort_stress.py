"""Abort requests at random points of the PD pipeline and classify every outcome.

Each request gets a unique rid; after a random delay an /abort_request for that rid is sent to the
prefill worker, the decode worker, both, or neither. A request that does not return within the
client timeout is counted as hung.

usage: abort_stress.py --out <json> [--n 240] [--concurrency 32] [--seed 0]
"""

import argparse
import asyncio
import collections
import json
import os
import random
import time

import aiohttp

parser = argparse.ArgumentParser()
parser.add_argument("--lb", default=os.environ.get("LB_URL"))
parser.add_argument("--prefill", default=os.environ.get("PREFILL_URL"))
parser.add_argument("--decode", default=os.environ.get("DECODE_URL"))
parser.add_argument("--n", type=int, default=240)
parser.add_argument("--concurrency", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--timeout", type=float, default=420)
parser.add_argument("--out", required=True)
cli = parser.parse_args()

TARGETS = {"prefill": ["prefill"], "decode": ["decode"], "both": ["prefill", "decode"], "none": []}


def make_jobs():
    rng = random.Random(cli.seed)
    jobs = []
    for i in range(cli.n):
        length = rng.choice([2048, 8192, 32768])
        jobs.append(
            {
                "rid": f"abort-{cli.seed}-{i}-{rng.getrandbits(32):08x}",
                "length": length,
                "ids": [rng.randrange(1000, 100000) for _ in range(length)],
                "max_new_tokens": rng.choice([16, 512]),
                "delay": 10 ** rng.uniform(-2, 0.5),
                "target": rng.choices(list(TARGETS), weights=[3, 3, 3, 1])[0],
            }
        )
    return jobs


async def send_abort(session, url, rid):
    try:
        async with session.post(url + "/abort_request", json={"rid": rid, "abort_all": False},
                                timeout=aiohttp.ClientTimeout(total=10)) as r:
            return r.status
    except Exception as e:
        return f"exc:{type(e).__name__}"


async def run_job(session, sem, job):
    async with sem:
        start = time.monotonic()
        payload = {
            "rid": job["rid"],
            "input_ids": job["ids"],
            "sampling_params": {"temperature": 0, "max_new_tokens": job["max_new_tokens"], "ignore_eos": True},
        }

        async def aborter():
            await asyncio.sleep(job["delay"])
            urls = {"prefill": cli.prefill, "decode": cli.decode}
            return [await send_abort(session, urls[t], job["rid"]) for t in TARGETS[job["target"]]]

        abort_task = asyncio.create_task(aborter())
        outcome, detail = "unknown", ""
        try:
            async with session.post(cli.lb + "/generate", json=payload,
                                    timeout=aiohttp.ClientTimeout(total=cli.timeout)) as r:
                text = await r.text()
                if r.status == 200:
                    body = json.loads(text)
                    reason = (body.get("meta_info") or {}).get("finish_reason") or {}
                    rtype = reason.get("type") if isinstance(reason, dict) else str(reason)
                    outcome = "aborted" if rtype == "abort" else "completed"
                    detail = rtype
                else:
                    outcome = "error_status"
                    detail = f"{r.status}:{text[:160]}"
        except asyncio.TimeoutError:
            outcome = "hung"
        except Exception as e:
            outcome = "client_exception"
            detail = f"{type(e).__name__}:{str(e)[:120]}"
        abort_status = await abort_task
        return {
            "rid": job["rid"], "length": job["length"], "max_new_tokens": job["max_new_tokens"],
            "delay": round(job["delay"], 3), "target": job["target"], "outcome": outcome,
            "detail": detail, "abort_status": abort_status, "elapsed": round(time.monotonic() - start, 3),
        }


async def main():
    jobs = make_jobs()
    sem = asyncio.Semaphore(cli.concurrency)
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*(run_job(session, sem, j) for j in jobs))
    by_target = collections.defaultdict(collections.Counter)
    for r in results:
        by_target[r["target"]][r["outcome"]] += 1
    error_details = collections.Counter(
        r["detail"].split(":")[0] + ":" + r["detail"][4:90] for r in results if r["outcome"] == "error_status"
    )
    summary = {
        "outcomes": dict(collections.Counter(r["outcome"] for r in results)),
        "by_target": {k: dict(v) for k, v in by_target.items()},
        "hung": sum(r["outcome"] == "hung" for r in results),
        "error_details_top": error_details.most_common(5),
    }
    json.dump({"summary": summary, "results": results}, open(cli.out, "w"), indent=1)
    print(json.dumps(summary))


asyncio.run(main())
