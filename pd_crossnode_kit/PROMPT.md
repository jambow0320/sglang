# Cross-node NIXL staging validation for sglang PR #41140

You are validating an sglang pull request on two GPU nodes connected by RDMA. Compare the PR with
its upstream base in one configuration: prefill TP2 on one node, decode TP1 on the other node, NIXL
KV transfer backend, staging buffer enabled. Run a correctness check and a performance A/B
comparison, then write a report. Do not modify sglang source code. If a step fails, keep the logs,
record the failure in the report, and continue with the next step.

## Background

PR: https://github.com/sgl-project/sglang/pull/41140 ("[PD] Consolidate sender and receiver
implementations"). In prefill/decode (PD) disaggregation, the prefill server writes the KV cache of
each request into the decode server's GPU memory through a transfer backend. The PR moves the
sender/receiver lifecycle of the Mooncake, NIXL and Mori backends into shared code, and changes how
NIXL signals transfer completion:

- Base: prefill attaches a NIXL notification to every RDMA write, and decode counts the
  notifications to decide that a request's KV has arrived.
- PR: prefill waits until all RDMA writes of the last chunk complete, then sends a `KV_STATUS=Success`
  message to decode over ZMQ (TCP). On the staging path, each staged chunk is announced to decode
  with a ZMQ `CHUNK_READY` message instead of a notification. NIXL notifications are no longer used.

When prefill and decode use different TP sizes and `SGLANG_DISAGG_STAGING_BUFFER=1` is set, prefill
packs each chunk into a staging buffer, writes it to the decode side's staging buffer, and decode
scatters it into its KV cache. When the staging buffer is full, a chunk is deferred and retried.
The PR was validated on a single node only; this run checks the staging path across nodes. Points to
watch: requests that never complete, completions reported before the KV data has landed (wrong
output tokens), KV-cache leaks after aborts, and added time-to-first-token (TTFT).

## Inputs (fill in before starting)

```
P_IP=            # prefill node IP, reachable from the decode node
D_IP=            # decode node IP, reachable from the prefill node
GPU type / count per node:   # prefill needs 2 GPUs, decode needs 1
RDMA NIC(s) closest to the GPUs used (for UCX_NET_DEVICES, e.g. mlx5_0:1):
MODEL=Qwen/Qwen3-8B   # local path or HF id, available on both nodes
```

## Ground rules

- Both nodes run the same version at any time. Never pair a PR prefill with a base decode or the
  reverse: the two versions use different completion signals, and a mixed pair hangs.
- Use the same nodes, GPUs, NICs, environment variables and server flags for both versions; only the
  checkout differs.
- Before each launch, stop all sglang and router processes on both nodes and confirm with
  `nvidia-smi` that the GPUs are free.
- Run every client (benchmark, eval and check scripts) with `PYTHONPATH=$BASE/python`, so the client
  code is identical for both versions.
- Do not post anything publicly and do not push to any repository.

## Step 0: Environment

1. On both nodes, at the same paths (or on a shared filesystem):

   ```bash
   git clone -b pr41140-crossnode-test https://github.com/jambow0320/sglang.git sglang-pr
   cd sglang-pr
   git diff --stat 540545d10e4e9f187249d928c62e78890c771204 HEAD   # must list only pd_crossnode_kit/
   git worktree add ../sglang-base f48ed2127e3457e88f5ee86debd4d12a7735c69f
   cd ..
   export PR=$PWD/sglang-pr BASE=$PWD/sglang-base KIT=$PWD/sglang-pr/pd_crossnode_kit
   ```

   `sglang-pr` is the PR commit `540545d` plus this kit, which adds no sglang code. `sglang-base` is
   the upstream commit the PR is based on. The PR does not change dependencies: install once with
   `pip install -e $BASE/python`, then select the version at launch time with
   `PYTHONPATH=<checkout>/python`. Also install `nixl` (the single-node run used 1.5.0; 1.3.0 or
   newer is required) and `sglang-router` (single-node: 0.3.2).
2. Save to `results/env/`: `nvidia-smi`, `nvidia-smi topo -m`, `ibv_devinfo`, `ucx_info -v`,
   `ucx_info -d | grep -E "Transport|Device"`, `pip freeze`, `lsmod | grep -E "nvidia_peermem|ib_"`,
   and `git -C $PR rev-parse HEAD` and `git -C $BASE rev-parse HEAD` on both nodes.
3. Pick the RDMA NIC(s) closest to the GPUs used (from `nvidia-smi topo -m`). Use one UCX setting for
   the whole run on both nodes, for example `UCX_NET_DEVICES=mlx5_0:1`. On RoCE you may also need
   `UCX_IB_GID_INDEX`. Record the final UCX variables in the report.

## Step 1: Launch template and RDMA preflight

Launch template (`VER` is `base` or `pr`, `CHECKOUT` is `$BASE` or `$PR`, `TAG` names the run).
`SGLANG_DISAGG_STAGING_BUFFER=1` is set on both nodes in every run. Keep the default
`SGLANG_DISAGG_STAGING_POOL_SIZE_MB` (4096): with 128 MB every request fails with "chunk exceeds
ring buffer total size" in both versions.

```bash
# prefill node (GPUs 0 and 1)
export PYTHONPATH=$INSTR$CHECKOUT/python SGLANG_HOST_IP=$P_IP SGLANG_DISAGG_STAGING_BUFFER=1  # plus UCX vars
python -c "import sglang; print('sglang from', sglang.__file__)" > logs/$TAG-$VER-prefill.log
python -m sglang.launch_server --model-path $MODEL --host $P_IP --port 30000 \
  --disaggregation-mode prefill --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 8998 --tp 2 --base-gpu-id 0 --enable-metrics $P_EXTRA \
  >> logs/$TAG-$VER-prefill.log 2>&1

# decode node (GPU 0)
export PYTHONPATH=$INSTR$CHECKOUT/python SGLANG_HOST_IP=$D_IP SGLANG_DISAGG_STAGING_BUFFER=1  # plus UCX vars
python -c "import sglang; print('sglang from', sglang.__file__)" > logs/$TAG-$VER-decode.log
python -m sglang.launch_server --model-path $MODEL --host $D_IP --port 30001 \
  --disaggregation-mode decode --disaggregation-transfer-backend nixl \
  --tp 1 --base-gpu-id 0 --enable-metrics \
  >> logs/$TAG-$VER-decode.log 2>&1

# router, on the prefill node or a client host
python -m sglang_router.launch_router --pd-disaggregation --mini-lb \
  --prefill http://$P_IP:30000 8998 --decode http://$D_IP:30001 --host 0.0.0.0 --port 8000
```

`INSTR` is `$KIT/scripts/staging_instrument:` for the correctness runs and empty for the performance
runs. The instrument counts staged and deferred chunks and prints lines such as
`[staging-instrument pid=...] {'nixl_staged': 256, 'nixl_deferred': ...}` to the prefill log.

Client environment: `export LB_URL=http://<router-host>:8000 PREFILL_URL=http://$P_IP:30000
DECODE_URL=http://$D_IP:30001 MODEL=$MODEL PYTHONPATH=$BASE/python`.

`SGLANG_HOST_IP` sets the address each server advertises. In the PR, the `Success` and `CHUNK_READY`
messages travel over TCP from the prefill node to a ZMQ port on the decode node, so the decode node
must accept TCP connections from the prefill node on arbitrary ports. If requests hang in the PR but
not in base, check this first.

Preflight, with the PR version and the instrument enabled:

1. Send one greedy request with 32768 random input token ids and `max_new_tokens=4` through the router.
2. Read `/sys/class/infiniband/<dev>/ports/1/counters/port_xmit_data` on the prefill node and
   `port_rcv_data` on the decode node before and after the request, summed over the NICs in
   `UCX_NET_DEVICES` (the counters are in 4-byte units). Qwen3-8B stores 147,456 bytes of KV per token
   (36 layers x 8 KV heads x 128 x 2 x 2 bytes), so the summed delta x 4 should be about 4.8 GB. If it
   is far smaller, the KV did not cross the RDMA NIC: fix the UCX configuration before going further.
3. Check that the prefill log shows `nixl_staged > 0`; otherwise the staging path was not used.

Save the counter values and the instrument line in `results/preflight/`.

## Step 2: Correctness

Run the sequence below for `base` and then for `pr`, each on a fresh launch with the instrument
enabled (`TAG=c2`):

```bash
python $KIT/scripts/determinism_check.py --out results/correctness/det-c2-$VER.json
python $KIT/scripts/eval_gsm8k.py --out results/correctness/gsm8k-full-c2-$VER.json
python $KIT/scripts/abort_stress.py --out results/correctness/abort-c2-$VER.json
bash $KIT/scripts/post_check.sh <prefill-log> <decode-log> > results/correctness/postcheck1-c2-$VER.txt
python $KIT/scripts/eval_gsm8k.py --num-examples 200 --out results/correctness/gsm8k-200-c2-$VER.json
bash $KIT/scripts/post_check.sh <prefill-log> <decode-log> > results/correctness/postcheck2-c2-$VER.txt
grep staging-instrument <prefill-log> | grep -v patched | tail -2 > results/correctness/staging-c2-$VER.txt
```

`post_check.sh` reads both server logs; copy the decode node's log to the client host first (or run
the grep on that node). Then compare the greedy outputs:

```bash
python $KIT/scripts/compare_determinism.py results/correctness/det-c2-base.json results/correctness/det-c2-pr.json
```

Expected behavior that is not a regression:

- An abort that reaches only the prefill server can leave the request open on decode until
  `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` (300 s by default) expires. This happens in both versions,
  so some requests in `abort_stress.py` take about 300 s. Its client timeout is 420 s; only requests
  that exceed 420 s count as hung.
- Most abort-stress requests end as `error_status` or `aborted`; that is the purpose of the test.
  Compare the outcome distributions between versions.
- `nixl_deferred` grows by one per retry attempt, so it can reach hundreds of thousands.

Pass criteria:

- Greedy outputs: PR token ids identical to base on all 18 prompts.
- Staging counters: `nixl_staged > 0` and `nixl_deferred > 0` in both versions.
- GSM8K full set (1319 questions): PR within 2 points of base.
- Abort stress: `hung` is 0. After idling, every `/health` returns 200, every queue gauge in
  `post_check.sh` is 0, and `leak` and `traceback` are 0 in both logs. The GSM8K-200 run after the
  storm is not lower than base by more than 3 points.

## Step 3: Performance A/B

Same configuration with the instrument disabled (`INSTR` empty) and `P_EXTRA=--disable-radix-cache`
on the prefill server (the same prompts repeat across reps). Each rep is one fresh launch of both
servers and the router, followed by:

```bash
bash $KIT/scripts/run_loads.sh $VER $REP results/perf
```

`run_loads.sh` runs a warmup and five loads (`NAME input_len output_len num_prompts concurrency`):
`P1 2048 4 300 1`, `P2 32768 4 20 1`, `P3c32 4096 256 256 32`, `P3c128 4096 256 512 128`,
`P4c32 16384 16 128 32`. It writes the benchmark output and Prometheus snapshots before and after
each load.

Run at least 3 reps per version, interleaved as base, pr, pr, base, base, pr. If the 95% interval of
the P1 TTFT difference is wider than +/-1 ms, add 3 more reps per version in the same pattern. Then:

```bash
python $KIT/scripts/summarize_perf.py results/perf --metrics all > results/perf/summary.txt
python $KIT/scripts/summarize_stages.py results/perf > results/perf/stages.txt
```

`summarize_perf.py` reports, per load and metric, the PR-minus-base difference of rep means with a
Welch 95% interval. `summarize_stages.py` does the same for per-stage server latencies; the stages
most related to this PR are `prefill_transfer_kv_cache` and `decode_transferred`.

## Deliverables

Write `REPORT.md` and pack `results/` and `logs/` into one archive. The report contains:

1. Environment: nodes, GPUs, NICs, driver and CUDA, NIXL and UCX versions, the UCX variables used,
   and both commit hashes as printed by each server log.
2. Transport check: the RDMA counter deltas and the staging counter from the preflight.
3. Correctness: determinism match count, GSM8K full and 200 scores, abort outcome counts and `hung`,
   post-check results and staging counters for both versions, and which pass criteria hold.
4. Performance: `summary.txt` and `stages.txt`, and the loads whose interval excludes 0.
5. Anomalies: every failure, hang, traceback or unexpected difference, with log excerpts and the
   commands that produced them.
6. Deviations from this plan and their reasons.
