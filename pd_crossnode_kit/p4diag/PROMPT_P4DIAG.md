# Follow-up: timing probes for the P4c32 transfer-stage difference (PR #41140)

In the previous cross-node run (prefill TP2 / decode TP1, NIXL, staging), the PR's
`prefill_transfer_kv_cache` on P4c32 was 1836 ms against 1460 ms for base. This run repeats
P4c32 only, with timing probes, to find which step of the staged transfer takes longer. Do not
modify sglang source code.

## Setup

Same nodes, GPUs, image, UCX variables and launch flags as the previous run, with two changes:

1. Replace the previous `shim/` with `$KIT/p4diag/` on `PYTHONPATH` on both nodes
   (`PYTHONPATH=$KIT/p4diag:<checkout>/python`). Its `sitecustomize.py`
   contains the same `slot_layer_ids` shim and adds the probes. The staging instrument is not used.
2. Set `P4DIAG_DIR=<shared-results>/p4diag/<ver>-r<rep>` on both nodes for each launch (create the
   directory first). Each scheduler process writes `<host>-<pid>.jsonl` there.

Prefill keeps `--disable-radix-cache`; `SGLANG_DISAGG_STAGING_BUFFER=1` on both nodes.

## Runs

Four fresh launches in the order base r1, pr r2, pr r3, base r4. After each launch:

```bash
LOADS_SPEC="P4c32 16384 16 128 32 106" timeout 600 bash $KIT/scripts/run_loads.sh $VER $REP results/p4diag-perf
```

The `timeout` bounds the base hang seen before (one request never completed); record whether it
fired. Before r1 and after r4, record the clock offset between the two nodes (for example
`chronyc tracking` on both, or the offset reported by your time service): some probe intervals
compare timestamps across nodes.

## Analysis and deliverables

```bash
for d in <shared-results>/p4diag/*-r*; do python3 $KIT/p4diag/p4_analyze.py $d; done > p4diag_summary.txt
python3 $KIT/scripts/summarize_stages.py results/p4diag-perf > results/p4diag-perf/stages.txt
```

Send `p4diag_summary.txt`, `stages.txt`, the `p4diag/` and `results/p4diag-perf/` directories, the
server logs and the clock offsets.
