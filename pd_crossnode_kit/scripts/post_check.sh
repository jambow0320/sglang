#!/bin/bash
# usage: post_check.sh <prefill-log> <decode-log>     env: LB_URL PREFILL_URL DECODE_URL
# Waits for idle, then reports health, queue gauges, and leak / error lines in the server logs.
# Every queue gauge must read 0 once idle. Copy or mount the remote node's log before calling.
P_LOG=$1
D_LOG=$2
sleep 15
for url in "$LB_URL" "$PREFILL_URL" "$DECODE_URL"; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$url/health")
    echo "health $url: $code"
done
for url in "$PREFILL_URL" "$DECODE_URL"; do
    curl -s -m 10 "$url/metrics" \
        | grep -E '^sglang:(num_running_reqs|num_queue_reqs|num_prefill_bootstrap_queue_reqs|num_prefill_inflight_queue_reqs|num_decode_prealloc_queue_reqs|num_decode_transfer_queue_reqs|num_transfer_failed_reqs_total|num_bootstrap_failed_reqs_total)\b' \
        | sed -E 's/\{[^}]*\}//' | sed "s|^|  $url |"
done
for f in "$P_LOG" "$D_LOG"; do
    n_leak=$(grep -c "leak detected" "$f")
    n_tb=$(grep -c "Traceback" "$f")
    n_err=$(grep -cE "\] (ERROR|Error)|KVTransferError" "$f")
    echo "$(basename "$f"): leak=$n_leak traceback=$n_tb error_lines=$n_err"
done
