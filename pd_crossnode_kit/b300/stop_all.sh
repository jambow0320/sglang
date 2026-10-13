#!/bin/bash
# Stop every sglang server and router in this pod, then wait until no process holds a GPU.
pkill -f "sglang.launch_server|sglang::|sglang_router" || true
for i in $(seq 1 60); do
    left=$(pgrep -c -f "sglang.launch_server|sglang::|sglang_router" || true)
    gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [[ ${left:-0} -eq 0 && $gpu -eq 0 ]]; then
        echo "stopped (${i}s)"
        exit 0
    fi
    [[ $i -eq 20 ]] && pkill -9 -f "sglang.launch_server|sglang::|sglang_router"
    sleep 1
done
echo "still running: processes=$left gpu_apps=$gpu"
exit 1
