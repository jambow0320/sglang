#!/bin/bash
# usage: launch_server.sh <prefill|decode> <base|pr> <tag>
# Starts one PD server in the background inside this pod and returns. Prefill is TP2 on GPUs 0-1,
# decode is TP1 on GPU 0, NIXL with staging, radix cache off on prefill.
# env: HOST_IP (this pod's IP, required), ROOT (default /root/pd41140), DIAG=1 enables the timing
#   probes, NIXL_BACKEND (default LIBFABRIC), EXTRA_ARGS appended to launch_server.
set -euo pipefail
ROLE=$1
VER=$2
TAG=$3
ROOT=${ROOT:-/root/pd41140}
HOST_IP=${HOST_IP:?set HOST_IP to this pod IP}
CHECKOUT=$ROOT/sglang-$VER
LOG=$ROOT/logs/$TAG-$VER-$ROLE.log
mkdir -p "$ROOT/logs"

export PYTHONPATH=$ROOT/sglang-pr/pd_crossnode_kit/p4diag:$CHECKOUT/python
export SGLANG_HOST_IP=$HOST_IP
export SGLANG_DISAGG_STAGING_BUFFER=1
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export SGLANG_DISAGGREGATION_NIXL_BACKEND=${NIXL_BACKEND:-LIBFABRIC}
export HF_HUB_OFFLINE=1
if [[ ${DIAG:-0} == 1 ]]; then
    export P4DIAG_DIR=$ROOT/p4diag/$TAG-$VER
    mkdir -p "$P4DIAG_DIR"
fi

case $ROLE in
    prefill)
        ARGS=(--disaggregation-mode prefill --port 30000 --disaggregation-bootstrap-port 8998
            --tp 2 --base-gpu-id 0 --disable-radix-cache)
        ;;
    decode)
        ARGS=(--disaggregation-mode decode --port 30001 --tp 1 --base-gpu-id 0)
        ;;
    *)
        echo "unknown role $ROLE" >&2
        exit 1
        ;;
esac

{
    echo "== $VER $ROLE tag=$TAG started=$(date -Is)"
    "$ROOT/venv/bin/python" -c "import sglang; print('sglang from', sglang.__file__)"
    git -C "$CHECKOUT" rev-parse HEAD
    env | grep -E '^(SGLANG_|FI_|NIXL_|UCX_|P4)' | sort || true
} > "$LOG" 2>&1

# shellcheck disable=SC2086
setsid nohup "$ROOT/venv/bin/python" -m sglang.launch_server --model-path "$ROOT/models/Qwen3-8B" \
    --host "$HOST_IP" --enable-metrics "${ARGS[@]}" ${EXTRA_ARGS:-} >> "$LOG" 2>&1 < /dev/null &
echo "started $ROLE $VER pid=$! log=$LOG"
