#!/bin/bash
# usage: run_loads.sh <base|pr> <rep> <out-dir>
# Runs the perf load set against an already running PD stack; one call is one sample per load.
# env: LB_URL PREFILL_URL DECODE_URL MODEL; PYTHONPATH must point at the base checkout so the
#   client is identical for both versions. Optional LOADS_SPEC="NAME IN OUT N C SEED;...".
set -uo pipefail
VER=$1
REP=$2
OUT=$3
S=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
mkdir -p "$OUT"

bench() {
    "$PY" -m sglang.benchmark.serving --backend sglang --base-url "$LB_URL" \
        --model "$MODEL" --tokenizer "$MODEL" \
        --dataset-name random-ids --random-range-ratio 1 --disable-tqdm "$@"
}

echo "== warmup"
bench --random-input-len 4096 --random-output-len 64 --num-prompts 64 --max-concurrency 16 --seed 7 > /dev/null 2>&1

DEFAULT_LOADS="P1 2048 4 300 1 101;P2 32768 4 20 1 102;P3c32 4096 256 256 32 103;P3c128 4096 256 512 128 105;P4c32 16384 16 128 32 106"
IFS=";" read -r -a LOADS <<< "${LOADS_SPEC:-$DEFAULT_LOADS}"
for load in "${LOADS[@]}"; do
    read -r NAME IN OUTLEN N C SEED <<< "$load"
    PREFIX="$OUT/$VER-r$REP-$NAME"
    "$PY" "$S/metrics_snapshot.py" "$PREFIX.metrics_before.json"
    bench --random-input-len "$IN" --random-output-len "$OUTLEN" --num-prompts "$N" --max-concurrency "$C" \
        --seed "$SEED" --output-file "$PREFIX.jsonl" > "$PREFIX.log" 2>&1
    "$PY" "$S/metrics_snapshot.py" "$PREFIX.metrics_after.json"
    grep -E "Successful requests|Mean TTFT|P99 TTFT|Output token throughput" "$PREFIX.log" \
        | tr -s ' ' | tr '\n' ';' | sed "s/^/$VER r$REP $NAME: /"
    echo
done
