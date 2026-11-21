#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/order_detect"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

COUNTS=("$@")
if [ "${#COUNTS[@]}" -eq 0 ]; then COUNTS=(3); fi
for count in "${COUNTS[@]}"; do
    if ! [[ "$count" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid window count: $count" >&2
        exit 1
    fi
done

mkdir -p runs
BATCH=$(mktemp -d "runs/three_pair_sweeps_$(date +%Y%m%d_%H%M%S)_XXXXXX")
printf '%s\n' "$BATCH" > runs/latest_three_pair_sweep.txt
printf '%s\n' "Window counts: ${COUNTS[*]}" > "$BATCH/settings.txt"

run_pair() {
    local pair="$1" source_run="$2" average_width
    average_width=$(cat "$source_run/average_window_width.txt")
    if ! [[ "$average_width" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid average width for $pair" >&2
        return 1
    fi
    local widths=("$average_width")
    if [ "$average_width" -ne 25 ]; then widths+=(25); fi
    echo "Starting $pair: counts=${COUNTS[*]}, widths=${widths[*]}"
    cp "$source_run/average_window_settings.json" "$BATCH/${pair}_average_settings.json"
    {
        time python window_sweep.py \
          --run-dir "$source_run" \
          --out "$BATCH/$pair" \
          --counts "${COUNTS[@]}" \
          --widths "${widths[@]}" \
          --include-whole
    } 2>&1 | tee "$BATCH/${pair}.log"
}

run_pair KGW_KGW runs/llama_full
run_pair PRW_PRW runs/final_PRW_PRW_n3000_actualmin25_max200
run_pair KGW_PRW runs/final_KGW_PRW_n3000_min25_max200
printf 'Completed all three pairs. Results: %s\n' "$BATCH"
