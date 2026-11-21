#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/order_detect"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

mkdir -p runs logs
BATCH=$(mktemp -d runs/average_windows_1_2_XXXXXX)

run_pair() {
    local pair="$1"
    local run="$2"
    local width_file="$run/average_window_width.txt"

    if [ ! -s "$width_file" ]; then
        echo "Missing average width: $width_file" >&2
        return 1
    fi

    local width
    width=$(cat "$width_file")

    if ! [[ "$width" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid average width for $pair: $width" >&2
        return 1
    fi

    echo "Running $pair: average width=$width tokens; window counts=1,2"

    {
        time python -u window_sweep.py \
          --run-dir "$run" \
          --out "$BATCH/$pair" \
          --counts 1 2 \
          --widths "$width"
    } 2>&1 | tee "$BATCH/${pair}.log"
}

run_pair PRW_PRW \
  runs/final_PRW_PRW_n3000_actualmin25_max200

run_pair KGW_PRW \
  runs/final_KGW_PRW_n3000_min25_max200

echo "Results saved in: $BATCH"
