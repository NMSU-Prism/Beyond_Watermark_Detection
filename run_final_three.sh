#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/order_detect"

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

for PAIR in KGW_KGW PRW_PRW KGW_PRW; do
    A="${PAIR%_*}"
    B="${PAIR#*_}"

    RUN="runs/final_${PAIR}_n3000_min100_max200"

    echo "Starting: $PAIR"
    date

    if [ ! -f "$RUN/config.json" ]; then
        python method_order.py prepare \
            --base-run runs/llama_full \
            --out "$RUN" \
            --a "$A" \
            --b "$B" \
            --count 3000 \
            --prw-fraction 0.5 \
            --strength 2.0 \
            --min-new-tokens 100 \
            --max-new-tokens 200
    fi

    python method_order.py preflight --run-dir "$RUN"

    bash run_method_order.sh "$RUN" dynamic4
    bash run_method_order.sh "$RUN" fixed59x4

    echo "Completed: $PAIR"
    date
done
