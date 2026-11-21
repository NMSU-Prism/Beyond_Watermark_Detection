#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/order_detect"

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

BASE="runs/llama_full"
ASSETS="assets/sir_llama"

for PAIR in KGW_KGW PRW_PRW KGW_PRW KGW_SIR PRW_SIR; do
    A="${PAIR%_*}"
    B="${PAIR#*_}"

    RUN="runs/final_${PAIR}_n3000_min100_max200"

    echo "Starting: $PAIR"
    date

    # Prepare once; preserve the saved configuration on subsequent runs.
    if [ ! -f "$RUN/config.json" ]; then
        EXTRA=()

        if [ "$A" = "SIR" ] || [ "$B" = "SIR" ]; then
            EXTRA=(--sir-assets "$ASSETS")
        fi

        python method_order.py prepare \
            --base-run "$BASE" \
            --out "$RUN" \
            --a "$A" \
            --b "$B" \
            --count 3000 \
            --prw-fraction 0.5 \
            --strength 2.0 \
            --min-new-tokens 100 \
            --max-new-tokens 200 \
            "${EXTRA[@]}"
    fi

    python method_order.py preflight --run-dir "$RUN"

    # First invocation generates text, scores it, and trains/evaluates.
    bash run_method_order.sh "$RUN" dynamic4

    # Reuses the generated text and scoring journal.
    bash run_method_order.sh "$RUN" fixed59x4

    echo "Completed: $PAIR"
    date
done
