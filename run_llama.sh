#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
MODE=${1:-test}
EXTRA=()
case "$MODE" in
 test) N=10; OUT=runs/llama_test ;;
 full)
  N=3000; OUT=runs/llama_full
  if [ -f runs/llama_test/prompts.jsonl ]; then EXTRA=(--exclude-prompts runs/llama_test/prompts.jsonl); fi ;;
 *) echo 'Usage: bash run_llama.sh test|full' >&2; exit 2 ;;
esac
python generate_dataset.py --model meta-llama/Llama-3.2-1B-Instruct \
 --n-per-class "$N" --out-dir "$OUT" --batch-size 1 \
 --prompt-style chat --dtype bfloat16 --min-new-tokens 25 --max-new-tokens 200 "${EXTRA[@]}"
python extract_features.py --run-dir "$OUT"
for F in windows_whole windows whole halves4; do
 if [ ! -f "$OUT/linreg_$F.joblib" ]; then python train_classifier.py --run-dir "$OUT" --features "$F"; fi
done
cat "$OUT/generation_timing.json"
