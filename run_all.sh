#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
MODE=${1:-test}
EXTRA=()
case "$MODE" in
  test) N=10; OUT=runs/test_fixed ;;
  full)
    N=1000; OUT=runs/full_fixed
    if [ -f runs/test_fixed/prompts.jsonl ]; then EXTRA=(--exclude-prompts runs/test_fixed/prompts.jsonl); fi ;;
  *) echo 'Usage: bash run_all.sh test|full' >&2; exit 2 ;;
esac
python generate_dataset.py --n-per-class "$N" --out-dir "$OUT" --batch-size 1 "${EXTRA[@]}"
python extract_features.py --run-dir "$OUT"
for F in windows_whole windows whole halves4; do
  if [ ! -f "$OUT/linreg_$F.joblib" ]; then
    python train_classifier.py --run-dir "$OUT" --features "$F"
  else
    echo "Reusing $OUT/linreg_$F.joblib"
  fi
done
cat "$OUT/generation_timing.json"
