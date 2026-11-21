#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
SOURCE=${1:-runs/llama_full}
OUTPUT=${2:-runs/four_windows_full}
python four_window_experiment.py extract --run-dir "$SOURCE" --out "$OUTPUT"
for MODE in dynamic4 fixed59x4; do
  CASE="$OUTPUT/$MODE"
  if [ ! -f "$CASE/linreg_windows_whole.joblib" ]; then
    python train_classifier.py --run-dir "$CASE" --features windows_whole
  fi
  if [ ! -f "$CASE/extra_windows_whole/comparison.json" ]; then
    python train_extra_classifiers.py --run-dir "$CASE" --features windows_whole --out "$CASE/extra_windows_whole"
  fi
  if [ ! -f "$CASE/metrics_windows_whole/all_metrics.json" ]; then
    python evaluate_three_models.py --run-dir "$CASE" --features windows_whole --out "$CASE/metrics_windows_whole"
  fi
done
