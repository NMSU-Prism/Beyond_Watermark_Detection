#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
RUN=${1:?Usage: bash run_method_order.sh RUN_DIRECTORY [dynamic4|fixed59x4]}
MODE=${2:-dynamic4}
python method_order.py generate --run-dir "$RUN"
python method_order.py score --run-dir "$RUN" --modes "$MODE"
if [ ! -f "$RUN/$MODE/linreg_windows_whole.joblib" ]; then
 python train_method_classifiers.py --case-dir "$RUN/$MODE"
fi
if [ ! -f "$RUN/$MODE/metrics_windows_whole/all_metrics.json" ]; then
 python evaluate_three_models.py --run-dir "$RUN/$MODE" --features windows_whole
fi
