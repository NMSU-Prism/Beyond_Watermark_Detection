#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
COUNTS=("$@")
if [ ${#COUNTS[@]} -eq 0 ]; then COUNTS=(1 2 3 4 5 6 7 8 9); fi
PAIRS=(KGW_KGW PRW_PRW KGW_PRW)
RUNS=(runs/llama_full runs/final_PRW_PRW_n3000_actualmin25_max200 runs/final_KGW_PRW_n3000_min25_max200)
# Check all sources before doing any training.
for run in "${RUNS[@]}"; do
    for file in config.json samples.jsonl COMPLETE.json average_window_width.txt; do
        if [ ! -s "$run/$file" ]; then echo "Missing $run/$file" >&2; exit 1; fi
    done
done
mkdir -p runs
BATCH=$(mktemp -d runs/windows_only_three_pairs_XXXXXX)
printf '%s\n' "$BATCH" > runs/latest_windows_only_sweep.txt
for i in "${!PAIRS[@]}"; do
    pair=${PAIRS[$i]}; run=${RUNS[$i]}
    width=$(cat "$run/average_window_width.txt")
    if ! [[ "$width" =~ ^[1-9][0-9]*$ ]]; then echo "Invalid width: $run" >&2; exit 1; fi
    widths=(25)
    if [ "$width" -ne 25 ]; then widths+=("$width"); fi
    echo "PAIR=$pair COUNTS=${COUNTS[*]} WIDTHS=${widths[*]} OUTPUT=$BATCH/$pair"
    cp "$run/average_window_width.txt" "$BATCH/${pair}_average_window_width.txt"
    {
        time python -u window_sweep_windows_only.py \
          --run-dir "$run" --out "$BATCH/$pair" \
          --counts "${COUNTS[@]}" --widths "${widths[@]}"
    } 2>&1 | tee "$BATCH/${pair}.log"
done
python - "$BATCH" <<'PY'
from pathlib import Path
import sys
import pandas as pd
root=Path(sys.argv[1])
for split in ('validation','test'):
    tables=[]
    for pair in ('KGW_KGW','PRW_PRW','KGW_PRW'):
        df=pd.read_csv(root/pair/f'{split}_comparison.csv')
        assert (df.n_features==2*df.windows).all()
        df.insert(0,'method_pair',pair)
        tables.append(df)
    pd.concat(tables,ignore_index=True).to_csv(root/f'{split}_all_methods.csv',index=False)
print('Saved windows-only results:',root)
PY
