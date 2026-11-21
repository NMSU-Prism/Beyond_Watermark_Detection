# End-to-end watermark-order experiments: KGW–KGW, PRW–PRW, KGW–PRW

This README covers preparation, smoke generation, full text generation, scoring, average-width calculation, classifier training, evaluation, and comparison. It assumes the existing `~/order_detect` project and its previously supplied add-ons are installed. It is not an installer for an empty server.

The commands below create new `end2end_*` runs and preserve your earlier results. Each pair uses the same source prompts, group splits, pinned Llama model revision, and minimum 25/maximum 200 new tokens. A/B share model weights but use different watermark keys. PRW uses fraction 0.5 and KGW uses 0.25 in this implementation; strength is 2.0. This is your order-classification experiment, not an exact reproduction of the collision paper's full setup.

## A. Check the project and environment

```bash
cd ~/order_detect
source ~/watermark-order/bin/activate
mkdir -p logs
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

python - <<'PY'
from pathlib import Path
import torch, transformers, pandas, sklearn
needed = [
    'generate_dataset.py', 'method_order.py', 'method_backends.py',
    'window_sweep.py', 'run_method_order.sh', 'train_method_classifiers.py',
    'evaluate_three_models.py', 'common.py', 'engine.py',
]
missing = [x for x in needed if not Path(x).is_file()]
if missing:
    raise SystemExit(f'Missing project/add-on files: {missing}')
print('PyTorch:', torch.__version__)
print('Transformers:', transformers.__version__)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable: resolve the GPU environment before generation.')
PY

python method_order.py prepare --help
python window_sweep.py --help
```

If method dependencies are missing, use your existing add-on requirement file:

```bash
python -m pip install -r requirements-methods.txt
```

Keep your working CUDA/PyTorch installation. Access to the Llama model must already be approved on Hugging Face. If authentication is needed, run `hf auth login` on the server. Do not put access tokens in this README or scripts.

## B. Select the shared model/prompt template

For your current server, reuse the configuration and 3,000 prompt groups already saved in `runs/llama_full`. Preparation copies prompts/configuration; it does not reuse those texts as PRW-generated passages.

```bash
python - <<'PY'
from pathlib import Path
import json
root = Path('runs/llama_full')
c = json.loads((root / 'config.json').read_text())['experiment']
p = root / 'prompts.jsonl'
if not p.is_file():
    p = root / 'prompts_used.jsonl'
rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
print('Model A:', c['A'])
print('Model B:', c['B'])
print('Prompt groups:', len(rows))
if len(rows) < 3000:
    raise SystemExit('Need a base with at least 3,000 prompt groups.')
print('Template available.')
PY
```

**Only if you do not already have a base run:** create one using the installed original generation engine. This command downloads/prepares source prompts and generates a full KGW base bank, so it is expensive; skip it on your current server because `runs/llama_full` exists.

```bash
python generate_dataset.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --revision 9213176726f574b556790deb65791e0c5aa438b6 \
  --n-per-class 3000 \
  --out-dir runs/llama_full \
  --batch-size 1 \
  --prompt-style chat --dtype bfloat16 \
  --min-new-tokens 25 --max-new-tokens 200 \
  --temperature 0.7 --top-p 0.95 --top-k 40 \
  --gamma 0.25 --delta 2.0 \
  --seeding-scheme selfhash --key-a 2024 --key-b 2023 --seed 0
```

The installed engine's default source is the English C4 shard `c4-train.00000-of-01024.json.gz`. For a local JSONL source, supply `--prompts-file /absolute/path/prompts.jsonl`; it must contain text records accepted by the engine. Do not change arguments when resuming an existing run.

## C. Smoke-test generation for every pair

Create a standalone script so failures do not close your SSH shell. Preparation requires new directories. Run preparation once; if interrupted after a directory is prepared, resume its `generate` command rather than preparing that directory again.

```bash
cat > smoke_three_pairs.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/order_detect"
for pair in KGW_KGW PRW_PRW KGW_PRW; do
    case "$pair" in
        KGW_KGW) A=KGW; B=KGW ;;
        PRW_PRW) A=PRW; B=PRW ;;
        KGW_PRW) A=KGW; B=PRW ;;
    esac
    RUN="runs/end2end_smoke_${pair}"
    python method_order.py prepare \
      --base-run runs/llama_full --out "$RUN" \
      --a "$A" --b "$B" --count 10 \
      --prw-fraction 0.5 --strength 2.0 \
      --min-new-tokens 25 --max-new-tokens 200
    python method_order.py preflight --run-dir "$RUN"
    time python method_order.py generate --run-dir "$RUN"
    python method_order.py score --run-dir "$RUN" --modes dynamic4
done
BASH

bash smoke_three_pairs.sh 2>&1 | tee logs/end2end_smoke_generation.log
```

Inspect intermediate and final text:

```bash
python - <<'PY'
from pathlib import Path
import json
for pair in ('KGW_KGW', 'PRW_PRW', 'KGW_PRW'):
    root = Path('runs') / f'end2end_smoke_{pair}'
    rows = [json.loads(x) for x in (root / 'samples.jsonl').read_text().splitlines() if x.strip()]
    print('\nPAIR:', pair, 'passages:', len(rows))
    for r in rows[:4]:
        print('\nLABEL:', r['label'])
        print('INTERMEDIATE:', r['intermediate_text'])
        print('FINAL:', r['final_text'])
PY
```

Check that rewriting preserves meaning, outputs are not dominated by repetition, both labels exist, and scores are finite. Native preflight checks implementation consistency, not order-attribution accuracy. Ten prompt groups are a software smoke test, not enough for a reliable accuracy claim. These smoke groups are subsets of the existing base prompts; for a strictly untouched final study, reserve separate development prompts before collecting the final bank.

Generation prints elapsed time and rough ETA. Use observed generation timing on the same server/settings to budget the full run; do not assume a previous pair's runtime applies exactly.

## D. Prepare full datasets (3,000 prompts and 6,000 final passages per pair)

This deliberately generates fresh text for all three pairs using the method adapter. If you only want to reuse existing completed banks, skip fresh generation and map the later source paths to those banks instead.

```bash
cat > prepare_full_three_pairs.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/order_detect"
for pair in KGW_KGW PRW_PRW KGW_PRW; do
    case "$pair" in
        KGW_KGW) A=KGW; B=KGW ;;
        PRW_PRW) A=PRW; B=PRW ;;
        KGW_PRW) A=KGW; B=PRW ;;
    esac
    RUN="runs/end2end_${pair}_n3000_min25_max200"
    python method_order.py prepare \
      --base-run runs/llama_full --out "$RUN" \
      --a "$A" --b "$B" --count 3000 \
      --prw-fraction 0.5 --strength 2.0 \
      --min-new-tokens 25 --max-new-tokens 200
    python method_order.py preflight --run-dir "$RUN"
done
BASH

bash prepare_full_three_pairs.sh
```

Check each newly saved config before generation. The sampling minimum must be 25 and maximum 200. KGW fraction is hard-coded to 0.25 by this adapter; do not pass the unsupported `--kgw-fraction` option.

## E. Generate and save full texts, then score an initial layout

```bash
cat > generate_full_three_pairs.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/order_detect"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
mkdir -p logs
for pair in KGW_KGW PRW_PRW KGW_PRW; do
    RUN="runs/end2end_${pair}_n3000_min25_max200"
    {
        time python method_order.py generate --run-dir "$RUN"
        time python method_order.py score --run-dir "$RUN" --modes dynamic4
    } 2>&1 | tee -a "logs/end2end_${pair}_generation_scoring.log"
done
BASH

bash generate_full_three_pairs.sh
```

For a long run, launch the script inside `screen -S watermark_generation`. Detach with Ctrl+A then D; reattach with `screen -r watermark_generation`.

Generation saves completed first-stage outputs and final passages, configuration, prompts, a manifest, and a completion marker. Resume an interrupted run by rerunning the same `method_order.py generate --run-dir ...` command. Do not rerun `prepare` on its existing directory or edit the run's configuration in place. Initial scoring saves final-text evidence and `dynamic4/features.csv`; this supplies training token lengths for the next stage. Scoring does not train classifiers.

Across all three full banks this produces 18,000 final passages. All histories from a prompt retain the same train/validation/test group assignment. The two order categories each contain 3,000 passages per pair.

# Continue: feature layouts, classifier training, and evaluation

The remaining steps use the new `end2end_*` banks created above. You can evaluate K=3 alone or K=3 through 9 without regenerating text.

## 1. Activate the environment

```bash
cd ~/order_detect
source ~/watermark-order/bin/activate
mkdir -p logs
python window_sweep.py --help
```

Use the updated `window_sweep.py` that accepts arbitrary positive token widths, rather than a version restricted to 25 and 59.

## 2. Check source runs

| Pair | Source run |
| --- | --- |
| KGW–KGW | `runs/end2end_KGW_KGW_n3000_min25_max200` |
| PRW–PRW | `runs/end2end_PRW_PRW_n3000_min25_max200` |
| KGW–PRW | `runs/end2end_KGW_PRW_n3000_min25_max200` |

Use the corrected PRW run above. The older directory named `final_PRW_PRW_n3000_min25_max200` actually used a minimum of 100 in the configuration previously inspected.

```bash
python - <<'PY'
import json
from pathlib import Path

for folder in (
    'end2end_KGW_KGW_n3000_min25_max200',
    'end2end_PRW_PRW_n3000_min25_max200',
    'end2end_KGW_PRW_n3000_min25_max200',
):
    root = Path('runs') / folder
    print('\nSOURCE:', root)
    for name in ('config.json', 'samples.jsonl', 'COMPLETE.json'):
        if not (root / name).is_file():
            raise SystemExit(f'Missing: {root / name}')
    config = json.loads((root / 'config.json').read_text())
    sampling = config['experiment']['sampling']
    print('Sampling:', sampling)
    if sampling['min_new_tokens'] != 25 or sampling['max_new_tokens'] != 200:
        raise SystemExit('Expected minimum 25 and maximum 200; inspect configuration.')
    with (root / 'samples.jsonl').open() as f:
        n = sum(bool(line.strip()) for line in f)
    print('Final passages:', n)
    if n != 6000:
        raise SystemExit('Expected 6,000 saved passages.')
print('\nSource-file checks passed. This does not verify generation quality or accuracy.')
PY
```

Do not change a completed run's configuration to relabel previously generated data. If source files are absent, finish the corresponding generation workflow first.

## 3. Create an initial feature table if needed

The average width is calculated from training rows in an existing feature table. Look for these tables:

```bash
find runs -name features.csv
```

If a pair has no feature table, run its initial layout. Run only the command needed for the missing pair:

```bash
bash run_method_order.sh runs/end2end_KGW_KGW_n3000_min25_max200 dynamic4
bash run_method_order.sh runs/end2end_PRW_PRW_n3000_min25_max200 dynamic4
bash run_method_order.sh runs/end2end_KGW_PRW_n3000_min25_max200 dynamic4
```

These commands use your existing project wrapper. For completed compatible runs, its generation stage should resume/reuse saved passages. Check the log before leaving a run unattended.

## 4. Calculate each pair's average width

Create the following helper. It calculates the mean from training rows only, checks A/B tokenizer-length agreement when available, and preserves a conflicting existing width for inspection.

```bash
cat > prepare_average_widths.py <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd

R = Path('runs')
EXPERIMENTS = {
    'KGW_KGW': ('end2end_KGW_KGW_n3000_min25_max200', []),
    'PRW_PRW': ('end2end_PRW_PRW_n3000_min25_max200', []),
    'KGW_PRW': ('end2end_KGW_PRW_n3000_min25_max200', []),
}

results = []
for pair, (folder, alternatives) in EXPERIMENTS.items():
    run = R / folder
    candidates = []
    for root in [run] + [R / x for x in alternatives]:
        candidates.extend([root / 'features.csv', root / 'dynamic4/features.csv',
                           root / 'dynamic3/features.csv', root / 'whole/features.csv'])
        candidates.extend(sorted(root.glob('*/features.csv')))
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise SystemExit(f'{pair}: no features.csv found; complete step 3.')
    df = pd.read_csv(source)
    if 'split' not in df:
        raise SystemExit(f'{source}: missing split column')
    if 'sample_id' in df:
        if df['sample_id'].isna().any() or df['sample_id'].duplicated().any():
            raise SystemExit(f'{source}: invalid sample IDs')
    elif {'idx', 'label'}.issubset(df.columns):
        if df.duplicated(['idx', 'label']).any():
            raise SystemExit(f'{source}: duplicate passages')
    train = df[df['split'].astype(str).str.strip().str.lower().eq('train')]
    if train.empty:
        raise SystemExit(f'{source}: no training passages')
    if {'A_n_tokens', 'B_n_tokens'}.issubset(train.columns):
        lengths = pd.to_numeric(train['A_n_tokens'], errors='raise')
        other = pd.to_numeric(train['B_n_tokens'], errors='raise')
        if not np.array_equal(lengths.to_numpy(), other.to_numpy()):
            raise SystemExit(f'{pair}: A/B token lengths differ; select tokenizer explicitly')
    elif 'n_tokens' in train:
        lengths = pd.to_numeric(train['n_tokens'], errors='raise')
    else:
        raise SystemExit(f'{source}: no token-count column')
    v = lengths.to_numpy(dtype=float)
    if (not np.isfinite(v).all() or (v <= 0).any()
            or not np.equal(v, np.floor(v)).all()):
        raise SystemExit(f'{source}: invalid token lengths')
    mean = float(v.mean())
    width = max(1, int(np.floor(mean + 0.5)))
    target = run / 'average_window_width.txt'
    if target.exists() and int(target.read_text().strip()) != width:
        raise SystemExit(f'{target}: existing width differs from calculated {width}; inspect first')
    results.append((run, dict(method_pair=pair, source_features=str(source),
        split='train', training_passages=len(train), mean_tokens=mean,
        window_width=width)))

# Write only after all three inputs pass validation.
for run, result in results:
    (run / 'average_window_width.txt').write_text(f"{result['window_width']}\n")
    (run / 'average_window_settings.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
PY

python prepare_average_widths.py
```

Use each pair's computed width. Do not copy the PRW width to KGW–PRW. An average passage length is an experimental window-width choice, not a demonstrated optimal setting.

## 5. Create the launcher

The launcher executes pairs sequentially. Each invocation creates a fresh batch directory so earlier sweep outputs remain unchanged. `set -e` is inside the script, so a failure stops the script without logging you out of your interactive SSH shell. Logs include elapsed wall-clock time from `time`.

```bash
cat > run_three_pair_sweeps.sh <<'BASH'
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

run_pair KGW_KGW runs/end2end_KGW_KGW_n3000_min25_max200
run_pair PRW_PRW runs/end2end_PRW_PRW_n3000_min25_max200
run_pair KGW_PRW runs/end2end_KGW_PRW_n3000_min25_max200
printf 'Completed all three pairs. Results: %s\n' "$BATCH"
BASH

chmod +x run_three_pair_sweeps.sh
```

## 6. Run the experiments

For three windows only:

```bash
bash run_three_pair_sweeps.sh 3
```

Alternatively, evaluate counts 3 through 9:

```bash
bash run_three_pair_sweeps.sh 3 4 5 6 7 8 9
```

Choose one command. No need to repeat counts already evaluated unless you want a new complete batch. A new output directory may require detector evidence to be calculated again if compatible cached evidence cannot be reused; it does not require new LLM generation.

For count 3, each pair has four layouts (whole, dynamic3, fixed25x3, fixedAveragex3) and three classifiers: 12 configurations per pair, 36 across the three pairs. If average width equals 25, the duplicate fixed-width layout is intentionally omitted.

For a long session, start a screen session before the run:

```bash
screen -S watermark_sweeps
source ~/watermark-order/bin/activate
cd ~/order_detect
bash run_three_pair_sweeps.sh 3
```

Detach with Ctrl+A followed by D. Reattach with `screen -r watermark_sweeps`. Do not launch the same batch twice concurrently.

## 7. Combine saved comparisons

Run after all three pairs finish:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

batch = Path(Path('runs/latest_three_pair_sweep.txt').read_text().strip())
for split in ('validation', 'test'):
    tables = []
    for pair in ('KGW_KGW', 'PRW_PRW', 'KGW_PRW'):
        source = batch / pair / f'{split}_comparison.csv'
        df = pd.read_csv(source)
        if df.duplicated(['layout', 'model']).any():
            raise SystemExit(f'Duplicate layout/model rows: {source}')
        df.insert(0, 'method_pair', pair)
        tables.append(df)
    combined = pd.concat(tables, ignore_index=True)
    target = batch / f'{split}_comparison_all_methods.csv'
    combined.to_csv(target, index=False)
    print('\n' + split.upper())
    print(combined[['method_pair', 'layout', 'model', 'accuracy', 'macro_f1', 'auc']]
          .to_string(index=False))
    print('Saved:', target)
print('\nBatch:', batch)
PY
```

The latest-batch pointer is updated each time the launcher starts. For an older batch, replace the pointer lookup in the comparison script with its explicit directory.

## 8. Locate outputs

```bash
BATCH=$(cat runs/latest_three_pair_sweep.txt)
ls -lh "$BATCH"
ls -lh "$BATCH/PRW_PRW"
```

Each pair's sweep directory contains comparison CSVs, per-layout features and classifier outputs, completion metadata, and the plots produced by your installed sweep script. The batch root contains per-pair timing logs, average-width settings, and the combined comparison CSVs from step 7.

## Interpretation and reporting
<<<<<<< HEAD

- Fit scaling and classifier parameters using training data only.
- Select layouts, counts, classifier settings, and any tuned decision threshold using validation data.
- Use the test partition to report the selected configuration's final performance. Mark the maximum across inspected test configurations as descriptive, not an independently selected result.
- Keep both histories from the same source prompt in the same partition, using the existing prompt-group split.
- Compare whole-text-only performance against whole-text-plus-window performance to measure the contribution of local scores.
- Equal accuracy across layouts means the same number of correct predictions; it does not necessarily mean identical predictions or identical AUC.
- The classifier addresses two known histories. Accuracy does not prove that the first watermark survived: identifying the last watermark can already help infer the order in this restricted setting.
- Use the corrected PRW minimum-length run for controlled comparisons. Do not mix its results with the earlier minimum-100 PRW generation.

## Troubleshooting

**Missing average_window_width.txt:** complete step 4. Never append `|| exit 1` directly to a command in your interactive SSH shell; it can terminate the session.

**Missing features.csv:** complete an initial scoring layout in step 3. Use tables belonging to the same generation run; especially avoid the old PRW minimum-100 tables.

**Only average-token results appear:** check that the launcher includes both widths. This guide supplies the average and 25 explicitly.

**Invalid width argument:** your sweep version may still restrict widths to 25 and 59. Use the previously updated arbitrary-width version before continuing.

**Missing COMPLETE.json:** the source run is not marked complete. Finish and verify that run before starting its sweep.

**Interrupted sweep:** retain its directory and logs. Consult your installed script's resume behavior before rerunning that exact invocation. Running this launcher again deliberately creates a fresh batch rather than resuming the old one.

## Validation status of this README

The embedded Python and shell blocks were checked for syntax when this file was created. The experiments were not executed on your remote servers as part of writing this README. Their actual success and timing depend on your installed project code, source files, and environment.

=======
## Troubleshooting
## Validation status of this README
>>>>>>> 91cec8f (added readme)
