from pathlib import Path
import json
import numpy as np
import pandas as pd

R = Path('runs')
EXPERIMENTS = {
    'KGW_KGW': ('llama_full', ['window_sweep_KGW']),
    'PRW_PRW': ('final_PRW_PRW_n3000_actualmin25_max200', [
        'window_sweep_PRW_PRW_actualmin25_average',
        'window_sweep_PRW_PRW_actualmin25_average_count3',
        'window_sweep_PRW_PRW_actualmin25_count3_avg_and25']),
    'KGW_PRW': ('final_KGW_PRW_n3000_min25_max200', [
        'window_sweep_KGW_PRW_average', 'window_sweep_KGW_PRW',
        'window_sweep_KGW_PRW_count3_avg_and25']),
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
