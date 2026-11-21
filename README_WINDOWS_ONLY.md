# Windows-only classification: all three pairs

This add-on uses only A/B local window scores. It removes A_whole/B_whole (also called whole_A/whole_B) from feature CSVs and model inputs. The detector may compute whole scores internally when producing cached token evidence, but they never enter these classifiers. Existing project files and previous results are preserved.

## Install in your existing project

Download windows_only_addon.zip to the server, then:

```bash
cd ~/order_detect
source ~/watermark-order/bin/activate
unzip -n ~/Downloads/windows_only_addon.zip
python window_sweep_windows_only.py --self-test
```

Use the actual downloaded archive location if different. This add-on requires the existing engine.py, method_backends.py, watermark implementations, and working scientific Python environment. It does not replace the original window_sweep.py.

## Sources and average widths

The launcher uses:

- KGW–KGW: runs/llama_full
- PRW–PRW: runs/final_PRW_PRW_n3000_actualmin25_max200
- KGW–PRW: runs/final_KGW_PRW_n3000_min25_max200

Each needs config.json, samples.jsonl, COMPLETE.json, and average_window_width.txt. Calculate average widths from training passages only using your existing prepare_average_widths.py. Do not use the older PRW minimum-100 bank. If you use end2end_* banks instead, edit the RUNS array in the launcher to match those banks and calculate averages from their own feature tables.

```bash
python prepare_average_widths.py
```

Run the preceding command only if average widths have not already been saved. It uses the helper previously supplied in the project; that helper is not bundled here.

## Run

Only counts 1 and 2:

```bash
bash run_windows_only_three_pairs.sh 1 2
```

Only count 3:

```bash
bash run_windows_only_three_pairs.sh 3
```

All counts 1–9:

```bash
bash run_windows_only_three_pairs.sh 1 2 3 4 5 6 7 8 9
```

For each pair and requested count K, run dynamicK, fixed25xK, and fixedAveragexK, with OLS linear regression, RBF SVM, and MLP. If the average is 25, duplicate layouts are omitted.

There are exactly 2K classifier features, ordered A_w1...A_wK then B_w1...B_wK. Masks, lengths, labels, hashes, and whole scores are not input features. Labels and split metadata remain in the feature CSV for evaluation. Missing windows use zero scores as in the original sweep. Fixed windows cover a prefix of at most K times width tokens; removing whole scores means omitted suffixes now contribute no features. Dynamic windows cover the full passage. Local scoring preserves the existing native evidence aggregation, including KGW deduplication and PRW counting conventions.

For K=1, dynamic1 covers the full passage, so its local score can be numerically identical to the whole-text score. It still occupies a window feature column. To study truly local-only evidence, inspect K>=2 as well.

## Saved outputs

Each launch creates runs/windows_only_three_pairs_XXXXXX with three pair subdirectories. Previous output directories are not changed. The latest location is recorded in runs/latest_windows_only_sweep.txt.

Every layout saves:

- features.csv and layout.json, listing the exact selected inputs.
- model_linear.joblib, model_svm.joblib, model_mlp.joblib.
- train/validation/test_predictions_linear.csv (and svm/mlp equivalents): prompt ID, true label, predicted label, continuous score.
- results_linear.json (and svm/mlp equivalents): per-class precision, recall, F1, confusion matrix, accuracy, macro metrics, AUC, elapsed training time, warnings.
- ROC points and PNG/PDF per classifier.

Pair directories save all_results.csv, validation_comparison.csv, test_comparison.csv, validation_selected_candidate.json, comparison plots, manifests and cached evidence. The batch saves combined validation_all_methods.csv and test_all_methods.csv plus timing logs.

Predictions here are held-out saved-dataset predictions, not a new-text CLI. Load joblib files only from trusted sources. Do not use old whole-plus-window inference scripts with these windows-only bundles: the input feature schema differs.

## Verify exclusion and feature counts

```bash
python - <<'PY'
from pathlib import Path
import json
import pandas as pd
root=Path(Path('runs/latest_windows_only_sweep.txt').read_text().strip())
for pair in ('KGW_KGW','PRW_PRW','KGW_PRW'):
    print('\n',pair)
    print(pd.read_csv(root/pair/'test_comparison.csv')[
        ['layout','model','n_features','accuracy','macro_f1','auc']].to_string(index=False))
    for path in (root/pair).glob('*/layout.json'):
        spec=json.loads(path.read_text())
        assert not spec['whole_scores_included']
        assert len(spec['columns'])==2*spec['k']
        assert all(c.startswith(('A_w','B_w')) and 'whole' not in c for c in spec['columns'])
        df=pd.read_csv(path.parent/'features.csv',nrows=1)
        assert 'A_whole' not in df and 'B_whole' not in df
print('\nPASS: only local window score columns are supplied to classifiers.')
PY
```

## Reuse and fair comparison

