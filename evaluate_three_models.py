"""Report saved held-out predictions without training or modifying classifiers."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, roc_auc_score, roc_curve)


def read_predictions(path, score_column):
    df = pd.read_csv(path, dtype={'idx': str})
    required = ['idx', 'true_label', 'predicted_label', score_column]
    if any(c not in df for c in required):
        raise ValueError(f'{path}: required columns {required}')
    if df[required].isna().any().any() or not np.isfinite(df[score_column].to_numpy(float)).all():
        raise ValueError(f'{path}: missing/nonfinite values')
    if not set(df.true_label) == {0, 1} or not set(df.predicted_label) <= {0, 1}:
        raise ValueError(f'{path}: expected labels A_first=0 and B_first=1')
    if df.duplicated(['idx', 'true_label']).any():
        raise ValueError(f'{path}: duplicate prompt/label records')
    return df.sort_values(['idx', 'true_label']).reset_index(drop=True)


def report_for(df, score_column):
    y, pred, score = df.true_label, df.predicted_label, df[score_column]
    report = classification_report(y, pred, labels=[0, 1],
        target_names=['A_first', 'B_first'], output_dict=True, zero_division=0)
    report.update(n=len(df), accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        roc_auc_B_first=float(roc_auc_score(y, score)),
        confusion_matrix_rows_true_columns_pred_A_first_B_first=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        positive_class='B_first (label 1)', undefined_metric_policy='zero_division=0')
    return report


def evaluate(args):
    root = Path(args.run_dir)
    extra = Path(args.extra_dir) if args.extra_dir else root / ('extra_' + args.features)
    out = Path(args.out) if args.out else root / ('metrics_' + args.features)
    if out.exists():
        raise FileExistsError(f'Existing outputs preserved: {out}; choose a new --out')
    sources = [
        ('linear_regression', 'Linear regression', root / f'test_predictions_{args.features}.csv', 'linear_score'),
        ('svm', 'SVM', extra / 'test_predictions_svm.csv', 'score'),
        ('mlp', 'Neural network', extra / 'test_predictions_mlp.csv', 'score')]
    # Read/validate everything before creating output files.
    tables = [(key, title, path, col, read_predictions(path, col)) for key, title, path, col in sources]
    reference = tables[0][-1][['idx', 'true_label']]
    for key, _, _, _, df in tables[1:]:
        if not reference.equals(df[['idx', 'true_label']]):
            raise ValueError(f'{key}: test cases differ; do not compare different splits')
    out.mkdir(parents=True)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    rows, reports = [], {}
    for key, title, path, col, df in tables:
        r = report_for(df, col)
        r['source_predictions'] = str(path)
        r['score_type'] = {'linear_regression':'OLS output, not probability',
            'svm':'SVM decision margin, not probability', 'mlp':'uncalibrated probability estimate'}[key]
        reports[key] = r
        (out / f'metrics_{key}.json').write_text(json.dumps(r, indent=2) + '\n')
        pd.DataFrame(r['confusion_matrix_rows_true_columns_pred_A_first_B_first'],
            index=['actual_A_first','actual_B_first'],
            columns=['predicted_A_first','predicted_B_first']).to_csv(out / f'confusion_matrix_{key}.csv')
        for category in ['A_first','B_first','macro avg','weighted avg']:
            entry = r[category]
            rows.append(dict(model=title, category=category, n=r['n'], support=entry['support'],
                precision=entry['precision'], recall=entry['recall'], f1=entry['f1-score'],
                accuracy=r['accuracy'], balanced_accuracy=r['balanced_accuracy'],
                roc_auc_B_first=r['roc_auc_B_first']))
        # Continuous scores, never hard predicted labels, determine ROC.
        fpr, tpr, thresholds = roc_curve(df.true_label, df[col], pos_label=1, drop_intermediate=False)
        pd.DataFrame({'false_positive_rate':fpr, 'true_positive_rate':tpr,
            'score_threshold':thresholds}).to_csv(out / f'roc_points_{key}.csv', index=False)
        label = f"{title} (AUC={r['roc_auc_B_first']:.4f})"
        ax.plot(fpr, tpr, label=label)
        single, sax = plt.subplots(figsize=(6.2, 5.2))
        sax.plot(fpr, tpr, label=label)
        format_roc(sax); single.tight_layout()
        single.savefig(out / f'roc_{key}.png', dpi=200)
        single.savefig(out / f'roc_{key}.pdf'); plt.close(single)
    format_roc(ax); fig.tight_layout()
    fig.savefig(out / 'roc_all_models.png', dpi=200)
    fig.savefig(out / 'roc_all_models.pdf'); plt.close(fig)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'classification_metrics.csv', index=False)
    (out / 'all_metrics.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(summary.to_string(index=False, float_format=lambda x:f'{x:.4f}'))
    print('\nSaved:', out)
    print('Recall A_first/B_first = per-class accuracy. Macro F1 treats both histories equally.')


def format_roc(ax):
    ax.plot([0,1], [0,1], '--', color='gray', label='Chance')
    ax.set(xlim=(0,1), ylim=(0,1.02), xlabel='False positive rate (B_first claim on A_first)',
        ylabel='True positive rate (B_first recall)', title='Held-out ROC: positive class B_first')
    ax.grid(alpha=.2); ax.legend(loc='lower right', fontsize=9)


def self_test():
    import tempfile
    df = pd.DataFrame({'idx':['a','b','c','d'], 'true_label':[0,0,1,1],
        'predicted_label':[0,1,1,1], 'score':[-1,.2,.8,2]})
    r = report_for(df, 'score')
    assert r['accuracy'] == .75 and r['B_first']['recall'] == 1
    assert np.isclose(r['B_first']['precision'], 2/3)
    assert np.isclose(r['B_first']['f1-score'], .8)
    assert r['roc_auc_B_first'] == 1
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp); extra = root/'extra_whole'; extra.mkdir()
        df.rename(columns={'score':'linear_score'}).to_csv(root/'test_predictions_whole.csv', index=False)
        for name in ['svm','mlp']: df.to_csv(extra/f'test_predictions_{name}.csv', index=False)
        args = argparse.Namespace(run_dir=str(root), extra_dir=None, features='whole', out=None)
        evaluate(args)
        assert (root/'metrics_whole/roc_all_models.png').stat().st_size > 0
        assert len(pd.read_csv(root/'metrics_whole/classification_metrics.csv')) == 12
    print('PASS: hand-checked metrics, continuous-score ROC, identical-case checks, exports. Synthetic only.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir'); p.add_argument('--extra-dir'); p.add_argument('--out')
    p.add_argument('--features', choices=['whole','windows','windows_whole','halves4'], default='windows_whole')
    p.add_argument('--self-test', action='store_true'); a = p.parse_args()
    if a.self_test: self_test()
    elif not a.run_dir: p.error('--run-dir required')
    else: evaluate(a)

