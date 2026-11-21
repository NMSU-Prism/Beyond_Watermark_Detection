"""Reuse order_detect feature tables: RBF SVM and small CPU neural network."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model(kind, seed):
    if kind == 'svm':
        estimator = SVC(C=1.0, kernel='rbf', gamma='scale', probability=False)
    else:
        estimator = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',
            solver='adam', alpha=0.01, batch_size='auto', learning_rate_init=0.001,
            max_iter=500, early_stopping=False, random_state=seed)
    return make_pipeline(StandardScaler(), estimator)


def scores(model, X, kind):
    return model.decision_function(X) if kind == 'svm' else model.predict_proba(X)[:, 1]


def evaluate(model, X, y, kind):
    pred = model.predict(X)
    s = scores(model, X, kind)
    return dict(n=len(y), accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        auc=float(roc_auc_score(y, s)) if len(set(y)) == 2 else None,
        confusion_matrix_rows_true_columns_pred_A_first_B_first=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def train(args):
    # Import existing package only when processing its real saved feature table.
    from common import get_X
    from train_classifier import validate_frame
    import engine
    root = Path(args.run_dir)
    out = Path(args.out) if args.out else root / ('extra_' + args.features)
    if out.exists():
        raise FileExistsError(f'Existing results preserved: {out}; choose a NEW --out')
    config = json.loads((root / 'config.json').read_text())
    manifest = json.loads((root / 'feature_manifest.json').read_text())
    raw = (root / 'features.csv').read_text()
    if manifest['version'] != engine.VERSION or manifest['config_hash'] != engine.fp(config['experiment']):
        raise ValueError('Feature/config mismatch')
    if manifest['csv_sha256'] != engine.th(raw):
        raise ValueError('Feature checksum mismatch')
    df = pd.read_csv(root / 'features.csv', dtype={'idx': str, 'text_hash': str})
    validate_frame(df)
    if len(df) != manifest['feature_rows']:
        raise ValueError('Incomplete feature table')
    X, columns = get_X(df, args.features)
    y = df['label'].to_numpy(dtype=int)
    splits = {name: np.flatnonzero(df['split'].eq(name)) for name in ('train','validation','test')}
    out.mkdir(parents=True)
    report = {'feature_mode': args.features, 'columns': columns, 'seed': args.seed,
        'features_sha256': hashlib.sha256(raw.encode()).hexdigest(), 'models': {},
        'scope': 'Closed world A_first=0/B_first=1. Fixed starting settings, no parameter selection or confidence calibration. Same stored prompt splits; test is evaluation only.'}
    for kind in ('svm', 'mlp'):
        started = time.monotonic()
        model = build_model(kind, args.seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(X[splits['train']], y[splits['train']])
        messages = [str(w.message) for w in caught]
        result = {'splits': {}, 'warnings': messages, 'parameters': model[-1].get_params(),
            'decision_cutoff': 0.0 if kind == 'svm' else 0.5,
            'score_type': 'uncalibrated SVM decision margin' if kind == 'svm' else 'uncalibrated MLP probability estimate'}
        for name, ix in splits.items():
            result['splits'][name] = evaluate(model, X[ix], y[ix], kind)
            result['splits'][name]['prompt_groups'] = int(df.iloc[ix]['idx'].nunique())
            baseline = (df.iloc[ix]['A_whole'].to_numpy() >= df.iloc[ix]['B_whole'].to_numpy()).astype(int)
            result['splits'][name]['stronger_final_z_baseline_accuracy'] = float(np.mean(baseline == y[ix]))
        result['elapsed_seconds'] = time.monotonic() - started
        if kind == 'mlp':
            result['iterations'] = int(model[-1].n_iter_)
            pd.DataFrame({'epoch': np.arange(1, len(model[-1].loss_curve_) + 1), 'training_loss': model[-1].loss_curve_}).to_csv(out / 'mlp_loss.csv', index=False)
        joblib.dump({'model': model, 'kind': kind, 'features': args.features, 'columns': columns,
            'experiment': config['experiment'], 'watermarks': config['watermarks'],
            'capacity': manifest['capacity'], 'window': manifest['window'],
            'labels': {0:'A_first',1:'B_first'}}, out / f'{kind}_model.joblib')
        ix = splits['test']
        pd.DataFrame({'idx': df.iloc[ix]['idx'].to_numpy(), 'true_label': y[ix],
            'predicted_label': model.predict(X[ix]), 'score': scores(model, X[ix], kind)}).to_csv(out / f'test_predictions_{kind}.csv', index=False)
        report['models'][kind] = result
        (out / f'results_{kind}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'classifier': kind, 'elapsed_seconds': result['elapsed_seconds'], 'test': result['splits']['test'], 'warnings': messages}), flush=True)
    (out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Saved:', out)


def self_test():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(160, 6)); y = (X[:, 0] > 0).astype(int)
    for kind in ('svm', 'mlp'):
        model = build_model(kind, 42).fit(X[:120], y[:120])
        assert np.allclose(model[0].mean_, X[:120].mean(axis=0))
        assert set(model.predict(X[120:])) <= {0, 1}
        assert evaluate(model, X[120:], y[120:], kind)['accuracy'] > 0.7
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'model.joblib'; joblib.dump(model, p)
            assert np.array_equal(model.predict(X[120:]), joblib.load(p).predict(X[120:]))
    print('PASS: synthetic SVM/MLP fit, train-only scaling, prediction, serialization. Not a watermark accuracy test.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir'); p.add_argument('--out')
    p.add_argument('--features', choices=['whole','windows','windows_whole','halves4'], default='windows_whole')
    p.add_argument('--seed', type=int, default=42); p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test: self_test()
    elif not a.run_dir: p.error('--run-dir is required')
    else: train(a)

