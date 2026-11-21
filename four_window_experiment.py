"""Four dynamic sections / four fixed 59-token windows; final-text features only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

SCHEMA = 'four-window-v1'


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def boundaries(length, mode):
    if mode == 'dynamic4':
        q, r = divmod(length, 4)
        ends = [0]
        for i in range(4):
            ends.append(ends[-1] + q + int(i < r))
        return list(zip(ends[:-1], ends[1:]))
    if length > 236:
        raise ValueError('fixed59x4 capacity exceeded: more than 236 final-text tokens')
    return [(min(i * 59, length), min((i + 1) * 59, length)) for i in range(4)]


def zscore(green, n, gamma):
    return (green - gamma * n) / math.sqrt(n * gamma * (1 - gamma)) if n else 0.0


def sections(tokens, traces, ngram_length, gamma, mode):
    result = []
    for start, end in boundaries(len(tokens), mode):
        unique = {}
        for item in traces:
            pos = item['position']
            if start <= pos < end:
                gram = tuple(tokens[pos - ngram_length + 1:pos + 1])
                unique.setdefault(gram, int(item['green']))
        n = len(unique); green = sum(unique.values())
        result.append(dict(start_token=start, end_token_exclusive=end,
            token_count=end-start, effective_n=n, green=green,
            z=zscore(green, n, gamma), available=int(n > 0)))
    return result


def recover(path):
    # Only an interrupted last append may be discarded. Interior corruption fails.
    if not path.exists():
        return []
    raw = path.read_bytes(); lines = raw.splitlines(keepends=True); records = []; end = 0
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if i != len(lines)-1 or line.endswith(b'\n'):
                raise ValueError(f'Corrupt journal at line {i+1}')
            with path.open('r+b') as f:
                f.truncate(end)
            break
        end += len(line)
    if records and raw and not raw.endswith(b'\n') and end == len(raw):
        with path.open('ab') as f:
            f.write(b'\n')
    return records


def score_text(text, tokenizer, detectors, gamma):
    # Same tokenizer and original full-passage watermark context for BOTH modes.
    from common import window_scores, tokenize_for_detection
    outputs = {mode: {} for mode in ('dynamic4', 'fixed59x4')}
    for identity, detector in detectors.items():
        tokens = tokenize_for_detection(detector, text).cpu().tolist()
        if len(tokens) < 4:
            raise ValueError('At least four final-text tokens required')
        if len(tokens) > 236:
            raise ValueError('Final text exceeds fixed59x4 capacity; no truncation performed')
        values = window_scores(detector, text, window=25)
        # window_scores checks whole z against the unmodified upstream detector.
        n = detector.context_width + 1 - int(detector.self_salt)
        for mode in outputs:
            local = sections(tokens, values['traces'], n, gamma, mode)
            outputs[mode][identity] = dict(whole_z=values['whole_z'],
                window_z=[s['z'] for s in local],
                window_available=[s['available'] for s in local],
                n_tokens=len(tokens), sections=local)
    return outputs


def load_detectors(config):
    import torch
    from transformers import AutoTokenizer
    from common import WMConfig, build_detector
    tok = AutoTokenizer.from_pretrained(config['A']['model'],
        revision=config['A']['revision'], use_fast=True)
    detectors = {}
    for identity in ('A', 'B'):
        wm = WMConfig(identity, config[identity]['key'], config['gamma'],
            config['delta'], config['seeding_scheme'])
        detectors[identity] = build_detector(tok, wm, torch.device(config['device']))
    return tok, detectors


def extract(args):
    import fcntl
    import os
    import pandas as pd
    import engine
    from common import feature_row, get_X
    from train_classifier import validate_frame
    source = Path(args.run_dir); out = Path(args.out)
    if source.resolve() == out.resolve():
        raise ValueError('Use a separate --out directory')
    conf = json.loads((source / 'config.json').read_text()); config = conf['experiment']
    rows = [json.loads(l) for l in (source / 'samples.jsonl').read_text().splitlines() if l.strip()]
    engine.validate_samples(rows)
    if not (source / 'COMPLETE.json').exists():
        raise ValueError('Generation has not completed')
    spec = dict(schema=SCHEMA, config_hash=engine.fp(config), samples_hash=engine.fp(rows),
        modes=['dynamic4','fixed59x4'], slots=4,
        classifier_features='A_w1..4,B_w1..4,A_whole,B_whole; masks excluded')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'extract.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = out / 'extraction_manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != spec:
                raise ValueError('Resume mismatch: source text/configuration changed')
        else:
            if (out / 'four_window_journal.jsonl').exists():
                raise ValueError('Journal exists without manifest')
            engine.save(manifest, spec)
        records = recover(out / 'four_window_journal.jsonl')
        done = {r['sample_id']: r for r in records}
        if len(done) != len(records) or not set(done) <= {r['sample_id'] for r in rows}:
            raise ValueError('Unexpected/duplicate scored sample IDs')
        started = time.monotonic(); added = 0; tok = detectors = None
        for row in rows:
            if row['sample_id'] in done:
                continue
            if tok is None:
                tok, detectors = load_detectors(config)
            values = score_text(row['final_text'], tok, detectors, config['gamma'])
            label = engine.LABELS[row['label']] if isinstance(row['label'], str) else int(row['label'])
            record = dict(sample_id=row['sample_id'], idx=str(row['prompt_id']),
                label=label, category={0:'A_first',1:'B_first'}[label], split=row['split'],
                text_hash=engine.th(row['final_text']), values=values)
            with (out / 'four_window_journal.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())
            done[row['sample_id']] = record; added += 1
            if added == 1 or added % 25 == 0 or len(done) == len(rows):
                elapsed = time.monotonic()-started
                progress = dict(completed=len(done), total=len(rows), session_seconds=round(elapsed,2),
                    rough_eta_seconds=round(elapsed/added*(len(rows)-len(done)),2))
                print(json.dumps(progress), flush=True); engine.save(out/'progress.json', progress)
        for mode in ('dynamic4','fixed59x4'):
            dest = out/mode; dest.mkdir(exist_ok=True)
            features = []; detail = []
            for row in rows:
                r = done[row['sample_id']]; scores = r['values'][mode]
                fr = feature_row(scores['A'], scores['B'], r['idx'], r['label'], r['category'])
                fr.update(split=r['split'], text_hash=r['text_hash']); features.append(fr)
                for identity in ('A','B'):
                    for i, s in enumerate(scores[identity]['sections'], 1):
                        detail.append(dict(sample_id=r['sample_id'], detector=identity, window=i, **s))
            df = pd.DataFrame(features); validate_frame(df)
            X, columns = get_X(df, 'windows_whole')
            if X.shape != (len(rows),10):
                raise AssertionError('Expected exactly 10 classifier features')
            # Compatibility with existing train_classifier/train_extra scripts.
            engine.save(dest/'config.json', conf)
            csv = df.to_csv(index=False); (dest/'features.csv').write_text(csv)
            engine.save(dest/'feature_manifest.json', dict(version=engine.VERSION,
                config_hash=engine.fp(config), window=('dynamic4' if mode=='dynamic4' else 59), capacity=4, feature_rows=len(df),
                csv_sha256=engine.th(csv), window_scheme=mode, classifier_columns=columns,
                note='Use four_window_experiment.py predict; original predict_category.py assumes fixed token blocks'))
            engine.save(dest/'window_spec.json', dict(schema=SCHEMA, mode=mode, slots=4,
                fixed_window_tokens=59 if mode=='fixed59x4' else None,
                columns=columns, masks_used_as_classifier_inputs=False,
                context='Full passage context retained; unique ngrams counted within each section'))
            pd.DataFrame(detail).to_csv(dest/'window_details.csv', index=False)
            print(json.dumps(dict(mode=mode, rows=len(df), features=columns, saved=str(dest))), flush=True)
        engine.save(out/'COMPLETE.json', dict(schema=SCHEMA, scored=len(done), session_seconds=time.monotonic()-started))


def predict(args):
    import joblib
    import pandas as pd
    from common import feature_row, get_X
    root = Path(args.case_dir); spec = json.loads((root/'window_spec.json').read_text())
    if spec['schema'] != SCHEMA:
        raise ValueError('Wrong window schema')
    if args.model == 'linear':
        bundle = joblib.load(root/'linreg_windows_whole.joblib')
    else:
        bundle = joblib.load(root/'extra_windows_whole'/f'{args.model}_model.joblib')
    if bundle['columns'] != spec['columns']:
        raise ValueError('Saved feature order mismatch')
    text = Path(args.text_file).read_text()
    tok, detectors = load_detectors(bundle['experiment'])
    values = score_text(text, tok, detectors, bundle['experiment']['gamma'])[spec['mode']]
    X, columns = get_X(pd.DataFrame([feature_row(values['A'], values['B'])]), 'windows_whole')
    if columns != bundle['columns']:
        raise ValueError('Inference feature mismatch')
    model = bundle['model']
    if args.model == 'linear':
        score = float(model.predict(X)[0]); label = int(score >= .5); cutoff = .5
    elif args.model == 'svm':
        score = float(model.decision_function(X)[0]); label = int(model.predict(X)[0]); cutoff = 0.
    else:
        score = float(model.predict_proba(X)[0,1]); label = int(model.predict(X)[0]); cutoff = .5
    result = dict(category={0:'A_first',1:'B_first'}[label], model=args.model,
        window_scheme=spec['mode'], score=score, cutoff=cutoff,
        calibrated_probability=False, exact_provenance_proof=False,
        features=dict(zip(columns, X[0].tolist())))
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


def self_test():
    assert [b-a for a,b in boundaries(30,'dynamic4')] == [8,8,7,7]
    assert [b-a for a,b in boundaries(50,'dynamic4')] == [13,13,12,12]
    assert boundaries(100,'fixed59x4') == [(0,59),(59,100),(100,100),(100,100)]
    try:
        boundaries(237,'fixed59x4')
    except ValueError:
        pass
    else:
        raise AssertionError('Overflow must fail')
    # Context begins before section boundary; duplicate ngrams counted once locally.
    tokens = [1,2,1,2,3,4,5,6]
    traces = [{'position':p,'green':v} for p,v in [(1,1),(2,0),(3,1),(4,1),(5,0),(6,1),(7,1)]]
    s = sections(tokens,traces,2,.25,'dynamic4')
    assert s[1]['effective_n']==2 and s[1]['green']==1
    assert math.isclose(s[1]['z'], (1-.5)/math.sqrt(.375))
    repeated = sections([1,1,1,1],[{'position':1,'green':1},{'position':2,'green':1},{'position':3,'green':1}],2,.25,'fixed59x4')
    assert repeated[0]['effective_n']==1 and repeated[0]['green']==1
    assert repeated[1]['available']==0 and repeated[1]['z']==0
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'journal'; p.write_bytes(b'{"a":1}\n{"broken":')
        assert recover(p)==[{'a':1}]
        assert p.read_bytes()==b'{"a":1}\n'
    print('PASS: dynamic/fixed boundaries, overflow rejection, full-context section evidence, deduplication, padding, journal recovery. No real watermark accuracy claim.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('extract'); a.add_argument('--run-dir',required=True); a.add_argument('--out',required=True)
    a=sub.add_parser('predict'); a.add_argument('--case-dir',required=True); a.add_argument('--model',choices=['linear','svm','mlp'],default='linear'); a.add_argument('--text-file',required=True); a.add_argument('--out',required=True)
    sub.add_parser('self-test'); args=p.parse_args()
    if args.cmd=='extract': extract(args)
    elif args.cmd=='predict': predict(args)
    else: self_test()
