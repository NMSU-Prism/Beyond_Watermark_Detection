"""Predict one whole final passage with any supported feature mode; export JSON."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
import torch
from transformers import AutoTokenizer
from common import WMConfig,build_detector,window_scores,feature_row,get_X,CATEGORIES
import engine


def predict(bundle,text):
    if bundle.get('version')!=engine.VERSION:raise ValueError('Legacy model; re-extract features and retrain')
    if not text.strip():raise ValueError('Empty passage')
    c=bundle['experiment'];tok=AutoTokenizer.from_pretrained(c['A']['model'],revision=c['A']['revision'],use_fast=True)
    scores={}
    for identity,cfg in bundle['watermarks'].items():
        d=build_detector(tok,WMConfig(**cfg),torch.device(c['device']))
        scores[identity]=window_scores(d,text,bundle['window'],bundle['capacity'])
    X,cols=get_X(pd.DataFrame([feature_row(scores['A'],scores['B'])]),bundle['features'])
    if cols!=bundle['columns']:raise ValueError('Training/prediction feature schema mismatch')
    value=float(bundle['model'].predict(X)[0]);label=int(value>=.5)
    return {'category':CATEGORIES[label],'label':label,'linear_score_not_probability':value,'threshold':.5,
        'detector_scores':scores,'scope':'Only A->B or B->A, both stages watermarked','exact_provenance_proof':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--features',default='windows_whole',choices=['windows_whole','windows','whole','halves4'])
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--text');group.add_argument('--text-file')
    p.add_argument('--out');a=p.parse_args();root=Path(a.run_dir)
    bundle=joblib.load(root/f'linreg_{a.features}.joblib') # Trusted files only.
    text=a.text if a.text is not None else Path(a.text_file).read_text(encoding='utf-8')
    result=predict(bundle,text);out=Path(a.out) if a.out else root/f'prediction_{a.features}.json'
    engine.save(out,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
