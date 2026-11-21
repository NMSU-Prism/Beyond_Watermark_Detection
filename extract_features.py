"""Resumable scoring. Export both detectors' token-aligned block scores and masks."""
import argparse
import json
from pathlib import Path
import math
import pandas as pd
import torch
import engine
from common import feature_row

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--window',type=int,choices=[25],default=25);p.add_argument('--device')
    a=p.parse_args();root=Path(a.run_dir);conf=json.loads((root/'config.json').read_text());cfg=conf['experiment']
    if a.device and torch.device(a.device).type!=torch.device(cfg['device']).type:raise ValueError('Generation/detection RNG device type must match')
    engine.score(cfg,root/'samples.jsonl',root/'scored')
    saved=engine.read(root/'scored/scoring_journal.jsonl');W=math.ceil(cfg['sampling']['max_new_tokens']/25)+1;rows=[]
    for r in saved:
        fr=r['feature_row'];scores={}
        for h,blocks in r['blocks'].items():
            v=[b['z'] for b in blocks];mask=[int(b['effective_n']>0) for b in blocks]
            scores[h]={'window_z':v+[0.]*(W-len(v)),'window_available':mask+[0]*(W-len(mask)),
                'whole_z':fr['features'][h+'_z'],'n_tokens':max((b['end_token_exclusive'] for b in blocks),default=0)}
        row=feature_row(scores['A'],scores['B'],fr['prompt_id'],engine.LABELS[fr['label']],fr['label'])
        row['split']=fr['split'];row['text_hash']=fr['text_hash'];rows.append(row)
    path=root/'features.csv';tmp=root/'features.csv.tmp';pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(path)
    spec={'version':engine.VERSION,'config_hash':engine.fp(cfg),'window':25,'capacity':W,'feature_rows':len(rows)}
    spec['csv_sha256']=engine.th(path.read_text());engine.save(root/'feature_manifest.json',spec)
    print('Saved:',path,'rows:',len(rows),'window slots per detector:',W)

if __name__=='__main__':main()
