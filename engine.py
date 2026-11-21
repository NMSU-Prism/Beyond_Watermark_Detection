#!/usr/bin/env python3
"""Two-order GPT-2 watermark experiment; four scores and ordinary linear regression."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,roc_auc_score,mean_squared_error

VERSION='order-detect-fixed-v1'
FEATURES=['A_z','B_z','A_mean_block25_z','B_mean_block25_z']
LABELS={'A_first':0,'B_first':1}
REWRITE_EXAMPLES=(
    'Original: The rain stopped and the sun appeared.\nParaphrase: After the rain ended, the sun came out.\n\n'
    'Original: Plants need sunlight and water to grow.\nParaphrase: Water and sunlight are necessary for plant growth.\n\n')

def read(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def fp(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def th(text):return hashlib.sha256(text.encode()).hexdigest()
def seed(pid,stage):return int(th(pid+'|'+stage)[:8],16)

def init(path,device):
    if Path(path).exists():raise FileExistsError('Use a new configuration path')
    from huggingface_hub import HfApi
    revision=HfApi().model_info('openai-community/gpt2').sha
    c={'A':{'model':'openai-community/gpt2','revision':revision,'key':2024},
       'B':{'model':'openai-community/gpt2','revision':revision,'key':2023},
       'device':device,'dtype':'float32','gamma':.25,'delta':2.,'seeding_scheme':'selfhash',
       'windows':[25],'block_size':25,'prompt_max_tokens':128,'rewrite_source_max_tokens':300,
       'sampling':{'temperature':.7,'top_p':.95,'top_k':40,'max_new_tokens':250,'min_new_tokens':50}}
    save(path,c);print('Pinned GPT-2 revision:',revision)

def prepare(count,out,input_path=None,exclude_path=None):
    if count<5:raise ValueError('Use at least 5 source prompts; recommended smoke count=10')
    if Path(out).exists():raise FileExistsError('Preserve previous prompts')
    if input_path:source=iter(read(input_path))
    else:
        from datasets import load_dataset
        url='https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz'
        source=iter(load_dataset('json',data_files={'train':url},split='train',streaming=True).shuffle(seed=42,buffer_size=10000))
    selected=[];seen=set();docs=set()
    if exclude_path:
        for r in read(exclude_path):
            docs.add(str(r['prompt_id']));seen.add(th(' '.join(str(r['text']).split())[:800]))
    for r in source:
        text=' '.join(str(r['text']).split())
        if len(text)<200:continue
        prefix=text[:800];pid=str(r.get('prompt_id') or th(text))
        if th(prefix) in seen or pid in docs:continue
        seen.add(th(prefix));docs.add(pid);selected.append({'prompt_id':pid,'text':prefix,'source':'local JSONL' if input_path else 'allenai/c4 English shard 00000'})
        if len(selected)==count:break
    if len(selected)!=count:raise ValueError('Insufficient unique prompts')
    order=np.random.default_rng(42).permutation(count);train_end=int(.7*count);val_end=train_end+max(1,int(.15*count))
    for rank,i in enumerate(order):selected[i]['split']='train' if rank<train_end else 'validation' if rank<val_end else 'test'
    dest=Path(out);dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.name+'.tmp')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected));tmp.replace(dest)
    print(json.dumps({'source_prompts':count,'final_passages':2*count,'final_per_category':count,
                      'prompt_splits':dict(Counter(r['split'] for r in selected))}))

SUPPORTED_MODELS={'openai-community/gpt2','meta-llama/Llama-3.2-1B','meta-llama/Llama-3.2-1B-Instruct'}
CHAT_INSTRUCTIONS={
    'first':'Write one clear self-contained paragraph based on the passage below. Preserve its stated information and do not add new facts. Return only the paragraph.\n\n',
    'second':'Paraphrase the passage below using different wording while preserving its meaning, names, numbers and factual details. Do not add information or commentary. Return only the rewritten passage.\n\n'}

def check_config(c):
    if c['A']['model'] not in SUPPORTED_MODELS or c['A']['model']!=c['B']['model'] or c['A']['revision']!=c['B']['revision']:
        raise ValueError('Use a supported pinned model shared by both watermark identities')
    if c['A']['key']==c['B']['key']:raise ValueError('Keys must differ')
    if c['block_size']!=25:raise ValueError('This experiment uses 25-token blocks')
    if c.get('prompt_style','completion') not in ('completion','chat'):raise ValueError('Invalid prompt style')

def context_limit(model,tokenizer):
    for obj,name in ((model.config,'max_position_embeddings'),(model.config,'n_positions'),(tokenizer,'model_max_length')):
        value=getattr(obj,name,None)
        if isinstance(value,int) and 0<value<10**9:return value
    raise ValueError('Unable to determine model context limit')

def encode_prompt(tokenizer,text,c,stage):
    if c.get('prompt_style','completion')=='chat':
        if not getattr(tokenizer,'chat_template',None):raise ValueError('Chat mode requires the model tokenizer chat template')
        instruction=c.get('chat_instructions',CHAT_INSTRUCTIONS)[stage]
        messages=[{'role':'system','content':'You rewrite text accurately and concisely.'},
                  {'role':'user','content':instruction+text}]
        return tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt')
    return tokenizer(text,return_tensors='pt').input_ids

def stopping_ids(model,tokenizer):
    base=getattr(getattr(model,'generation_config',None),'eos_token_id',None)
    values=list(base) if isinstance(base,(list,tuple)) else [base] if base is not None else []
    eos=getattr(tokenizer,'eos_token_id',None)
    if isinstance(eos,int):values.append(eos)
    vocab=tokenizer.get_vocab()
    if '<|eot_id|>' in vocab:values.append(int(vocab['<|eot_id|>']))
    return list(dict.fromkeys(int(v) for v in values))

def append(path,row):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def recover(path):
    path=Path(path)
    if not path.exists():return []
    data=path.read_bytes()
    if data and not data.endswith(b'\n'):
        backup=path.with_name(path.name+'.tail-backup-'+str(time.time_ns()))
        backup.write_bytes(data);path.write_bytes(data[:data.rfind(b'\n')+1])
    rows=read(path)
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate IDs in journal: '+str(path))
    return rows

class Progress:
    def __init__(self,root,stage,total,done):
        self.root=Path(root);self.stage=stage;self.total=total;self.done=done;self.session_done=0;self.start=time.monotonic()
        self.path=self.root/(stage+'_timing.json')
        self.base=json.loads(self.path.read_text()).get('total_active_seconds',0.) if self.path.exists() else 0.
    def update(self,done,**extra):
        self.done=done;self.session_done+=1;elapsed=time.monotonic()-self.start
        rate=self.session_done/elapsed if elapsed else 0.
        report={'stage':self.stage,'completed':done,'total':self.total,'session_elapsed_seconds':round(elapsed,2),
            'total_active_seconds':round(self.base+elapsed,2),'estimated_remaining_seconds':round((self.total-done)/rate,2) if rate else None,
            'estimated_finish_unix':time.time()+(self.total-done)/rate if rate else None,**extra}
        save(self.path,report);append(self.root/(self.stage+'_progress.jsonl'),report)
        print(json.dumps(report),flush=True)
        return report

def generate(c,prompts_path,out):
    import fcntl
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    with (root/'generation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _generate(c,prompts_path,root)

def _generate(c,prompts_path,root):
    import torch
    import transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM,LogitsProcessorList,set_seed
    from packaging.version import Version
    from wm.extended_watermark_processor import WatermarkLogitsProcessor
    check_config(c);prompts=read(prompts_path)
    if len({r['prompt_id'] for r in prompts})!=len(prompts) or len({r['text'] for r in prompts})!=len(prompts):raise ValueError('Duplicate source prompts')
    if any(r['split'] not in ('train','validation','test') for r in prompts):raise ValueError('Invalid split')
    manifest={'version':VERSION,'config':c,'prompt_hash':fp(prompts),'expected_per_category':len(prompts),'rewrite_examples':REWRITE_EXAMPLES}
    mp=root/'manifest.json'
    if mp.exists():
        if json.loads(mp.read_text())!=manifest:raise ValueError('Resume manifest mismatch; use original script or a NEW output directory')
    else:
        if (root/'samples.jsonl').exists() or (root/'first_stage.jsonl').exists():raise ValueError('Existing journals lack manifest')
        save(mp,manifest)
    if not (root/'prompts_used.jsonl').exists():
        (root/'prompts_used.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in prompts))
    import platform
    from importlib.metadata import version,PackageNotFoundError
    packages={}
    for package in ('torch','transformers','numpy','scikit-learn','joblib','datasets','huggingface_hub'):
        try:packages[package]=version(package)
        except PackageNotFoundError:packages[package]='not installed'
    save(root/'environment.json',{'python':platform.python_version(),'architecture':platform.machine(),'packages':packages})
    journal=root/'samples.jsonl';bank=root/'first_stage.jsonl'
    rows=recover(journal);done={r['sample_id']:r for r in rows};cached={r['sample_id']:r for r in recover(bank)}
    expected={p['prompt_id']+'_'+h for p in prompts for h in LABELS}
    if not set(done)<=expected or not set(cached)<=expected:raise ValueError('Unexpected sample IDs')
    if set(done)==expected:
        save(root/'COMPLETE.json',{'samples':len(done),'manifest_hash':fp(manifest)})
        print(json.dumps({'complete':True,'samples':len(done),'path':str(journal),'generation_skipped':True}));return
    progress=Progress(root,'generation',4*len(prompts),len(cached)+len(done))
    load_start=time.monotonic()
    tok=AutoTokenizer.from_pretrained(c['A']['model'],revision=c['A']['revision']);tok.pad_token=tok.eos_token
    arg='dtype' if Version(transformers.__version__)>=Version('4.56') else 'torch_dtype'
    model=AutoModelForCausalLM.from_pretrained(c['A']['model'],revision=c['A']['revision'],**{arg:getattr(torch,c['dtype'])}).to(c['device']).eval()
    append(root/'model_load_times.jsonl',{'seconds':time.monotonic()-load_start,'time_unix':time.time()})
    def call(text,identity,s,stage):
        set_seed(s);ids=encode_prompt(tok,text,c,stage).to(c['device'])
        if ids.shape[1]+c['sampling']['max_new_tokens']>context_limit(model,tok):raise ValueError('Model context budget exceeded')
        proc=WatermarkLogitsProcessor(vocab=list(tok.get_vocab().values()),gamma=c['gamma'],delta=c['delta'],
              seeding_scheme=c['seeding_scheme'],hash_key=c[identity]['key'])
        kw=dict(c['sampling'],do_sample=True,pad_token_id=tok.pad_token_id,attention_mask=torch.ones_like(ids),logits_processor=LogitsProcessorList([proc]))
        stops=stopping_ids(model,tok)
        if stops:kw['eos_token_id']=stops
        t=time.monotonic()
        with torch.inference_mode():v=model.generate(ids,**kw)
        text=tok.decode(v[0,ids.shape[1]:],skip_special_tokens=True).strip()
        if not text:raise ValueError('Empty generated passage; inspect before retrying')
        return text,time.monotonic()-t
    for p in prompts:
        for history in LABELS:
            sid=p['prompt_id']+'_'+history
            if sid in done:continue
            first='A' if history=='A_first' else 'B';second='B' if first=='A' else 'A'
            if sid not in cached:
                prompt=tok.decode(tok(p['text'],add_special_tokens=False)['input_ids'][:c['prompt_max_tokens']])
                intermediate,seconds=call(prompt,first,seed(str(c.get('seed',0))+'|'+p['prompt_id'],'first'),'first')
                record={'sample_id':sid,'prompt_id':p['prompt_id'],'label':history,'split':p['split'],
                    'first_identity':first,'intermediate_text':intermediate,'generation_seconds':seconds}
                append(bank,record);cached[sid]=record
                progress.update(len(cached)+len(done),sample_id=sid,operation='first_stage',call_seconds=seconds)
            intermediate=cached[sid]['intermediate_text']
            source=tok.decode(tok(intermediate,add_special_tokens=False)['input_ids'][:c['rewrite_source_max_tokens']])
            rewrite_input=source if c.get('prompt_style','completion')=='chat' else REWRITE_EXAMPLES+'Original: '+source+'\nParaphrase:'
            final,seconds=call(rewrite_input,second,seed(str(c.get('seed',0))+'|'+p['prompt_id'],'second'),'second')
            row={'sample_id':sid,'prompt_id':p['prompt_id'],'split':p['split'],'label':history,
                 'first_identity':first,'second_identity':second,'intermediate_text':intermediate,'rewrite_source':source,
                 'final_text':final,'first_seed':seed(str(c.get('seed',0))+'|'+p['prompt_id'],'first'),'second_seed':seed(str(c.get('seed',0))+'|'+p['prompt_id'],'second'),
                 'first_generation_seconds':cached[sid]['generation_seconds'],'second_generation_seconds':seconds}
            append(journal,row);done[sid]=row
            progress.update(len(cached)+len(done),sample_id=sid,operation='second_stage',final_passages_completed=len(done),call_seconds=seconds)
    save(root/'COMPLETE.json',{'samples':len(done),'manifest_hash':fp(manifest)})


def blocks(tokens,trace,n,gamma):
    from common import _z as zscore
    result=[]
    for start in range(0,len(tokens),25):
        end=min(start+25,len(tokens));seen=set();green=0
        for r in trace:
            pos=r['position']
            if start<=pos<end:
                gram=tuple(tokens[pos-n+1:pos+1])
                if gram not in seen:seen.add(gram);green+=int(r['green'])
        result.append({'start_token':start,'end_token_exclusive':end,'full_25_token_block':end-start==25,
                       'green':green,'effective_n':len(seen),'z':zscore(green,len(seen),gamma)})
    return result

def regression_names(c,mode):
    if mode=='four':return FEATURES
    count=math.ceil(c['sampling']['max_new_tokens']/25)+1
    return ['A_z','B_z']+[f'{h}_block25_{i:02d}_{k}' for h in ('A','B') for i in range(count) for k in ('z','available')]

def extract(scorer,text,c):
    aggregate,traces=scorer.score(text);features={i+'_z':aggregate[i+'_z'] for i in ('A','B')};local={}
    for identity,(tok,det) in scorer.items.items():
        ids=tok(text,add_special_tokens=False)['input_ids']
        if ids and ids[0]==tok.bos_token_id:ids=ids[1:]
        n=det.context_width+1-int(det.self_salt);local[identity]=blocks(ids,traces[identity],n,c['gamma'])
        full=[w['z'] for w in local[identity] if w['full_25_token_block'] and w['effective_n']>0]
        if not full:raise ValueError('Passage has no complete scored 25-token block; inspect generation')
        features[identity+'_mean_block25_z']=float(np.mean(full))
        count=math.ceil(c['sampling']['max_new_tokens']/25)+1
        if len(local[identity])>count:raise ValueError('More blocks than fixed feature capacity; choose a reviewed configuration')
        for i in range(count):
            available=i<len(local[identity]) and local[identity][i]['effective_n']>0
            features[f'{identity}_block25_{i:02d}_z']=float(local[identity][i]['z']) if available else 0.
            features[f'{identity}_block25_{i:02d}_available']=float(available)
    return features,local,traces

def validate_samples(raw):
    ids=set();groups={};splits={}
    for r in raw:
        if r['sample_id'] in ids:raise ValueError('Duplicate sample ID')
        ids.add(r['sample_id']);pid=r['prompt_id'];split=r['split']
        if split not in ('train','validation','test'):raise ValueError('Invalid split')
        if pid in splits and splits[pid]!=split:raise ValueError('Prompt split leakage')
        splits[pid]=split;groups.setdefault(pid,[]).append(r['label'])
    if not groups or any(sorted(v)!=sorted(LABELS) for v in groups.values()):raise ValueError('Each prompt must have exactly both order labels')

def score(c,samples,out):
    import fcntl
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    with (root/'scoring.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _score(c,samples,root)

def _score(c,samples,out):
    from detector_adapter import Scorer
    check_config(c);root=Path(samples).parent;manifest=json.loads((root/'manifest.json').read_text())
    if manifest['config']!=c or not (root/'COMPLETE.json').exists():raise ValueError('Incomplete or incompatible generation')
    raw=read(samples);validate_samples(raw)
    spec={'version':VERSION,'config_hash':fp(c),'samples_hash':fp(raw)};mp=out/'scoring_manifest.json'
    if mp.exists():
        if json.loads(mp.read_text())!=spec:raise ValueError('Scoring resume mismatch')
    else:
        if (out/'scoring_journal.jsonl').exists():raise ValueError('Scoring journal without manifest')
        save(mp,spec)
    journal=out/'scoring_journal.jsonl';done={r['sample_id']:r for r in recover(journal)}
    if not set(done)<={r['sample_id'] for r in raw}:raise ValueError('Unexpected scored IDs')
    progress=Progress(out,'scoring',len(raw),len(done));scorer=None
    for r in raw:
        if r['sample_id'] in done:continue
        if scorer is None:scorer=Scorer(c)
        t=time.monotonic();features,local,traces=extract(scorer,r['final_text'],c)
        row={k:r[k] for k in ('sample_id','prompt_id','split','label')}
        row.update(features=features,config_hash=fp(c),text_hash=th(r['final_text']))
        record={'sample_id':r['sample_id'],'feature_row':row,'blocks':local,'traces':traces}
        append(journal,record);done[r['sample_id']]=record
        progress.update(len(done),sample_id=r['sample_id'],scoring_seconds=time.monotonic()-t)
    # Rebuild exports from the durable journal; no detector rescoring on resume.
    import csv
    handles={name:(out/(name+'.tmp')).open('w',encoding='utf-8',newline='') for name in ('features.jsonl','block25_scores.jsonl','token_traces.jsonl','four_scores.csv','all_scores.csv','block25_scores.csv')}
    four=csv.DictWriter(handles['four_scores.csv'],fieldnames=['sample_id','prompt_id','split','label']+FEATURES);four.writeheader()
    all_names=FEATURES+[n for n in regression_names(c,'blocks') if n not in FEATURES]
    all_writer=csv.DictWriter(handles['all_scores.csv'],fieldnames=['sample_id','prompt_id','split','label']+all_names);all_writer.writeheader()
    bw=csv.DictWriter(handles['block25_scores.csv'],fieldnames=['sample_id','label','detector','start_token','end_token_exclusive','full_25_token_block','green','effective_n','z']);bw.writeheader()
    try:
        for source in raw:
            r=done[source['sample_id']];row=r['feature_row'];sid=r['sample_id']
            handles['features.jsonl'].write(json.dumps(row)+'\n')
            handles['block25_scores.jsonl'].write(json.dumps({'sample_id':sid,'blocks':r['blocks']})+'\n')
            handles['token_traces.jsonl'].write(json.dumps({'sample_id':sid,'traces':r['traces']})+'\n')
            meta={k:row[k] for k in ('sample_id','prompt_id','split','label')}
            four.writerow({**meta,**{k:row['features'][k] for k in FEATURES}})
            all_writer.writerow({**meta,**{k:row['features'][k] for k in all_names}})
            for h,items in r['blocks'].items():
                for item in items:bw.writerow({'sample_id':sid,'label':row['label'],'detector':h,**item})
    finally:
        for f in handles.values():f.close()
    for name in handles:(out/(name+'.tmp')).replace(out/name)
    save(out/'COMPLETE.json',{'scored':len(raw),'config_hash':fp(c)})

def matrix(rows,names=FEATURES):
    x=np.asarray([[r['features'][k] for k in names] for r in rows],float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite features')
    return x

def metrics(rows,p):
    y=np.array([LABELS[r['label']] for r in rows]);pred=(p>=.5).astype(int)
    return {'n':len(rows),'accuracy':float(accuracy_score(y,pred)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'confusion_matrix_rows_true_columns_pred_A_first_B_first':confusion_matrix(y,pred,labels=[0,1]).tolist(),
        'auroc_B_first':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        'mse':float(mean_squared_error(y,p)),'raw_regression_range':[float(np.min(p)),float(np.max(p))]}

def train(c,path,out,feature_mode="four"):
    started=time.monotonic();check_config(c);rows=read(path);validate_samples(rows);names=regression_names(c,feature_mode)
    marker=json.loads((Path(path).parent/'COMPLETE.json').read_text())
    if marker.get('scored')!=len(rows) or any(r['config_hash']!=fp(c) for r in rows):raise ValueError('Incomplete/mismatched scoring')
    # Reject identical final passages crossing split boundaries.
    seen={}
    for r in rows:
        h=r['text_hash']
        if h in seen and seen[h]!=r['split']:raise ValueError('Identical text across splits')
        seen[h]=r['split']
    parts={s:[r for r in rows if r['split']==s] for s in ('train','validation','test')}
    if any({r['label'] for r in rs}!=set(LABELS) for rs in parts.values()):raise ValueError('Each split must contain both labels')
    model=make_pipeline(StandardScaler(),LinearRegression())
    model.fit(matrix(parts['train'],names),[LABELS[r['label']] for r in parts['train']])
    report={'version':VERSION,'task':'CLOSED WORLD: input is guaranteed to have exactly both watermark stages in one of two orders',
        'features':names,'feature_mode':feature_mode,'numeric_labels':LABELS,'decision_threshold':.5,
        'threshold_origin':'Fixed midpoint of regression target labels 0 and 1; not false-claim calibration',
        'prompt_groups':{s:len({r['prompt_id'] for r in rs}) for s,rs in parts.items()},'splits':{}}
    for split,rs in parts.items():
        p=model.predict(matrix(rs,names));report['splits'][split]=metrics(rs,p)
        baseline=np.array([0 if r['features']['B_z']>r['features']['A_z'] else 1 for r in rs])
        report['splits'][split]['stronger_final_z_rule_accuracy']=float(np.mean(baseline==[LABELS[r['label']] for r in rs]))
    report['caveat']='Ordinary linear regression score is not a probability and can fall outside [0,1]. No unknown class or false-claim calibration. Cannot establish earlier watermarking against single-watermark alternatives. GPT-2 paraphrase fidelity must be inspected.'
    out=Path(out)
    if out.exists():raise FileExistsError('Choose a new model directory')
    out.mkdir(parents=True);joblib.dump({'version':VERSION,'model':model,'config':c,'features':names},out/'linear_order.joblib');save(out/'metrics.json',report)
    with (out/'test_predictions.jsonl').open('w') as f:
        for r,p in zip(parts['test'],model.predict(matrix(parts['test'],names))):
            f.write(json.dumps({'sample_id':r['sample_id'],'true_label':r['label'],'prediction':'B_first' if p>=.5 else 'A_first','linear_score':float(p)})+'\n')
    report['training_evaluation_elapsed_seconds']=time.monotonic()-started;save(out/'metrics.json',report)
    print(json.dumps(report,indent=2));return model,report

def predict(path,textfile,out):
    from detector_adapter import Scorer
    b=joblib.load(path) # Trusted files only.
    if b.get('version')!=VERSION:raise ValueError('Wrong model version')
    text=Path(textfile).read_text(encoding='utf-8');features,local,_=extract(Scorer(b['config']),text,b['config'])
    p=float(b['model'].predict(np.array([[features[k] for k in b['features']]]))[0])
    result={'prediction':'B_first' if p>=.5 else 'A_first','linear_score_not_probability':p,'threshold':.5,
        'features':features,'block25_scores':local,'scope':'Only A->B or B->A, both stages watermarked; not authenticated history','exact_provenance_proof':False}
    save(out,result);print(json.dumps(result,indent=2));return result

def estimate(root,target):
    root=Path(root);first=read(root/'first_stage.jsonl');final=read(root/'samples.jsonl')
    a=[float(r['generation_seconds']) for r in first];b=[float(r['second_generation_seconds']) for r in final]
    if not a or not b:raise ValueError('Need completed first and second stages from a smoke run')
    seconds=2*target*(float(np.mean(a))+float(np.mean(b)))
    result={'target_source_prompts':target,'target_final_passages':2*target,
        'mean_first_stage_seconds':float(np.mean(a)),'mean_second_stage_seconds':float(np.mean(b)),
        'estimated_generation_seconds':seconds,'estimated_generation_hours':seconds/3600,
        'caveat':'Rough extrapolation from completed smoke calls on same hardware/settings. Excludes initial downloads, model loading, scoring and training; not a deadline guarantee.'}
    print(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('init');q.add_argument('--out',required=True);q.add_argument('--device',default='cuda:0')
    q=sub.add_parser('prompts');q.add_argument('--count',type=int,default=1000);q.add_argument('--input');q.add_argument('--exclude');q.add_argument('--out',required=True)
    for cmd in ('generate','score','train'):
        q=sub.add_parser(cmd);q.add_argument('--config',required=True);q.add_argument('--out',required=True)
        q.add_argument('--prompts' if cmd=='generate' else '--input',required=True)
        if cmd=='train':q.add_argument('--feature-mode',choices=['blocks','four'],default='blocks')
    q=sub.add_parser('estimate');q.add_argument('--smoke-run',required=True);q.add_argument('--target-prompts',type=int,default=1000)
    q=sub.add_parser('predict');q.add_argument('--model',required=True);q.add_argument('--text-file',required=True);q.add_argument('--out',required=True)
    a=p.parse_args()
    if a.cmd=='estimate':estimate(a.smoke_run,a.target_prompts)
    elif a.cmd=='init':init(a.out,a.device)
    elif a.cmd=='prompts':prepare(a.count,a.out,a.input,a.exclude)
    elif a.cmd=='predict':predict(a.model,a.text_file,a.out)
    else:
        c=json.loads(Path(a.config).read_text())
        if a.cmd=='generate':generate(c,a.prompts,a.out)
        elif a.cmd=='score':score(c,a.input,a.out)
        else:train(c,a.input,a.out,a.feature_mode)

if __name__=='__main__':main()
