"""Collision KGW/PRW/SIR backends for final-text order classification."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import time
import fcntl
from method_backends import Backend,UPSTREAM_COMMIT,file_sha,feature_scores
VERSION='collision-order-v1'
LABELS={'A_first':0,'B_first':1}


def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def save(p,r):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def append(p,r):
    with Path(p).open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def assets(a):
    import urllib.request
    from huggingface_hub import HfApi
    from transformers import AutoConfig
    import torch
    from collision_vendor.sir.train_watermark_model import TransformModel
    root=Path(a.out).resolve()
    if root.exists():raise FileExistsError('Use a new asset directory')
    root.mkdir(parents=True)
    target=root/'transform_model_cbert.pth'
    if a.checkpoint:
        import shutil
        shutil.copyfile(a.checkpoint,target)
    else:
        url='https://raw.githubusercontent.com/THU-BPM/Robust_Watermark/main/model/transform_model_cbert.pth'
        print('Downloading official SIR transformation checkpoint:',url,flush=True)
        urllib.request.urlretrieve(url,target)
    wm=TransformModel();wm.load_state_dict(torch.load(target,map_location='cpu',weights_only=True))
    base=json.loads((Path(a.base_run)/'config.json').read_text())['experiment']['A']
    model_config=AutoConfig.from_pretrained(base['model'],revision=base['revision'])
    vocab=model_config.vocab_size
    rng=random.Random(a.mapping_seed)
    save(root/'mapping.json',[rng.randint(0,299) for _ in range(vocab)])
    embedding='perceptiveshawty/compositional-bert-large-uncased'
    revision=HfApi().model_info(embedding).sha
    save(root/'sir_assets.json',dict(embedding_model=embedding,embedding_revision=revision,
        target_model=base['model'],target_revision=base['revision'],vocab_size=vocab,
        mapping_seed=a.mapping_seed,files={p.name:file_sha(p) for p in [target,root/'mapping.json']},
        note='Persist this mapping for BOTH generation and detection. SIR context has no per-user watermark key.'))
    print('SIR assets ready:',root)


def prepare(a):
    from transformers import AutoConfig
    import engine
    root=Path(a.out)
    if root.exists():raise FileExistsError('Use a new output directory')
    if a.a=='SIR' and a.b=='SIR':raise ValueError('Paper excludes SIR-SIR: context mode has no distinct keyed identities')
    base=Path(a.base_run);old=json.loads((base/'config.json').read_text());c=copy.deepcopy(old['experiment'])
    prompt_file=base/'prompts.jsonl'
    if not prompt_file.exists():prompt_file=base/'prompts_used.jsonl'
    prompts=read(prompt_file)
    if a.count:
        if a.count<5:raise ValueError('At least5sourceprompts')
        if a.count>len(prompts):raise ValueError('Not enough existing source prompts')
        # Preserve ALL source-group split assignments. Stratified deterministic subset.
        buckets={s:[p for p in prompts if p['split']==s] for s in ['train','validation','test']}
        take={'train':int(.7*a.count),'validation':max(1,int(.15*a.count))}
        take['test']=a.count-take['train']-take['validation']
        if any(take[s]>len(buckets[s]) for s in take):raise ValueError('Insufficient groups in existing splits')
        prompts=[p for s in take for p in buckets[s][:take[s]]]
    if a.min_new_tokens is not None:c['sampling']['min_new_tokens']=a.min_new_tokens
    if a.max_new_tokens is not None:
        c['sampling']['max_new_tokens']=a.max_new_tokens
        c['rewrite_source_max_tokens']=a.max_new_tokens
    if not 4<=c['sampling']['min_new_tokens']<=c['sampling']['max_new_tokens']:raise ValueError('Invalid min/max tokens')
    c['prompt_style']='chat';c['scheme_extension']=VERSION
    c['vocab_size']=AutoConfig.from_pretrained(c['A']['model'],revision=c['A']['revision']).vocab_size
    c['methods']={};sir_asset=None
    if 'SIR' in [a.a,a.b]:
        if not a.sir_assets:raise ValueError('--sir-assets required')
        sir_path=Path(a.sir_assets).resolve();sir_asset=json.loads((sir_path/'sir_assets.json').read_text())
        if sir_asset['vocab_size']!=c['vocab_size'] or sir_asset['target_revision']!=c['A']['revision']:raise ValueError('SIR assets incompatible with base run')
    for identity,kind in [('A',a.a),('B',a.b)]:
        spec=dict(identity=identity,method=kind,key=c[identity]['key'] if kind!='SIR' else None,
            gamma=.5 if kind=='SIR' else (a.prw_fraction if kind=='PRW' else .25),
            delta=((a.strength_a if a.strength_a is not None else a.strength) if identity=='A' else (a.strength_b if a.strength_b is not None else a.strength)),
            seeding_scheme='selfhash')
        if kind=='SIR':spec.update(assets=str(sir_path),asset_manifest=sir_asset,chunk_length=10)
        c['methods'][identity]=spec
    conf=dict(package_version=engine.VERSION,extension=VERSION,experiment=c,
        watermarks=c['methods'],upstream_commit=UPSTREAM_COMMIT,
        note='Same chosen Llama weights, new watermark methods. Not a full paper-model/data/TPR reproduction.')
    root.mkdir(parents=True);save(root/'config.json',conf)
    (root/'prompts.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in prompts))
    print(json.dumps(dict(prepared=str(root),source_prompts=len(prompts),final_passages=2*len(prompts),A=a.a,B=a.b,prw_fraction=a.prw_fraction)))
    if sir_asset:print('SIR may have no score on short texts: native detector requires >=40 jieba segments of preceding context. No score will be recorded as unavailable, not falsely reported as watermark evidence.')


def progress(start,added,done,total):
    elapsed=time.monotonic()-start
    print(json.dumps(dict(completed=done,total=total,session_seconds=round(elapsed,2),rough_eta_seconds=round(elapsed/added*(total-done),2) if added else None)),flush=True)


def verify_assets(c):
    for m in c['methods'].values():
        if m['method']=='SIR':
            root=Path(m['assets'])
            if json.loads((root/'sir_assets.json').read_text())!=m['asset_manifest']:raise ValueError('SIR manifest changed')
            for name,sha in m['asset_manifest']['files'].items():
                if file_sha(root/name)!=sha:raise ValueError('SIR asset bytes changed')


def generate(a):
    import engine
    import torch,transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM,LogitsProcessorList,set_seed
    from packaging.version import Version
    root=Path(a.run_dir);c=json.loads((root/'config.json').read_text())['experiment'];verify_assets(c)
    with (root/'generate.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prompts=read(root/'prompts.jsonl');manifest=dict(extension=VERSION,config=c,prompt_hash=engine.fp(prompts))
        mp=root/'manifest.json'
        if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Generation resume mismatch')
        save(mp,manifest)
        journal=root/'samples.jsonl';bank=root/'first_stage.jsonl'
        previous=engine.recover(journal);firsts=engine.recover(bank)
        done={r['sample_id']:r for r in previous};cached={r['sample_id']:r for r in firsts}
        expected={p['prompt_id']+'_'+h for p in prompts for h in LABELS}
        if len(done)!=len(previous) or len(cached)!=len(firsts) or not set(done)<=expected or not set(cached)<=expected:raise ValueError('Unexpected sample IDs')
        if set(done)==expected:save(root/'COMPLETE.json',dict(samples=len(done)));print('Generation already complete');return
        tok=AutoTokenizer.from_pretrained(c['A']['model'],revision=c['A']['revision']);tok.pad_token=tok.eos_token
        arg='dtype' if Version(transformers.__version__)>=Version('4.56') else 'torch_dtype'
        model=AutoModelForCausalLM.from_pretrained(c['A']['model'],revision=c['A']['revision'],**{arg:getattr(torch,c['dtype'])}).to(c['device']).eval()
        if model.config.vocab_size!=c['vocab_size']:raise ValueError('Model vocab changed')
        backs={h:Backend(c['methods'][h],tok,c['vocab_size'],c['device']) for h in ['A','B']}
        started=time.monotonic();added=0;total=4*len(prompts)
        def call(text,h,s,stage):
            set_seed(s)
            instruction=('Write one clear paragraph based on the passage. Preserve its information. Return only the paragraph.' if stage=='first' else 'Paraphrase the passage with different wording. Preserve meaning, names, numbers and details. Do not summarize. Return only the rewritten passage.')
            ids=tok.apply_chat_template([dict(role='system',content='You rewrite text accurately and concisely.'),dict(role='user',content=instruction+'\n\n'+text)],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(c['device'])
            limit=getattr(model.config,'max_position_embeddings',getattr(model.config,'n_positions',None))
            if limit is None or ids.shape[1]+c['sampling']['max_new_tokens']>limit:raise ValueError('Context budget exceeded')
            kw=dict(c['sampling'],do_sample=True,pad_token_id=tok.pad_token_id,attention_mask=torch.ones_like(ids),logits_processor=LogitsProcessorList([backs[h].processor]))
            stops=model.generation_config.eos_token_id
            stops=[stops] if isinstance(stops,int) else list(stops or [])
            if tok.eos_token_id is not None:stops.append(tok.eos_token_id)
            if '<|eot_id|>' in tok.get_vocab():stops.append(tok.get_vocab()['<|eot_id|>'])
            if stops:kw['eos_token_id']=list(dict.fromkeys(stops))
            before=time.monotonic()
            with torch.inference_mode():out=model.generate(ids,**kw)
            text=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()
            if not text:raise ValueError('Empty output')
            return text,time.monotonic()-before
        for p in prompts:
            for label in LABELS:
                sid=p['prompt_id']+'_'+label
                if sid in done:continue
                first='A' if label=='A_first' else 'B';second='B' if first=='A' else 'A'
                fs=engine.seed(str(c.get('seed',0))+'|'+p['prompt_id'],'first');ss=engine.seed(str(c.get('seed',0))+'|'+p['prompt_id'],'second')
                if sid not in cached:
                    source=tok.decode(tok(p['text'],add_special_tokens=False)['input_ids'][:c['prompt_max_tokens']])
                    text,sec=call(source,first,fs,'first')
                    r=dict(sample_id=sid,prompt_id=p['prompt_id'],split=p['split'],label=label,intermediate_text=text,generation_seconds=sec)
                    append(bank,r);cached[sid]=r;added+=1;progress(started,added,len(done)+len(cached),total)
                source=tok.decode(tok(cached[sid]['intermediate_text'],add_special_tokens=False)['input_ids'][:c['rewrite_source_max_tokens']])
                text,sec=call(source,second,ss,'second')
                r=dict(sample_id=sid,prompt_id=p['prompt_id'],split=p['split'],label=label,first_identity=first,second_identity=second,
                    intermediate_text=cached[sid]['intermediate_text'],rewrite_source=source,final_text=text,first_seed=fs,second_seed=ss,
                    first_generation_seconds=cached[sid]['generation_seconds'],second_generation_seconds=sec,methods=c['methods'])
                append(journal,r);done[sid]=r;added+=1;progress(started,added,len(done)+len(cached),total)
        save(root/'COMPLETE.json',dict(samples=len(done),session_seconds=time.monotonic()-started))


def score(a):
    import engine
    import pandas as pd
    from common import feature_row,get_X
    from train_classifier import validate_frame
    from transformers import AutoTokenizer
    root=Path(a.run_dir);conf=json.loads((root/'config.json').read_text());c=conf['experiment'];verify_assets(c)
    if not (root/'COMPLETE.json').exists():raise ValueError('Finish generation first')
    rows=read(root/'samples.jsonl');engine.validate_samples(rows)
    out=root/'scoring';out.mkdir(exist_ok=True)
    with (out/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=dict(config_hash=engine.fp(c),samples_hash=engine.fp(rows),extension=VERSION)
        p=out/'manifest.json'
        if p.exists() and json.loads(p.read_text())!=manifest:raise ValueError('Scoring resume mismatch')
        save(p,manifest)
        previous=engine.recover(out/'journal.jsonl');done={r['sample_id']:r for r in previous}
        if len(done)!=len(previous) or not set(done)<={r['sample_id'] for r in rows}:raise ValueError('Bad scoring IDs')
        tok=None;started=time.monotonic();added=0
        for row in rows:
            if row['sample_id'] in done:continue
            if tok is None:
                tok=AutoTokenizer.from_pretrained(c['A']['model'],revision=c['A']['revision'],use_fast=True)
                backs={h:Backend(c['methods'][h],tok,c['vocab_size'],c['device'],need_processor=False) for h in ['A','B']}
            values={h:backs[h].score(row['final_text']) for h in ['A','B']}
            r=dict(sample_id=row['sample_id'],values=values)
            append(out/'journal.jsonl',r);done[row['sample_id']]=r;added+=1
            if added==1 or added%25==0 or len(done)==len(rows):progress(started,added,len(done),len(rows))
        for mode in a.modes:
            dest=root/mode;dest.mkdir(exist_ok=True);features=[]
            for row in rows:
                raw=done[row['sample_id']]['values']
                v={h:feature_scores(c['methods'][h]['method'],raw[h],c['methods'][h]['gamma'],mode) for h in ['A','B']}
                fr=feature_row(v['A'],v['B'],row['prompt_id'],LABELS[row['label']],row['label'])
                fr.update(split=row['split'],text_hash=engine.th(row['final_text']),A_whole_available=v['A']['whole_available'],B_whole_available=v['B']['whole_available'])
                features.append(fr)
            df=pd.DataFrame(features);validate_frame(df)
            X,cols=get_X(df,'windows_whole')
            if X.shape[1]!=10:raise AssertionError('Expected10features')
            csv=df.to_csv(index=False);(dest/'features.csv').write_text(csv);save(dest/'config.json',conf)
            save(dest/'feature_manifest.json',dict(version=engine.VERSION,config_hash=engine.fp(c),window=mode,capacity=4,feature_rows=len(df),csv_sha256=engine.th(csv)))
            save(dest/'method_window_spec.json',dict(extension=VERSION,mode=mode,columns=cols,native_methods=c['methods'],missing_score_policy='zero placeholder; availability saved but NOT classifier input'))
            report={h:dict(method=c['methods'][h]['method'],whole_available_fraction=float(df[h+'_whole_available'].mean())) for h in ['A','B']}
            report_by_group=df.groupby(['split','category'])[['A_whole_available','B_whole_available']].mean().reset_index().to_dict(orient='records')
            save(dest/'score_availability_by_split_category.json',report_by_group)
            save(dest/'score_availability.json',report);print(mode,json.dumps(report),flush=True)
            if any(v['whole_available_fraction']<1 for v in report.values()):print('CAUTION: native detector unavailable for some passages. Zero placeholders are not observed evidence; report availability with accuracy.')
        save(out/'COMPLETE.json',dict(scored=len(done),session_seconds=time.monotonic()-started))


def preflight(a):
    import torch
    import numpy as np
    from transformers import AutoTokenizer
    c=json.loads((Path(a.run_dir)/'config.json').read_text())['experiment'];verify_assets(c)
    tok=AutoTokenizer.from_pretrained(c['A']['model'],revision=c['A']['revision'],use_fast=True)
    text=' '.join(['A mango becomes sweeter during ripening because stored starch is converted into sugars.']*15)
    ids=torch.tensor([tok(text,add_special_tokens=False)['input_ids']],device=c['device'])
    for h in ['A','B']:
        backend=Backend(c['methods'][h],tok,c['vocab_size'],c['device']);scores=torch.zeros((1,c['vocab_size']),device=c['device'])
        with torch.inference_mode():biased=backend.processor(ids,scores)
        if biased.shape!=scores.shape or not torch.isfinite(biased).all():raise AssertionError('Processor shape/nonfinite failure')
        r=backend.score(text)
        print(json.dumps(dict(identity=h,method=backend.kind,whole_score=r['whole_score'],score_type=r['native_score_type'],evidence_count=len(r['evidence']))))
        if r['whole_score'] is None:raise ValueError('Preflight sample has no evidence')
    print('PASS native processors/detectors and whole-score reconstruction; no generation/order accuracy claim.')


def predict(a):
    import joblib,pandas as pd
    from transformers import AutoTokenizer
    from common import feature_row,get_X
    case=Path(a.case_dir);spec=json.loads((case/'method_window_spec.json').read_text())
    path=case/'linreg_windows_whole.joblib' if a.classifier=='linear' else case/'extra_windows_whole'/f'{a.classifier}_model.joblib'
    bundle=joblib.load(path);c=bundle['experiment'];verify_assets(c)
    if c['methods']!=spec['native_methods']:raise ValueError('Method specification mismatch')
    tok=AutoTokenizer.from_pretrained(c['A']['model'],revision=c['A']['revision'],use_fast=True);text=Path(a.text_file).read_text()
    raw={h:Backend(c['methods'][h],tok,c['vocab_size'],c['device'],False).score(text) for h in ['A','B']}
    v={h:feature_scores(c['methods'][h]['method'],raw[h],c['methods'][h]['gamma'],spec['mode']) for h in ['A','B']}
    X,cols=get_X(pd.DataFrame([feature_row(v['A'],v['B'])]),'windows_whole')
    if cols!=bundle['columns']:raise ValueError('Feature mismatch')
    model=bundle['model']
    if a.classifier=='linear':s=float(model.predict(X)[0]);label=int(s>=.5)
    else:
        label=int(model.predict(X)[0]);s=float(model.decision_function(X)[0]) if a.classifier=='svm' else float(model.predict_proba(X)[0,1])
    result=dict(category={0:'A_first',1:'B_first'}[label],score_not_calibrated_probability=s,
        whole_scores={h:raw[h]['whole_score'] for h in raw},exact_provenance_proof=False,
        scope='Known candidate pair; exactly both stages in one of two orders')
    save(a.out,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('sir-assets');q.add_argument('--base-run',required=True);q.add_argument('--out',required=True);q.add_argument('--checkpoint');q.add_argument('--mapping-seed',type=int,default=42)
    q=sub.add_parser('prepare');q.add_argument('--base-run',required=True);q.add_argument('--out',required=True);q.add_argument('--a',choices=['KGW','PRW','SIR'],required=True);q.add_argument('--b',choices=['KGW','PRW','SIR'],required=True);q.add_argument('--count',type=int);q.add_argument('--sir-assets');q.add_argument('--prw-fraction',type=float,choices=[.25,.5],default=.5);q.add_argument('--strength',type=float,choices=[2.,5.],default=2.);q.add_argument('--strength-a',type=float,choices=[2.,5.]);q.add_argument('--strength-b',type=float,choices=[2.,5.]);q.add_argument('--min-new-tokens',type=int);q.add_argument('--max-new-tokens',type=int)
    for cmd in ['preflight','generate','score']:
        q=sub.add_parser(cmd);q.add_argument('--run-dir',required=True)
        if cmd=='score':q.add_argument('--modes',nargs='+',choices=['dynamic4','fixed59x4'],default=['dynamic4'])
    q=sub.add_parser('predict');q.add_argument('--case-dir',required=True);q.add_argument('--classifier',choices=['linear','svm','mlp'],default='linear');q.add_argument('--text-file',required=True);q.add_argument('--out',required=True)
    a=p.parse_args();{'sir-assets':assets,'prepare':prepare,'generate':generate,'score':score,'preflight':preflight,'predict':predict}[a.cmd](a)
