"""Explicitly reuse legacy text without generation; source revision must be supplied."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import engine


def migrate(source,out,revision):
    source=Path(source);out=Path(out)
    if out.exists():raise FileExistsError('Migration output must be a new directory')
    if not re.fullmatch(r'[0-9a-fA-F]{40}',revision):raise ValueError('Supply the exact source model/tokenizer commit SHA, not main')
    old=json.loads((source/'config.json').read_text());a=old['args'];wm=old['watermarks']
    if a['model']!='openai-community/gpt2':raise ValueError('Expected GPT-2 source run')
    for key in ('gamma','delta','seeding_scheme'):
        if wm['A'][key]!=wm['B'][key]:raise ValueError('Migration requires matching watermark scheme/strength')
    raw={};duplicates=0
    for label in ('A_first','B_first'):
        values={}
        for r in engine.read(source/(label+'.jsonl')):
            idx=str(r['idx'])
            if idx in values:
                if values[idx]!=r:raise ValueError('Conflicting duplicate '+label+' '+idx+'; inspect manually')
                duplicates+=1;continue
            if not r['final_text'].strip() or not r['stage1_text'].strip():raise ValueError('Empty legacy passage')
            if r.get('category',label)!=label:raise ValueError('Legacy category mismatch')
            if int(r['label'])!=engine.LABELS[label]:raise ValueError('Legacy label mismatch')
            values[idx]=r
        raw[label]=values
    if not raw['A_first'] or set(raw['A_first'])!=set(raw['B_first']):raise ValueError('Both categories must contain identical source IDs')
    ids=sorted(raw['A_first']);prompts=[];seen=set()
    for idx in ids:
        x,y=raw['A_first'][idx],raw['B_first'][idx]
        if x['prompt_text']!=y['prompt_text']:raise ValueError('Different source text for paired ID')
        pid=engine.th(x['prompt_text'])
        if pid in seen:raise ValueError('Different IDs duplicate the same source prompt; inspect manually')
        seen.add(pid);prompts.append({'prompt_id':pid,'text':x['prompt_text'],'legacy_idx':idx})
    n=len(prompts)
    if n<5:raise ValueError('At least five paired source groups required')
    for rank,i in enumerate(np.random.default_rng(42).permutation(n)):
        prompts[i]['split']='train' if rank<int(.7*n) else 'validation' if rank<int(.7*n)+max(1,int(.15*n)) else 'test'
    c={'A':{'model':a['model'],'revision':revision,'key':wm['A']['key']},'B':{'model':a['model'],'revision':revision,'key':wm['B']['key']},
        'device':a['device'],'dtype':'float32','gamma':wm['A']['gamma'],'delta':wm['A']['delta'],'seeding_scheme':wm['A']['seeding_scheme'],
        'seed':a.get('seed',0),'windows':[25],'block_size':25,'prompt_max_tokens':a['prompt_len'],
        'rewrite_source_max_tokens':a['max_new_tokens'],'sampling':{k:a[k] for k in ('temperature','top_p','top_k','min_new_tokens','max_new_tokens')}}
    engine.check_config(c);out.mkdir(parents=True)
    (out/'prompts.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in prompts))
    for p in prompts:
        for label in ('A_first','B_first'):
            r=raw[label][p['legacy_idx']]
            first='A' if label=='A_first' else 'B';second='B' if first=='A' else 'A'
            engine.append(out/'samples.jsonl',{'sample_id':p['prompt_id']+'_'+label,'prompt_id':p['prompt_id'],'split':p['split'],
                'label':label,'first_identity':first,'second_identity':second,'intermediate_text':r['stage1_text'],'final_text':r['final_text'],
                'legacy_idx':r['idx'],'legacy_record':r})
    manifest={'version':engine.VERSION,'config':c,'migration':True,'revision_source':'User-declared original commit; old script did not pin it',
              'legacy_config':old,'exact_duplicate_records_removed':duplicates,'expected_per_category':n}
    engine.save(out/'manifest.json',manifest)
    engine.save(out/'config.json',{'package_version':engine.VERSION,'args':{'migration_only':True},'experiment':c,'watermarks':wm})
    engine.save(out/'COMPLETE.json',{'samples':2*n,'manifest_hash':engine.fp(manifest)})
    print('Migrated existing final texts:',2*n,'No model generation was performed. Re-extract features and retrain.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--revision',required=True,help='Exact actual model/tokenizer source SHA; do not guess it')
    a=p.parse_args();migrate(a.source,a.out_dir,a.revision)
