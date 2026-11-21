"""Generate exactly two watermark orders with GPT-2 or Llama 3.2, saving every completed stage."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi
import torch
import engine
from common import DEFAULT_WM_A,DEFAULT_WM_B,WMConfig

PARAPHRASE_TEMPLATE=engine.REWRITE_EXAMPLES+'Original: {text}\nParaphrase:'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='openai-community/gpt2');p.add_argument('--revision')
    p.add_argument('--dataset',default='allenai/c4');p.add_argument('--prompts-file');p.add_argument('--exclude-prompts')
    p.add_argument('--n-per-class',type=int,default=1000);p.add_argument('--out-dir',default='runs/full')
    p.add_argument('--batch-size',type=int,choices=[1],default=1,help='Stage-safe generation currently processes one passage at a time')
    p.add_argument('--prompt-len',type=int,default=128);p.add_argument('--max-new-tokens',type=int,default=200)
    p.add_argument('--min-new-tokens',type=int,default=None);p.add_argument('--temperature',type=float,default=.7)
    p.add_argument('--top-p',type=float,default=.95);p.add_argument('--top-k',type=int,default=40)
    p.add_argument('--gamma',type=float,default=.25);p.add_argument('--delta',type=float,default=2.)
    p.add_argument('--seeding-scheme',default='selfhash');p.add_argument('--key-a',type=int,default=DEFAULT_WM_A.key)
    p.add_argument('--key-b',type=int,default=DEFAULT_WM_B.key);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--dtype',choices=['float32','float16','bfloat16'])
    p.add_argument('--prompt-style',choices=['auto','completion','chat'],default='auto')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args()
    if a.model not in engine.SUPPORTED_MODELS:p.error('Unsupported model; use GPT-2 or Llama-3.2-1B[-Instruct]')
    if a.prompt_style=='auto':a.prompt_style='chat' if a.model.endswith('-Instruct') else 'completion'
    if a.dtype is None:a.dtype='bfloat16' if a.model.startswith('meta-llama/') and torch.device(a.device).type=='cuda' else 'float32'
    if a.min_new_tokens is None:a.min_new_tokens=25 if a.model.startswith('meta-llama/') else 200
    if a.key_a==a.key_b:p.error('Watermark keys must differ')
    if not 25<=a.min_new_tokens<=a.max_new_tokens:p.error('Require 25 <= min-new-tokens <= max-new-tokens')
    if a.n_per_class<5:p.error('Use at least 5 source prompts')
    if a.dataset!='allenai/c4' and not a.prompts_file:p.error('Use --prompts-file for another/local dataset; legacy dataset scripts are not loaded')
    root=Path(a.out_dir);root.mkdir(parents=True,exist_ok=True);cfgpath=root/'config.json';args=vars(a).copy()
    # Lock configuration AND prompt preparation as well as stage writes.
    import fcntl
    with (root/'preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if cfgpath.exists():
            saved=json.loads(cfgpath.read_text())
            if saved.get('package_version')!=engine.VERSION:raise ValueError('Legacy run: preserve it and choose a NEW output directory')
            compare={k:v for k,v in args.items() if k!='revision'}
            previous={k:v for k,v in saved['args'].items() if k!='revision'}
            previous.setdefault('dtype',saved['experiment'].get('dtype','float32'))
            previous.setdefault('prompt_style',saved['experiment'].get('prompt_style','completion'))
            if compare!=previous:raise ValueError('Resume argument mismatch; existing configuration preserved')
            revision=saved['experiment']['A']['revision']
            if a.revision is not None and a.revision!=saved['args'].get('revision') and a.revision!=revision:raise ValueError('Revision mismatch')
            cfg=saved['experiment']
        else:
            if any((root/name).exists() for name in ('A_first.jsonl','B_first.jsonl','samples.jsonl','manifest.json','first_stage.jsonl','prompts.jsonl')):
                raise ValueError('Existing unmanifested data; choose a fresh output directory')
            revision=HfApi().model_info(a.model,revision=a.revision or 'main').sha
            cfg={'A':{'model':a.model,'revision':revision,'key':a.key_a},'B':{'model':a.model,'revision':revision,'key':a.key_b},
                'device':a.device,'dtype':a.dtype,'prompt_style':a.prompt_style,'gamma':a.gamma,'delta':a.delta,'seeding_scheme':a.seeding_scheme,
                'seed':a.seed,'windows':[25],'block_size':25,'prompt_max_tokens':a.prompt_len,'rewrite_source_max_tokens':a.max_new_tokens,
                'sampling':{'temperature':a.temperature,'top_p':a.top_p,'top_k':a.top_k,'min_new_tokens':a.min_new_tokens,'max_new_tokens':a.max_new_tokens}}
            if a.prompt_style=='chat':cfg['chat_instructions']=dict(engine.CHAT_INSTRUCTIONS)
            engine.save(cfgpath,{'package_version':engine.VERSION,'args':args,'experiment':cfg,
                'watermarks':{h:WMConfig(h,cfg[h]['key'],a.gamma,a.delta,a.seeding_scheme).asdict() for h in ('A','B')},
                'paraphrase_template':cfg.get('chat_instructions',PARAPHRASE_TEMPLATE)})
        prompts=root/'prompts.jsonl'
        if not prompts.exists():engine.prepare(a.n_per_class,prompts,a.prompts_file,a.exclude_prompts)
        if len(engine.read(prompts))!=a.n_per_class:raise ValueError('Unexpected prompt count')
        engine.generate(cfg,prompts,root)
        # Compatibility category exports are rebuilt from canonical durable records.
        rows=engine.read(root/'samples.jsonl')
        for label in ('A_first','B_first'):
            tmp=root/(label+'.jsonl.tmp')
            tmp.write_text(''.join(json.dumps(dict(r,idx=r['prompt_id'],category=r['label'],label=engine.LABELS[r['label']]),ensure_ascii=False)+'\n' for r in rows if r['label']==label))
            tmp.replace(root/(label+'.jsonl'))
        print('Complete. Canonical text bank:',root/'samples.jsonl')
        print('Timing:',root/'generation_timing.json')

if __name__=='__main__':main()
