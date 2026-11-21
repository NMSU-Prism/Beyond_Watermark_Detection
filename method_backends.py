"""Native collision-repository scores, not a shared KGW approximation."""
import bisect
import hashlib
import json
import math
from pathlib import Path

UPSTREAM_COMMIT='bf9470f436a6161d4e66a22dfa030b5121a00b86'


def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def bounds(n, mode):
    if mode=='dynamic4':
        q,r=divmod(n,4);ends=[0]
        for i in range(4):ends.append(ends[-1]+q+int(i<r))
        return list(zip(ends[:-1],ends[1:]))
    if n>236:raise ValueError('fixed59x4 exceeds236tokens; choose dynamic4; no truncation')
    return [(min(i*59,n),min((i+1)*59,n)) for i in range(4)]


def aggregate(kind, evidence, gamma):
    if not evidence:return 0.,0
    if kind=='KGW':
        unique={tuple(r['unit']):r['value'] for r in evidence};v=list(unique.values())
    else:v=[r['value'] for r in evidence]
    if kind=='SIR':return sum(v)/len(v),len(v)
    return (sum(v)-gamma*len(v))/math.sqrt(len(v)*gamma*(1-gamma)),len(v)


def feature_scores(kind, record, gamma, mode):
    zs=[];availability=[];details=[]
    for a,b in bounds(record['n_tokens'],mode):
        v=[r for r in record['evidence'] if a<=r['position']<b]
        z,n=aggregate(kind,v,gamma);zs.append(z);availability.append(int(n>0))
        details.append(dict(start_token=a,end_token_exclusive=b,score=z,effective_n=n))
    return dict(whole_z=record['whole_score'] if record['whole_score'] is not None else 0.,
        window_z=zs,window_available=availability,n_tokens=record['n_tokens'],sections=details,
        whole_available=int(record['whole_score'] is not None))


class Backend:
    def __init__(self, spec, tokenizer, vocab_size, device, need_processor=True):
        import torch
        self.spec=spec;self.kind=spec['method'];self.tok=tokenizer;self.vocab_size=vocab_size
        self.processor=None;self.detector=None
        self.device=torch.device(device)
        if self.kind=='KGW':
            from common import WMConfig,build_processor,build_detector
            c=WMConfig(spec['identity'],spec['key'],spec['gamma'],spec['delta'],spec['seeding_scheme'])
            if need_processor:self.processor=build_processor(tokenizer,c)
            self.detector=build_detector(tokenizer,c,self.device)
        elif self.kind=='PRW':
            from collision_vendor.prw.gptwm import GPTWatermarkDetector,GPTWatermarkLogitsWarper
            kw=dict(fraction=spec['gamma'],strength=spec['delta'],vocab_size=vocab_size,watermark_key=spec['key'])
            self.detector=GPTWatermarkDetector(**kw)
            if need_processor:self.processor=GPTWatermarkLogitsWarper(**kw)
        else:
            import numpy as np
            from transformers import AutoTokenizer,BertModel
            from collision_vendor.sir.watermark import WatermarkContext,WatermarkBase,WatermarkLogitsProcessor
            from collision_vendor.sir.train_watermark_model import TransformModel
            assets=Path(spec['assets']);manifest=json.loads((assets/'sir_assets.json').read_text())
            if manifest['vocab_size']!=vocab_size:raise ValueError('SIR mapping/model vocab mismatch')
            for name,sha in manifest['files'].items():
                if file_sha(assets/name)!=sha:raise ValueError('SIR asset checksum mismatch: '+name)
            # Avoid upstream's mutable cwd mapping and random mapping auto-creation.
            wm=WatermarkContext.__new__(WatermarkContext)
            WatermarkBase.__init__(wm,spec['gamma'],spec['delta'],tokenizer,vocab_size)
            wm.device=self.device;wm.chunk_length=10
            wm.embedding_tokenizer=AutoTokenizer.from_pretrained(manifest['embedding_model'],revision=manifest['embedding_revision'])
            wm.embedding_model=BertModel.from_pretrained(manifest['embedding_model'],revision=manifest['embedding_revision']).to(self.device).eval()
            wm.transform_model=TransformModel()
            state=torch.load(assets/'transform_model_cbert.pth',map_location='cpu',weights_only=True)
            wm.transform_model.load_state_dict(state);wm.transform_model.to(self.device).eval()
            wm.mapping=json.loads((assets/'mapping.json').read_text())
            if len(wm.mapping)!=vocab_size or any(not 0<=int(x)<300 for x in wm.mapping):raise ValueError('Invalid SIR mapping')
            self.detector=wm
            if need_processor:self.processor=WatermarkLogitsProcessor(wm)

    def score(self,text):
        import torch
        import numpy as np
        with torch.inference_mode():return self._score(text,np)

    def _score(self,text,np):
        ids=self.tok(text,add_special_tokens=False)['input_ids']
        if self.kind=='KGW':
            from common import window_scores,tokenize_for_detection
            ids=tokenize_for_detection(self.detector,text).cpu().tolist()
            v=window_scores(self.detector,text,25);n=self.detector.context_width+1-int(self.detector.self_salt)
            ev=[dict(position=r['position'],value=int(r['green']),unit=ids[r['position']-n+1:r['position']+1]) for r in v['traces']]
            whole=v['whole_z'] if ev else None
        elif self.kind=='PRW':
            mask=self.detector.green_list_mask.tolist()
            ev=[dict(position=i,value=int(mask[t])) for i,t in enumerate(ids)]
            whole=float(self.detector.detect(ids)) if ids else None
        else:
            import jieba
            wm=self.detector;groups=wm.get_text_split(text);ev=[];all_values=[]
            encoding=self.tok(text,add_special_tokens=False,return_offsets_mapping=True)
            offsets=encoding['offset_mapping'];ends=[b for a,b in offsets];cursor=0
            for i,g in enumerate(groups):
                raw=''.join(g);start=text.find(raw,cursor)
                if start<0:raise ValueError('SIR segmentation cannot be aligned to original final text')
                cursor=start+len(raw)
                if i==0:continue
                context=''.join(''.join(x) for x in groups[:i]).strip()
                current=raw.strip()
                if len(list(jieba.cut(context)))<40:continue  # EXACT upstream eligibility rule
                embedding=wm.get_embedding(context)
                output=wm.transform_model(embedding).cpu()[0].numpy()
                similarity=wm.scale_vector(output)[wm.mapping]
                chunk=self.tok(current,add_special_tokens=False,return_offsets_mapping=True)
                chunk_start=start+len(raw)-len(raw.lstrip())
                for token,(a,b) in zip(chunk['input_ids'],chunk['offset_mapping']):
                    value=-float(similarity[token]);all_values.append(value)
                    # Chunk re-tokenization is native SIR behavior. Assign its evidence
                    # to the containing full-text token by character midpoint.
                    midpoint=chunk_start+(a+b)/2
                    pos=min(bisect.bisect_right(ends,midpoint),max(0,len(ids)-1))
                    ev.append(dict(position=pos,value=value))
            whole=float(np.mean(all_values)) if all_values else None
            native=float(wm.detect(text)) if all_values else None
            if whole is not None and not np.isclose(whole,native,atol=1e-6):raise AssertionError('SIR native whole-score parity failed')
        if whole is not None and not math.isfinite(whole):raise ValueError('Nonfinite detector score')
        expected,_=aggregate(self.kind,ev,self.spec['gamma'])
        if whole is not None and not math.isclose(whole,expected,abs_tol=1e-5):raise AssertionError('Whole-score reconstruction mismatch')
        return dict(n_tokens=len(ids),whole_score=whole,evidence=ev,native_score_type='mean semantic bias' if self.kind=='SIR' else 'z-score')
