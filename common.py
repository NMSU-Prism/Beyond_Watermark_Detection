"""
Shared helpers for the "which watermark came first?" experiment.

Watermark + detection score are taken from the collision paper repo
(AInnovateLab/watermark-collision, KGW folder; vendored in ./wm):
  * KGW logit watermark, gamma=0.25, delta=2.0, seeding_scheme="selfhash"
  * detector: z-score of green-token count, ignore_repeated_ngrams=True, no normalizers
  * the detector re-tokenizes the FINAL TEXT (like the repo's detect_text)

A and B are two *different* watermarks (different secret key). Everything else is
identical, so the classifier has to learn ORDER, not watermark strength.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch

from wm.extended_watermark_processor import (
    WatermarkLogitsProcessor,
    WatermarkDetector,
    ngrams,
)

CATEGORIES = {0: "A_first", 1: "B_first"}
# A_first : Model writes text with watermark A, then paraphrases it with watermark B
# B_first : Model writes text with watermark B, then paraphrases it with watermark A


@dataclass
class WMConfig:
    name: str
    key: int
    gamma: float = 0.25
    delta: float = 2.0
    seeding_scheme: str = "selfhash"
    z_threshold: float = 4.0

    def asdict(self):
        return asdict(self)


# Keys follow the repo's scripts (first stage 2024 / rephrase stage 2023)
DEFAULT_WM_A = WMConfig(name="A", key=2024)
DEFAULT_WM_B = WMConfig(name="B", key=2023)


def build_processor(tokenizer, cfg: WMConfig) -> WatermarkLogitsProcessor:
    return WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=cfg.gamma,
        delta=cfg.delta,
        seeding_scheme=cfg.seeding_scheme,
        hash_key=cfg.key,
    )


def build_detector(tokenizer, cfg: WMConfig, device) -> WatermarkDetector:
    # device must match the device used for generation (RNG device matters for the greenlist)
    return WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=cfg.gamma,
        seeding_scheme=cfg.seeding_scheme,
        device=device,
        tokenizer=tokenizer,
        z_threshold=cfg.z_threshold,
        normalizers=[],
        ignore_repeated_ngrams=True,
        hash_key=cfg.key,
    )


def _z(green: int, T: int, gamma: float) -> float:
    if T <= 0:
        return 0.0
    return (green - gamma * T) / math.sqrt(T * gamma * (1 - gamma))


def tokenize_for_detection(detector: WatermarkDetector, text: str) -> torch.Tensor:
    ids = detector.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    bos = detector.tokenizer.bos_token_id
    if len(ids) > 0 and bos is not None and ids[0] == bos:
        ids = ids[1:]
    return ids.to(detector.device)


@torch.no_grad()
def window_scores(detector, text, window=25, num_windows=None):
    """Raw-text token blocks with full-text contexts; missing != observed zero.

    A final partial block is retained. Capacity overflow is an error, never truncation.
    Whole z is checked against the unmodified upstream detector.
    """
    if window<1:raise ValueError('window must be positive')
    ids=tokenize_for_detection(detector,text);tokens=ids.cpu().tolist();n_tok=len(tokens)
    n=detector.context_width+1-int(detector.self_salt)
    traces=[];uniq={}
    if n_tok>detector.context_width:
        lookup,_=detector._score_ngrams_in_passage(ids)
        grams=list(ngrams(tokens,n))
        for j,g in enumerate(grams):
            bit=int(bool(lookup[g]));uniq.setdefault(g,bit)
            traces.append({'position':j+n-1,'green':bit})
    whole=_z(sum(uniq.values()),len(uniq),detector.gamma)
    if n_tok>detector.context_width:
        upstream=detector.detect(tokenized_text=ids,return_z_at_T=False)
        if abs(float(upstream['z_score'])-whole)>1e-6:raise AssertionError('Upstream whole-score mismatch')
    blocks=[]
    for start in range(0,n_tok,window):
        end=min(start+window,n_tok);u={}
        for r in traces:
            pos=r['position']
            if start<=pos<end:u.setdefault(tuple(tokens[pos-n+1:pos+1]),r['green'])
        blocks.append({'start_token':start,'end_token_exclusive':end,'full_block':end-start==window,
            'effective_n':len(u),'green':sum(u.values()),'z':_z(sum(u.values()),len(u),detector.gamma)})
    zs=[b['z'] for b in blocks];fractions=[b['green']/b['effective_n'] if b['effective_n'] else 0. for b in blocks]
    avail=[int(b['effective_n']>0) for b in blocks]
    if num_windows is not None:
        if len(blocks)>num_windows:raise ValueError('Passage exceeds trained window capacity; no scores discarded')
        zs+= [0.]*(num_windows-len(zs));fractions += [0.]*(num_windows-len(fractions));avail += [0]*(num_windows-len(avail))
    return {'whole_z':whole,'window_z':zs,'window_green':fractions,'window_available':avail,
            'n_tokens':n_tok,'blocks':blocks,'traces':traces}


def feature_row(sa,sb,idx=None,label=None,category=None):
    row={}
    if idx is not None:row['idx']=str(idx)
    if label is not None:row['label']=int(label)
    if category is not None:row['category']=category
    row['n_tokens']=sa['n_tokens']
    for identity,score in [('A',sa),('B',sb)]:
        row.update({f'{identity}_w{i+1}':v for i,v in enumerate(score['window_z'])})
        row.update({f'{identity}_m{i+1}':v for i,v in enumerate(score['window_available'])})
        row[identity+'_whole']=score['whole_z']
    return row


def get_X(df,mode):
    """One feature builder shared by training and inference."""
    import re
    import numpy as np
    def columns(identity):
        return sorted([c for c in df.columns if re.fullmatch(identity+r'_w\d+',c)],key=lambda c:int(c.split('_w')[1]))
    A,B=columns('A'),columns('B')
    if not A or len(A)!=len(B):raise ValueError('Missing/mismatched window features')
    if mode=='windows':cols=A+B
    elif mode=='windows_whole':cols=A+B+['A_whole','B_whole']
    elif mode=='whole':cols=['A_whole','B_whole']
    elif mode=='halves4':
        if len(A)<2:raise ValueError('halves4 requires at least 2 window slots')
        def mean(cols):
            values=df[cols].to_numpy(float);masks=[]
            for col in cols:
                identity,i=col.split('_w');name=identity+'_m'+i
                if name not in df:raise ValueError('Missing window availability; re-extract features')
                masks.append(df[name].to_numpy(float))
            mask=np.array(masks).T
            return (values*mask).sum(axis=1)/np.maximum(1,mask.sum(axis=1))
        h=len(A)//2
        x=np.c_[mean(A[:h]),mean(A[h:]),mean(B[:h]),mean(B[h:])]
        return x,['A_first_half','A_second_half','B_first_half','B_second_half']
    else:raise ValueError('Unsupported feature mode')
    x=df[cols].to_numpy(float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite feature values')
    if len(cols)!=len(set(cols)):raise AssertionError('Duplicate feature columns')
    return x,cols
