"""Bridge the persistent engine to vendored upstream KGW. No external order package."""
from transformers import AutoTokenizer
import torch
from common import WMConfig,build_detector,window_scores

class Scorer:
    def __init__(self,config):
        self.config=config;self.items={}
        tok=AutoTokenizer.from_pretrained(config['A']['model'],revision=config['A']['revision'],use_fast=True)
        for identity in ('A','B'):
            c=WMConfig(identity,config[identity]['key'],config['gamma'],config['delta'],config['seeding_scheme'])
            self.items[identity]=(tok,build_detector(tok,c,torch.device(config['device'])))
    def score(self,text):
        summaries={};traces={}
        for h,(tok,d) in self.items.items():
            values=window_scores(d,text,25);summaries[h+'_z']=values['whole_z']
            enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
            offsets=enc['offset_mapping'];ids=enc['input_ids']
            if ids and ids[0]==tok.bos_token_id:offsets=offsets[1:]
            traces[h]=[dict(r,char_start=offsets[r['position']][0],char_end=offsets[r['position']][1]) for r in values['traces']]
        return summaries,traces
