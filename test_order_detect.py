import contextlib,io,json,tempfile,types,sys,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
import pandas as pd
import numpy as np
import torch
import common,engine
import generate_dataset as generation
import train_classifier as training
import predict_category as prediction

def config():
    return {'A':{'model':'openai-community/gpt2','revision':'fixed','key':2024},'B':{'model':'openai-community/gpt2','revision':'fixed','key':2023},'block_size':25,'gamma':.25,'delta':2.,'seeding_scheme':'selfhash','device':'cpu','dtype':'float32','windows':[25],'prompt_max_tokens':128,'rewrite_source_max_tokens':200,'sampling':{'max_new_tokens':200,'min_new_tokens':200}}

class Tokenizer:
    bos_token_id=None
    def get_vocab(self):return {str(i):i for i in range(64)}
    def __call__(self,text,**kw):return {'input_ids':torch.arange(200).remainder(63).unsqueeze(0)}

class OrderTests(unittest.TestCase):
    def test_upstream_score_and_window_boundaries(self):
        det=common.build_detector(Tokenizer(),common.WMConfig('A',2024),torch.device('cpu'))
        r=common.window_scores(det,'synthetic',25,9)
        raw=det.detect(tokenized_text=common.tokenize_for_detection(det,'synthetic'),return_z_at_T=False)
        self.assertAlmostEqual(r['whole_z'],float(raw['z_score']))
        self.assertEqual(len(r['blocks']),8);self.assertEqual(r['blocks'][0]['start_token'],0)
        self.assertEqual(r['blocks'][0]['end_token_exclusive'],25)
        self.assertEqual(r['blocks'][0]['effective_n'],22)
        self.assertEqual(r['window_available'],[1]*8+[0])
        with self.assertRaises(ValueError):common.window_scores(det,'synthetic',25,7)

    def test_feature_columns_and_halves(self):
        r={f'{h}_w{i}':float(i) for h in ('A','B') for i in range(1,9)}
        r.update({f'{h}_m{i}':1 for h in ('A','B') for i in range(1,9)});r.update(A_whole=100.,B_whole=200.)
        df=pd.DataFrame([r])
        self.assertEqual(common.get_X(df,'windows')[0].shape,(1,16))
        self.assertEqual(common.get_X(df,'windows_whole')[0].shape,(1,18))
        np.testing.assert_allclose(common.get_X(df,'halves4')[0],[[2.5,6.5,2.5,6.5]])

    def make_run(self,root):
        c=config();wm={h:common.WMConfig(h,c[h]['key']).asdict() for h in ('A','B')}
        engine.save(root/'config.json',{'experiment':c,'watermarks':wm});rows=[]
        for i in range(30):
            for y in (0,1):
                r={'idx':str(i),'split':'train' if i<20 else 'validation' if i<25 else 'test','label':y,'text_hash':f'{i}_{y}'}
                for h in ('A','B'):
                    z=float(y if h=='A' else 1-y);r[h+'_whole']=z*8
                    for k in range(1,10):r[h+'_w'+str(k)]=z if k<9 else 0.;r[h+'_m'+str(k)]=int(k<9)
                rows.append(r)
        pd.DataFrame(rows).to_csv(root/'features.csv',index=False)
        engine.save(root/'feature_manifest.json',{'version':engine.VERSION,'config_hash':engine.fp(c),'window':25,'capacity':9,'feature_rows':len(rows),'csv_sha256':engine.th((root/'features.csv').read_text())})

    def test_all_feature_modes_predict_and_training_only_saved_model(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);self.make_run(root)
            def scores(h):return {'n_tokens':200,'whole_z':0. if h=='A' else 8.,'window_z':([0.]*9 if h=='A' else [1.]*8+[0.]),'window_available':[1]*8+[0],'blocks':[],'traces':[]}
            for mode in ('windows_whole','windows','whole','halves4'):
                b,r=training.train(root,mode)
                self.assertEqual(r['splits']['test']['accuracy'],1.)
                self.assertEqual(b['model'][0].n_samples_seen_,40)
                with patch.object(prediction.AutoTokenizer,'from_pretrained',return_value=MagicMock()),patch.object(prediction,'build_detector',side_effect=lambda tok,cfg,dev:cfg.name),patch.object(prediction,'window_scores',side_effect=lambda det,*args:scores(det)):
                    result=prediction.predict(b,'whole multiline\npassage')
                self.assertEqual(result['category'],'A_first')

    def test_duplicates_and_prompt_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.make_run(root);df=pd.read_csv(root/'features.csv',dtype={'idx':str,'text_hash':str})
            with self.assertRaises(ValueError):training.validate_frame(pd.concat([df,df.iloc[:1]]))
            df.loc[0,'split']='test'
            with self.assertRaises(ValueError):training.validate_frame(df)

    def test_legacy_configuration_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old={'sentinel':'legacy'};engine.save(root/'config.json',old)
            with patch.object(sys,'argv',['generate_dataset.py','--n-per-class','5','--out-dir',d,'--device','cpu']):
                with self.assertRaises(ValueError):generation.main()
            self.assertEqual(json.loads((root/'config.json').read_text()),old)

    def test_scoring_resume_and_feature_export(self):
        import extract_features as extraction
        tok=MagicMock();tok.bos_token_id=None;tok.side_effect=lambda *a,**kw:{'input_ids':list(range(60))}
        det=types.SimpleNamespace(context_width=4,self_salt=True)
        trace=[{'position':i,'green':int(i%2==0)} for i in range(3,60)]
        scorer=MagicMock();scorer.items={h:(tok,det) for h in ('A','B')}
        scorer.score.return_value=({'A_z':1.,'B_z':2.},{'A':trace,'B':trace})
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);c=config();raw=[{'sample_id':h,'prompt_id':'p','split':'train','label':h,'final_text':'FINAL '+h} for h in engine.LABELS]
            (root/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw))
            engine.save(root/'manifest.json',{'config':c});engine.save(root/'COMPLETE.json',{})
            engine.save(root/'config.json',{'experiment':c})
            with patch('detector_adapter.Scorer',return_value=scorer),patch.object(sys,'argv',['extract_features.py','--run-dir',d]):
                extraction.main();extraction.main()
            self.assertEqual(scorer.score.call_count,2)
            df=pd.read_csv(root/'features.csv')
            self.assertEqual(len(df),2)
            self.assertEqual(len(common.get_X(df,'windows_whole')[1]),20)
            self.assertTrue((root/'scored/token_traces.jsonl').exists())
            self.assertTrue((root/'scored/block25_scores.csv').exists())

    def test_torn_journal_tail_recovered(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rows.jsonl';p.write_bytes(b'{"sample_id":"saved"}\n{"partial":')
            self.assertEqual(engine.recover(p),[{'sample_id':'saved'}])
            self.assertTrue(list(Path(d).glob('*.tail-backup-*')))

    def test_explicit_migration_reuses_text_and_rejects_conflicts(self):
        import migrate_legacy
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);old=root/'old';old.mkdir()
            args={'model':'openai-community/gpt2','device':'cpu','seed':0,'prompt_len':128,
                  'max_new_tokens':200,'min_new_tokens':200,'temperature':.7,'top_p':.95,'top_k':40}
            wm={h:common.WMConfig(h,k).asdict() for h,k in [('A',2024),('B',2023)]}
            engine.save(old/'config.json',{'args':args,'watermarks':wm})
            for label,y in engine.LABELS.items():
                rows=[{'idx':i,'label':y,'category':label,'prompt_text':'source '+str(i),
                       'stage1_text':'first '+str(i),'final_text':label+' final '+str(i)} for i in range(5)]
                (old/(label+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            migrate_legacy.migrate(old,root/'migrated','a'*40)
            rows=engine.read(root/'migrated/samples.jsonl');self.assertEqual(len(rows),10)
            self.assertTrue(all(r['final_text']==r['legacy_record']['final_text'] for r in rows))
            duplicate=engine.read(old/'A_first.jsonl')[0];duplicate['final_text']='CONFLICT'
            engine.append(old/'A_first.jsonl',duplicate)
            with self.assertRaises(ValueError):migrate_legacy.migrate(old,root/'bad','a'*40)

    def test_partial_stage_resume_no_duplicate_labels(self):
        class Tensor:
            shape=(1,8)
            def to(self,*a):return self
            def __getitem__(self,k):return [1,2,3]
        class Tok:
            eos_token='EOS';pad_token_id=0
            def get_vocab(self):return {'a':0,'b':1}
            def __call__(self,text,return_tensors=None,**kw):return types.SimpleNamespace(input_ids=Tensor()) if return_tensors else {'input_ids':list(range(10))}
            def decode(self,*a,**kw):return 'generated passage '*25
        class Model:
            config=types.SimpleNamespace(n_positions=1024);calls=0
            def to(self,*a):return self
            def eval(self):return self
            def generate(self,*a,**kw):
                Model.calls+=1
                if Model.calls==2:raise RuntimeError('interrupt rewrite')
                return Tensor()
        fake=types.ModuleType('transformers');fake.__version__='4.57.0';fake.AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tok());fake.AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a,**k:Model());fake.LogitsProcessorList=list;fake.set_seed=lambda s:None
        ft=types.ModuleType('torch');ft.float32='float32';ft.ones_like=lambda x:x;ft.inference_mode=contextlib.nullcontext
        wm=types.ModuleType('wm.extended_watermark_processor');keys=[];wm.WatermarkLogitsProcessor=lambda **kw:keys.append(kw['hash_key'])
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);p=root/'p.jsonl';p.write_text(json.dumps({'prompt_id':'p','text':'source','split':'train'})+'\n')
            with patch.dict(sys.modules,{'transformers':fake,'torch':ft,'wm.extended_watermark_processor':wm}):
                with self.assertRaises(RuntimeError):engine.generate(config(),p,root/'run')
                engine.generate(config(),p,root/'run');engine.generate(config(),p,root/'run')
            self.assertEqual(len(engine.read(root/'run/samples.jsonl')),2)
            self.assertEqual(len(engine.read(root/'run/first_stage.jsonl')),2)
            self.assertEqual(keys,[2024,2023,2023,2023,2024])

if __name__=='__main__':unittest.main()
