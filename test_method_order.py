"""CPU-only semantic tests; heavy model execution is not implied by these tests."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from method_backends import Backend,bounds,aggregate,feature_scores


class MethodTests(unittest.TestCase):
    def test_boundaries_and_aggregation(self):
        self.assertEqual([b-a for a,b in bounds(30,'dynamic4')],[8,8,7,7])
        with self.assertRaises(ValueError):bounds(237,'fixed59x4')
        ev=[dict(position=0,value=1,unit=[1,2]),dict(position=1,value=1,unit=[1,2])]
        _,kgw_n=aggregate('KGW',ev,.25);_,prw_n=aggregate('PRW',ev,.25)
        self.assertEqual((kgw_n,prw_n),(1,2))
        v=feature_scores('SIR',dict(n_tokens=30,whole_score=None,evidence=[]),.5,'fixed59x4')
        self.assertEqual(v['window_available'],[0]*4);self.assertEqual(v['whole_available'],0)

    def test_native_prw_mask_processor_and_score(self):
        class Array(np.ndarray):
            def to(self,*args):return self
        fake=types.ModuleType('torch');fake.Tensor=Array;fake.FloatTensor=Array;fake.float32=np.float32
        fake.tensor=lambda x,dtype=None:np.asarray(x,dtype=dtype).view(Array)
        transformers=types.ModuleType('transformers');transformers.LogitsWarper=object
        path=Path(__file__).parent/'collision_vendor/prw/gptwm.py'
        with patch.dict(sys.modules,{'torch':fake,'transformers':transformers}):
            spec=importlib.util.spec_from_file_location('prw_test_native',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
            kw=dict(fraction=.25,strength=2.,vocab_size=100,watermark_key=2024)
            det=m.GPTWatermarkDetector(**kw);processor=m.GPTWatermarkLogitsWarper(**kw)
            self.assertEqual(int(det.green_list_mask.sum()),25)
            self.assertTrue(np.array_equal(det.green_list_mask,processor.green_list_mask))
            self.assertFalse(np.array_equal(det.green_list_mask,m.GPTWatermarkDetector(**dict(kw,watermark_key=2023)).green_list_mask))
            ids=[0,1,1,2,3,4,5,6,7,8]
            evidence=[dict(position=i,value=int(det.green_list_mask[t])) for i,t in enumerate(ids)]
            self.assertAlmostEqual(float(det.detect(ids)),aggregate('PRW',evidence,.25)[0])
            logits=np.zeros((1,100)).view(Array)
            self.assertTrue(np.array_equal(processor(None,logits),2*det.green_list_mask[None,:]))

    def test_sir_native_score_alignment_and_missing(self):
        # Execute the actual upstream detect/split/scale methods with a lightweight
        # embedding/transform fixture; no replacement of the score formula.
        jieba=types.SimpleNamespace(cut=lambda text:list(text))
        tree=ast.parse((Path(__file__).parent/'collision_vendor/sir/watermark.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WatermarkContext')
        node.bases=[];node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in ['detect','get_text_split','scale_vector']]
        scope={'np':np,'jieba':jieba}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'native_sir_subset','exec'),scope)
        class Tok:
            def __call__(self,text,**kw):
                return dict(input_ids=[ord(x)%300 for x in text],offset_mapping=[(i,i+1) for i in range(len(text))])
            def encode(self,text,**kw):return [[ord(x)%300 for x in text]]
        class Value:
            def __init__(self,a):self.a=a
            def cpu(self):return self
            def __getitem__(self,i):return Value(self.a[i])
            def numpy(self):return self.a
        wm=scope['WatermarkContext'].__new__(scope['WatermarkContext']);wm.chunk_length=10;wm.mapping=list(range(300));wm.target_tokenizer=Tok()
        wm.get_embedding=lambda text:len(text)
        wm.transform_model=lambda n:Value(np.sin(np.arange(300)[None,:]+n))
        backend=Backend.__new__(Backend);backend.kind='SIR';backend.tok=wm.target_tokenizer;backend.detector=wm;backend.spec={'gamma':.5}
        with patch.dict(sys.modules,{'jieba':jieba}):
            text='Mango ripening changes starch into sugars and preserves the sweet taste. '*4
            record=backend._score(text,np)
            self.assertAlmostEqual(record['whole_score'],float(wm.detect(text)))
            self.assertTrue(all(0<=r['position']<len(text) for r in record['evidence']))
            self.assertIsNone(backend._score('short text',np)['whole_score'])


if __name__=='__main__':unittest.main()
