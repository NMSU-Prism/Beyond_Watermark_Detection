"""Compare dynamic K sections and K fixed token windows on saved order data."""
import argparse,json,math,time,hashlib,warnings,os,fcntl
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,roc_auc_score,roc_curve,accuracy_score,balanced_accuracy_score,confusion_matrix
VERSION='window-sweep-windows-only-v1'


def fp(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def save(p,r):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def append(p,r):
 with Path(p).open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def spans(n,k,width=None):
 if width is not None:return [(min(i*width,n),min((i+1)*width,n)) for i in range(k)]
 q,r=divmod(n,k);ends=[0]
 for i in range(k):ends.append(ends[-1]+q+int(i<r))
 return list(zip(ends[:-1],ends[1:]))


def aggregate(kind,ev,gamma):
 if not ev:return 0.,0
 vals=list({tuple(r['unit']):r['value'] for r in ev}.values()) if kind=='KGW' else [r['value'] for r in ev]
 if kind=='SIR':return float(np.mean(vals)),len(vals)
 return float((sum(vals)-gamma*len(vals))/math.sqrt(len(vals)*gamma*(1-gamma))),len(vals)


def validate(df):
 if df.duplicated(['idx','label']).any():raise ValueError('Duplicate prompt/label')
 for _,g in df.groupby('idx'):
  if len(g)!=2 or set(g.label)!={0,1} or g.split.nunique()!=1:raise ValueError('Prompt pairs/splits invalid')
 if set(df.split)!={'train','validation','test'}:raise ValueError('Need all3originalsplits')
 if df.groupby('text_hash').split.nunique().max()>1:raise ValueError('Identical final text crosses splits')


def normalized_rows(raw):
 result=[]
 for r in raw:
  label={'A_first':0,'B_first':1}[r['label']] if isinstance(r['label'],str) else int(r['label'])
  if label not in [0,1]:raise ValueError('Invalid label')
  result.append(dict(sample_id=r['sample_id'],idx=str(r['prompt_id']),label=label,
   category=['A_first','B_first'][label],split=r['split'],text_hash=hashlib.sha256(r['final_text'].encode()).hexdigest()))
 validate(pd.DataFrame(result));return result


def load_evidence(root,out,c,raw):
 from engine import recover
 normalized=normalized_rows(raw)
 journal=out/'evidence.jsonl';previous=recover(journal);done={r['sample_id']:r for r in previous}
 if len(done)!=len(previous) or not set(done)<={r['sample_id'] for r in raw}:raise ValueError('Bad cache IDs')
 native=root/'scoring/journal.jsonl';manifest=root/'scoring/manifest.json';reused=0
 if native.exists() and manifest.exists():
  old=json.loads(manifest.read_text())
  if old.get('config_hash')!=fp(c) or old.get('samples_hash')!=fp(raw):raise ValueError('Existing method scoring/config/data mismatch')
  for r in read(native):
   if r['sample_id'] not in done:
    append(journal,r);done[r['sample_id']]=r;reused+=1
  print(json.dumps({'reused_native_evidence':reused}),flush=True)
 started=time.monotonic();added=0;backends=None
 for r in raw:
  if r['sample_id'] in done:continue
  if backends is None:
   from transformers import AutoTokenizer,AutoConfig
   from method_backends import Backend
   tok=AutoTokenizer.from_pretrained(c['A']['model'],revision=c['A']['revision'],use_fast=True)
   vocab=c.get('vocab_size') or AutoConfig.from_pretrained(c['A']['model'],revision=c['A']['revision']).vocab_size
   methods=c.get('methods') or {h:dict(identity=h,method='KGW',key=c[h]['key'],gamma=c['gamma'],delta=c['delta'],seeding_scheme=c['seeding_scheme']) for h in ['A','B']}
   backends={h:Backend(methods[h],tok,vocab,c['device'],False) for h in ['A','B']}
  record=dict(sample_id=r['sample_id'],values={h:backends[h].score(r['final_text']) for h in ['A','B']})
  append(journal,record);done[r['sample_id']]=record;added+=1
  if added==1 or added%25==0 or len(done)==len(raw):
   elapsed=time.monotonic()-started
   print(json.dumps({'scored':len(done),'total':len(raw),'session_seconds':round(elapsed,2),'rough_eta_seconds':round(elapsed/added*(len(raw)-len(done)),2)}),flush=True)
 return normalized,done


def table(normalized,evidence,c,k,width):
 rows=[];cols=[]
 if k:cols += [f'{h}_w{i+1}' for h in ['A','B'] for i in range(k)]
 for meta in normalized:
  row=meta.copy()
  for h in ['A','B']:
   v=evidence[meta['sample_id']]['values'][h]
   method=c.get('methods',{}).get(h,dict(method='KGW',gamma=c.get('gamma',.25)))
   row[h+'_whole']=v['whole_score'] if v['whole_score'] is not None else 0.
   row[h+'_whole_available']=int(v['whole_score'] is not None);row[h+'_n_tokens']=v['n_tokens']
   for i,(a,b) in enumerate(spans(v['n_tokens'],k,width) if k else []):
    ev=[r for r in v['evidence'] if a<=r['position']<b]
    z,n=aggregate(method['method'],ev,method['gamma'])
    row[f'{h}_w{i+1}']=z;row[f'{h}_m{i+1}']=int(n>0);row[f'{h}_effective_n{i+1}']=n
   covered=min(v['n_tokens'],k*width) if k and width else v['n_tokens']
   row[h+'_window_omitted_tokens']=v['n_tokens']-covered
  rows.append(row)
 df=pd.DataFrame(rows);validate(df)
 df=df.drop(columns=['A_whole','B_whole','A_whole_available','B_whole_available'])
 if not np.isfinite(df[cols].to_numpy()).all():raise ValueError('Nonfinite features')
 return df,cols


def metrics(y,pred,s):
 report=classification_report(y,pred,labels=[0,1],target_names=['A_first','B_first'],output_dict=True,zero_division=0)
 return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
  auc=float(roc_auc_score(y,s)),macro_precision=report['macro avg']['precision'],macro_recall=report['macro avg']['recall'],macro_f1=report['macro avg']['f1-score'],
  classification_report=report,confusion_matrix_rows_true_columns_pred_A_first_B_first=confusion_matrix(y,pred,labels=[0,1]).tolist())


def fit_case(dest,df,cols,config,spec):
 dest.mkdir(exist_ok=True);df.to_csv(dest/'features.csv',index=False)
 summary=[];X=df[cols].to_numpy(float);y=df.label.to_numpy();tr=np.flatnonzero(df.split.eq('train'))
 for kind in ['linear','svm','mlp']:
  reportpath=dest/f'results_{kind}.json'
  if (dest/f'{kind}_COMPLETE.json').exists():
   report=json.loads(reportpath.read_text());summary.extend(report['summary_rows']);continue
  start=time.monotonic()
  estimator=LinearRegression() if kind=='linear' else SVC(C=1.,kernel='rbf',gamma='scale') if kind=='svm' else MLPClassifier(hidden_layer_sizes=(32,16),alpha=.01,max_iter=500,early_stopping=False,random_state=42)
  model=make_pipeline(StandardScaler(),estimator)
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always');model.fit(X[tr],y[tr])
  report=dict(columns=cols,specification=spec,model=kind,warnings=[str(w.message) for w in ws],splits={},summary_rows=[])
  for split in ['train','validation','test']:
   ix=np.flatnonzero(df.split.eq(split));xx=X[ix];yy=y[ix]
   score=model.predict(xx) if kind=='linear' else model.decision_function(xx) if kind=='svm' else model.predict_proba(xx)[:,1]
   pred=(score>=.5).astype(int) if kind=='linear' else model.predict(xx)
   m=metrics(yy,pred,score);report['splits'][split]=m
   entry=dict(layout=spec['layout'],windows=spec['k'],token_width=spec['width'],n_features=len(cols),model=kind,split=split,
    **{key:m[key] for key in ['n','accuracy','balanced_accuracy','auc','macro_precision','macro_recall','macro_f1']})
   summary.append(entry);report['summary_rows'].append(entry)
   pd.DataFrame(dict(idx=df.iloc[ix].idx.to_numpy(),true_label=yy,predicted_label=pred,score=score)).to_csv(dest/f'{split}_predictions_{kind}.csv',index=False)
   if split=='test':
    fpr,tpr,thresholds=roc_curve(yy,score,pos_label=1)
    pd.DataFrame(dict(fpr=fpr,tpr=tpr,threshold=thresholds)).to_csv(dest/f'roc_points_{kind}.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,5));ax.plot(fpr,tpr,label=f'{kind} AUC={m["auc"]:.4f}');ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'{spec["layout"]}: positive class B_first');ax.legend();fig.tight_layout()
    fig.savefig(dest/f'roc_{kind}.png',dpi=180);fig.savefig(dest/f'roc_{kind}.pdf');plt.close(fig)
  joblib.dump(dict(version=VERSION,model=model,kind=kind,columns=cols,experiment=config,specification=spec),dest/f'model_{kind}.joblib')
  report['elapsed_seconds']=time.monotonic()-start;save(reportpath,report)
  if kind=='mlp':pd.DataFrame({'epoch':np.arange(1,len(estimator.loss_curve_)+1),'loss':estimator.loss_curve_}).to_csv(dest/'mlp_loss.csv',index=False)
  save(dest/f'{kind}_COMPLETE.json',{'completed':True})
  print(json.dumps(dict(layout=spec['layout'],model=kind,elapsed_seconds=round(report['elapsed_seconds'],2),validation_accuracy=report['splits']['validation']['accuracy'],test_accuracy=report['splits']['test']['accuracy'])),flush=True)
 return summary


def comparison(summary,out):
 df=pd.DataFrame(summary);df.to_csv(out/'all_results.csv',index=False)
 df[df.split.eq('test')].to_csv(out/'test_comparison.csv',index=False)
 val=df[df.split.eq('validation')].sort_values(['balanced_accuracy','macro_f1','n_features'],ascending=[False,False,True])
 val.to_csv(out/'validation_comparison.csv',index=False)
 save(out/'validation_selected_candidate.json',val.iloc[0].astype(object).where(pd.notna(val.iloc[0]),None).to_dict())
 for split in ['validation','test']:
  selected=df[df.split.eq(split)];fig,ax=plt.subplots(figsize=(12,5))
  for kind,g in selected.groupby('model',sort=False):
   ax.plot(g.layout,g.accuracy,'o-',label=kind)
  ax.set(ylabel='Accuracy',title=f'Window representations — {split}');ax.tick_params(axis='x',rotation=65);ax.legend();ax.grid(alpha=.2);fig.tight_layout()
  fig.savefig(out/f'accuracy_{split}.png',dpi=180);fig.savefig(out/f'accuracy_{split}.pdf');plt.close(fig)
 print('\nVALIDATION RESULTS\n'+val[['layout','model','n_features','accuracy','macro_f1','auc']].to_string(index=False))
 print('\nSaved comparisons:',out)


def run(a):
 root=Path(a.run_dir);out=Path(a.out);conf=json.loads((root/'config.json').read_text());c=conf['experiment']
 if not (root/'COMPLETE.json').exists():raise ValueError('Finish source generation first')
 raw=read(root/'samples.jsonl');normalized_rows(raw)
 if any(k<1 for k in a.counts):raise ValueError('Window counts must be positive')
 if any(w<1 for w in a.widths):raise ValueError('Window widths must be positive')
 if len(set(a.counts))!=len(a.counts) or len(set(a.widths))!=len(a.widths):raise ValueError('Duplicate counts/widths')
 specification=dict(version=VERSION,config_hash=fp(c),samples_hash=fp(raw),counts=a.counts,widths=a.widths,include_whole=a.include_whole,
  overflow_policy='Fixed slots cover prefix only; whole scores excluded from classifier',models='OLS;RBF SVC C1 gamma-scale;MLP32,16 alpha.01 max500 seed42',masks_in_model=False)
 out.mkdir(parents=True,exist_ok=True)
 with (out/'sweep.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  p=out/'sweep_manifest.json'
  if p.exists() and json.loads(p.read_text())!=specification:raise ValueError('Sweep resume mismatch; use new --out')
  save(p,specification)
  normalized,ev=load_evidence(root,out,c,raw)
  layouts=([('whole',0,None)] if a.include_whole else [])
  layouts += [(f'dynamic{k}',k,None) for k in a.counts]
  layouts += [(f'fixed{width}x{k}',k,width) for width in a.widths for k in a.counts]
  all_summary=[];start=time.monotonic()
  for index,(name,k,width) in enumerate(layouts,1):
   df,cols=table(normalized,ev,c,k,width);spec=dict(layout=name,k=k,width=width,whole_scores_included=False,columns=cols)
   dest=out/name;dest.mkdir(exist_ok=True);save(dest/'layout.json',spec)
   coverage={h:dict(mean_tokens=float(df[h+'_n_tokens'].mean()),fraction_with_omitted_window_tokens=float((df[h+'_window_omitted_tokens']>0).mean()),mean_omitted_window_tokens=float(df[h+'_window_omitted_tokens'].mean())) for h in ['A','B']}
   save(dest/'coverage.json',coverage)
   all_summary+=fit_case(dest,df,cols,c,spec)
   print(json.dumps({'layouts_completed':index,'layouts_total':len(layouts),'training_session_seconds':round(time.monotonic()-start,2)}),flush=True)
   comparison_df=pd.DataFrame(all_summary);comparison_df.to_csv(out/'all_results_partial.csv',index=False)
  comparison(all_summary,out);save(out/'COMPLETE.json',{'layouts':len(layouts),'classifiers':3*len(layouts)})


def self_test():
 assert [b-a for a,b in spans(30,4)]==[8,8,7,7]
 assert spans(130,4,25)[-1]==(75,100)
 assert len(spans(30,9))==9
 ev=[{'position':i,'value':int(i%3==0),'unit':[i]} for i in range(130)]
 record={'whole_score':2.,'n_tokens':130,'evidence':ev}
 meta=[dict(sample_id=f'{i}_{y}',idx=str(i),label=y,category=['A_first','B_first'][y],split=s,text_hash=f'{i}_{y}') for i,s in enumerate(['train','validation','test']) for y in [0,1]]
 cache={r['sample_id']:{'values':{'A':record,'B':record}} for r in meta}
 for k in range(1,10):
  for width in [None,25,59]:
   df,cols=table(meta,cache,{'gamma':.25},k,width)
   assert len(cols)==2*k and 'A_whole' not in df and 'B_whole' not in df
   assert df.A_window_omitted_tokens.eq(max(0,130-k*width) if width else 0).all()
 m=metrics(np.array([0,0,1,1]),np.array([0,1,1,1]),np.array([-.1,.6,.8,1.2]))
 assert m['accuracy']==.75 and m['auc']==1
 print('PASS boundaries, explicit prefix coverage, feature counts, whole-score exclusion, and metrics. Synthetic only.')


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir');p.add_argument('--out')
 p.add_argument('--counts',nargs='+',type=int,default=list(range(4,10)));p.add_argument('--widths',nargs='+',type=int,default=[59,25]);p.set_defaults(include_whole=False);p.add_argument('--self-test',action='store_true')
 a=p.parse_args()
 if a.self_test:self_test()
 elif not a.run_dir or not a.out:p.error('--run-dir and --out required')
 else:run(a)
