"""OLS classifier using saved prompt splits; saved model is the evaluated training fit."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,roc_auc_score
from common import get_X
import engine


def evaluate(y,s):
    pred=(s>=.5).astype(int)
    return {'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'auc':float(roc_auc_score(y,s)) if len(set(y))==2 else None,
        'confusion_matrix_rows_true_columns_pred_A_first_B_first':confusion_matrix(y,pred,labels=[0,1]).tolist(),
        'raw_regression_range':[float(min(s)),float(max(s))]}


def validate_frame(df):
    if df[['idx','label']].duplicated().any():raise ValueError('Duplicate prompt/category rows')
    if not set(df['label'])=={0,1}:raise ValueError('Need two numeric labels')
    if set(df['split'])!={'train','validation','test'}:raise ValueError('Need stored source-group splits')
    for _,rs in df.groupby('idx'):
        if len(rs)!=2 or set(rs['label'])!={0,1} or rs['split'].nunique()!=1:raise ValueError('Prompt-pair/split mismatch')
    for _,rs in df.groupby('text_hash'):
        if rs['split'].nunique()>1:raise ValueError('Identical final text crosses splits')


def train(run,mode):
    start=time.monotonic();run=Path(run);conf=json.loads((run/'config.json').read_text());spec=json.loads((run/'feature_manifest.json').read_text())
    if spec['version']!=engine.VERSION or spec['config_hash']!=engine.fp(conf['experiment']):raise ValueError('Feature/config mismatch')
    if spec['csv_sha256']!=engine.th((run/'features.csv').read_text()):raise ValueError('Features changed after extraction')
    df=pd.read_csv(run/'features.csv',dtype={'idx':str,'text_hash':str});validate_frame(df)
    if len(df)!=spec['feature_rows']:raise ValueError('Incomplete CSV')
    X,cols=get_X(df,mode);y=df['label'].to_numpy();tr=np.flatnonzero(df['split'].eq('train'));va=np.flatnonzero(df['split'].eq('validation'));te=np.flatnonzero(df['split'].eq('test'))
    mk=lambda:make_pipeline(StandardScaler(),LinearRegression())
    model=mk().fit(X[tr],y[tr]);res={'version':engine.VERSION,'features':mode,'columns':cols,'n_features':len(cols),'threshold':.5,
        'threshold_origin':'Fixed midpoint of OLS labels 0 and 1; no confidence calibration','splits':{}}
    for name,ix in [('train',tr),('validation',va),('test',te)]:
        res['splits'][name]={'n':len(ix),'prompt_groups':df.iloc[ix]['idx'].nunique(),**evaluate(y[ix],model.predict(X[ix]))}
        baseline=(df.iloc[ix]['A_whole'].to_numpy()>=df.iloc[ix]['B_whole'].to_numpy()).astype(int)
        res['splits'][name]['stronger_final_z_baseline_accuracy']=float(np.mean(baseline==y[ix]))
    groups=df.iloc[tr]['idx'].to_numpy();n=len(set(groups));cv=[]
    if n>=2:
        for a,b in GroupKFold(n_splits=min(5,n)).split(X[tr],y[tr],groups):
            m=mk().fit(X[tr[a]],y[tr[a]]);cv.append(evaluate(y[tr[b]],m.predict(X[tr[b]])))
    res['cv_training_groups_only']=cv
    res['caveat']='Closed world, exactly both watermark stages in one of two orders. OLS scores are not probabilities. Inspect GPT-2 paraphrase quality. No refit on test data.'
    model_path=run/f'linreg_{mode}.joblib'
    if model_path.exists():raise FileExistsError('Existing model preserved; use a separate run copy for another fit')
    bundle={'version':engine.VERSION,'model':model,'columns':cols,'features':mode,'experiment':conf['experiment'],
        'watermarks':conf['watermarks'],'window':spec['window'],'capacity':spec['capacity']}
    tmp=model_path.with_suffix('.joblib.tmp');joblib.dump(bundle,tmp);tmp.replace(model_path)
    pd.DataFrame({'feature':cols,'coef_on_standardized_x':model[-1].coef_}).to_csv(run/f'coefficients_{mode}.csv',index=False)
    pred=model.predict(X[te]);pd.DataFrame({'idx':df.iloc[te]['idx'].to_numpy(),'true_label':y[te],
        'linear_score':pred,'predicted_label':(pred>=.5).astype(int)}).to_csv(run/f'test_predictions_{mode}.csv',index=False)
    res['elapsed_seconds']=time.monotonic()-start;engine.save(run/f'results_{mode}.json',res)
    print(json.dumps(res,indent=2));return bundle,res


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--features',choices=['windows_whole','windows','whole','halves4'],default='windows_whole')
    a=p.parse_args();train(a.run_dir,a.features)

if __name__=='__main__':main()
