"""Train OLS/SVM/MLP on10 native method scores; no invalid cross-scale baseline."""
import argparse,json,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,confusion_matrix


def train(a):
    import engine
    from common import get_X
    from train_classifier import validate_frame
    from method_order import save
    root=Path(a.case_dir);conf=json.loads((root/'config.json').read_text());spec=json.loads((root/'feature_manifest.json').read_text())
    if not (root/'method_window_spec.json').exists():raise ValueError('Expected method-order case directory')
    raw=(root/'features.csv').read_text()
    if spec['config_hash']!=engine.fp(conf['experiment']) or spec['csv_sha256']!=engine.th(raw):raise ValueError('Feature/config checksum mismatch')
    df=pd.read_csv(root/'features.csv',dtype={'idx':str,'text_hash':str});validate_frame(df)
    if len(df)!=spec['feature_rows']:raise ValueError('Incomplete table')
    X,cols=get_X(df,'windows_whole');y=df.label.to_numpy();tr=np.flatnonzero(df.split.eq('train'))
    if len(cols)!=10:raise ValueError('Expected10features')
    kinds=['linear','svm','mlp'];paths=[root/'linreg_windows_whole.joblib',root/'extra_windows_whole/svm_model.joblib',root/'extra_windows_whole/mlp_model.joblib']
    if any(p.exists() for p in paths):raise FileExistsError('Existing models preserved; use a separate case copy to refit')
    (root/'extra_windows_whole').mkdir(exist_ok=True)
    for kind,path in zip(kinds,paths):
        estimator=LinearRegression() if kind=='linear' else SVC(C=1.,kernel='rbf',gamma='scale') if kind=='svm' else MLPClassifier(hidden_layer_sizes=(32,16),alpha=.01,max_iter=500,early_stopping=False,random_state=42)
        model=make_pipeline(StandardScaler(),estimator);start=time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(X[tr],y[tr])
        report=dict(classifier=kind,columns=cols,splits={},warnings=[str(w.message) for w in caught],
            caveat='Native z-scores and SIR mean bias differ in scale. Raw stronger-score baseline deliberately omitted. Missing native scores are zero placeholders; see availability report. Closed-world two histories only.')
        for split in ['train','validation','test']:
            ix=np.flatnonzero(df.split.eq(split));xx=X[ix];yy=y[ix]
            score=model.predict(xx) if kind=='linear' else model.decision_function(xx) if kind=='svm' else model.predict_proba(xx)[:,1]
            pred=(score>=.5).astype(int) if kind=='linear' else model.predict(xx)
            report['splits'][split]=dict(n=len(ix),prompt_groups=int(df.iloc[ix].idx.nunique()),accuracy=float(accuracy_score(yy,pred)),balanced_accuracy=float(balanced_accuracy_score(yy,pred)),auc=float(roc_auc_score(yy,score)),confusion_matrix_rows_true_columns_pred_A_first_B_first=confusion_matrix(yy,pred,labels=[0,1]).tolist())
            if split=='test':
                dest=root/'test_predictions_windows_whole.csv' if kind=='linear' else path.parent/f'test_predictions_{kind}.csv'
                pd.DataFrame(dict(idx=df.iloc[ix].idx.to_numpy(),true_label=yy,predicted_label=pred,**{'linear_score' if kind=='linear' else 'score':score})).to_csv(dest,index=False)
        bundle=dict(model=model,columns=cols,features='windows_whole',experiment=conf['experiment'],watermarks=conf['watermarks'],capacity=4,window=spec['window'],kind=kind,extension='collision-order-v1')
        joblib.dump(bundle,path);report['elapsed_seconds']=time.monotonic()-start
        if kind=='mlp':pd.DataFrame(dict(epoch=np.arange(1,len(estimator.loss_curve_)+1),training_loss=estimator.loss_curve_)).to_csv(path.parent/'mlp_loss.csv',index=False)
        dest=root/'results_windows_whole.json' if kind=='linear' else path.parent/f'results_{kind}.json'
        save(dest,report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case-dir',required=True);train(p.parse_args())
