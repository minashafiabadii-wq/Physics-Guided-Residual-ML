from __future__ import annotations
import sys, json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,'/mnt/data')
from Geomech_ML_reproducible_pipeline import build_data, make_models, NUM_FEATURES, TARGETS, WELL_MAP, MAX_TRAIN_SAMPLES
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings('ignore')

blind=sys.argv[1]
outdir=Path('/mnt/data/geomech_fold_outputs'); outdir.mkdir(exist_ok=True)
data=build_data(); features=NUM_FEATURES+['Lithology_Group']
train=data['Well']!=blind; test=data['Well']==blind
train_frame=data.loc[train].sort_values(['Well','Depth_m'])
if len(train_frame)>MAX_TRAIN_SAMPLES:
    sel=np.unique(np.linspace(0,len(train_frame)-1,MAX_TRAIN_SAMPLES,dtype=int)); fit_index=train_frame.index.to_numpy()[sel]
else: fit_index=train_frame.index.to_numpy()
prep=ColumnTransformer([
 ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),NUM_FEATURES),
 ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),['Lithology_Group'])],remainder='drop',sparse_threshold=0)
prep.fit(data.loc[train,features]); Xtr=prep.transform(data.loc[fit_index,features]); Xte=prep.transform(data.loc[test,features])
out=data.loc[test,['Sample_ID','Well','Depth_m','Lithology']].copy(); metrics=[]
for target in TARGETS:
    prior=data.loc[test,f'{target}_Prior'].to_numpy(); ref=data.loc[test,f'{target}_Reference'].to_numpy(); ytr=data.loc[fit_index,f'{target}_TrueResidual'].to_numpy()
    out[f'{target}_Prior']=prior; out[f'{target}_Reference']=ref; out[f'{target}_TrueResidual']=ref-prior
    for name,model in make_models().items():
        print(blind,target,name,flush=True)
        model.fit(Xtr,ytr); pr=np.asarray(model.predict(Xte),float); ps=prior+pr; err=ps-ref
        out[f'{target}_{name}_PredResidual']=pr; out[f'{target}_{name}_PredStress']=ps; out[f'{target}_{name}_Error']=err
        metrics.append({'Model':name,'Blind_Well':blind,'Target':target,'N':len(out),'RMSE_MPa':float(np.sqrt(mean_squared_error(ref,ps))),'MAE_MPa':float(mean_absolute_error(ref,ps)),'R2':float(r2_score(ref,ps)),'Bias_MPa':float(np.mean(err))})
out.to_csv(outdir/f"{blind.replace(' ','_')}_predictions.csv",index=False)
pd.DataFrame(metrics).to_csv(outdir/f"{blind.replace(' ','_')}_metrics.csv",index=False)
# preprocessing records
imp=prep.named_transformers_['num'].named_steps['imputer']; sc=prep.named_transformers_['num'].named_steps['scaler']
pd.DataFrame([{'Blind_Well':blind,'Feature':f,'Training_Median':float(m),'Scaled_Mean':float(mu),'Scaled_SD':float(sd)} for f,m,mu,sd in zip(NUM_FEATURES,imp.statistics_,sc.mean_,sc.scale_)]).to_csv(outdir/f"{blind.replace(' ','_')}_preprocess.csv",index=False)
print('SAVED',blind,len(out),flush=True)
