from __future__ import annotations
import os, json, math, textwrap, zipfile, shutil, warnings, sys, platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from scipy.stats import friedmanchisquare, wilcoxon
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED']='42'
SEED=42
MAX_TRAIN_SAMPLES=1000
INPUT=Path('/mnt/data/Geomech_4Wells_Before_After(5).xlsx')
OUT=Path('/mnt/data/geomech_outputs_final')
PKG=Path('/mnt/data/geomech_ml_package_final')
OUT.mkdir(exist_ok=True)
(PKG/'models').mkdir(parents=True,exist_ok=True)

WELL_MAP={'F2':'Well A','F6':'Well B','F8':'Well C','F15':'Well D'}
ALIASES={
'Depth_m':['Depth_m'],'BitSize':['Input_BS'],'GR':['Input_GR'],'RT':['Input_RT'],
'Input_PHIE':['Input_PHIE','Input_PHIEV_V'],'Input_PHIT':['Input_PHIT','Input_PHITV_V'],
'DT_used_usft':['DT_used_usft','Input_DT','Input_DTUS_F'],
'PHI_before_vv':['PHI_before_vv'],'PHI_after_vv':['PHI_after_vv'],'Delta_PHI_pu':['Delta_PHI_pu'],
'RHOB_before_gcc':['RHOB_before_gcc','Input_RHOB','Input_RHOBG_C3'],
'RHOB_after_gcc':['RHOB_after_gcc'],'Delta_RHOB_gcc':['Delta_RHOB_gcc'],
'Vsh':['Vsh','Input_Vsh','Input_VSH'],'Calcite':['Calcite','Input_CALCITE'],
'Dolomite':['Dolomite','Input_DOLOM'],'Quartz':['Quartz','Input_QUARTZ'],
'Vp_km_s':['Vp_km_s'],'Vs_km_s':['Vs_km_s'],
'E_before_GPa':['E_before_GPa'],'E_after_GPa':['E_after_GPa'],'Delta_E_GPa':['Delta_E_GPa'],
'G_before_GPa':['G_before_GPa'],'G_after_GPa':['G_after_GPa'],'Delta_G_GPa':['Delta_G_GPa'],
'K_before_GPa':['K_before_GPa'],'K_after_GPa':['K_after_GPa'],'Delta_K_GPa':['Delta_K_GPa'],
'Nu_before':['Nu_before'],'Nu_after':['Nu_after'],'Delta_Nu':['Delta_Nu'],
'Biot_before':['Biot_before'],'Biot_after':['Biot_after'],'Delta_Biot':['Delta_Biot'],
'Sv_before_MPa':['Sv_before_MPa'],'Sv_after_MPa':['Sv_after_MPa'],'Delta_Sv_MPa':['Delta_Sv_MPa']}
NUM_FEATURES=list(ALIASES)
TARGETS={
'Shmin_Eaton':('Shmin_Eaton_before_MPa','Shmin_Eaton_after_MPa'),
'SHmax_Eaton':('SHmax_Eaton_before_MPa','SHmax_Eaton_after_MPa'),
'Shmin_Bowers':('Shmin_Bowers_before_MPa','Shmin_Bowers_after_MPa'),
'SHmax_Bowers':('SHmax_Bowers_before_MPa','SHmax_Bowers_after_MPa')}

def build_data():
    frames=[]
    for original,anon in WELL_MAP.items():
        raw=pd.read_excel(INPUT,sheet_name=original)
        d=pd.DataFrame(index=raw.index)
        d['Well']=anon; d['Original_Well']=original
        d['Sample_ID']=[f"{anon.replace(' ','')}_{i+1:05d}" for i in range(len(raw))]
        d['Lithology']=raw['Lithology'].astype('string').fillna('Unknown').str.strip().replace('','Unknown')
        low=d['Lithology'].str.lower()
        d['Lithology_Group']=np.select(
            [low.str.contains('lime'),low.str.contains('dolom'),low.str.contains('sand'),low.str.contains('shale'),low.str.contains('marl')],
            ['Limestone','Dolomite','Sandstone','Shale','Marl'],default=d['Lithology'].astype(str))
        for new,cands in ALIASES.items():
            found=next((c for c in cands if c in raw.columns),None)
            d[new]=pd.to_numeric(raw[found],errors='coerce') if found else np.nan
        for t,(p,r) in TARGETS.items():
            d[f'{t}_Prior']=pd.to_numeric(raw[p],errors='coerce')
            d[f'{t}_Reference']=pd.to_numeric(raw[r],errors='coerce')
            d[f'{t}_TrueResidual']=d[f'{t}_Reference']-d[f'{t}_Prior']
        frames.append(d)
    data=pd.concat(frames,ignore_index=True).replace([np.inf,-np.inf],np.nan)
    required=['Depth_m']+[f'{t}_{x}' for t in TARGETS for x in ('Prior','Reference','TrueResidual')]
    return data.dropna(subset=required).reset_index(drop=True)

def make_models():
    return {
      'Ridge':Ridge(alpha=1.0),
      'KNN':KNeighborsRegressor(n_neighbors=15,weights='distance',p=2,n_jobs=-1),
      'SVR':SVR(C=5.0,epsilon=0.015,gamma='scale',kernel='rbf',cache_size=1200),
      'Random Forest':RandomForestRegressor(n_estimators=30,max_depth=18,min_samples_leaf=2,max_features='sqrt',random_state=SEED,n_jobs=-1),
      'Extra Trees':ExtraTreesRegressor(n_estimators=30,max_depth=None,min_samples_leaf=2,max_features=0.8,random_state=SEED,n_jobs=-1),
      'AdaBoost':AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=4,min_samples_leaf=3,random_state=SEED),n_estimators=40,learning_rate=0.06,loss='linear',random_state=SEED),
      'XGBoost':XGBRegressor(n_estimators=25,max_depth=4,learning_rate=0.08,subsample=0.85,colsample_bytree=0.85,reg_lambda=1.0,objective='reg:squarederror',tree_method='hist',max_bin=128,random_state=SEED,n_jobs=4,verbosity=0),
      'LightGBM':LGBMRegressor(n_estimators=30,num_leaves=31,learning_rate=0.07,subsample=0.85,colsample_bytree=0.85,reg_lambda=1.0,random_state=SEED,n_jobs=4,verbosity=-1),
      'CatBoost':CatBoostRegressor(iterations=30,depth=6,learning_rate=0.07,loss_function='RMSE',random_seed=SEED,verbose=False,allow_writing_files=False,thread_count=4),
      'MLP':MLPRegressor(hidden_layer_sizes=(48,24),activation='relu',solver='adam',alpha=0.001,learning_rate_init=0.001,max_iter=80,early_stopping=True,validation_fraction=0.15,n_iter_no_change=15,random_state=SEED)}

def run():
    data=build_data(); feature_cols=NUM_FEATURES+['Lithology_Group']; wells=list(WELL_MAP.values())
    prep_template=ColumnTransformer([
      ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),NUM_FEATURES),
      ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),['Lithology_Group'])],remainder='drop',sparse_threshold=0)
    model_names=list(make_models())
    for t in TARGETS:
      for m in model_names:
        for suffix in ('PredResidual','PredStress','Error'):
          data[f'{t}_{m}_{suffix}']=np.nan
    task=[]; prep_rows=[]
    for blind in wells:
      train=data['Well']!=blind; test=data['Well']==blind
      train_frame=data.loc[train].sort_values(['Well','Depth_m'])
      if len(train_frame)>MAX_TRAIN_SAMPLES:
        sel=np.unique(np.linspace(0,len(train_frame)-1,MAX_TRAIN_SAMPLES,dtype=int))
        fit_index=train_frame.index.to_numpy()[sel]
      else:
        fit_index=train_frame.index.to_numpy()
      prep=clone(prep_template)
      prep.fit(data.loc[train,feature_cols])
      Xtr=prep.transform(data.loc[fit_index,feature_cols]); Xte=prep.transform(data.loc[test,feature_cols])
      imp=prep.named_transformers_['num'].named_steps['imputer']; sc=prep.named_transformers_['num'].named_steps['scaler']
      for f,med,mean,sd in zip(NUM_FEATURES,imp.statistics_,sc.mean_,sc.scale_):
        prep_rows.append({'Blind_Well':blind,'Feature':f,'Training_Median':float(med),'Scaled_Mean':float(mean),'Scaled_SD':float(sd)})
      for t in TARGETS:
        ytr=data.loc[fit_index,f'{t}_TrueResidual'].to_numpy(); prior=data.loc[test,f'{t}_Prior'].to_numpy(); ref=data.loc[test,f'{t}_Reference'].to_numpy()
        for name,model in make_models().items():
          model.fit(Xtr,ytr); pr=np.asarray(model.predict(Xte),float); ps=prior+pr; err=ps-ref
          data.loc[test,f'{t}_{name}_PredResidual']=pr; data.loc[test,f'{t}_{name}_PredStress']=ps; data.loc[test,f'{t}_{name}_Error']=err
          task.append({'Model':name,'Blind_Well':blind,'Target':t,'N':int(test.sum()),'RMSE_MPa':float(np.sqrt(mean_squared_error(ref,ps))),'MAE_MPa':float(mean_absolute_error(ref,ps)),'R2':float(r2_score(ref,ps)),'Bias_MPa':float(np.mean(err))})
      print('Completed',blind,flush=True)
    task=pd.DataFrame(task); task['Task_Rank']=task.groupby(['Blind_Well','Target'])['RMSE_MPa'].rank(method='average')
    agg=(task.groupby('Model',sort=False).agg(Mean_RMSE_MPa=('RMSE_MPa','mean'),SD_RMSE_MPa=('RMSE_MPa','std'),Mean_MAE_MPa=('MAE_MPa','mean'),Mean_R2=('R2','mean'),Mean_Abs_Bias_MPa=('Bias_MPa',lambda s:np.mean(np.abs(s))),Mean_Task_Rank=('Task_Rank','mean')).reset_index().sort_values('Mean_RMSE_MPa').reset_index(drop=True))
    agg.insert(0,'RMSE_Rank',np.arange(1,len(agg)+1))
    pivot=task.pivot(index=['Blind_Well','Target'],columns='Model',values='RMSE_MPa')
    fs,fp=friedmanchisquare(*[pivot[m].to_numpy() for m in model_names]); best=agg.iloc[0]['Model']; second=agg.iloc[1]['Model']; ws,wp=wilcoxon(pivot[best],pivot[second],zero_method='wilcox')
    stats=pd.DataFrame({'Statistic':['Best_Model','Best_Mean_RMSE_MPa','Second_Model','Second_Mean_RMSE_MPa','Friedman_ChiSquare','Friedman_p','Wilcoxon_Best_vs_Second_W','Wilcoxon_Best_vs_Second_p','Number_of_Tasks'],'Value':[best,float(agg.iloc[0]['Mean_RMSE_MPa']),second,float(agg.iloc[1]['Mean_RMSE_MPa']),float(fs),float(fp),float(ws),float(wp),int(len(pivot))]})
    # CSV outputs
    canonical_cols=['Sample_ID','Well','Original_Well','Depth_m','Lithology','Lithology_Group']+NUM_FEATURES[1:]
    for t in TARGETS: canonical_cols += [f'{t}_Prior',f'{t}_Reference',f'{t}_TrueResidual']
    data[canonical_cols].to_csv(OUT/'Canonical_Logs.csv',index=False)
    for t in TARGETS:
      cols=['Sample_ID','Well','Depth_m','Lithology',f'{t}_Prior',f'{t}_Reference',f'{t}_TrueResidual']
      for m in model_names: cols += [f'{t}_{m}_PredResidual',f'{t}_{m}_PredStress',f'{t}_{m}_Error']
      data[cols].to_csv(OUT/f'Predictions_{t}.csv',index=False)
    task.to_csv(OUT/'Task_Metrics.csv',index=False); agg.to_csv(OUT/'Aggregate_Metrics.csv',index=False); pd.DataFrame(prep_rows).to_csv(OUT/'Preprocess_Parameters.csv',index=False); stats.to_csv(OUT/'Statistics.csv',index=False)
    hypers=[]
    for n,m in make_models().items(): hypers.append({'Model':n,'Parameters':json.dumps(m.get_params(),default=str,ensure_ascii=False,sort_keys=True)})
    pd.DataFrame(hypers).to_csv(OUT/'Hyperparameters.csv',index=False)
    manifest={'Input_File':INPUT.name,'Samples':len(data),'Wells':wells,'Targets':list(TARGETS),'Models':model_names,'Validation':'Leave-One-Well-Out','Random_State':SEED,'Max_Training_Samples_Per_Fold':MAX_TRAIN_SAMPLES,'Best_Model':best,'Best_Mean_RMSE_MPa':float(agg.iloc[0]['Mean_RMSE_MPa']),'Python':sys.version,'Platform':platform.platform()}
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return data,task,agg,stats,manifest

if __name__=='__main__':
    data,task,agg,stats,manifest=run()
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
