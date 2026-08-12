from pathlib import Path
import sys, json, zipfile, shutil, platform
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
sys.path.insert(0,'/mnt/data')
from Geomech_ML_reproducible_pipeline import build_data, TARGETS, NUM_FEATURES, make_models, MAX_TRAIN_SAMPLES

fold_dir=Path('/mnt/data/geomech_fold_outputs')
out=Path('/mnt/data/geomech_outputs_final'); out.mkdir(exist_ok=True)
pkg=Path('/mnt/data/geomech_ml_package_final'); (pkg/'models').mkdir(parents=True,exist_ok=True)

# Merge fold outputs and metrics
pred_files=sorted(fold_dir.glob('Well_*_predictions.csv'))
met_files=sorted(fold_dir.glob('Well_*_metrics.csv'))
prep_files=sorted(fold_dir.glob('Well_*_preprocess.csv'))
all_pred=pd.concat([pd.read_csv(f) for f in pred_files],ignore_index=True).sort_values(['Well','Depth_m']).reset_index(drop=True)
task=pd.concat([pd.read_csv(f) for f in met_files],ignore_index=True)
prep=pd.concat([pd.read_csv(f) for f in prep_files],ignore_index=True)
model_order=list(make_models())
task['Task_Rank']=task.groupby(['Blind_Well','Target'])['RMSE_MPa'].rank(method='average')
agg=(task.groupby('Model',sort=False).agg(Mean_RMSE_MPa=('RMSE_MPa','mean'),SD_RMSE_MPa=('RMSE_MPa','std'),Mean_MAE_MPa=('MAE_MPa','mean'),Mean_R2=('R2','mean'),Mean_Abs_Bias_MPa=('Bias_MPa',lambda s:np.mean(np.abs(s))),Mean_Task_Rank=('Task_Rank','mean')).reset_index().sort_values('Mean_RMSE_MPa').reset_index(drop=True))
agg.insert(0,'RMSE_Rank',np.arange(1,len(agg)+1))
pivot=task.pivot(index=['Blind_Well','Target'],columns='Model',values='RMSE_MPa')
fs,fp=friedmanchisquare(*[pivot[m].to_numpy() for m in model_order]); best=agg.iloc[0]['Model']; second=agg.iloc[1]['Model']; ws,wp=wilcoxon(pivot[best],pivot[second],zero_method='wilcox')
stats=pd.DataFrame({'Statistic':['Best_Model','Best_Mean_RMSE_MPa','Second_Model','Second_Mean_RMSE_MPa','Friedman_ChiSquare','Friedman_p','Wilcoxon_Best_vs_Second_W','Wilcoxon_Best_vs_Second_p','Number_of_Tasks'],'Value':[best,float(agg.iloc[0]['Mean_RMSE_MPa']),second,float(agg.iloc[1]['Mean_RMSE_MPa']),float(fs),float(fp),float(ws),float(wp),int(len(pivot))]})

# Canonical logs rebuilt from original file
base=build_data()
canon_cols=['Sample_ID','Well','Original_Well','Depth_m','Lithology','Lithology_Group']+NUM_FEATURES[1:]
for t in TARGETS: canon_cols += [f'{t}_Prior',f'{t}_Reference',f'{t}_TrueResidual']
base[canon_cols].to_csv(out/'Canonical_Logs.csv',index=False)

# Split wide outputs by target
for t in TARGETS:
    cols=['Sample_ID','Well','Depth_m','Lithology',f'{t}_Prior',f'{t}_Reference',f'{t}_TrueResidual']
    for m in model_order: cols += [f'{t}_{m}_PredResidual',f'{t}_{m}_PredStress',f'{t}_{m}_Error']
    all_pred[cols].to_csv(out/f'Predictions_{t}.csv',index=False)

task.to_csv(out/'Task_Metrics.csv',index=False)
agg.to_csv(out/'Aggregate_Metrics.csv',index=False)
prep.to_csv(out/'Preprocess_Parameters.csv',index=False)
stats.to_csv(out/'Statistics.csv',index=False)
hypers=pd.DataFrame([{'Model':n,'Parameters':json.dumps(m.get_params(),default=str,ensure_ascii=False,sort_keys=True)} for n,m in make_models().items()])
hypers.to_csv(out/'Hyperparameters.csv',index=False)
manifest={'Input_File':'Geomech_4Wells_Before_After(5).xlsx','Samples':len(base),'Wells':['Well A','Well B','Well C','Well D'],'Targets':list(TARGETS),'Models':model_order,'Validation':'Strict Leave-One-Well-Out','Training_Sampling':f'Depth-balanced systematic sample, maximum {MAX_TRAIN_SAMPLES} training rows per fold; predictions retained at every available depth sample','Random_State':42,'Best_Model':best,'Best_Mean_RMSE_MPa':float(agg.iloc[0]['Mean_RMSE_MPa']),'Python':sys.version,'Platform':platform.platform()}
(out/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

# Method-specific code files
model_snippets={
'01_Ridge':"""from sklearn.linear_model import Ridge\n\ndef build_model(random_state=42):\n    return Ridge(alpha=1.0)\n""",
'02_KNN':"""from sklearn.neighbors import KNeighborsRegressor\n\ndef build_model(random_state=42):\n    return KNeighborsRegressor(n_neighbors=15, weights='distance', p=2, n_jobs=-1)\n""",
'03_SVR':"""from sklearn.svm import SVR\n\ndef build_model(random_state=42):\n    return SVR(C=5.0, epsilon=0.015, gamma='scale', kernel='rbf', cache_size=1200)\n""",
'04_Random_Forest':"""from sklearn.ensemble import RandomForestRegressor\n\ndef build_model(random_state=42):\n    return RandomForestRegressor(n_estimators=30, max_depth=18, min_samples_leaf=2, max_features='sqrt', random_state=random_state, n_jobs=-1)\n""",
'05_Extra_Trees':"""from sklearn.ensemble import ExtraTreesRegressor\n\ndef build_model(random_state=42):\n    return ExtraTreesRegressor(n_estimators=30, max_depth=None, min_samples_leaf=2, max_features=0.8, random_state=random_state, n_jobs=-1)\n""",
'06_AdaBoost':"""from sklearn.ensemble import AdaBoostRegressor\nfrom sklearn.tree import DecisionTreeRegressor\n\ndef build_model(random_state=42):\n    base=DecisionTreeRegressor(max_depth=4, min_samples_leaf=3, random_state=random_state)\n    return AdaBoostRegressor(estimator=base, n_estimators=40, learning_rate=0.06, loss='linear', random_state=random_state)\n""",
'07_XGBoost':"""from xgboost import XGBRegressor\n\ndef build_model(random_state=42):\n    return XGBRegressor(n_estimators=25, max_depth=4, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, objective='reg:squarederror', tree_method='hist', max_bin=128, random_state=random_state, n_jobs=4, verbosity=0)\n""",
'08_LightGBM':"""from lightgbm import LGBMRegressor\n\ndef build_model(random_state=42):\n    return LGBMRegressor(n_estimators=30, num_leaves=31, learning_rate=0.07, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, random_state=random_state, n_jobs=4, verbosity=-1)\n""",
'09_CatBoost':"""from catboost import CatBoostRegressor\n\ndef build_model(random_state=42):\n    return CatBoostRegressor(iterations=30, depth=6, learning_rate=0.07, loss_function='RMSE', random_seed=random_state, verbose=False, allow_writing_files=False, thread_count=4)\n""",
'10_MLP':"""from sklearn.neural_network import MLPRegressor\n\ndef build_model(random_state=42):\n    return MLPRegressor(hidden_layer_sizes=(48,24), activation='relu', solver='adam', alpha=0.001, learning_rate_init=0.001, max_iter=80, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=random_state)\n""",
}
for name,code in model_snippets.items():
    (pkg/'models'/f'{name}.py').write_text(code,encoding='utf-8')
    (pkg/'models'/f'{name}.txt').write_text(code,encoding='utf-8')
# Main scripts
shutil.copy2('/mnt/data/Geomech_ML_reproducible_pipeline.py',pkg/'run_pipeline_full.py')
shutil.copy2('/mnt/data/run_geomech_fold.py',pkg/'run_single_fold.py')
shutil.copy2('/mnt/data/merge_geomech_outputs.py',pkg/'merge_outputs.py')
(pkg/'requirements.txt').write_text('numpy\npandas\nscikit-learn\nscipy\nxgboost\nlightgbm\ncatboost\nopenpyxl\nartifact-tool\n',encoding='utf-8')
(pkg/'README_FA.txt').write_text(f'''بسته بازتولید تحلیل تنش ژئومکانیکی\n\nورودی: Geomech_4Wells_Before_After(5).xlsx\nچاه‌ها در خروجی: Well A تا Well D\nروش اعتبارسنجی: Leave-One-Well-Out\nفرمول مدل هیبرید: تنش پیش‌بینی‌شده = تنش تحلیلی قبل + باقیمانده پیش‌بینی‌شده با ML\nبرای محدودکردن زمان اجرا، در هر fold حداکثر {MAX_TRAIN_SAMPLES} نمونه آموزشی به‌صورت منظم و متوازن در عمق انتخاب شده است؛ پیش‌بینی برای تمام نمونه‌های عمقی تولید شده است.\n\nترتیب اجرا:\n1) run_single_fold.py برای هر Well\n2) merge_outputs.py\nیا run_pipeline_full.py برای اجرای یکپارچه\n\nبهترین مدل اجرای بازتولید فعلی: {best}\nMean RMSE: {float(agg.iloc[0]['Mean_RMSE_MPa']):.6f} MPa\n''',encoding='utf-8')
# zip package
zip_path=Path('/mnt/data/Geomech_ML_Code_Package.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for f in pkg.rglob('*'):
        if f.is_file(): z.write(f,f.relative_to(pkg))
print('BEST',best,float(agg.iloc[0]['Mean_RMSE_MPa']))
print('ROWS',len(base),'FILES',len(list(out.glob('*'))),'ZIP',zip_path.stat().st_size)
