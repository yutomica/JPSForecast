import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta
import gc
import warnings
import argparse
import json
import sys
import ast
from joblib import Parallel, delayed
import MySQLdb

# プロジェクトのルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 設定 ---
# MLflow設定
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
# 出力先ディレクトリ
OUTPUT_DIR = "./data/backtest"

# 警告抑制
warnings.filterwarnings("ignore")


start_date = '2025-01-01'
end_date = '2026-12-04'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()
conn = MySQLdb.connect(
    user='root',
    passwd='root',
    host='127.0.0.1',
    port=3306,
    charset='utf8',
)


# load data
master_dir = './data/master/'
meta_df = pd.read_parquet(master_dir + "index_meta.parquet")
meta_df = meta_df.reset_index(drop=True)
all_features = pd.read_json(master_dir + "feature_names.json", typ='series').tolist()
features_mmap = np.memmap(
    master_dir + "features.npy", 
    dtype='float32', 
    mode='r', 
    shape=(len(meta_df), len(all_features))
)

# load models
print("\nLoading Staging models from MLflow Registry...")
registered_models = client.search_registered_models(filter_string="name LIKE '%'")
models = []
for rm in registered_models:
    for v in rm.latest_versions:
        if v.current_stage == 'Staging':
            models.append(v)
            print(f"  - Found: {v.name} (Version: {v.version}, Stage: {v.current_stage})")
            break

date_idx = sorted(meta_df['date'].unique())
mask = (meta_df['date'] >= start_date) & (meta_df['date'] <= end_date)
dates = sorted(meta_df.loc[mask, 'date'].unique())
output = meta_df.loc[mask, ['date', 'scode', 'Future_High_Tac', 'Future_Low_Tac', 'Future_Close_Tac', 'Future_High_Str', 'Future_Low_Str', 'Future_Close_Str']].copy()

for model_version in models:
    model_name = model_version.name
    model_uri = f"models:/{model_name}/Staging"
    print(f"\nScoring with {model_name}...")
    try:
        # MLflowのRunパラメータからモデルのドメインとカテゴリ（時系列かTableか）を取得
        run = client.get_run(model_version.run_id)
        params = run.data.params
        # domain のパース（dictが文字列化されている場合に対応）
        domain_raw = params.get('domain.name', params.get('domain/name', params.get('domain', 'STR')))
        if isinstance(domain_raw, str) and domain_raw.strip().startswith('{'):
            try:
                domain_dict = ast.literal_eval(domain_raw)
                domain = domain_dict.get('name', 'STR').upper()
            except (ValueError, SyntaxError):
                domain = 'STR'
        else:
            domain = str(domain_raw).upper()
        # data_category のパース（dictが文字列化されている場合に対応）
        model_raw = params.get('model.data_category', params.get('model/data_category', params.get('model', 'tabular')))
        if isinstance(model_raw, str) and model_raw.strip().startswith('{'):
            try:
                model_dict = ast.literal_eval(model_raw)
                data_category = model_dict.get('data_category', 'tabular').lower()
            except (ValueError, SyntaxError):
                data_category = 'tabular'
        else:
            data_category = str(model_raw).lower()
            
        print(f"  - Domain: {domain}, Category: {data_category}")
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        local_path = client.download_artifacts(run_id=model_version.run_id, path="configs/feature_cols.json")
        with open(local_path) as f:
            config = json.load(f)
        feature_cols = config['feature_cols']
        col_indices = [all_features.index(c) for c in feature_cols]
        domain_col = f'is_candidate_{domain.lower()}'
        
        ref_idx = meta_df.index[(meta_df['date'].isin(dates)) & (meta_df[domain_col] == True)]
        if len(ref_idx) == 0:
            print("  - No candidate records found. Skipping.")
            continue
            
        print(f"  - Predicting for {len(ref_idx):,} records...")
        pipeline = loaded_model.unwrap_python_model()
        all_preds = np.zeros(len(ref_idx), dtype=np.float32)
        batch_size = 10000
        
        for i in tqdm(range(0, len(ref_idx), batch_size), desc="  Predicting batches"):
            batch_idx = ref_idx[i : i + batch_size].values
            batch_preds_folds = []
            
            for fold_pipe in pipeline.fold_pipelines:
                X_processed = fold_pipe.preprocessor.transform(features_mmap, row_indices=batch_idx, col_indices=col_indices)
                p = fold_pipe.model.predict(X_processed)
                batch_preds_folds.append(p)
                
            all_preds[i : i + batch_size] = np.mean(batch_preds_folds, axis=0)
            
        output[f'scores_{model_name}'] = np.nan
        output.loc[ref_idx, f'scores_{model_name}'] = all_preds

        print(f"  - Finished scoring with {model_name}.")
    except Exception as e:
        print(f"  - Failed to score with {model_name}. Error: {e}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, f"backtest_results_{start_date.replace('-','')}_{end_date.replace('-','')}.csv")
output.to_csv(output_path, index=False)
print(f"\n✅ All predictions saved to: {output_path}")