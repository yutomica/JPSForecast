import os
import sys
import argparse
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import gc
import json
import ast
from tqdm import tqdm

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_loader.loader import DataLoader

# 警告抑制
warnings.filterwarnings("ignore")

def get_staging_model_info(client, model_name):
    """
    指定されたモデル名のStagingエイリアスを持つモデル情報を取得する
    """
    try:
        # ステージで最新のものを取得
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        if versions:
            v = versions[0]
            # モデルのカテゴリ（tabular/timeseries）を取得
            run = client.get_run(v.run_id)
            params = run.data.params
            model_raw = params.get('model.data_category', params.get('model/data_category', 'tabular'))
            if isinstance(model_raw, str) and model_raw.strip().startswith('{'):
                try:
                    model_dict = ast.literal_eval(model_raw)
                    data_category = model_dict.get('data_category', 'tabular').lower()
                except:
                    data_category = 'tabular'
            else:
                data_category = str(model_raw).lower()

            return {
                "name": model_name,
                "version": v.version,
                "run_id": v.run_id,
                "model_uri": f"models:/{model_name}/{v.version}",
                "data_category": data_category
            }
    except Exception as e:
        print(f"Warning: Model {model_name} (Staging) not found: {e}")
    return None

def load_master_meta(master_dir, start_date, end_date):
    """
    index_meta.parquetを読み込み、指定期間でフィルタリングする
    """
    master_dir = Path(master_dir)
    print(f"Loading master meta data from {master_dir}...")
    meta_df = pd.read_parquet(master_dir / "index_meta.parquet")
    meta_df['date'] = pd.to_datetime(meta_df['date'])
    
    mask = (meta_df['date'] >= pd.to_datetime(start_date)) & \
           (meta_df['date'] <= pd.to_datetime(end_date))
    meta_eval = meta_df[mask].copy()
    print(f" - Filtered eval meta: {len(meta_eval)} records.")
    return meta_eval

def load_master_features_optimized(master_dir, feature_cols, start_idx, end_idx):
    """
    指定された行範囲の特徴量を一括ロードする（高速化版）
    """
    master_dir = Path(master_dir)
    chunk_files = sorted((master_dir / "features").glob("features_chunk_*.parquet"))
    
    features_list = []
    current_start = 0
    for cf in tqdm(chunk_files, desc="Loading feature chunks", leave=False):
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(cf)
        chunk_len = parquet_file.metadata.num_rows
        chunk_end = current_start + chunk_len - 1
        
        if chunk_end >= start_idx and current_start <= end_idx:
            read_start = max(0, start_idx - current_start)
            read_end = min(chunk_len, end_idx - current_start + 1)
            
            df_chunk = pd.read_parquet(cf, columns=feature_cols)
            features_list.append(df_chunk.iloc[read_start:read_end])
            
        current_start += chunk_len
        if current_start > end_idx:
            break
            
    if not features_list:
        return np.array([])
    return pd.concat(features_list).values.astype(np.float32)

def generate_alpha_scores(master_dir, meta_df, model_info):
    """
    Masterデータから特徴量をロードし、モデルによるスコアリングを行う
    """
    domain = "TAC" if "tac" in model_info['name'].lower() else "STR"
    candidate_col = "is_candidate_tac" if domain == "TAC" else "is_candidate_str"
    # 対象銘柄に絞り込み
    eval_df = meta_df[meta_df[candidate_col] == True].copy()
    if eval_df.empty:
        print(f"No candidate records for {domain} in the given period.")
        return pd.DataFrame()
    # 重複排除
    eval_df = eval_df.drop_duplicates(subset=['date', 'scode'])
    print(f"Scoring {len(eval_df)} records with {model_info['name']}...")
    # 特徴量リストを取得
    client = MlflowClient()
    local_path = client.download_artifacts(run_id=model_info['run_id'], path="configs/feature_cols.json")
    with open(local_path) as f:
        config = json.load(f)
    feature_cols = config['feature_cols']
    # 特徴量をロード (時系列モデルの場合は履歴が必要)
    if model_info['data_category'] == 'timeseries':
        min_idx = eval_df.index.min()
        # 履歴を確保するためのバッファ (50万行程度)
        load_start_idx = max(0, min_idx - 500000)
        load_end_idx = eval_df.index.max()
        features_arr = load_master_features_optimized(master_dir, feature_cols, load_start_idx, load_end_idx)
        ref_indices = eval_df.index.values - load_start_idx
    else:
        load_start_idx = eval_df.index.min()
        load_end_idx = eval_df.index.max()
        features_arr = load_master_features_optimized(master_dir, feature_cols, load_start_idx, load_end_idx)
        ref_indices = eval_df.index.values - load_start_idx
    # 推論 (バッチ処理)
    model = mlflow.pyfunc.load_model(model_info['model_uri'])
    pipeline = model.unwrap_python_model()
    all_preds = np.zeros(len(ref_indices), dtype=np.float32)
    batch_size = 10000
    col_indices = list(range(len(feature_cols)))
    for i in tqdm(range(0, len(ref_indices), batch_size), desc="  Predicting"):
        batch_ref_idx = ref_indices[i : i + batch_size]
        batch_preds_folds = []
        for fold_pipe in pipeline.fold_pipelines:
            X_processed = fold_pipe.preprocessor.transform(features_arr, row_indices=batch_ref_idx, col_indices=col_indices)
            p = fold_pipe.model.predict(X_processed)
            batch_preds_folds.append(p)
            if isinstance(X_processed, str) and X_processed.endswith('.zarr'):
                import shutil
                shutil.rmtree(X_processed, ignore_errors=True)
        all_preds[i : i + batch_size] = np.mean(batch_preds_folds, axis=0)
        gc.collect()
    eval_df['alpha_score'] = all_preds
    return eval_df[['date', 'scode', 'alpha_score']]



def main():
    parser = argparse.ArgumentParser(description="Alphalens evaluation for Staging models.")
    parser.add_argument("--start_date", type=str, default="2025-04-01", help="Evaluation start date.")
    parser.add_argument("--end_date", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="Evaluation end date.")
    parser.add_argument("--models", type=str, nargs="+", default=["tac_alpha", "str_alpha"], help="Model names in MLflow to evaluate.")
    parser.add_argument("--master_dir", type=str, default="data/master", help="Master data directory.")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save reports.")
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    loader = DataLoader()
    
    model_names = args.models
    target_models = []
    for name in model_names:
        info = get_staging_model_info(client, name)
        if info:
            target_models.append(info)
            
    if not target_models:
        print("No staging models found. Exiting.")
        return

    # Master Metaのロード
    meta_eval = load_master_meta(args.master_dir, args.start_date, args.end_date)
    
    for model_info in target_models:
        # スコア生成（Master Featuresを使用）
        pred_df = generate_alpha_scores(args.master_dir, meta_eval, model_info) 
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        model_name = model_info['name']
        if model_name.find('tac') != -1:
            meta = meta_eval[['date', 'scode', 'Future_Close_Tac', 'Future_High_Tac', 'Future_Low_Tac']]
            meta = meta.rename(columns={'Future_Close_Tac': 'Future_Close', 'Future_High_Tac': 'Future_High', 'Future_Low_Tac': 'Future_Low'})
        else:
            meta = meta_eval[['date', 'scode', 'Future_Close_Str', 'Future_High_Str', 'Future_Low_Str']]
            meta = meta.rename(columns={'Future_Close_Str': 'Future_Close', 'Future_High_Str': 'Future_High', 'Future_Low_Str': 'Future_Low'})
        df = pd.merge(pred_df, meta, on=['date', 'scode'], how='left')

        # -- BIN分析 --
        df['bin'] = df.groupby('date')['alpha_score'].transform(
            lambda x: pd.qcut(x, 20, labels=False, duplicates='drop')
        )
        # 各ビンについて統計量を算出
        def q1(x): return x.quantile(0.01)
        def q5(x): return x.quantile(0.05)
        def q50(x): return x.quantile(0.50)
        def q95(x): return x.quantile(0.95)
        def q99(x): return x.quantile(0.99)
        agg_funcs = ['count', 'mean', 'std', 'min', q1, q5, q50, q95, q99, 'max']
        res = df.groupby('bin')[['Future_Low', 'Future_High', 'Future_Close']].agg(agg_funcs)
        # カラム名をフラット化
        res.columns = [f"{c[0]}_{c[1] if isinstance(c[1], str) else c[1].__name__}" for c in res.columns]
        res = res.reset_index()
        res.to_csv(args.output_dir+model_name+'_bin_analysis.csv',index=False)

        # -- Top10銘柄分析 --
        # 日別上位10銘柄を選出し、Future_Closeの統計量を算出
        if model_name.find('alpha') != -1:
            top10_daily = df.sort_values(['date', 'alpha_score'], ascending=[True, False]).groupby('date').head(10)
        else:
            top10_daily = df.sort_values(['date', 'alpha_score'], ascending=[True, False]).groupby('date').tail(10)
        res2 = top10_daily.groupby('date')['Future_Close'].agg(['min', q5, q50, q95, 'max']).reset_index()
        # 全銘柄平均リターンの算出と追加
        market_avg = meta.groupby('date')['Future_Close'].mean().reset_index().rename(columns={'Future_Close': 'market_avg_return'})
        res2 = pd.merge(res2, market_avg, on='date', how='left')

        res2['cum_return'] = res2['q50'].cumsum()
        top10_lists = top10_daily.groupby('date')['scode'].apply(set).reset_index()
        top10_lists['prev_scode_set'] = top10_lists['scode'].shift(1)
        def calc_overlap(row):
            if row['prev_scode_set'] is None or pd.isna(row['prev_scode_set']):
                return np.nan
            return len(row['scode'] & row['prev_scode_set'])
        top10_lists['overlap_count'] = top10_lists.apply(calc_overlap, axis=1)
        res3 = top10_lists[['date', 'overlap_count']].dropna()
        res2 = pd.merge(res2, res3, on='date', how='left')
        res2.to_csv(args.output_dir+model_name+'_top10_analysis.csv',index=False)

    loader.close()
    print("Evaluation completed.")

if __name__ == "__main__":
    main()
