import os
import numpy as np
import pandas as pd
import glob
import mlflow
from mlflow.tracking import MlflowClient
import gc
from scipy.special import erfinv
from tqdm import tqdm
import random
import pyarrow.parquet as pq
import polars as pl
from pathlib import Path
from src.features.engineer import FeatureEngineer
from src.data_loader.loader import DataLoader
from src.data_loader.filter import FinancialUniverseEngine
import logging

# MLflow (alembic) のログを抑制
logging.getLogger("alembic").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_DIR / 'data/intermediate'
OUTPUT_DIR = PROJECT_DIR / 'data/master'
SAMPLE_OUTPUT_DIR = PROJECT_DIR / 'data/sample' # サンプル出力先


def main(mode = "full"):
    loader = DataLoader()
    filter = FinancialUniverseEngine()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chunk_files = sorted(glob.glob(f"{INPUT_DIR}/date_chunks/*.parquet"))
    n_stocks = 200 # サンプル作成時の銘柄数

    # 銘柄リストの取得とサンプリング
    if mode == "sample":
        print("Scanning unique scodes...")
        all_scodes = set()
        for f in chunk_files[:10]: # 高速化のため最初の数ファイルから銘柄を抽出
            tmp = pd.read_parquet(f, columns=['scode'])
            all_scodes.update(tmp['scode'].unique())
        selected_scodes = random.sample(list(all_scodes), min(n_stocks, len(all_scodes)))
        print(f"Selected {len(selected_scodes)} stocks for sampling.")
    
    # サイズ確定のためのダミー実行
    print("Pre-scanning to determine feature list...")
    if chunk_files:
        pf = pq.ParquetFile(chunk_files[0])
        sample_df = pf.read_row_group(0).to_pandas().head(1)
        engineer = FeatureEngineer(sample_df)
        pipe = (
            engineer
            .apply_bulk_cross_sectional()
        )
        sample_df = pipe.get_df()
        feature_cols = [x for x in sample_df.columns if x.startswith(('MOM_', 'VOL_', 'LIQ_', 'VAL_', 'QLT_', 'SIZ_', 'SPD_', 'BET_', 'SEA_', 'EVT_', 'CON_', 'GOV_'))]
        del sample_df
        gc.collect()
    num_features = len(feature_cols)
    total_rows = 0
    for f in chunk_files:
        tmp_meta = pd.read_parquet(f, columns=['scode']) # メモリ節約のため scode のみ
        if mode == "sample":
            total_rows += tmp_meta['scode'].isin(selected_scodes).sum()
        else:
            total_rows += len(tmp_meta)
    print(f"Total rows to process: {total_rows}, Total features: {num_features}")
    
    # memmap の事前割当 (float32)
    if mode == "sample":
        features_path = os.path.join(SAMPLE_OUTPUT_DIR, "features.npy")
    else:
        features_path = os.path.join(OUTPUT_DIR, "features.npy")
    mmap_array = np.memmap(features_path, dtype='float32', mode='w+', shape=(total_rows, num_features))
    
    # チャンク処理と書き込み
    current_row = 0
    meta_list = []
    # MLflowの初期設定
    abs_path = os.path.expanduser("~/JPSForecast/mlflow_runs")
    os.makedirs(abs_path, exist_ok=True)
    mlflow_db_path = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(mlflow_db_path)
    if mode == "sample":
        experiment_name = "Sample_Data_Creation"
    else:
        experiment_name = "Master_Data_Creation"
    client = MlflowClient()
    existing_exp = client.get_experiment_by_name(experiment_name)
    if existing_exp is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=f"file://{abs_path}"
        )
    elif existing_exp.lifecycle_stage == 'deleted':
        print(f"Restoring deleted experiment: {experiment_name}")
        client.restore_experiment(existing_exp.experiment_id)
    mlflow.set_experiment(experiment_name)

    raw_buffer_df = pd.DataFrame()
    with mlflow.start_run(run_name="Create_Master_Data"):
        for f in chunk_files:
            print(f"Processing chunk: {os.path.basename(f)}")
            df = pd.read_parquet(f)
            # 銘柄フィルタリング (サンプルモードの場合)
            if mode == "sample":
                df = df[df['scode'].isin(selected_scodes)].reset_index(drop=True)
                if len(df) == 0: continue
            # メモリ削減: float64 -> float32
            f_cols = df.select_dtypes(include=['float64']).columns
            if len(f_cols) > 0:
                df[f_cols] = df[f_cols].astype('float32')
            df['sector33_code'] = df['sector33_code'].astype(str)
            # --- Buffer Logic for Rolling Metrics ---
            filter_cols = ['date', 'scode', 'close', 'filt_Return', 'filt_Median_ADV_20', 
                        'filt_Tick_Sensitivity', 'filt_ATR_Ratio', 'filt_Is_Stop_Day']
            if not raw_buffer_df.empty:
                min_dt = df['date'].min() - pd.Timedelta(days=100)
                raw_buffer_df = raw_buffer_df[raw_buffer_df['date'] >= min_dt]
                calc_df = pd.concat([raw_buffer_df, df[filter_cols]], axis=0, ignore_index=True)
            else:
                calc_df = df[filter_cols].copy()
            # --- フィルタリング ---
            calc_df = filter.calc_relative_metrics(calc_df)
            current_flags = calc_df.iloc[-len(df):]
            df['is_candidate_tac'] = current_flags['is_candidate_tac'].values
            df['is_candidate_str'] = current_flags['is_candidate_str'].values
            # --- クロスセクショナル特徴量＆ターゲットの生成 ---
            engineer = FeatureEngineer(df)
            pipe = (
                engineer
                .apply_bulk_cross_sectional()
            )
            df = pipe.get_df()
            del calc_df, engineer
            gc.collect()
            # バッファ更新 (生のdfを使用)
            raw_buffer_df = pd.concat([raw_buffer_df, df[filter_cols]], axis=0, ignore_index=True)
            # フィルタ通過状況のログ出力
            n_tac = df['is_candidate_tac'].sum()
            n_str = df['is_candidate_str'].sum()
            if n_tac > 0 or n_str > 0:
                print(f"  [Filter Stats] TAC: {n_tac}, STR: {n_str} / {len(df)} rows")
            # --- 特徴量とメタデータの書き込み ---
            future_cols = ['Future_High_Tac','Future_Low_Tac','Future_Close_Tac','Future_High_Str','Future_Low_Str','Future_Close_Str']
            for col in future_cols:
                df[col] = df[col]/df['Entry_Price']
            data_to_write = df[feature_cols].values.astype('float32')
            mmap_array[current_row : current_row + len(df)] = data_to_write
            meta_cols = ['date', 'scode', 'is_candidate_tac', 'is_candidate_str', 'log_market_cap'] + future_cols + [c for c in df.columns if c.startswith('target_')]
            meta_list.append(df[meta_cols])
            current_row += len(df)
            mmap_array.flush()
            del df, data_to_write
            gc.collect()

        # 成果物の保存
        meta_df = pd.concat(meta_list)
        if mode == "sample":
            meta_path = os.path.join(SAMPLE_OUTPUT_DIR, "index_meta.parquet")
            pd.Series(feature_cols).to_json(os.path.join(SAMPLE_OUTPUT_DIR, "feature_names.json"), orient='records')
            mlflow.log_artifact(os.path.join(SAMPLE_OUTPUT_DIR, "feature_names.json"), "metadata")
        else:
            meta_path = os.path.join(OUTPUT_DIR, "index_meta.parquet")
            pd.Series(feature_cols).to_json(os.path.join(OUTPUT_DIR, "feature_names.json"), orient='records')
            mlflow.log_artifact(os.path.join(OUTPUT_DIR, "feature_names.json"), "metadata")
        meta_df.to_parquet(meta_path)
        mlflow.log_param("total_rows", total_rows)
        mlflow.log_artifact(meta_path, "metadata")

    print(f"✅ Master data creation complete. Total rows: {total_rows}")

if __name__ == "__main__":
    import sys
    if "--sample" in sys.argv:
        main(mode="sample")
    else:
        main()