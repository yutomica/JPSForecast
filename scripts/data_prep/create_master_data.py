import os
import numpy as np
import pandas as pd
import glob
import mlflow
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

def create_sample_data(n_stocks=50):
    """
    銘柄(scode)をランダムにサンプリングして軽量な動作確認用データを作成する
    """
    print(f"--- Creating Sample Data (Target: {n_stocks} stocks) ---")
    os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)
    
    loader = DataLoader()
    engineer = FeatureEngineer()
    filter = FinancialUniverseEngine()
    chunk_files = sorted(glob.glob(f"{INPUT_DIR}/date_chunks/*.parquet"))

    # 1. 銘柄リストの取得とサンプリング
    print("Scanning unique scodes...")
    all_scodes = set()
    for f in chunk_files[:10]: # 高速化のため最初の数ファイルから銘柄を抽出
        tmp = pd.read_parquet(f, columns=['scode'])
        all_scodes.update(tmp['scode'].unique())
    
    selected_scodes = random.sample(list(all_scodes), min(n_stocks, len(all_scodes)))
    print(f"Selected {len(selected_scodes)} stocks for sampling.")

    # 2. 列名と総行数の確定
    pf = pq.ParquetFile(chunk_files[0])
    sample_df = pf.read_row_group(0).to_pandas().head(1)
    # ダミー実行で特徴量リストを確定
    df_sector_indices = loader.fetch_sector_return()
    sample_df = sample_df.merge(df_sector_indices, on=['date', 'sector33_code'], how='left')
    _ = engineer.add_cross_sectional_features(sample_df)
    feature_cols = engineer.feature_list
    num_features = len(feature_cols)

    total_sample_rows = 0
    print("Calculating total rows for sample...")
    for f in chunk_files:
        tmp = pd.read_parquet(f, columns=['scode'])
        total_sample_rows += tmp['scode'].isin(selected_scodes).sum()
    
    print(f"Total sample rows: {total_sample_rows}, Features: {num_features}")

    # 3. memmap の割当
    os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)
    features_path = os.path.join(SAMPLE_OUTPUT_DIR, "features.npy")
    mmap_array = np.memmap(features_path, dtype='float32', mode='w+', shape=(total_sample_rows, num_features))
    
    # 4. データの抽出と書き込み
    current_row = 0
    meta_list = []
    # df_targets = pd.read_parquet(INPUT_DIR / 'orthogonalized_targets.parquet')
    raw_buffer_df = pd.DataFrame()

    for f in tqdm(chunk_files, desc="Processing chunks for sample"):
        df = pd.read_parquet(f)
        df['sector33_code'] = df['sector33_code'].astype('object')
        # 銘柄フィルタリング
        df = df[df['scode'].isin(selected_scodes)].reset_index(drop=True)
        if len(df) == 0: continue
        # --- Buffer Logic for Rolling Metrics ---
        if not raw_buffer_df.empty:
            min_dt = df['date'].min() - pd.Timedelta(days=180)
            raw_buffer_df = raw_buffer_df[raw_buffer_df['date'] >= min_dt]
            calc_df = pd.concat([raw_buffer_df, df], axis=0, ignore_index=True)
        else:
            calc_df = df.copy()
        
        calc_res = filter.calc_relative_metrics(calc_df)
        current_flags = calc_res.iloc[-len(df):]
        df['is_candidate_tac'] = current_flags['is_candidate_tac'].values
        df['is_candidate_str'] = current_flags['is_candidate_str'].values
        raw_buffer_df = pd.concat([raw_buffer_df, df], axis=0, ignore_index=True)
        df = df.drop(columns=[c for c in df.columns if c.startswith('filt_')], errors='ignore')
        # ----------------------------------------
        df = df.merge(df_sector_indices, on=['date', 'sector33_code'], how='left')
        df = engineer.add_cross_sectional_features(df)
        # df = df.merge(df_targets[['date', 'scode', 'target_reg', 'target_cls']], on=['date', 'scode'], how='left')
        # 保存用処理（本番用 create_master_data.py と同一）
        future_cols = ['Future_High_Tac','Future_Low_Tac','Future_Close_Tac','Future_High_Str','Future_Low_Str','Future_Close_Str']
        for col in future_cols:
            df[col] = df[col]/df['Entry_Price']
        data_to_write = df[feature_cols].values.astype('float32')
        mmap_array[current_row : current_row + len(df)] = data_to_write
        meta_cols = ['date', 'scode', 'is_candidate_tac', 'is_candidate_str', 'log_market_cap'] + engineer.target_cols
        meta_list.append(df[meta_cols])
        current_row += len(df)
        mmap_array.flush()
        del df; gc.collect()

    # 5. 成果物の保存
    meta_df = pd.concat(meta_list)
    meta_df.to_parquet(os.path.join(SAMPLE_OUTPUT_DIR, "index_meta.parquet"))
    pd.Series(feature_cols).to_json(os.path.join(SAMPLE_OUTPUT_DIR, "feature_names.json"))
    print(f"✅ Sample data created at {SAMPLE_OUTPUT_DIR}")


def main():
    loader = DataLoader()
    engineer = FeatureEngineer()
    filter = FinancialUniverseEngine()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load sector_return
    df_sector_indices = loader.fetch_sector_return()
    df_sector_indices['sector33_code'] = df_sector_indices['sector33_code'].astype(str)
    # load orthogonalized targets
    # df_targets = pd.read_parquet(INPUT_DIR / 'orthogonalized_targets.parquet')

    chunk_files = sorted(glob.glob(f"{INPUT_DIR}/date_chunks/*.parquet"))

    # --- 【重要】サイズ確定のためのダミー実行（Pass 0） ---
    print("Pre-scanning to determine feature list...")
    if chunk_files:
        pf = pq.ParquetFile(chunk_files[0])
        sample_df = pf.read_row_group(0).to_pandas().head(1)
        sample_df = sample_df.merge(df_sector_indices, on=['date', 'sector33_code'], how='left')
        _ = engineer.add_cross_sectional_features(sample_df)
        del sample_df
        gc.collect()
    feature_cols = engineer.feature_list
    num_features = len(feature_cols)
    # 1. 事前スキャン（行数のみ確定）
    total_rows = 0
    for f in chunk_files:
        tmp_meta = pd.read_parquet(f, columns=['scode']) # メモリ節約のため scode のみ
        total_rows += len(tmp_meta)
    print(f"Total rows to process: {total_rows}, Total features: {num_features}")
    
    # 2. memmap の事前割当 (float32)
    features_path = os.path.join(OUTPUT_DIR, "features.npy")
    mmap_array = np.memmap(features_path, dtype='float32', mode='w+', shape=(total_rows, num_features))
    
    # 3. チャンク処理と書き込み
    current_row = 0
    meta_list = []
    # MLflowの初期設定
    abs_path = os.path.expanduser("~/JPSForecast/mlflow_runs")
    os.makedirs(abs_path, exist_ok=True)
    mlflow_db_path = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(mlflow_db_path)
    raw_buffer_df = pd.DataFrame()
    with mlflow.start_run(run_name="Create_Master_Data"):
        for f in chunk_files:
            print(f"Processing chunk: {os.path.basename(f)}")
            df = pd.read_parquet(f)
            df['sector33_code'] = df['sector33_code'].astype(str)
            # --- Buffer Logic for Rolling Metrics ---
            if not raw_buffer_df.empty:
                min_dt = df['date'].min() - pd.Timedelta(days=180)
                raw_buffer_df = raw_buffer_df[raw_buffer_df['date'] >= min_dt]
                calc_df = pd.concat([raw_buffer_df, df], axis=0, ignore_index=True)
            else:
                calc_df = df.copy()
            calc_res = filter.calc_relative_metrics(calc_df)
            current_flags = calc_res.iloc[-len(df):]
            df['is_candidate_tac'] = current_flags['is_candidate_tac'].values
            df['is_candidate_str'] = current_flags['is_candidate_str'].values
            raw_buffer_df = pd.concat([raw_buffer_df, df], axis=0, ignore_index=True)
            df = df.drop(columns=[c for c in df.columns if c.startswith('filt_')], errors='ignore')
            # ----------------------------------------
            df = df.merge(df_sector_indices, on=['date', 'sector33_code'], how='left')
            # --- 横断的計算の実行 ---
            df = engineer.add_cross_sectional_features(df)
            # --- 戦略モデルターゲットの結合 ---
            # df = df.merge(df_targets[['date', 'scode', 'target_reg', 'target_cls']], on=['date', 'scode'], how='left')
            # --- フィルタリングの適用 ---
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
            meta_cols = ['date', 'scode', 'is_candidate_tac', 'is_candidate_str', 'log_market_cap'] + future_cols + engineer.target_cols
            meta_list.append(df[meta_cols])
            
            current_row += len(df)
            mmap_array.flush()
            
            del df, data_to_write
            gc.collect()

        # 4. 成果物の保存
        meta_df = pd.concat(meta_list)
        meta_path = os.path.join(OUTPUT_DIR, "index_meta.parquet")
        meta_df.to_parquet(meta_path)
        
        # 候補フラグの年別集計と保存
        meta_df['year'] = meta_df['date'].dt.year
        cand_cols = ['is_candidate_tac', 'is_candidate_str']
        valid_cand_cols = [c for c in cand_cols if c in meta_df.columns]
        if valid_cand_cols:
            cand_stats = meta_df.groupby('year')[valid_cand_cols].sum().astype(int)
            cand_stats['total_rows'] = meta_df.groupby('year').size()
            cand_stats = cand_stats.reset_index()
            cand_path = os.path.join(OUTPUT_DIR, "candidate_counts_by_year.csv")
            cand_stats.to_csv(cand_path, index=False)
            mlflow.log_artifact(cand_path, "metadata")
            print(f"✅ Candidate counts saved to {cand_path}")
        
        # 特徴量名のリストを保存
        pd.Series(feature_cols).to_json(os.path.join(OUTPUT_DIR, "feature_names.json"))
        
        # MLflowへのログ記録
        mlflow.log_param("total_rows", total_rows)
        mlflow.log_artifact(meta_path, "metadata")
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "feature_names.json"), "metadata")

    print(f"✅ Master data creation complete. Total rows: {total_rows}")

if __name__ == "__main__":
    import sys
    # コマンドライン引数で --sample が指定された場合はサンプル作成のみ実行
    if "--sample" in sys.argv:
        create_sample_data(n_stocks=30)
    else:
        main()