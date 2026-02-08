import os
import numpy as np
import pandas as pd
import glob
import mlflow
import gc
from scipy.special import erfinv
from tqdm import tqdm
import pyarrow.parquet as pq
from pathlib import Path
from src.features.engineer import FeatureEngineer
from src.data_loader.filter import RuleBasedFilter, RuleBasedFilter_STR

PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_DIR / 'data/intermediate'
OUTPUT_DIR = PROJECT_DIR / 'data/master'

def main():
    engineer = FeatureEngineer()
    filter_tac = RuleBasedFilter()
    filter_str = RuleBasedFilter_STR()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load sector_return
    df_sector_indices = pd.read_parquet(INPUT_DIR / 'sector_return.parquet')
    # load orthogonalized targets
    df_targets = pd.read_parquet(INPUT_DIR / 'orthogonalized_targets.parquet')

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
    with mlflow.start_run(run_name="Create_Master_Data"):
        for f in chunk_files:
            print(f"Processing chunk: {os.path.basename(f)}")
            df = pd.read_parquet(f)
            df = df.merge(df_sector_indices, on=['date', 'sector33_code'], how='left')
            
            # --- 横断的計算の実行 ---
            df = engineer.add_cross_sectional_features(df)
            # --- 戦略モデルターゲットの結合 ---
            df = df.merge(df_targets[['date', 'scode', 'target_reg', 'target_cls']], on=['date', 'scode'], how='left')
            # --- フィルタリングの適用 ---
            df = filter_tac.apply(df)
            df = filter_str.apply(df)
            
            # --- 特徴量とメタデータの書き込み ---
            future_cols = ['Future_High_Tac','Future_Low_Tac','Future_Close_Tac','Future_High_Str','Future_Low_Str','Future_Close_Str']
            for col in future_cols:
                df[col] = df[col]/df['Entry_Price']
            data_to_write = df[feature_cols].values.astype('float32')
            mmap_array[current_row : current_row + len(df)] = data_to_write
            meta_cols = ['date', 'scode', 'is_candidate_tac', 'is_candidate_str'] + future_cols + engineer.target_cols
            meta_list.append(df[meta_cols])
            
            current_row += len(df)
            mmap_array.flush()
            
            del df, data_to_write
            gc.collect()

        # 4. 成果物の保存
        meta_df = pd.concat(meta_list)
        meta_path = os.path.join(OUTPUT_DIR, "index_meta.parquet")
        meta_df.to_parquet(meta_path)
        
        # 特徴量名のリストを保存
        pd.Series(feature_cols).to_json(os.path.join(OUTPUT_DIR, "feature_names.json"))
        
        # MLflowへのログ記録
        mlflow.log_param("total_rows", total_rows)
        mlflow.log_artifact(meta_path, "metadata")
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "feature_names.json"), "metadata")

    print(f"✅ Master data creation complete. Total rows: {total_rows}")

if __name__ == "__main__":
    main()