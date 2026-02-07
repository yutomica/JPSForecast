import os
import numpy as np
import pandas as pd
import json
import gc

def verify_master_data(master_dir):
    print(f"--- Starting Data Validation for: {master_dir} ---")
    
    # 1. ファイルの存在確認
    meta_path = os.path.join(master_dir, "index_meta.parquet")
    features_path = os.path.join(master_dir, "features.npy")
    names_path = os.path.join(master_dir, "feature_names.json")
    
    for p in [meta_path, features_path, names_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    # 2. 特徴量名のロード
    with open(names_path, 'r') as f:
        feature_names = json.load(f)
        if isinstance(feature_names, dict): # Series.to_json の形式に対応
            feature_names = list(feature_names.values())
    
    num_features = len(feature_names)
    print(f"✅ Loaded {num_features} feature names.")

    # 3. メタデータの検証
    meta_df = pd.read_parquet(meta_path)
    total_rows = len(meta_df)
    print(f"✅ Meta data loaded: {total_rows} rows.")

    # 4. memmap のロード (メモリを消費しない)
    features_mmap = np.memmap(features_path, dtype='float32', mode='r', 
                              shape=(total_rows, num_features))
    
    # 5. 基本的な整合性チェック
    print("\n--- Integrity Checks ---")
    if len(meta_df) == features_mmap.shape[0]:
        print(f"✅ Row count match: {len(meta_df)}")
    else:
        print(f"❌ ROW COUNT MISMATCH! Meta: {len(meta_df)}, Features: {features_mmap.shape[0]}")

    # 6. ターゲットの欠損値チェック (index_meta 内)
    target_cols = [c for c in meta_df.columns if c.startswith('target_')]
    print("\n--- Target Quality (NaN check) ---")
    for col in target_cols:
        nan_count = meta_df[col].isna().sum()
        nan_pct = (nan_count / total_rows) * 100
        print(f"Column '{col}': {nan_count} NaNs ({nan_pct:.2f}%)")
        
        # フィルタ通過行 (is_entry_tac=True) でターゲットが欠損していないか
        if 'is_entry_tac' in meta_df.columns:
            target_nan_in_candidate = meta_df[meta_df['is_entry_tac'] == True][col].isna().sum()
            if target_nan_in_candidate > 0:
                print(f"  ⚠️ Warning: {target_nan_in_candidate} NaNs found in candidate rows!")

    # 7. 特徴量の異常値チェック (サンプルチェック)
    # 8GB環境のため、全データをなめるのではなく先頭・中間・末尾からサンプリング
    print("\n--- Feature Quality (Sample check) ---")
    sample_indices = [0, total_rows // 2, total_rows - 1]
    for idx in sample_indices:
        sample_row = features_mmap[idx]
        num_nans = np.isnan(sample_row).sum()
        num_infs = np.isinf(sample_row).sum()
        print(f"Row {idx:10}: NaNs={num_nans}, Infs={num_infs}")

    # 8. 重複チェック
    dup_count = meta_df.duplicated(subset=['date', 'scode']).sum()
    if dup_count > 0:
        print(f"❌ FATAL: {dup_count} duplicated (date, scode) pairs found!")
    else:
        print("✅ No duplicates found (date, scode).")

    print("\n--- Validation Summary ---")
    print(f"Features: {num_features} columns")
    print(f"Time Range: {meta_df['date'].min()} to {meta_df['date'].max()}")
    print("Verification completed.")

if __name__ == "__main__":
    # プロジェクト構造に合わせたパス指定
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    master_data_dir = os.path.join(base_dir, "data/master")
    verify_master_data(master_data_dir)