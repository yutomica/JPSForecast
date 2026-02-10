import os
import numpy as np
import pandas as pd
import json
import gc
import time

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
    print("\n--- Feature Quality (Sample check) ---")
    sample_indices = [0, total_rows // 2, total_rows - 1]
    for idx in sample_indices:
        sample_row = features_mmap[idx]
        num_nans = np.isnan(sample_row).sum()
        num_infs = np.isinf(sample_row).sum()
        print(f"Row {idx:10}: NaNs={num_nans}, Infs={num_infs}")

    # --- [追加] 8. 全特徴量の欠損率チェック (全数抽出・高速版) ---
    print("\n--- Global Feature Missing Rate Scan (Full Data) ---")
    start_time = time.time()
    # 8GB RAM を考慮したチャンクサイズ (約20万行 * 特徴量数 * float32)
    chunk_size = 200000 
    nan_counts = np.zeros(num_features, dtype=np.int64)
    inf_counts = np.zeros(num_features, dtype=np.int64)
    
    for i in range(0, total_rows, chunk_size):
        chunk = features_mmap[i : i + chunk_size]
        nan_counts += np.isnan(chunk).sum(axis=0)
        inf_counts += np.isinf(chunk).sum(axis=0)
        
        # 進捗表示
        if (i // chunk_size) % 5 == 0:
            progress = min((i + chunk_size) / total_rows * 100, 100.0)
            print(f"  Scanning: {progress:.1f}% complete...")

    elapsed = time.time() - start_time
    print(f"✅ Scan completed in {elapsed:.1f} seconds.")

    # 結果の集計
    missing_df = pd.DataFrame({
        'feature': feature_names,
        'nan_count': nan_counts,
        'inf_count': inf_counts,
        'missing_rate_pct': (nan_counts / total_rows) * 100
    }).sort_values(by='missing_rate_pct', ascending=False)

    # 欠損率が高い上位10件を表示
    print("\nTop 10 features with highest missing rates (NaN):")
    print(missing_df.head(10)[['feature', 'nan_count', 'missing_rate_pct']].to_string(index=False))

    # 全結果をCSV保存
    report_path = os.path.join(master_dir, "feature_missing_report.csv")
    missing_df.to_csv(report_path, index=False)
    print(f"\n✅ Full missing rate report saved to: {report_path}")

    # 9. 重複チェック
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
    # scripts/validation/validate_master_data.py から見た data/master の相対パス
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    master_data_dir = os.path.join(base_dir, "data/master")
    
    if os.path.exists(master_data_dir):
        verify_master_data(master_data_dir)
    else:
        print(f"❌ Master data directory not found at: {master_data_dir}")