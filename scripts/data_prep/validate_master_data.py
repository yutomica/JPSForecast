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
        if isinstance(feature_names, dict):
            feature_names = list(feature_names.values())
    
    num_features = len(feature_names)
    print(f"✅ Loaded {num_features} feature names.")

    # 3. メタデータの検証（インデックスをリセットして物理行と同期させる）
    meta_df = pd.read_parquet(meta_path).reset_index(drop=True)
    total_rows = len(meta_df)
    print(f"✅ Meta data loaded: {total_rows} rows.")

    # 4. memmap のロード
    features_mmap = np.memmap(features_path, dtype='float32', mode='r', 
                              shape=(total_rows, num_features))
    
    # sector_return の欠損チェックと出力
    check_col = 'sector_return'
    if check_col in feature_names:
        print(f"\n--- Checking for missing {check_col} ---")
        col_idx = feature_names.index(check_col)
        col_data = features_mmap[:, col_idx]
        missing_mask = np.isnan(col_data) | np.isinf(col_data)
        missing_count = np.sum(missing_mask)
        print(f"Found {missing_count} records with missing {check_col}.")
        
        if missing_count > 0:
            missing_rows = meta_df.iloc[np.where(missing_mask)[0]][['date', 'scode']]
            out_path = os.path.join(master_dir, f"missing_{check_col}_records.csv")
            missing_rows.to_csv(out_path, index=False)
            print(f"✅ Exported missing {check_col} records to: {out_path}")

    # ターゲット列の特定
    target_cols = [c for c in meta_df.columns if c.startswith('target_')]
    
    # 年カラムの作成
    meta_df['year'] = pd.to_datetime(meta_df['date']).dt.year

    # 7. 候補フラグの年別集計 (サンプリング適用)
    print("\n--- Candidate Flag Counts by Year (with Sampling) ---")
    # train.pyのapply_samplingと同様のロジックを適用
    # is_candidate_tac: interval=5, is_candidate_str: interval=20
    sampling_rules = {'is_candidate_tac': 5, 'is_candidate_str': 20}
    valid_cols = [c for c in sampling_rules.keys() if c in meta_df.columns]
    
    if valid_cols:
        # 集計用にコピーを作成してソート (元のmeta_dfの順序はmmapとの対応のため維持する)
        temp_df = meta_df[['date', 'scode', 'year'] + valid_cols].copy()
        temp_df = temp_df.sort_values(['scode', 'date'])
        temp_df['seq_id'] = temp_df.groupby('scode').cumcount()
        
        candidate_stats = temp_df.groupby('year').size().reset_index(name='total_rows')
        
        for col in valid_cols:
            interval = sampling_rules[col]
            # 1. フラグがTrueのレコードに絞る
            df_true = temp_df[temp_df[col]].copy()
            
            # サンプリング前の件数を集計
            raw_counts = df_true.groupby('year').size().reset_index(name=f'{col}_raw')
            candidate_stats = candidate_stats.merge(raw_counts, on='year', how='left').fillna(0)
            candidate_stats[f'{col}_raw'] = candidate_stats[f'{col}_raw'].astype(int)
            
            # 2. 残った行のdateの値を参照し、連続する行のdateがinterval期間以上となるようにデータを間引く
            if not df_true.empty:
                scodes = df_true['scode'].values
                dates = df_true['date'].values
                keep_mask = np.zeros(len(df_true), dtype=bool)
                
                last_scode = None
                last_date = np.datetime64('1900-01-01')
                interval_td = np.timedelta64(interval, 'D')
                
                for i in range(len(df_true)):
                    if scodes[i] != last_scode or (dates[i] - last_date) >= interval_td:
                        keep_mask[i] = True
                        last_scode = scodes[i]
                        last_date = dates[i]
                
                counts = df_true[keep_mask].groupby('year').size().reset_index(name=col)
            else:
                counts = pd.DataFrame(columns=['year', col])

            candidate_stats = candidate_stats.merge(counts, on='year', how='left').fillna(0)
            candidate_stats[col] = candidate_stats[col].astype(int)
            print(f"Applied sampling interval={interval} for {col}")
            
        print(candidate_stats)
        cand_out_path = os.path.join(master_dir, "candidate_counts_by_year.csv")
        candidate_stats.to_csv(cand_out_path, index=False)
        print(f"✅ Candidate counts saved to: {cand_out_path}")
        del temp_df
        gc.collect()

    # 8. 年別・全項目の欠損率チェック
    print("\n--- Global Missing Rate Scan by Year (Features & Targets) ---")
    start_time = time.time()
    
    years = sorted(meta_df['year'].unique())
    
    chunk_size = 200000 
    yearly_reports = []

    for year in years:
        print(f"  Processing Year: {year}...")
        year_mask = meta_df['year'] == year
        # 物理的な行番号(0, 1, 2...)を取得
        year_pos = np.where(year_mask)[0]
        year_total_rows = len(year_pos)
        
        if year_total_rows == 0: continue

        nan_counts_feat = np.zeros(num_features, dtype=np.int64)
        inf_counts_feat = np.zeros(num_features, dtype=np.int64)
        
        # チャンクごとに memmap から読み込み
        for i in range(0, year_total_rows, chunk_size):
            batch_indices = year_pos[i : i + chunk_size]
            chunk = features_mmap[batch_indices, :]
            
            nan_counts_feat += np.isnan(chunk).sum(axis=0)
            inf_counts_feat += np.isinf(chunk).sum(axis=0)

        # ターゲット（Meta DataFrame側）の欠損集計
        year_meta_subset = meta_df.iloc[year_pos]
        target_nan_counts = year_meta_subset[target_cols].isna().sum().values
        target_inf_counts = np.isinf(year_meta_subset[target_cols].select_dtypes(include=[np.number])).sum().values

        # 特徴量とターゲットの結果を結合
        combined_names = feature_names + target_cols
        combined_nans = np.concatenate([nan_counts_feat, target_nan_counts])
        combined_infs = np.concatenate([inf_counts_feat, target_inf_counts])
        combined_types = ['feature'] * num_features + ['target'] * len(target_cols)

        # 実質欠損率 = (NaN + Inf) / 全行数
        year_df = pd.DataFrame({
            'year': year,
            'column_name': combined_names,
            'type': combined_types,
            'nan_count': combined_nans,
            'inf_count': combined_infs,
            'missing_rate_pct': ((combined_nans + combined_infs) / year_total_rows) * 100
        })
        yearly_reports.append(year_df)

    # 保存処理
    full_missing_df = pd.concat(yearly_reports, ignore_index=True)
    report_path = os.path.join(master_dir, "missing_report_by_year.csv")
    full_missing_df.to_csv(report_path, index=False)
    
    print(f"✅ Full report saved to: {report_path}")

if __name__ == "__main__":
    # 実行パスは環境に合わせて調整してください
    master_data_dir = "data/master"
    if os.path.exists(master_data_dir):
        verify_master_data(master_data_dir)