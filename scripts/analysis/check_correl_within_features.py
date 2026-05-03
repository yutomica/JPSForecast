import pandas as pd
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    master_dir = Path(cfg.data.path)
    
    # 1. 特徴量リストの取得
    feature_cols = cfg.features.get('feature_cols', [])
    if not feature_cols:
        print("Error: feature_cols not found in config.")
        return
    print(f"Loaded {len(feature_cols)} features from config.")

    # ドメイン設定からフィルタ列名を作成
    domain_name = "tac"
    if "domain" in cfg:
        if isinstance(cfg.domain, str):
            domain_name = cfg.domain
        elif "name" in cfg.domain:
            domain_name = cfg.domain.name
            
    candidate_col = f"is_candidate_{domain_name.lower()}"
    print(f"Domain filter column: {candidate_col}")

    # 2. データの読み込み
    features_dir = master_dir / "features"
    meta_path = master_dir / "index_meta.parquet"
    
    if not features_dir.exists() or not meta_path.exists():
        print(f"Error: Data paths do not exist.")
        return

    # 先にメタデータを読み込み、有効な行のマスク（ドメイン候補）を作成する
    meta_df = pd.read_parquet(meta_path, columns=[candidate_col])
    global_valid_mask = (meta_df[candidate_col] == True)

    chunk_files = sorted(features_dir.glob("features_chunk_*.parquet"))
    if not chunk_files:
        print(f"Error: No chunk files found in {features_dir}.")
        return
    
    df_list = []
    total_valid_rows = 0
    max_rows = 100000  # 計算資源・メモリを考慮し10万行程度に制限
    
    print("Loading data for correlation check...")
    current_idx = 0
    for cf in chunk_files:
        df_chunk = pd.read_parquet(cf, columns=feature_cols)
        chunk_len = len(df_chunk)
        
        chunk_mask = global_valid_mask.iloc[current_idx : current_idx + chunk_len]
        
        if chunk_mask.any():
            df_list.append(df_chunk[chunk_mask.values])
            total_valid_rows += chunk_mask.sum()
            
        current_idx += chunk_len
        
        if total_valid_rows > max_rows:
            break
            
    X_df = pd.concat(df_list, ignore_index=True)
    
    if len(X_df) > max_rows:
        print(f"Sampling {max_rows} rows from {len(X_df)} valid rows...")
        X_df = X_df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        
    print(f"Dataset shape for analysis: X={X_df.shape}")

    # 3. 相関の計算
    print("Calculating Spearman correlation matrix...")
    corr_matrix = X_df.corr(method='spearman').abs()
    
    # 欠損値は0で埋める
    corr_matrix = corr_matrix.fillna(0.0)
    
    # 自身との相関(対角成分)は対象外とするため0にする
    np.fill_diagonal(corr_matrix.values, 0.0)

    # 4. 最大値の出力
    max_corr = corr_matrix.max().max()
    max_col = corr_matrix.max().idxmax()
    max_row = corr_matrix[max_col].idxmax()

    print(f"\n--- Correlation Check Result ---")
    print(f"Features checked : {len(feature_cols)}")
    print(f"Max Spearman Cor : {max_corr:.4f}")
    print(f"Feature Pair     : '{max_row}' and '{max_col}'")

if __name__ == "__main__":
    main()