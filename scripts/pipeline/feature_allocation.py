import os
import yaml
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import average_precision_score
from pathlib import Path
import hydra
from omegaconf import DictConfig

def _calc_weighted_ap_score(X: pd.DataFrame, y_class: pd.Series) -> pd.Series:
    """
    tac_risk_class (0, 1, 2, 3) 用の予測力スコア。
    各閾値 (>=1, >=2, >=3) に対する AP の加重平均を計算。
    特徴量ごとに正負両方の方向を試し、最大値を取る。
    """
    # ターゲットを数値化
    y_class = pd.to_numeric(y_class, errors='coerce')
    valid_y = y_class.notna()
    y_class = y_class[valid_y].values
    X = X.iloc[valid_y.values]

    y_5 = (y_class >= 1).astype(int)
    y_7 = (y_class >= 2).astype(int)
    y_10 = (y_class >= 3).astype(int)

    weights = [0.25, 0.35, 0.40]
    targets = [y_5, y_7, y_10]

    scores = {}

    print(f"Calculating Weighted AP for {len(X.columns)} features...")
    for col in X.columns:
        # 特徴量を数値化し、明示的に float 配列に変換
        x_numeric = pd.to_numeric(X[col], errors='coerce')
        x = x_numeric.values.astype(float)
        mask = np.isfinite(x)
        
        if not mask.any():
            scores[col] = 0.0
            continue

        xt = x[mask]

        feat_ap = 0.0
        for w, yt_full in zip(weights, targets):
            yt = yt_full[mask]
            if len(np.unique(yt)) < 2:
                continue

            # 正負両方向で高い方のAPを取る
            try:
                # 警告を避けるため、極端なケースをチェック
                if np.unique(xt).size < 2:
                    continue
                ap_pos = average_precision_score(yt, xt)
                ap_neg = average_precision_score(yt, -xt)
                feat_ap += w * max(ap_pos, ap_neg)
            except Exception:
                pass

        scores[col] = feat_ap

    return pd.Series(scores)

def purge_correlated_features(
    X: pd.DataFrame, 
    y: pd.Series, 
    corr_threshold: float = 0.85,
    task_type: str = 'regression',
    target_name: str = 'unknown'
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    ターゲットとの予測力（IC または AP）を元に評価し、
    特徴量間の相関に基づく階層的クラスタリングを行って多重共線性を排除します。
    """
    print(f"Calculating Predictive Power (Task: {task_type}, Target: {target_name})...")

    # 1. 予測力（重要度）の計算
    if task_type == 'multiclass' and 'risk' in target_name:
        # risk ターゲットの場合は AP ベースのスコアを使用
        ic_scores = _calc_weighted_ap_score(X, y)
    else:
        # 通常（Alpha等）は Spearman IC の絶対値を使用
        ic_scores = X.corrwith(y, method='spearman').abs()

    # 分散0などでNaNになった場合は0とする
    ic_scores = ic_scores.fillna(0.0)

    print("Calculating Feature Correlation Matrix (Spearman)...")
    # 2. 距離行列の計算
    rho = X.corr(method='spearman')
    rho = rho.fillna(0.0)

    # 距離行列 D = 1 - abs(rho)
    D = 1.0 - rho.abs()
    D = (D + D.T) / 2.0
    np.fill_diagonal(D.values, 0.0)
    D = D.clip(lower=0.0)

    print("Performing Hierarchical Clustering...")
    # 3. 階層制クラスタリング
    condensed_D = squareform(D.values, checks=False)
    linkage_matrix = hierarchy.linkage(condensed_D, method='complete')

    # 4. フラットクラスタの抽出
    t = 1.0 - corr_threshold
    cluster_labels = hierarchy.fcluster(linkage_matrix, t, criterion='distance')

    cluster_info = pd.DataFrame({
        'feature': X.columns,
        'cluster_id': cluster_labels,
        'ic_score': ic_scores.values
    })

    # 5. チャンピオンの選出
    cluster_info = cluster_info.sort_values(
        by=['cluster_id', 'ic_score', 'feature'], 
        ascending=[True, False, True]
    )

    candidate_features = cluster_info.groupby('cluster_id').first()['feature'].tolist()

    print("Performing Post-Clustering Greedy Purge...")
    candidate_info = cluster_info[cluster_info['feature'].isin(candidate_features)]
    candidate_info = candidate_info.sort_values(by=['ic_score', 'feature'], ascending=[False, True])

    kept_features = []
    for f in candidate_info['feature']:
        conflict = False
        for k in kept_features:
            if abs(rho.loc[f, k]) > corr_threshold:
                conflict = True
                break
        if not conflict:
            kept_features.append(f)

    purged_features = [f for f in X.columns if f not in kept_features]

    cluster_info['is_kept'] = cluster_info['feature'].isin(kept_features)
    cluster_info = cluster_info.sort_index().reset_index(drop=True)

    return kept_features, purged_features, cluster_info


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg: DictConfig):
    master_dir = Path(cfg.data.path)

    feature_cols = cfg.features.get('feature_cols', [])
    if not feature_cols:
        print("Error: feature_cols not found in config.")
        return
    print(f"Loaded {len(feature_cols)} features from config.")

    target_col = cfg.target.get('column')
    task_type = cfg.target.get('task_type', 'regression')
    target_name = cfg.target.get('name', 'unknown')

    if not target_col:
        print("Error: target column not found in config.")
        return
    print(f"Target: {target_name} ({target_col}), Task: {task_type}")

    domain_name = "tac"
    if "domain" in cfg:
        if isinstance(cfg.domain, str):
            domain_name = cfg.domain
        elif "name" in cfg.domain:
            domain_name = cfg.domain.name

    candidate_col = f"is_candidate_{domain_name.lower()}"
    print(f"Domain filter column: {candidate_col}")

    features_dir = master_dir / "features"
    meta_path = master_dir / "index_meta.parquet"

    if not features_dir.exists() or not meta_path.exists():
        print(f"Error: Data paths do not exist.")
        return

    meta_df = pd.read_parquet(meta_path, columns=[target_col, candidate_col])
    global_valid_mask = meta_df[target_col].notna() & (meta_df[candidate_col] == True)

    chunk_files = sorted(features_dir.glob("features_chunk_*.parquet"))

    df_list = []
    y_list = []
    total_valid_rows = 0
    max_rows = 100000 

    print("Loading data for analysis...")
    current_idx = 0
    for cf in chunk_files:
        df_chunk = pd.read_parquet(cf, columns=feature_cols)
        chunk_len = len(df_chunk)

        chunk_mask = global_valid_mask.iloc[current_idx : current_idx + chunk_len]

        if chunk_mask.any():
            df_list.append(df_chunk[chunk_mask.values])
            y_list.append(meta_df[target_col].iloc[current_idx : current_idx + chunk_len][chunk_mask.values])
            total_valid_rows += chunk_mask.sum()

        current_idx += chunk_len

        if total_valid_rows > max_rows:
            break

    X_df = pd.concat(df_list, ignore_index=True)
    y = pd.concat(y_list, ignore_index=True)

    if len(X_df) > max_rows:
        print(f"Sampling {max_rows} rows from {len(X_df)} valid rows...")
        sample_idx = np.random.RandomState(42).choice(len(X_df), max_rows, replace=False)
        sample_idx.sort()
        X_df = X_df.iloc[sample_idx].reset_index(drop=True)
        y = y.iloc[sample_idx].reset_index(drop=True)

    print(f"Dataset shape for analysis: X={X_df.shape}, y={y.shape}")

    corr_threshold = cfg.get("corr_threshold", 0.85)
    print(f"Purging correlated features (Threshold: {corr_threshold})...")

    kept, purged, cluster_info = purge_correlated_features(
        X_df, y, 
        corr_threshold=corr_threshold,
        task_type=task_type,
        target_name=target_name
    )

    print(f"\n--- Purge Summary ---")
    print(f"Original features : {len(feature_cols)}")
    print(f"Kept features     : {len(kept)}")
    print(f"Purged features   : {len(purged)}")

    output_csv = f"feature_purge_{target_name}_info.csv"
    cluster_info.to_csv(output_csv, index=False)
    print(f"\nSaved cluster info to {output_csv}")

    output_yaml = f"purged_features_{target_name}.yaml"
    with open(output_yaml, 'w') as f:
        yaml.dump({"feature_cols": sorted(kept)}, f, default_flow_style=False, sort_keys=False)
    print(f"Saved kept features to {output_yaml}")

if __name__ == "__main__":
    main()