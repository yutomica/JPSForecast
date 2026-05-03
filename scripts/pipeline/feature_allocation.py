import os
import yaml
import pandas as pd
import numpy as np
from typing import Tuple, List
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from pathlib import Path
import hydra
from omegaconf import DictConfig

def purge_correlated_features(
    X: pd.DataFrame, 
    y: pd.Series, 
    corr_threshold: float = 0.85
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    ターゲットとのIC(Information Coefficient)を元に予測力を評価し、
    特徴量間の相関に基づく階層的クラスタリングを行って多重共線性を排除します。
    各クラスタからは最も|IC|が高い特徴量（チャンピオン）のみが選出されます。

    Args:
        X (pd.DataFrame): 特徴量のデータフレーム (NaNを含む可能性がある)
        y (pd.Series): ターゲット変数のシリーズ (NaNを含まないことが望ましい)
        corr_threshold (float, optional): 同一クラスタとみなすスピアマン相関の閾値. Defaults to 0.85.

    Returns:
        Tuple[List[str], List[str], pd.DataFrame]: 
            - kept_features: パージを生き残った特徴量のリスト
            - purged_features: 削除された特徴量のリスト
            - cluster_info: 各特徴量のクラスタIDやICスコア等をまとめた検証用データフレーム
    """
    print("Calculating Information Coefficient (IC)...")
    # 1. ICの計算
    # y と X の各特徴量とのスピアマン順位相関の絶対値（|IC|）を計算
    ic_scores = X.corrwith(y, method='spearman').abs()
    # 分散0などでNaNになった場合は0とする
    ic_scores = ic_scores.fillna(0.0)

    print("Calculating Feature Correlation Matrix...")
    # 2. 距離行列の計算
    # Xの特徴量間のスピアマン順位相関行列（rho）を計算
    rho = X.corr(method='spearman')
    rho = rho.fillna(0.0)
    
    # 距離行列 D = 1 - abs(rho)
    D = 1.0 - rho.abs()
    
    # 数値誤差による非対称性や負の値を修正
    D = (D + D.T) / 2.0
    np.fill_diagonal(D.values, 0.0)
    D = D.clip(lower=0.0)

    print("Performing Hierarchical Clustering...")
    # 3. 階層的クラスタリング
    # scipyのlinkageに渡すため、距離行列の上三角成分を一次元配列に変換
    condensed_D = squareform(D.values, checks=False)
    
    # 完全連結法（complete linkage）でクラスタリング
    linkage_matrix = hierarchy.linkage(condensed_D, method='complete')

    # 4. フラットクラスタの抽出
    # 距離閾値 t = 1 - corr_threshold を基準としてフラットなクラスタIDを割り当て
    t = 1.0 - corr_threshold
    cluster_labels = hierarchy.fcluster(linkage_matrix, t, criterion='distance')

    # クラスタリング結果をDataFrameにまとめる
    cluster_info = pd.DataFrame({
        'feature': X.columns,
        'cluster_id': cluster_labels,
        'ic_score': ic_scores.values
    })

    # 5. チャンピオンの選出（パージの実行）
    # ICスコア降順、同一の場合は特徴量名の昇順でソート (決定論的処理)
    cluster_info = cluster_info.sort_values(
        by=['cluster_id', 'ic_score', 'feature'], 
        ascending=[True, False, True]
    )

    # 各クラスタで最初に出現する特徴量（チャンピオン）を候補とする
    candidate_features = cluster_info.groupby('cluster_id').first()['feature'].tolist()

    print("Performing Post-Clustering Greedy Purge...")
    # 6. 完全連結法の性質によるクラスタ間高相関ペアの最終パージ
    # 完全連結法は連鎖パージを防ぐ反面、他メンバーの影響で別クラスタになった特徴量間に
    # 閾値以上の相関が残るケースがあります。ICスコア順に最終的な総当たりチェックを行います。
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

    # 検証用DataFrameの体裁を整える
    cluster_info['is_kept'] = cluster_info['feature'].isin(kept_features)
    cluster_info = cluster_info.sort_index().reset_index(drop=True)

    return kept_features, purged_features, cluster_info


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg: DictConfig):
    master_dir = Path(cfg.data.path)
    
    # 1. 特徴量リストの取得
    feature_cols = cfg.features.get('feature_cols', [])
    if not feature_cols:
        print("Error: feature_cols not found in config.")
        return
    print(f"Loaded {len(feature_cols)} features from config.")

    # 2. ターゲット名称の取得
    target_col = cfg.target.get('column')
    if not target_col:
        print("Error: target column not found in config.")
        return
    print(f"Target column: {target_col}")

    # ドメイン設定からフィルタ列名を作成
    domain_name = "tac"
    if "domain" in cfg:
        if isinstance(cfg.domain, str):
            domain_name = cfg.domain
        elif "name" in cfg.domain:
            domain_name = cfg.domain.name
            
    candidate_col = f"is_candidate_{domain_name.lower()}"
    print(f"Domain filter column: {candidate_col}")

    # 3. データの読み込み
    features_dir = master_dir / "features"
    meta_path = master_dir / "index_meta.parquet"
    
    if not features_dir.exists() or not meta_path.exists():
        print(f"Error: Data paths do not exist.")
        return

    # 先にメタデータを読み込み、有効な行のマスク（NaN除外 ＆ ドメイン候補）を作成する
    meta_df = pd.read_parquet(meta_path, columns=[target_col, candidate_col])
    global_valid_mask = meta_df[target_col].notna() & (meta_df[candidate_col] == True)

    chunk_files = sorted(features_dir.glob("features_chunk_*.parquet"))
    
    df_list = []
    y_list = []
    total_valid_rows = 0
    max_rows = 100000  # 計算資源・メモリを考慮し10万行程度に制限
    
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

    # 4. パージの実行
    # Hydraの閾値設定があれば利用（デフォルトは0.85）
    corr_threshold = cfg.get("corr_threshold", 0.85)
    print(f"Purging correlated features (Threshold: {corr_threshold})...")
    
    kept, purged, cluster_info = purge_correlated_features(X_df, y, corr_threshold=corr_threshold)
    
    print(f"\n--- Purge Summary ---")
    print(f"Original features : {len(feature_cols)}")
    print(f"Kept features     : {len(kept)}")
    print(f"Purged features   : {len(purged)}")

    # 5. 結果の保存
    # 役割名（tac_alpha 等）を取得してファイル名に含める
    target_name = cfg.target.get('name', 'unknown')
    output_csv = f"feature_purge_{target_name}_info.csv"
    cluster_info.to_csv(output_csv, index=False)
    print(f"\nSaved cluster info to {output_csv}")

    output_yaml = f"purged_features_{target_name}.yaml"
    with open(output_yaml, 'w') as f:
        yaml.dump({"feature_cols": sorted(kept)}, f, default_flow_style=False, sort_keys=False)
    print(f"Saved kept features to {output_yaml}")

if __name__ == "__main__":
    main()