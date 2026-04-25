import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from pathlib import Path
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg: DictConfig):
    master_dir = Path(cfg.data.path)
    
    # 1. yamlからfeature_colsを読み込む
    feature_cols = cfg.features.get('feature_cols', [])
    if not feature_cols:
        print("feature_cols not found in config.")
        return
        
    print(f"Loaded {len(feature_cols)} features from config.")

    # 2. parquetデータ読み込み
    features_dir = master_dir / "features"
    if not features_dir.exists():
        print(f"Features directory not found: {features_dir}")
        return

    chunk_files = sorted(features_dir.glob("features_chunk_*.parquet"))
    
    df_list = []
    total_rows = 0
    max_rows = 200000  # 計算資源・メモリを考慮し20万行程度に制限
    
    print("Loading data for analysis...")
    for cf in chunk_files:
        df_chunk = pd.read_parquet(cf, columns=feature_cols)
        df_list.append(df_chunk)
        total_rows += len(df_chunk)
        if total_rows > max_rows:
            break
            
    df = pd.concat(df_list, ignore_index=True)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    
    print(f"Dataset shape for analysis: {df.shape}")
    
    # 数値型の列のみを抽出（文字列などのカテゴリカル変数は除外）
    df = df.select_dtypes(include=[np.number, 'bool'])
    dropped_cols = set(feature_cols) - set(df.columns)
    if dropped_cols:
        print(f"Dropped {len(dropped_cols)} non-numeric columns: {list(dropped_cols)}")
    feature_cols = df.columns.tolist()

    # 欠損値処理
    df = df.fillna(0)
    
    # --- スピアマン順位相関の算出とCSV出力 ---
    print("Calculating Spearman correlation...")
    corr, _ = spearmanr(df)
    corr = np.nan_to_num(corr, nan=0.0)
    
    corr_df = pd.DataFrame(corr, index=feature_cols, columns=feature_cols)
    corr_output_path = 'spearman_correlation.csv'
    corr_df.to_csv(corr_output_path)
    print(f"Saved Spearman correlation matrix to {corr_output_path}")

    # --- PCAによるEffective Rank決定 ---
    print("Calculating PCA...")
    # ゼロ除算を避けるために標準偏差が0の列は1にする
    std = df.std().replace(0, 1)
    df_std = (df - df.mean()) / std
    
    pca = PCA()
    pca.fit(df_std)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Explained Variance')
    plt.title('Cumulative Explained Variance by PCA')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True)
    
    pca_output_path = 'pca_cumulative_variance.png'
    plt.savefig(pca_output_path)
    print(f"Saved PCA plot to {pca_output_path}")
    
    # 累積寄与率が80%を超える最初の主成分数を取得
    n_components_80 = int(np.argmax(cumulative_variance_ratio >= 0.8) + 1)
    print(f"Number of components explaining 80% variance (Effective Rank estimate): {n_components_80}")
    
    # --- 階層クラスタリング ---
    # 距離行列: 1 - |相関|
    distance_matrix = 1 - np.abs(corr)
    # 数値誤差を吸収して完全な対称行列にする
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    condensed_distance = squareform(distance_matrix, checks=False)
    
    print("Performing hierarchical clustering...")
    linkage_matrix = hierarchy.linkage(condensed_distance, method='complete')
    
    plt.figure(figsize=(15, 8))
    hierarchy.dendrogram(linkage_matrix, labels=feature_cols, leaf_rotation=90, leaf_font_size=8)
    plt.title('Hierarchical Clustering Dendrogram (Features)')
    plt.xlabel('Features')
    plt.ylabel('Distance (1 - |Spearman Correlation|)')
    plt.tight_layout()
    
    dendro_output_path = 'hierarchical_clustering_dendrogram.png'
    plt.savefig(dendro_output_path)
    print(f"Saved Dendrogram plot to {dendro_output_path}")
    
    # --- ユーザーからのクラスタ数入力 ---
    print("\n--- Clustering Setup ---")
    print(f"Suggested Effective Rank (Clusters): {n_components_80}")
    
    # Hydraからの指定があれば利用し、なければ標準入力を求める
    n_clusters = cfg.get("n_clusters", None)
    if n_clusters is None:
        try:
            val = input(f"Enter the number of clusters to form (default={n_components_80}): ")
            n_clusters = int(val) if val.strip() else n_components_80
        except (EOFError, ValueError):
            print(f"Invalid input or environment, defaulting to {n_components_80}")
            n_clusters = n_components_80
            
    print(f"Using {n_clusters} clusters.")
    
    cluster_labels = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    grouped_features = {}
    for i in range(1, n_clusters + 1):
        grouped_features[f"group_{i}"] = []
        
    for i, col in enumerate(feature_cols):
        grouped_features[f"group_{cluster_labels[i]}"].append(col)
        
    # 空のグループを削除
    grouped_features = {k: v for k, v in grouped_features.items() if len(v) > 0}
    
    output_yaml = "clustered_features.yaml"
    with open(output_yaml, 'w') as f:
        yaml.dump({"feature_groups": grouped_features}, f, default_flow_style=False, sort_keys=False)
        
    print(f"Saved clustered feature groups to {output_yaml}")
    print("Done.")

if __name__ == "__main__":
    main()
