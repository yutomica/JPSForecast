import numpy as np
import pandas as pd
import os
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
MASTER_DIR = PROJECT_DIR / 'data/master'

def find_high_spearman_correlations(file_path, names_path=None, threshold=0.95, output_csv=None, sample_size=100000):
    """
    features.npyを読み込み、Spearman相関を計算し、
    指定した閾値以上のペアをリストアップする関数。
    """
    
    # ファイルの存在確認
    if not os.path.exists(file_path):
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
        # テスト用にダミーデータを作成するか尋ねるなどの処理も可能ですが、
        # ここでは処理を中断します。
        return

    try:
        # 1. 特徴量名の読み込み
        feature_names = None
        if names_path and os.path.exists(names_path):
            with open(names_path, 'r') as f:
                names_data = json.load(f)
                # pd.Series.to_json() の形式 (dict) または list に対応
                if isinstance(names_data, dict):
                    feature_names = list(names_data.values())
                elif isinstance(names_data, list):
                    feature_names = names_data

        # 2. データの読み込み
        print(f"Loading {file_path}...")
        # create_master_data.py で生成された features.npy は Raw Binary (memmap) 形式のため、
        # np.load ではなく np.memmap で読み込む必要があります。
        if feature_names:
            n_features = len(feature_names)
            dtype = 'float32'
            itemsize = np.dtype(dtype).itemsize
            file_size = os.path.getsize(file_path)
            
            # ファイルサイズからサンプル数を計算
            if file_size % (n_features * itemsize) == 0:
                n_samples = file_size // (n_features * itemsize)
                features = np.memmap(file_path, dtype=dtype, mode='r', shape=(n_samples, n_features))
            else:
                # サイズが合わない場合は通常のnpy形式として試行
                print("Warning: File size implies not a raw binary with known features. Trying np.load...")
                features = np.load(file_path)
        else:
            # 特徴量名ファイルがない場合は通常のnpy形式として試行
            features = np.load(file_path)

        print(f"Data shape: {features.shape} (Samples: {features.shape[0]}, Features: {features.shape[1]})")

        # 3. Pandas DataFrameへ変換
        # 高速化・省メモリ化のため、データ数が多い場合はサンプリングを行う
        n_samples_total = features.shape[0]
        if sample_size and n_samples_total > sample_size:
            print(f"Sampling {sample_size} rows from {n_samples_total} total rows for faster calculation...")
            # ランダムにインデックスを選択し、ソートしてアクセス効率を良くする
            indices = np.random.choice(n_samples_total, sample_size, replace=False)
            indices.sort()
            # memmapから必要な部分だけをメモリにロード
            data_subset = features[indices]
            df = pd.DataFrame(data_subset, columns=feature_names)
        else:
            df = pd.DataFrame(features, columns=feature_names)

        # 4. Spearman順位相関の計算
        print("Calculating Spearman correlation...")
        corr_matrix = df.corr(method='spearman')

        # 5. 閾値以上のペアを抽出
        # 重複（A-BとB-A）と自己相関（対角成分）を除くため、上三角行列のみを取得します
        # k=1 は対角成分の一つ上から開始することを意味します
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # スタックしてSeries形式に変換し、NaN（下三角部分）を除去
        corr_pairs = upper_tri.stack()

        # 閾値 (0.95) 以上でフィルタリング
        high_corr_pairs = corr_pairs[corr_pairs >= threshold]

        # 6. 結果の出力
        print(f"\n--- Pairs with Spearman Correlation >= {threshold} ---")
        print(f"Total pairs found: {len(high_corr_pairs)}")
        
        if len(high_corr_pairs) > 0:
            # 相関係数が高い順にソートして表示
            sorted_pairs = high_corr_pairs.sort_values(ascending=False)
            
            print(f"{'Feature A':<12} | {'Feature B':<12} | {'Correlation':<12}")
            print("-" * 42)
            for (feat_a, feat_b), corr_value in sorted_pairs.items():
                print(f"{feat_a:<12} | {feat_b:<12} | {corr_value:.6f}")
            
            # CSV出力
            if output_csv:
                df_res = sorted_pairs.reset_index()
                df_res.columns = ['Feature_A', 'Feature_B', 'Correlation']
                df_res.to_csv(output_csv, index=False)
                print(f"\nSaved correlation results to: {output_csv}")
        else:
            print("該当するペアはありませんでした。")

    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    # 実行
    target_file = MASTER_DIR / 'features.npy'
    names_file = MASTER_DIR / 'feature_names.json'
    output_file = MASTER_DIR / 'high_correlation_pairs.csv'
    
    find_high_spearman_correlations(target_file, names_path=names_file, output_csv=output_file)
