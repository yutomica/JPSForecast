import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description="指定したYAMLから特徴量を抽出し、新しいmemmapデータを作成します。")
    parser.add_argument(
        "configs",
        nargs="+",
        help="config/features配下のYAMLファイル名（例: features1.yaml features2）"
    )
    args = parser.parse_args()

    # パスの設定
    project_root = Path(__file__).resolve().parent.parent.parent
    data_master_dir = project_root / "data" / "master"
    config_features_dir = project_root / "config" / "features"

    features_npy_path = data_master_dir / "features.npy"
    feature_names_path = data_master_dir / "feature_names.json"
    
    out_npy_path = data_master_dir / "features_select.npy"
    out_names_path = data_master_dir / "features_select_names.json"

    # 1. 抽出対象のカラム名の取得・重複削除
    selected_features = set()
    for config_name in args.configs:
        # .yaml 拡張子を補完
        if not config_name.endswith(".yaml"):
            config_name += ".yaml"
            
        yaml_path = config_features_dir / config_name
        if not yaml_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {yaml_path}")
            
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config and "feature_cols" in config:
                selected_features.update(config["feature_cols"])
            else:
                print(f"警告: {yaml_path} に 'feature_cols' が定義されていません。")

    if not selected_features:
        raise ValueError("抽出対象の特徴量が1つも指定されていません。")

    # 2. 元データのカラム名リストを取得
    if not feature_names_path.exists():
        raise FileNotFoundError(f"元データのカラム名定義が見つかりません: {feature_names_path}")
        
    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names_data = json.load(f)
        if isinstance(feature_names_data, dict):
            original_feature_names = list(feature_names_data.values())
        else:
            original_feature_names = feature_names_data

    original_count = len(original_feature_names)

    print(f"元データ読み込み: {features_npy_path}")
    try:
        # ヘッダー付きの標準 .npy として読み込みを試行
        original_mmap = np.load(features_npy_path, mmap_mode="r")
    except ValueError:
        # ヘッダーなしの生バイナリ（raw memmap）として読み込むフォールバック
        file_size = os.path.getsize(features_npy_path)
        bytes_per_row = original_count * 4  # float32 = 4 bytes
        if file_size % bytes_per_row != 0:
            raise ValueError(f"ファイルサイズ({file_size})が1行あたりのバイト数({bytes_per_row})の倍数ではありません。")
        
        n_samples = file_size // bytes_per_row
        original_mmap = np.memmap(
            features_npy_path, dtype='float32', mode='r', shape=(n_samples, original_count)
        )

    # 元データの順番を維持しつつ、抽出対象のカラムをリスト化
    final_selected_features = [f for f in original_feature_names if f in selected_features]
    
    missing_features = selected_features - set(final_selected_features)
    if missing_features:
        raise ValueError(
            f"エラー: 以下の抽出対象特徴量が元データに存在しません。設定(YAML)のタイポ等を確認してください:\n{missing_features}\n"
        )

    if not final_selected_features:
        raise ValueError("有効な抽出対象特徴量が1つもありませんでした。")

    selected_indices = [original_feature_names.index(f) for f in final_selected_features]

    # 3. memmapとして新データを作成
    n_samples = original_mmap.shape[0]
    n_features = len(final_selected_features)
    new_shape = (n_samples, n_features)

    print(f"新しいmemmapを作成中... 形状: {new_shape}")
    out_mmap = np.memmap(
        out_npy_path, dtype=original_mmap.dtype, mode="w+", shape=new_shape
    )

    # メモリを圧迫しないようチャンクごとにコピー (1万行ずつ)
    chunk_size = 10000
    for i in range(0, n_samples, chunk_size):
        end_idx = min(i + chunk_size, n_samples)
        out_mmap[i:end_idx, :] = original_mmap[i:end_idx, selected_indices]
    out_mmap.flush()
    print(f"データ抽出完了: {out_npy_path}")

    # 4. 抽出後のカラム名を保存
    with open(out_names_path, "w", encoding="utf-8") as f:
        json.dump(final_selected_features, f, indent=4, ensure_ascii=False)
    print(f"特徴量名リスト保存完了: {out_names_path}")

if __name__ == "__main__":
    main()