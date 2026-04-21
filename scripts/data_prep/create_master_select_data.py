import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


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

    features_dir = data_master_dir / "features"
    feature_names_path = data_master_dir / "feature_names.json"
    
    out_features_dir = data_master_dir / "features_select"
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

    print(f"元データディレクトリ: {features_dir}")

    # 元データの順番を維持しつつ、抽出対象のカラムをリスト化
    final_selected_features = [f for f in original_feature_names if f in selected_features]
    
    missing_features = selected_features - set(final_selected_features)
    if missing_features:
        raise ValueError(
            f"エラー: 以下の抽出対象特徴量が元データに存在しません。設定(YAML)のタイポ等を確認してください:\n{missing_features}\n"
        )

    if not final_selected_features:
        raise ValueError("有効な抽出対象特徴量が1つもありませんでした。")

    # 3. Parquetチャンクとして新データを作成
    out_features_dir.mkdir(parents=True, exist_ok=True)
    chunk_files = sorted(features_dir.glob("features_chunk_*.parquet"))
    
    print(f"新しいParquetチャンクを作成中... 抽出特徴量数: {len(final_selected_features)}")
    for cf in tqdm(chunk_files, desc="Processing chunks"):
        df = pd.read_parquet(cf)
        cols_to_save = [c for c in ['scode', 'date'] if c in df.columns] + final_selected_features
        
        out_chunk_path = out_features_dir / cf.name
        df[cols_to_save].to_parquet(out_chunk_path, index=False)
        
    print(f"データ抽出完了: {out_features_dir}")

    # 4. 抽出後のカラム名を保存
    with open(out_names_path, "w", encoding="utf-8") as f:
        json.dump(final_selected_features, f, indent=4, ensure_ascii=False)
    print(f"特徴量名リスト保存完了: {out_names_path}")

if __name__ == "__main__":
    main()