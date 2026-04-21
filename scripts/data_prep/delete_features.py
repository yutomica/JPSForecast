import os
import json
import numpy as np
import pandas as pd
import argparse
import gc
import shutil
from pathlib import Path
from tqdm import tqdm

# プロジェクトのルートディレクトリ設定
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MASTER_DIR = PROJECT_DIR / 'data/master'
DELETE_LIST_PATH = PROJECT_DIR / 'data/master/delete_features.csv'

def load_delete_features(csv_path):
    """
    削除対象の特徴量リストを読み込む
    ヘッダーがあることを前提とし、1列目を特徴量名として扱う
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"削除リストが見つかりません: {csv_path}")
    
    # ヘッダーありとして読み込み (header=0)
    df = pd.read_csv(csv_path, header=0)
    if df.empty:
        return set()
    
    # 1列目を使用する
    delete_list = set(df.iloc[:, 0].astype(str).values)
    print(f"Delete list loaded from {csv_path}: {len(delete_list)} features to drop.")
    return delete_list

def drop_features(target_dir, delete_csv_path=DELETE_LIST_PATH):
    target_dir = Path(target_dir)
    print(f"--- Starting Drop Features Process for: {target_dir} ---")

    # ファイルパス定義
    names_path = target_dir / "feature_names.json"
    features_dir = target_dir / "features"
    backup_dir = target_dir / "features_backup"

    # 1. 必須ファイルの存在確認
    if not names_path.exists() or not features_dir.exists():
        print(f"Error: Required files or directories not found in {target_dir}")
        return

    # 2. 特徴量名(JSON)のロード
    with open(names_path, 'r') as f:
        feature_names = json.load(f)
        # dict形式の場合はvaluesをリスト化、listならそのまま
        if isinstance(feature_names, dict):
            feature_names = list(feature_names.values())
    
    original_count = len(feature_names)
    print(f"Original feature count: {original_count}")

    # 3. 削除リストのロードとフィルタリング
    features_to_delete = load_delete_features(delete_csv_path)
    
    if not features_to_delete:
        print("No features to delete. Exiting.")
        return

    # 保持するインデックスと名前を決定
    keep_indices = []
    new_feature_names = []
    dropped_count = 0

    for idx, name in enumerate(feature_names):
        if name in features_to_delete:
            dropped_count += 1
        else:
            keep_indices.append(idx)
            new_feature_names.append(name)
    
    if dropped_count == 0:
        print("指定された削除対象の特徴量は、現在のデータに含まれていません。処理を終了します。")
        return

    print(f"Features to drop: {dropped_count}")
    print(f"Features to keep: {len(new_feature_names)}")

    chunk_files = sorted(features_dir.glob("features_chunk_*.parquet"))
    if not chunk_files:
        print("No parquet chunks found.")
        return

    # 安全のため、既存のfeaturesディレクトリをバックアップ
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(features_dir, backup_dir)
    print(f"Backed up original features to: {backup_dir}")

    try:
        # チャンクごとに不要な列を削除
        for cf in tqdm(chunk_files, desc="Processing chunks"):
            df = pd.read_parquet(cf)
            cols_to_drop = [c for c in features_to_delete if c in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                df.to_parquet(cf, index=False)

        gc.collect()
        
        # 5. feature_names.json の更新
        with open(names_path, 'w') as f:
            json.dump(new_feature_names, f)
        print(f"✅ Updated {names_path}")

        # バックアップの削除（成功した場合）
        # ※ 安全のため削除したくない場合はコメントアウトしてください
        # os.remove(backup_path)
        # print("✅ Process completed successfully. Backup removed.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("Restoring backup...")
        if features_dir.exists():
            shutil.rmtree(features_dir)
        shutil.copytree(backup_dir, features_dir)
        print("Restored original features.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drop specified features from features.npy and feature_names.json")
    parser.add_argument("--target_dir", type=str, default=str(DEFAULT_MASTER_DIR),
                        help="Target directory containing features.npy (default: data/master)")
    parser.add_argument("--delete_list", type=str, default=str(DELETE_LIST_PATH),
                        help="Path to CSV containing features to delete (default: data/master/delete_features.csv)")
    
    args = parser.parse_args()
    
    if os.path.exists(args.target_dir):
        drop_features(args.target_dir, args.delete_list)
    else:
        print(f"Target directory does not exist: {args.target_dir}")
