import os
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

def main():
    # MLflowのトラッキングURIを設定 (predict.pyなどと合わせる)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    master_dir = Path("./data/master")
    output_dir = Path("./data/stacking_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("MLflowからProductionモデルのOOFスコアを取得中...")
    registered_models = client.search_registered_models(filter_string="name LIKE '%'")
    production_versions = [v for rm in registered_models for v in rm.latest_versions if v.current_stage == 'Production']
    stacking_df = None
    for v in production_versions:
        try:
            local_dir = client.download_artifacts(v.run_id, "oof_data")
            csv_files = glob.glob(os.path.join(local_dir, "*.csv"))
            if not csv_files:
                continue
            oof_df = pd.read_csv(csv_files[0])
            if 'score' in oof_df.columns:
                oof_sub = oof_df[['date', 'scode', 'score']].rename(columns={'score': f'score_{v.name}'})
                if stacking_df is None:
                    stacking_df = oof_sub
                else:
                    stacking_df = pd.merge(stacking_df, oof_sub, on=['date', 'scode'], how='outer')
        except Exception as e:
            print(f"⚠️ モデル {v.name} のOOFデータ取得に失敗しました: {e}")
    stacking_df = stacking_df.dropna()
    stacking_df['date'] = pd.to_datetime(stacking_df['date'])
    if stacking_df is None or stacking_df.empty:
        print("❌ OOFデータが見つかりませんでした。処理を終了します。")
        return

    print("マスターデータとの結合とマーケット特徴量の抽出中...")
    master_meta_path = master_dir / "index_meta.parquet"
    features_dir = master_dir / "features"
    feature_names_path = master_dir / "feature_names.json"
    try:
        original_meta = pd.read_parquet(master_meta_path)
    except Exception as e:
        print(f"❌ 基礎データ ({master_meta_path}) の読み込みに失敗しました。")
        return
    # OOFデータが存在するレコードのみを残す（内部結合）
    merged_meta = pd.merge(original_meta, stacking_df, on=['date', 'scode'], how='inner')
    # マーケット特徴量の抽出 (features.npy から効率的に抽出)
    target_features = [
        'BET_MarketReturn_RAW', 'BET_MarketReturn_TSZ_20D', 
        'BET_MarketTrendIdx_RAW', 'BET_MarketTrendIdx_TSR_252D', 
        'BET_MarketHV20_RAW', 'BET_MarketHV20_TSR_252D', 
        'BET_MarketVolChange_RAW', 'BET_MarketVolChange_TSR_252D', 
        'BET_SectorReturn_RAW', 'VOL_HistVol20_TSR_252D'
    ]
    if features_dir.exists() and feature_names_path.exists():
        all_features = pd.read_json(feature_names_path, typ='series').tolist()
        
        cols_to_load = [f for f in target_features if f in all_features]
        chunk_files = sorted(features_dir.glob("features_chunk_*.parquet"))
        loaded_chunks = []
        for cf in chunk_files:
            df_chunk = pd.read_parquet(cf, columns=cols_to_load)
            loaded_chunks.append(df_chunk)
        features_df = pd.concat(loaded_chunks, ignore_index=True)
        
        # merged_meta の行に対応するインデックス（元のoriginal_metaにおける行番号）を使って抽出
        original_indices = merged_meta.index.values # merge前のindexは保持されないので下記で対応
        original_meta['original_idx'] = np.arange(len(original_meta))
        merged_meta = pd.merge(merged_meta, original_meta[['date', 'scode', 'original_idx']], on=['date', 'scode'], how='left')
        for feature in target_features:
            if feature in cols_to_load:
                merged_meta[feature] = features_df[feature].values[merged_meta['original_idx'].values]
            else:
                merged_meta[feature] = np.nan
        merged_meta = merged_meta.drop(columns=['original_idx'])

    # 特特徴追加
    print("特徴量を追加...")
    score_cols = [c for c in merged_meta.columns if c.startswith('score_')]
    for col in score_cols:
        merged_meta[f"RNK_{col}"] = merged_meta.groupby('date')[col].rank(pct=True)
        target_features.append(f"RNK_{col}")
    rank_cols = [c for c in merged_meta.columns if c.startswith('RNK_')]
    merged_meta['RankMean'] = merged_meta[rank_cols].mean(axis=1)
    target_features.append('RankMean')
    merged_meta['RankStd'] = merged_meta[rank_cols].std(axis=1)
    target_features.append('RankStd')
    # merged_meta['ModelDisagree'] = (merged_meta['RNK_score_LightGBM_target_tac_gauss_rank'] - merged_meta['RNK_score_TabNet_target_tac_gauss_rank']).abs()
    # target_features.append('ModelDisagree')
    merged_meta = merged_meta.sort_values(['date', 'scode']).reset_index(drop=True)
    
    print("スタッキング用データセットの保存中...")
    feature_cols = [c for c in merged_meta.columns if c.startswith('score_') or c in target_features]
    print(f"スタッキング用特徴量 ({len(feature_cols)}件): {feature_cols}")
    
    # 特徴量の保存 (Train.py に合わせて Parquet チャンクとして出力)
    out_features_dir = output_dir / "features"
    out_features_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = out_features_dir / "features_chunk_00.parquet"
    chunk_cols = ['scode', 'date'] + feature_cols
    merged_meta[chunk_cols].to_parquet(chunk_path, index=False)
    
    with open(output_dir / "feature_names.json", 'w') as f:
        json.dump(feature_cols, f)
    # メタデータの保存 (特徴量カラムを除外して保存)
    merged_meta.drop(columns=feature_cols).to_parquet(output_dir / "index_meta.parquet", index=False)
    
    print(f"✅ スタッキング用データセットの生成が完了しました！出力先: {output_dir}")
    print(f" - サンプル数: {len(merged_meta):,}")

if __name__ == "__main__":
    main()
