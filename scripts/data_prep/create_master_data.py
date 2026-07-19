import shutil
import pyarrow.dataset as ds
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)
import glob
import mlflow
from mlflow.tracking import MlflowClient
import gc
from scipy.special import erfinv
from tqdm import tqdm
import random
import pyarrow.parquet as pq
from pathlib import Path
from src.features.engineer import FeatureEngineer
from src.data_loader.filter import FinancialUniverseEngine
import logging

# MLflow (alembic) のログを抑制
logging.getLogger("alembic").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_DIR / 'data/intermediate'
OUTPUT_DIR = PROJECT_DIR / 'data/master'
SAMPLE_OUTPUT_DIR = PROJECT_DIR / 'data/sample' # サンプル出力先
CANDIDATE_COLS = [
    'is_candidate_5d',
    'is_candidate_10d',
    'is_candidate_20d',
    'is_candidate_40d',
    'is_candidate_60d',
]
LEGACY_CANDIDATE_COLS = ['is_candidate_tac', 'is_candidate_str']


def main(mode = "full"):
    filter = FinancialUniverseEngine()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chunk_files = sorted(glob.glob(f"{INPUT_DIR}/date_chunks/*.parquet"))
    n_stocks = 200 # サンプル作成時の銘柄数

    # 銘柄リストの取得とサンプリング
    if mode == "sample":
        print("Scanning unique scodes...")
        all_scodes = set()
        for f in chunk_files[:10]: # 高速化のため最初の数ファイルから銘柄を抽出
            tmp = pd.read_parquet(f, columns=['scode'])
            all_scodes.update(tmp['scode'].unique())
        selected_scodes = random.sample(list(all_scodes), min(n_stocks, len(all_scodes)))
        print(f"Selected {len(selected_scodes)} stocks for sampling.")
    
    # サイズ確定のためのダミー実行
    print("Pre-scanning to determine feature list...")
    if chunk_files:
        pf = pq.ParquetFile(chunk_files[0])
        sample_df = pf.read_row_group(0).to_pandas().head(1)
        sample_df['sector33_code'] = sample_df['sector33_code'].astype(str)
        engineer = FeatureEngineer(sample_df)
        pipe = (
            engineer
            .apply_bulk_cross_sectional()
            .apply_crosssectional_targets()
        )
        sample_df = pipe.get_df()
        feature_cols = [x for x in sample_df.columns if x.startswith(('MOM_', 'VOL_', 'LIQ_', 'VAL_', 'QLT_', 'SIZ_', 'SPD_', 'BET_', 'SEA_', 'EVT_', 'CON_', 'GOV_'))]
        del sample_df
        gc.collect()
    num_features = len(feature_cols)
    total_rows = 0
    for f in chunk_files:
        tmp_meta = pd.read_parquet(f, columns=['scode', 'date']) # フィルタリングのため date も読み込む
        tmp_meta = tmp_meta[tmp_meta['date'] >= pd.to_datetime('2017-01-01')]
        if mode == "sample":
            total_rows += tmp_meta['scode'].isin(selected_scodes).sum()
        else:
            total_rows += len(tmp_meta)
    print(f"Total rows to process: {total_rows}, Total features: {num_features}")
    
    # 出力先ディレクトリ
    if mode == "sample":
        out_features_dir = SAMPLE_OUTPUT_DIR / "features"
    else:
        out_features_dir = OUTPUT_DIR / "features"
    out_features_dir.mkdir(parents=True, exist_ok=True)

    # OOM回避のための一時ディレクトリ作成
    temp_dir = OUTPUT_DIR / "temp_features_buffer"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    meta_dfs = []
    # MLflowの初期設定
    abs_path = os.path.expanduser("~/JPSForecast/mlflow_runs")
    os.makedirs(abs_path, exist_ok=True)
    mlflow_db_path = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(mlflow_db_path)
    if mode == "sample":
        experiment_name = "Sample_Data_Creation"
    else:
        experiment_name = "Master_Data_Creation"
    client = MlflowClient()
    existing_exp = client.get_experiment_by_name(experiment_name)
    if existing_exp is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=f"file://{abs_path}"
        )
    elif existing_exp.lifecycle_stage == 'deleted':
        print(f"Restoring deleted experiment: {experiment_name}")
        client.restore_experiment(existing_exp.experiment_id)
    mlflow.set_experiment(experiment_name)

    raw_buffer_df = pd.DataFrame()
    with mlflow.start_run(run_name="Create_Master_Data"):
        for f in chunk_files:
            print(f"Processing chunk: {os.path.basename(f)}")
            df = pd.read_parquet(f).reset_index(drop=True)
            # 銘柄フィルタリング (サンプルモードの場合)
            if mode == "sample":
                df = df[df['scode'].isin(selected_scodes)].reset_index(drop=True)
                if len(df) == 0: continue
            # メモリ削減: float64 -> float32
            f_cols = df.select_dtypes(include=['float64']).columns
            if len(f_cols) > 0:
                df[f_cols] = df[f_cols].astype('float32')
            df['sector33_code'] = df['sector33_code'].astype(str)
            # --- Buffer Logic for Rolling Metrics ---
            filter_cols = ['date', 'scode', 'close', 'filt_Return', 'filt_Median_ADV_20', 
                        'filt_Tick_Sensitivity', 'filt_ATR_Ratio', 'filt_Is_Stop_Day']
            if not raw_buffer_df.empty:
                min_dt = df['date'].min() - pd.Timedelta(days=100)
                raw_buffer_df = raw_buffer_df[raw_buffer_df['date'] >= min_dt]
                calc_df = pd.concat([raw_buffer_df, df[filter_cols]], axis=0, ignore_index=True)
            else:
                calc_df = df[filter_cols].copy()
            # --- フィルタリング ---
            calc_df = filter.calc_relative_metrics(calc_df)
            current_flags = calc_df.iloc[-len(df):]
            for col in [*CANDIDATE_COLS, *LEGACY_CANDIDATE_COLS]:
                df[col] = current_flags[col].values
            # --- クロスセクショナル特徴量＆ターゲットの生成 ---
            engineer = FeatureEngineer(df)
            pipe = (
                engineer
                .apply_bulk_cross_sectional()
                .apply_crosssectional_targets()
            )
            df = pipe.get_df()
            del calc_df, engineer
            gc.collect()
            # バッファ更新 (生のdfを使用)
            raw_buffer_df = pd.concat([raw_buffer_df, df[filter_cols]], axis=0, ignore_index=True)

            # 2017年1月1日以降のデータのみ残す
            df = df[df['date'] >= pd.to_datetime('2017-01-01')].reset_index(drop=True)
            if len(df) == 0:
                continue

            # フィルタ通過状況のログ出力
            candidate_counts = df[CANDIDATE_COLS].sum()
            if candidate_counts.any():
                stats = ", ".join(
                    f"{col.removeprefix('is_candidate_').upper()}: {int(candidate_counts[col])}"
                    for col in CANDIDATE_COLS
                )
                print(f"  [Filter Stats] {stats} / {len(df)} rows")
            # --- 特徴量とメタデータの書き込み ---
            future_cols = [
                'Future_High_Tac', 'Future_Low_Tac', 'Future_Close_Tac',
                'Future_High_10d', 'Future_Low_10d', 'Future_Close_10d',
                'Future_High_20d', 'Future_Low_20d', 'Future_Close_20d',
                'Future_High_40d', 'Future_Low_40d', 'Future_Close_40d',
                'Future_High_Str', 'Future_Low_Str', 'Future_Close_Str',
            ]
            missing_future_cols = [col for col in future_cols if col not in df.columns]
            if missing_future_cols:
                raise KeyError(
                    "Future horizon columns are missing from standardized data: "
                    f"{missing_future_cols}"
                )
            for col in future_cols:
                df[col] = df[col]/df['Entry_Price']

            meta_cols = ['date', 'scode'] + CANDIDATE_COLS + LEGACY_CANDIDATE_COLS + ['log_market_cap'] + future_cols + [c for c in df.columns if c.startswith('target_')]
            
            # メモリ節約のため必要なカラムだけ保持
            save_cols = list(set(meta_cols + feature_cols))
            
            # 特徴量計算等でfloat64になったカラムを再度float32にキャスト
            f_cols = df[save_cols].select_dtypes(include=['float64']).columns
            if len(f_cols) > 0:
                df[f_cols] = df[f_cols].astype('float32')
            
            # --- メモリに乗せきれないため一時ファイルとして保存 ---
            temp_path = temp_dir / f"temp_chunk_{len(meta_dfs)}.parquet"
            df[save_cols].to_parquet(temp_path, index=False)
            
            # メタデータだけは結合用に保持
            meta_dfs.append(df[meta_cols].copy())
            
            del df
            gc.collect()

        # --- 全チャンク処理後、一時ファイルからPyArrowで部分ロードしてチャンク分割 ---
        print("Concatenating metadata...")
        meta_df = pd.concat(meta_dfs, ignore_index=True)
        del meta_dfs
        gc.collect()

        print("Filtering out stocks that are never candidates...")
        # scode ごとに期間全体でいずれかの horizon が True になったかを確認
        candidate_counts = meta_df.groupby('scode')[CANDIDATE_COLS].sum()
        valid_scodes = candidate_counts.index[candidate_counts.gt(0).any(axis=1)]
        
        initial_scode_count = meta_df['scode'].nunique()
        initial_row_count = len(meta_df)
        meta_df = meta_df[meta_df['scode'].isin(valid_scodes)]
        
        print(f"  - Filtered out {initial_scode_count - len(valid_scodes)} stocks. Remaining: {len(valid_scodes)} stocks.")
        print(f"  - Rows reduced from {initial_row_count} to {len(meta_df)}.")

        print("Sorting metadata by scode and date...")
        meta_df = meta_df.sort_values(['scode', 'date']).reset_index(drop=True)

        # 銘柄ごとにチャンク分割 (例: 20チャンク)
        n_chunks = 20
        unique_scodes = meta_df['scode'].unique()
        scodes_split = np.array_split(unique_scodes, n_chunks)

        print(f"Loading temporary data via PyArrow and saving into {n_chunks} parquet chunks...")
        total_rows_processed = len(meta_df)
        dataset = ds.dataset(temp_dir, format="parquet")
        
        for i, scode_group in enumerate(tqdm(scodes_split, desc="Writing Parquet Chunks")):
            scode_list = scode_group.tolist()
            # 該当銘柄群のみをロード
            chunk_table = dataset.to_table(filter=ds.field('scode').isin(scode_list))
            chunk_df = chunk_table.to_pandas()
            
            # scode, date順にソート
            chunk_df = chunk_df.sort_values(['scode', 'date']).reset_index(drop=True)
            
            chunk_path = out_features_dir / f"features_chunk_{i:02d}.parquet"
            # 結合キーを含めて出力
            chunk_cols = ['scode', 'date'] + feature_cols
            chunk_df[chunk_cols].to_parquet(chunk_path, index=False)
            
            del chunk_table, chunk_df
            gc.collect()
        
        if mode == "sample":
            meta_path = os.path.join(SAMPLE_OUTPUT_DIR, "index_meta.parquet")
            names_path = os.path.join(SAMPLE_OUTPUT_DIR, "feature_names.json")
        else:
            meta_path = os.path.join(OUTPUT_DIR, "index_meta.parquet")
            names_path = os.path.join(OUTPUT_DIR, "feature_names.json")
            
        meta_df.to_parquet(meta_path, index=False)
        pd.Series(feature_cols).to_json(names_path, orient='records')
        
        mlflow.log_artifact(names_path, "metadata")
        mlflow.log_param("total_rows", total_rows_processed)
        mlflow.log_artifact(meta_path, "metadata")

        # 一時ディレクトリの削除
        shutil.rmtree(temp_dir)

    print(f"✅ Master data creation complete. Total rows: {total_rows_processed}")

if __name__ == "__main__":
    import sys
    if "--sample" in sys.argv:
        main(mode="sample")
    else:
        main()
