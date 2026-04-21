import os
# 並列処理時のスレッド競合（オーバーサブスクリプション）を防ぐための環境変数設定
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from tqdm import tqdm
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta
import gc
import warnings
import argparse
import json
import sys
import ast
import MySQLdb

# プロジェクトのルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader.loader import DataLoader
from src.features.engineer import FeatureEngineer
from src.data_loader.filter import FinancialUniverseEngine

# --- 設定 ---
# MLflow設定
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
# 出力先ディレクトリ
OUTPUT_DIR = "./predictions"

# 警告抑制
warnings.filterwarnings("ignore")

def fetch_prediction_data(loader: DataLoader, target_date: str) -> pd.DataFrame:
    print(f"Fetching data for prediction date: {target_date}")
    start_date = (pd.to_datetime(target_date) - timedelta(days=365)).strftime('%Y-%m-%d')
    print(f"Data fetch range: {start_date} to {target_date}")
    print(' - Fetching symbols...')
    all_symbols = loader.get_latest_symbols(target_date)
    if len(all_symbols) == 0:
        print("Warning: No symbols found from DataLoader.")
        return pd.DataFrame()
    print(' - Fetching market data...')
    df_topix = loader.fetch_topix_data(start_date=start_date)
    print(' - Fetching N225 data...')
    df_n225 = loader.fetch_n225_data(start_date=start_date)
    print(' - Fetching financial data...')
    df_fins = loader.fetch_financial(start_date=start_date)
    df_fins = df_fins.sort_values('published_date')
    print(' - Fetching investor data...')
    df_investor = loader.fetch_investor_types(start_date=start_date)
    print(' - Fetching margin data...')
    df_margin_weekly = loader.fetch_margin_weekly(start_date=start_date)
    df_margin = df_margin_weekly.copy()
    df_margin['available_date'] = pd.to_datetime(df_margin['date']) + pd.Timedelta(days=4)
    df_margin = df_margin.sort_values('available_date')
    df_shrt_sector = loader.fetch_short_selling_sector(start_date=start_date)
    df_shrt_sector = df_shrt_sector.sort_values('date')
    print(' - Fetching sector data...')
    df_sector_indices = loader.fetch_sector_return(start_date=start_date)
    print(' - Fetching OHLCV data...')
    df_ohlcv = loader.fetch_batch_data(all_symbols['scode'].tolist(), start_date=start_date)
    if df_ohlcv.empty:
        print("Warning: No OHLCV data found for the specified date range.")
        return pd.DataFrame()
    df_merged = pd.merge(df_ohlcv, all_symbols, on='scode', how='left')
    df_merged = pd.merge(df_merged, df_topix, on='date', how='left', suffixes=('', '_mkt'))
    df_merged = pd.merge(df_merged, df_n225, on='date', how='left')
    df_merged = pd.merge(df_merged, df_investor, on='date', how='left')
    df_merged['date'] = pd.to_datetime(df_merged['date'])
    df_merged = df_merged.sort_values('date')
    df_merged = pd.merge_asof(df_merged, df_fins, left_on='date', right_on='published_date', by='scode', direction='backward')
    df_merged = pd.merge_asof(df_merged, df_margin[['scode', 'available_date', 'long_margin_trade_balance_share', 'short_margin_trade_balance_share']], left_on='date', right_on='available_date', by='scode', direction='backward')
    df_merged = pd.merge_asof(df_merged, df_shrt_sector, left_on='date', right_on='date', by='sector33_code', direction='backward')
    df_merged = pd.merge(df_merged, df_sector_indices, on=['date', 'sector33_code'], how='left')
    df_merged = df_merged.sort_values(['scode', 'date'])
    print(f"Data fetching and merging complete. Total rows: {len(df_merged)}")
    return df_merged


def create_features(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    print("Creating time-series features (vectorized)...")

    # 時系列処理のために全体を銘柄・日付でソートし、インデックスをリセット
    df = df.sort_values(['scode', 'date']).reset_index(drop=True)
    engineer = FeatureEngineer(df)
    pipe = (
        engineer
        .apply_momentum_block()
        .apply_volatility_block()
        .apply_liquidity_block()
        .apply_value_block()
        .apply_quality_block()
        .apply_size_block()
        .apply_supplydemand_bloc()
        .apply_beta_block()
        .apply_seasonality_block()
        .apply_event_block()
        .apply_consensus_block()
        .apply_governance_block()
        .apply_bulk_time_series()
    )
    full_df = pipe.get_df()
    if full_df.empty:
        print("Warning: No data after time-series feature engineering.")
        return pd.DataFrame()
    gc.collect()
    # TCN等の時系列モデル推論に必要な「過去252営業日分」のデータに絞り込み、不要な過去データの横断面加工をスキップ
    target_dt = pd.to_datetime(target_date)
    valid_dates = full_df[full_df['date'] <= target_dt]['date'].drop_duplicates().sort_values(ascending=False)
    required_dates = valid_dates.head(252)
    if not required_dates.empty:
        print(f"Filtering data to latest {len(required_dates)} trading days for cross-sectional processing.")
        full_df = full_df[full_df['date'].isin(required_dates)].reset_index(drop=True)
    # 2. 横断面加工 (create_master_data.pyと同様)
    print("Applying cross-sectional transformations...")
    engineer_cs = FeatureEngineer(full_df)
    pipe_cs = engineer_cs.apply_bulk_cross_sectional()
    final_df = pipe_cs.get_df()
    return final_df


def predict(target_date: str):
    """
    推論処理のメイン関数
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    conn = MySQLdb.connect(
        user='root',
        passwd='root',
        host='127.0.0.1',
        port=3306,
        charset='utf8',
    )

    # load data
    loader = DataLoader()
    raw_df = fetch_prediction_data(loader, target_date)
    loader.close()
    if raw_df.empty:
        print("No data to process. Exiting.")
        return

    # create features
    features_df = create_features(raw_df, target_date)
    if features_df.empty:
        print(f"No features could be generated for {target_date}. Exiting.")
        return

    # filtering
    filter = FinancialUniverseEngine()
    features_df = filter.calc_intrinsic_metrics(features_df)
    features_df = filter.calc_relative_metrics(features_df)

    # scoring
    sql = "select scode,sname,market,gyoshu,Close from jps.scode_list"
    output = pd.read_sql(sql, conn)
    print("\nLoading Production models from MLflow Registry...")
    features_df = features_df.sort_values(['scode', 'date']).reset_index(drop=True)
    latest_date = features_df['date'].max()
    all_scores = features_df[features_df['date'] == latest_date][['date', 'scode']].drop_duplicates(subset='scode').copy()
    registered_models = client.search_registered_models(filter_string="name LIKE '%'")
    production_models = []
    for rm in registered_models:
        for v in rm.latest_versions:
            if v.current_stage == 'Production':
                production_models.append(v)
                print(f"  - Found: {v.name} (Version: {v.version}, Stage: {v.current_stage})")
                break
    if not production_models:
        print("No models found in 'Production' stage. Exiting.")
        return

    for model_version in production_models:
        model_name = model_version.name
        model_uri = f"models:/{model_name}/Production"
        print(f"\nScoring with {model_name}...")
        try:
            # MLflowのRunパラメータからモデルのドメインとカテゴリ（時系列かTableか）を取得
            run = client.get_run(model_version.run_id)
            params = run.data.params
            # domain のパース（dictが文字列化されている場合に対応）
            domain_raw = params.get('domain.name', params.get('domain/name', params.get('domain', 'STR')))
            if isinstance(domain_raw, str) and domain_raw.strip().startswith('{'):
                try:
                    domain_dict = ast.literal_eval(domain_raw)
                    domain = domain_dict.get('name', 'STR').upper()
                except (ValueError, SyntaxError):
                    domain = 'STR'
            else:
                domain = str(domain_raw).upper()
            # data_category のパース（dictが文字列化されている場合に対応）
            model_raw = params.get('model.data_category', params.get('model/data_category', params.get('model', 'tabular')))
            if isinstance(model_raw, str) and model_raw.strip().startswith('{'):
                try:
                    model_dict = ast.literal_eval(model_raw)
                    data_category = model_dict.get('data_category', 'tabular').lower()
                except (ValueError, SyntaxError):
                    data_category = 'tabular'
            else:
                data_category = str(model_raw).lower()

            is_candidate_col = 'is_candidate_tac' if domain == 'TAC' else 'is_candidate_str'
            print(f"  - Domain: {domain}, Category: {data_category}")
            print(f"  - Target candidate col: {is_candidate_col}")

            # 最新日の対象銘柄（候補フラグがTrueの銘柄）を抽出
            latest_candidates_mask = (features_df['date'] == latest_date) & (features_df[is_candidate_col] == True)
            target_scodes = features_df[latest_candidates_mask]['scode'].unique()
            if len(target_scodes) == 0:
                print(f"  - WARNING: No candidate stocks found for {domain} on {latest_date.strftime('%Y-%m-%d')}.")
                all_scores[f'score_{model_name}'] = np.nan
                continue
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            local_path = client.download_artifacts(run_id=model_version.run_id, path="configs/feature_cols.json")
            with open(local_path) as f:
                config = json.load(f)
            feature_cols = config['feature_cols']
            
            features_arr = features_df[feature_cols].values
            col_indices = list(range(len(feature_cols)))
            ref_idx = features_df.index[latest_candidates_mask].values
            print(f"  - Predicting for {len(ref_idx):,} records...")
            pipeline = loaded_model.unwrap_python_model()
            all_preds = np.zeros(len(ref_idx), dtype=np.float32)
            batch_size = 10000
            for i in tqdm(range(0, len(ref_idx), batch_size), desc="  Predicting batches"):
                batch_idx = ref_idx[i : i + batch_size]
                batch_preds_folds = []
                for fold_pipe in pipeline.fold_pipelines:
                    X_processed = fold_pipe.preprocessor.transform(features_arr, row_indices=batch_idx, col_indices=col_indices)
                    p = fold_pipe.model.predict(X_processed)
                    batch_preds_folds.append(p)
                    # 中間生成されたZarrキャッシュのクリーンアップ
                    if isinstance(X_processed, str) and X_processed.endswith('.zarr') and os.path.exists(X_processed):
                        import shutil
                        shutil.rmtree(X_processed, ignore_errors=True)
                all_preds[i : i + batch_size] = np.mean(batch_preds_folds, axis=0)
                
            latest_scores = pd.DataFrame({
                'scode': features_df.loc[ref_idx, 'scode'].values,
                f'score_{model_name}': all_preds
            }).drop_duplicates(subset='scode')
            output = pd.merge(output, latest_scores, on='scode', how='left')
            print(f"  - Scoring complete. Average score: {np.nanmean(all_preds):.4f}")
        except Exception as e:
            print(f"  - Failed to score with {model_name}. Error: {e}")
            all_scores[f'score_{model_name}'] = np.nan
            continue

    # 4. スタッキングモデルのスコアリング (未実装)
    # print("\nScoring with Stacking model (placeholder)...")
    # try:
    #     stacking_model_uri = "models:/stacking_model_name/Production"
    #     stacking_model = mlflow.pyfunc.load_model(stacking_model_uri)
    #     X_stacking = all_scores.filter(like='score_')
    #     stacking_score = stacking_model.predict(X_stacking)
    #     all_scores['score_stacking'] = stacking_score
    #     print("  - Stacking model scoring complete.")
    #     all_scores['score_stacking'] = np.nan # プレースホルダー
    # except Exception as e:
    #     print(f"  - Stacking model not found or failed to score: {e}")
    #     all_scores['score_stacking'] = np.nan

    # 5. 結果をCSVに出力
    output = output.dropna()
    output = output.drop_duplicates(subset='scode')
    output_filename = f"predictions_{target_date.replace('-', '')}.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    output.to_csv(output_path, index=False, encoding='shift-jis')
    print(f"\n✅ All predictions saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction script for a given date.")
    parser.add_argument(
        "--date",
        type=str,
        default=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
        help="Target date for prediction in YYYY-MM-DD format. Defaults to yesterday."
    )
    args = parser.parse_args()

    predict(args.date)