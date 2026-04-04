import os
import pandas as pd
from pathlib import Path
import glob
import gc
from src.data_loader.loader import DataLoader
from src.features.engineer import FeatureEngineer
from src.data_loader.filter import FinancialUniverseEngine
import warnings
from tqdm import tqdm
# pandas_ta等の警告抑制
warnings.filterwarnings("ignore")

# ==========================================
# 設定 (Configuration)
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parents[2]
TEMP_DIR = PROJECT_DIR / 'data/temp_scode'
OUTPUT_PATH = PROJECT_DIR / 'data/intermediate/date_chunks'
BATCH_SIZE = 50  # 銘柄バッチ処理サイズ (メモリ制約に応じて調整)

def standardize_raw_data():
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    loader = DataLoader()
    filter = FinancialUniverseEngine()
    print("Fetching reference data for all symbols...")
    all_symbols = loader.get_all_symbols()
    print(f"Total unique symbols to process: {len(all_symbols)}")
    df_topix = loader.fetch_topix_data()
    print("Fetching additional market and financial data...")
    df_n225 = loader.fetch_n225_data()
    df_fins = loader.fetch_financial()
    df_investor_types = loader.fetch_investor_types()
    df_margin_weekly = loader.fetch_margin_weekly()
    # 信用残高は通常「金曜締め」→「翌週火曜公表」
    # そのため、marginデータのDateに +4日 (火曜日) 加算してから結合する
    df_margin = df_margin_weekly.copy()
    df_margin['available_date'] = pd.to_datetime(df_margin['date']) + pd.Timedelta(days=4)
    df_shrt_sector = loader.fetch_short_selling_sector()
    df_sector_indices = loader.fetch_sector_return()
    df_sector_indices['sector33_code'] = df_sector_indices['sector33_code'].astype(str)

    # --- 銘柄別ループ (時系列計算) ---
    for i in tqdm(range(0, all_symbols.shape[0], BATCH_SIZE), desc="Processing Batches"):
        batch_symbols = list(all_symbols.iloc[i : i + BATCH_SIZE,0]) # scode_list
        df_batch = loader.fetch_batch_data(batch_symbols) # 銘柄別OHLCVデータ
        if df_batch.empty:
            continue
        df_batch = pd.merge(df_batch, all_symbols, on='scode', how='left')
        df_batch = pd.merge(df_batch, df_topix, on='date', how='left', suffixes=('', '_mkt'))
        df_batch = pd.merge(df_batch, df_n225, on='date', how='left')
        df_batch = pd.merge(df_batch, df_investor_types, on='date', how='left')
        df_batch['date'] = pd.to_datetime(df_batch['date'])
        df_batch = df_batch.sort_values('date')
        # 財務データの結合
        batch_fins = df_fins[df_fins['scode'].isin(batch_symbols)].sort_values(['published_date'])
        df_batch = pd.merge_asof(
            df_batch,
            batch_fins,
            left_on='date',
            right_on='published_date',
            by='scode',
            direction='backward'
        )
        # 信用取引データの結合
        batch_margin = df_margin[df_margin['scode'].isin(batch_symbols)].sort_values('available_date')
        df_batch = pd.merge_asof(
            df_batch,
            batch_margin[['scode', 'available_date', 'long_margin_trade_balance_share', 'short_margin_trade_balance_share']],
            left_on='date',
            right_on='available_date',
            by='scode',
            direction='backward'
        )
        # 業種別空売り比率データの結合
        batch_shrt_sector = df_shrt_sector[df_shrt_sector['sector33_code'].isin(df_batch['sector33_code'].unique())].sort_values('date')
        df_batch = pd.merge_asof(
            df_batch,
            batch_shrt_sector,
            left_on='date',
            right_on='date',
            by='sector33_code',
            direction='backward'
        )
        # セクターインデックスの結合
        df_batch = pd.merge(df_batch,df_sector_indices,on=['date', 'sector33_code'],how='left')
        for symbol, df_stock in df_batch.groupby('scode'):
            df_stock = df_stock.sort_values('date').reset_index(drop=True)
            engine = FeatureEngineer(df_stock)
            pipe = (
                engine
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
                .apply_tempfeat()
                .apply_bulk_time_series()
                .apply_timeseries_targets()
            )
            df_feat = pipe.get_df()
            if df_feat.empty: continue
            # 過去データの除外、momentum_12_1基準で
            df_feat = df_feat.dropna(subset='MOM_Momentum12-1_RAW')
            # 上場間も無い銘柄を除外、Dist_SMA75基準で
            df_feat = df_feat.dropna(subset='MOM_DistSMA75_RAW')
            # 直近データの除外、Future_X_Str基準で
            df_feat = df_feat.dropna(subset='Future_High_Str')
            # filter
            df_feat = filter.calc_intrinsic_metrics(df_feat)
            # 一時保存
            df_feat.to_parquet(f"{TEMP_DIR}/{symbol}.parquet")

    del df_topix, df_fins, df_investor_types, df_margin_weekly, df_margin, df_shrt_sector, df_sector_indices
    gc.collect()

    # --- 日付別ループ (チャンク化) ---
    # 全銘柄の「MA_250計算済みデータ」を日付でまとめて保存し直す
    print("Regrouping data into date chunks...")
    all_temp_files = glob.glob(f"{TEMP_DIR}/*.parquet")
    if not all_temp_files:
        print("No temporary files found.")
        return

    print("Determining date range from temp files...")
    min_dates = []
    max_dates = []
    for f in all_temp_files:
        # メモリ節約と高速化のため、date列のみを読み込む
        df_date = pd.read_parquet(f, columns=['date'])
        if not df_date.empty:
            min_dates.append(df_date['date'].min())
            max_dates.append(df_date['date'].max())

    if not min_dates:
        print("No valid dates found in temp files.")
        return

    global_min_date = min(min_dates)
    global_max_date = max(max_dates)
    print(f"Data range found: {global_min_date.date()} to {global_max_date.date()}")

    # 最小日付が属する四半期の初日を計算
    q_start_month = (global_min_date.month - 1) // 3 * 3 + 1
    start_q = pd.Timestamp(year=global_min_date.year, month=q_start_month, day=1)
    
    # 四半期ごとの開始日を生成
    dates = pd.date_range(start=start_q, end=global_max_date, freq='QS')

    for start_date in dates:
        end_date = start_date + pd.DateOffset(months=3)
        chunk_list = []
        for f in all_temp_files:
            stock_chunk = pd.read_parquet(f)
            # 期間内のみ抽出
            mask = (stock_chunk['date'] >= start_date) & (stock_chunk['date'] < end_date)
            if mask.any():
                chunk_list.append(stock_chunk[mask])
        if chunk_list:
            final_chunk = pd.concat(chunk_list)
            chunk_name = f"standardized_{start_date.strftime('%Y%m')}.parquet"
            final_chunk.to_parquet(f"{OUTPUT_PATH}/{chunk_name}")
            print(f"✅ Created chunk: {chunk_name}")
        del chunk_list
        gc.collect()

    print("🎉 All raw data standardized and chunked.")

if __name__ == "__main__":
    standardize_raw_data()