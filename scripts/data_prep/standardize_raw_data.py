import os
import pandas as pd
from pathlib import Path
import glob
import gc
import shutil
import exchange_calendars as ecals
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
PRICE_ROW_MARKER = '__jps_observed_price_row__'
ORIGINAL_SCODE_COL = '__jps_original_scode__'

def _build_xtks_market_sessions(price_dates):
    if price_dates is None or price_dates.empty or 'date' not in price_dates.columns:
        raise RuntimeError("Database price trading dates are missing or empty.")
    dates = pd.to_datetime(price_dates['date'], errors='coerce', format='mixed')
    if dates.isna().any():
        raise RuntimeError("Database price trading dates contain invalid dates.")
    database_dates = pd.DatetimeIndex(dates.dt.normalize().unique()).sort_values()

    try:
        calendar = ecals.get_calendar('XTKS')
        official_sessions = calendar.sessions_in_range(
            database_dates.min(), database_dates.max()
        )
    except Exception as exc:
        raise RuntimeError("Unable to build the XTKS official session calendar.") from exc
    if official_sessions.tz is not None:
        official_sessions = official_sessions.tz_convert(None)
    official_sessions = official_sessions.normalize()

    missing_sessions = official_sessions.difference(database_dates)
    non_session_dates = database_dates.difference(official_sessions)
    if len(missing_sessions) or len(non_session_dates):
        raise RuntimeError(
            "Database price dates do not match XTKS official sessions: "
            f"missing_sessions={missing_sessions[:5].tolist()}, "
            f"non_session_dates={non_session_dates[:5].tolist()}"
        )
    return official_sessions

def _prepare_topix_reference(df_topix, market_sessions):
    if df_topix is None or df_topix.empty or 'date' not in df_topix.columns:
        raise RuntimeError("TOPIX reference data is missing or empty.")
    result = df_topix.copy()
    result['date'] = _normalize_source_dates(result, 'date')
    official_mask = result['date'].isin(market_sessions)
    non_session_count = int((~official_mask).sum())
    result = (
        result.loc[official_mask]
        .sort_values('date')
        .drop_duplicates('date', keep='last')
        .reset_index(drop=True)
    )
    if result.empty:
        raise RuntimeError(
            f"TOPIX has no XTKS session rows; non_session_count={non_session_count}."
        )

    expected_sessions = market_sessions[
        (market_sessions >= result['date'].min())
        & (market_sessions <= result['date'].max())
    ]
    observed_sessions = pd.DatetimeIndex(result['date'].unique()).sort_values()
    missing_sessions = expected_sessions.difference(observed_sessions)
    if len(missing_sessions):
        raise RuntimeError(
            "TOPIX is missing XTKS sessions within its observed range: "
            f"missing_count={len(missing_sessions)}, "
            f"non_session_count={non_session_count}, "
            f"missing_sessions={missing_sessions[:5].tolist()}"
        )
    return result

def _reindex_batch_to_xtks_sessions(
    price_data,
    market_sessions,
    marker_col=PRICE_ROW_MARKER,
):
    required_cols = ['scode', 'date']
    if price_data is None or price_data.empty:
        raise RuntimeError("Batch price data is empty.")
    missing_cols = [col for col in required_cols if col not in price_data.columns]
    if missing_cols:
        raise RuntimeError(
            f"Batch price data is missing required columns: {missing_cols}"
        )
    internal_cols = [marker_col, ORIGINAL_SCODE_COL]
    collisions = [col for col in internal_cols if col in price_data.columns]
    if collisions:
        raise RuntimeError(f"Internal price-grid columns already exist: {collisions}")

    result = price_data.copy()
    result['date'] = pd.to_datetime(
        result['date'], errors='coerce', format='mixed'
    ).dt.normalize()
    if result['date'].isna().any() or result['scode'].isna().any():
        raise RuntimeError("Batch price data contains invalid scode/date keys.")
    if result.duplicated(subset=['scode', 'date']).any():
        raise RuntimeError("Batch price data contains duplicate scode/date keys.")

    outside_calendar = pd.DatetimeIndex(result['date'].unique()).difference(market_sessions)
    if len(outside_calendar):
        raise RuntimeError(
            f"Batch price data contains dates outside XTKS sessions: {outside_calendar[:5].tolist()}"
        )

    original_cols = result.columns.tolist()
    result[ORIGINAL_SCODE_COL] = result['scode']
    result[marker_col] = True
    reindexed_groups = []
    for scode, group in result.groupby('scode', sort=False):
        group = group.sort_values('date')
        entity_sessions = market_sessions[
            (market_sessions >= group['date'].iloc[0])
            & (market_sessions <= group['date'].iloc[-1])
        ]
        if entity_sessions.empty:
            raise RuntimeError(f"No XTKS sessions found for scode={scode} observation range.")
        reindexed = group.set_index('date').reindex(entity_sessions)
        reindexed.index.name = 'date'
        reindexed[ORIGINAL_SCODE_COL] = scode
        reindexed[marker_col] = reindexed[marker_col].eq(True)
        episode_start = reindexed[marker_col] & ~reindexed[marker_col].shift(
            1, fill_value=True
        )
        episode = episode_start.cumsum()
        reindexed['scode'] = (
            str(scode) + '__jps_episode_' + episode.astype(str)
        )
        reindexed_groups.append(reindexed.reset_index())

    return (
        pd.concat(reindexed_groups, ignore_index=True)
        .sort_values(['scode', 'date'])
        .reset_index(drop=True)[[*original_cols, marker_col, ORIGINAL_SCODE_COL]]
    )

def _drop_price_placeholders(
    data,
    marker_col=PRICE_ROW_MARKER,
    restore_scode=True,
):
    required_internal_cols = [marker_col, ORIGINAL_SCODE_COL]
    missing_cols = [col for col in required_internal_cols if col not in data.columns]
    if missing_cols:
        raise RuntimeError(f"Internal price-grid columns are missing: {missing_cols}")
    result = data.loc[data[marker_col].eq(True)].copy()
    result = result.drop(columns=[marker_col])
    if not restore_scode:
        return result.sort_values(
            [ORIGINAL_SCODE_COL, 'scode', 'date']
        ).reset_index(drop=True)

    result['scode'] = result[ORIGINAL_SCODE_COL]
    return (
        result.drop(columns=[ORIGINAL_SCODE_COL])
        .sort_values(['scode', 'date'])
        .reset_index(drop=True)
    )

def _invalidate_incomplete_forward_labels(data, marker_col=PRICE_ROW_MARKER):
    if marker_col not in data.columns:
        raise RuntimeError(f"Internal price-row marker is missing: {marker_col}")

    result = data.copy()
    horizon_columns = {
        5: ['Future_High_Tac', 'Future_Low_Tac', 'Future_Close_Tac', 'target_ret_5'],
        10: ['Future_High_10d', 'Future_Low_10d', 'Future_Close_10d', 'target_ret_10'],
        20: ['Future_High_20d', 'Future_Low_20d', 'Future_Close_20d', 'target_ret_20'],
        40: ['Future_High_40d', 'Future_Low_40d', 'Future_Close_40d', 'target_ret_40'],
        60: ['Future_High_Str', 'Future_Low_Str', 'Future_Close_Str', 'target_ret_60'],
    }
    tb_cols = [col for col in result.columns if col.startswith('target_tac_tb_')]
    if not tb_cols:
        raise RuntimeError("TAC triple-barrier target columns are missing.")
    horizon_columns[5].extend(tb_cols)

    missing_cols = [
        col
        for columns in horizon_columns.values()
        for col in columns
        if col not in result.columns
    ]
    if missing_cols:
        raise RuntimeError(f"Required forward-label columns are missing: {missing_cols}")

    grouped_marker = result.groupby('scode', sort=False)[marker_col]
    for horizon, columns in horizon_columns.items():
        forward_observed_count = grouped_marker.transform(
            lambda marker: (
                marker.astype('int8')
                .iloc[::-1]
                .rolling(horizon, min_periods=horizon)
                .sum()
                .iloc[::-1]
                .shift(-1)
            )
        )
        complete = forward_observed_count.eq(horizon)
        result.loc[~complete, columns] = float('nan')
    return result

def _normalize_source_dates(data, source_col):
    if data is None or source_col not in data.columns:
        raise RuntimeError(f"Required source-date column is missing: {source_col}")
    source_dates = pd.to_datetime(
        data[source_col], errors='coerce', format='mixed'
    ).dt.normalize()
    if source_dates.isna().any():
        raise RuntimeError(f"Invalid source date found in column: {source_col}")
    return source_dates

def _assign_next_session_availability(
    data,
    market_sessions,
    source_col,
    availability_col,
):
    source_dates = _normalize_source_dates(data, source_col)
    positions = market_sessions.searchsorted(source_dates, side='right')
    valid = positions < len(market_sessions)
    result = data.loc[valid].copy()
    result[source_col] = source_dates.loc[valid]
    result[availability_col] = market_sessions.take(positions[valid]).to_numpy()
    return result

def _assign_following_week_second_session_availability(
    data,
    market_sessions,
    source_col,
    availability_col,
):
    source_dates = _normalize_source_dates(data, source_col)
    current_week_monday = source_dates - pd.to_timedelta(source_dates.dt.weekday, unit='D')
    next_week_monday = current_week_monday + pd.Timedelta(days=7)
    positions = market_sessions.searchsorted(next_week_monday, side='left') + 1
    valid = positions < len(market_sessions)
    result = data.loc[valid].copy()
    result[source_col] = source_dates.loc[valid]
    result[availability_col] = market_sessions.take(positions[valid]).to_numpy()
    return result

def _deduplicate_availability_rows(
    data,
    availability_col,
    source_col,
    entity_cols=None,
):
    entity_cols = list(entity_cols or [])
    logical_key_cols = [*entity_cols, availability_col, source_col]
    key_cols = [*entity_cols, availability_col]
    deduplicated = data.drop_duplicates().copy()
    conflicting = deduplicated.duplicated(subset=logical_key_cols, keep=False)
    if conflicting.any():
        conflict_keys = deduplicated.loc[conflicting, logical_key_cols].drop_duplicates()
        raise RuntimeError(
            "Conflicting payloads found for identical availability source keys: "
            f"{conflict_keys.to_dict(orient='records')}"
        )
    return (
        deduplicated.sort_values([*key_cols, source_col], kind='mergesort')
        .drop_duplicates(subset=key_cols, keep='last')
        .sort_values([availability_col, *entity_cols], kind='mergesort')
        .reset_index(drop=True)
    )

def _assert_availability_not_after_observation(
    data,
    observation_col,
    availability_cols,
):
    required_cols = [observation_col, *availability_cols]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise RuntimeError(f"Required availability validation columns are missing: {missing_cols}")

    observation_dates = pd.to_datetime(data[observation_col], errors='coerce')
    if observation_dates.isna().any():
        raise RuntimeError(f"Invalid observation date found in column: {observation_col}")

    for availability_col in availability_cols:
        availability_dates = pd.to_datetime(data[availability_col], errors='coerce')
        invalid = data[availability_col].notna() & availability_dates.isna()
        if invalid.any():
            raise RuntimeError(
                f"Invalid availability date found in column: {availability_col}"
            )
        future = availability_dates.notna() & availability_dates.gt(observation_dates)
        if future.any():
            raise RuntimeError(
                f"Availability date exceeds observation date in column: {availability_col}"
            )

def standardize_raw_data():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    loader = DataLoader()
    filter = FinancialUniverseEngine()
    print("Fetching reference data for all symbols...")
    all_symbols = loader.get_all_symbols()
    if all_symbols is None or all_symbols.empty:
        raise RuntimeError("No source symbols returned by DataLoader.get_all_symbols().")
    print(f"Total unique symbols to process: {len(all_symbols)}")
    price_trading_dates = loader.fetch_price_trading_dates()
    market_sessions = _build_xtks_market_sessions(price_trading_dates)
    df_topix = loader.fetch_topix_data()
    df_topix = _prepare_topix_reference(df_topix, market_sessions)
    print("Fetching additional market and financial data...")
    df_n225 = loader.fetch_n225_data()
    df_fins = loader.fetch_financial()
    df_fins = _assign_next_session_availability(
        df_fins,
        market_sessions,
        'published_date',
        'financial_available_date',
    )
    df_fins = _deduplicate_availability_rows(
        df_fins,
        'financial_available_date',
        'published_date',
        entity_cols=['scode'],
    )
    df_investor_types = loader.fetch_investor_types()
    df_investor_types = _assign_next_session_availability(
        df_investor_types,
        market_sessions,
        'investor_source_date',
        'investor_available_date',
    )
    df_investor_types = _deduplicate_availability_rows(
        df_investor_types,
        'investor_available_date',
        'investor_source_date',
    )
    df_margin_weekly = loader.fetch_margin_weekly()
    df_margin = df_margin_weekly.rename(columns={'date': 'margin_source_date'}).copy()
    df_margin = _assign_following_week_second_session_availability(
        df_margin,
        market_sessions,
        'margin_source_date',
        'margin_available_date',
    )
    df_margin = _deduplicate_availability_rows(
        df_margin,
        'margin_available_date',
        'margin_source_date',
        entity_cols=['scode'],
    )
    df_shrt_sector = loader.fetch_short_selling_sector().rename(
        columns={'date': 'short_selling_source_date'}
    )
    df_shrt_sector = _assign_next_session_availability(
        df_shrt_sector,
        market_sessions,
        'short_selling_source_date',
        'short_selling_available_date',
    )
    df_shrt_sector = _deduplicate_availability_rows(
        df_shrt_sector,
        'short_selling_available_date',
        'short_selling_source_date',
        entity_cols=['sector33_code'],
    )
    df_sector_indices = loader.fetch_sector_return()
    df_sector_indices['sector33_code'] = df_sector_indices['sector33_code'].astype(str)

    # --- 銘柄別ループ (時系列計算) ---
    for i in tqdm(range(0, all_symbols.shape[0], BATCH_SIZE), desc="Processing Batches"):
        batch_symbols = list(all_symbols.iloc[i : i + BATCH_SIZE,0]) # scode_list
        df_batch = loader.fetch_batch_data(batch_symbols) # 銘柄別OHLCVデータ
        if df_batch is None or df_batch.empty:
            raise RuntimeError(f"No batch data returned for symbols: {batch_symbols}")
        df_batch = pd.merge(df_batch, all_symbols, on='scode', how='left')
        df_batch['date'] = pd.to_datetime(df_batch['date']).dt.normalize()
        df_batch = pd.merge(df_batch, df_topix, on='date', how='left', suffixes=('', '_mkt'))
        df_batch = pd.merge(df_batch, df_n225, on='date', how='left')
        df_batch = df_batch.sort_values('date')
        df_batch = pd.merge_asof(
            df_batch,
            df_investor_types,
            left_on='date',
            right_on='investor_available_date',
            direction='backward',
        )
        # 財務データの結合
        batch_fins = df_fins[df_fins['scode'].isin(batch_symbols)].sort_values(
            ['financial_available_date', 'scode']
        )
        df_batch = pd.merge_asof(
            df_batch.sort_values(['date', 'scode']),
            batch_fins,
            left_on='date',
            right_on='financial_available_date',
            by='scode',
            direction='backward'
        )
        # 信用取引データの結合
        batch_margin = df_margin[df_margin['scode'].isin(batch_symbols)].sort_values(
            ['margin_available_date', 'scode']
        )
        df_batch = pd.merge_asof(
            df_batch.sort_values(['date', 'scode']),
            batch_margin[[
                'scode',
                'margin_source_date',
                'margin_available_date',
                'long_margin_trade_balance_share',
                'short_margin_trade_balance_share',
            ]],
            left_on='date',
            right_on='margin_available_date',
            by='scode',
            direction='backward'
        )
        # 業種別空売り比率データの結合
        batch_shrt_sector = df_shrt_sector[
            df_shrt_sector['sector33_code'].isin(df_batch['sector33_code'].unique())
        ].sort_values(['short_selling_available_date', 'sector33_code'])
        df_batch = pd.merge_asof(
            df_batch.sort_values(['date', 'sector33_code']),
            batch_shrt_sector,
            left_on='date',
            right_on='short_selling_available_date',
            by='sector33_code',
            direction='backward'
        )
        _assert_availability_not_after_observation(
            df_batch,
            'date',
            [
                'financial_available_date',
                'investor_available_date',
                'margin_available_date',
                'short_selling_available_date',
            ],
        )
        # セクターインデックスの結合
        df_batch = pd.merge(df_batch,df_sector_indices,on=['date', 'sector33_code'],how='left')
        df_batch = _reindex_batch_to_xtks_sessions(df_batch, market_sessions)

        df_batch = df_batch.sort_values(['scode', 'date']).reset_index(drop=True)
        engine = FeatureEngineer(df_batch)
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
        df_feat_all = _invalidate_incomplete_forward_labels(pipe.get_df())
        df_feat_all = _drop_price_placeholders(df_feat_all, restore_scode=False)
        
        for symbol, df_stock in df_feat_all.groupby(ORIGINAL_SCODE_COL):
            if df_stock.empty: continue
            # 過去データの除外、momentum_12_1基準で
            df_stock = df_stock.dropna(subset='MOM_Momentum12-1_RAW')
            # 上場間も無い銘柄を除外、Dist_SMA75基準で
            df_stock = df_stock.dropna(subset='MOM_DistSMA75_RAW')
            # filter
            df_stock = filter.calc_intrinsic_metrics(df_stock)
            df_stock['scode'] = df_stock[ORIGINAL_SCODE_COL]
            df_stock = (
                df_stock.drop(columns=[ORIGINAL_SCODE_COL])
                .sort_values('date')
                .reset_index(drop=True)
            )
            # 一時保存
            df_stock.to_parquet(f"{TEMP_DIR}/{symbol}.parquet")

    del df_topix, df_fins, df_investor_types, df_margin_weekly, df_margin, df_shrt_sector, df_sector_indices
    gc.collect()

    # --- 日付別ループ (チャンク化) ---
    # 全銘柄の「MA_250計算済みデータ」を日付でまとめて保存し直す
    print("Regrouping data into date chunks...")
    all_temp_files = glob.glob(f"{TEMP_DIR}/*.parquet")
    if not all_temp_files:
        raise RuntimeError(f"No temporary parquet files found in {TEMP_DIR}.")

    print("Determining date range from temp files...")
    min_dates = []
    max_dates = []
    for f in all_temp_files:
        # メモリ節約と高速化のため、date列のみを読み込む
        df_date = pd.read_parquet(f, columns=['date'])
        valid_dates = pd.to_datetime(df_date['date'], errors='coerce').dropna()
        if not valid_dates.empty:
            min_dates.append(valid_dates.min())
            max_dates.append(valid_dates.max())

    if not min_dates:
        raise RuntimeError(f"No valid dates found in temporary parquet files under {TEMP_DIR}.")

    global_min_date = min(min_dates)
    global_max_date = max(max_dates)
    print(f"Data range found: {global_min_date.date()} to {global_max_date.date()}")

    # 最小日付が属する四半期の初日を計算
    q_start_month = (global_min_date.month - 1) // 3 * 3 + 1
    start_q = pd.Timestamp(year=global_min_date.year, month=q_start_month, day=1)
    
    # 四半期ごとの開始日を生成
    dates = pd.date_range(start=start_q, end=global_max_date, freq='QS')

    if OUTPUT_PATH.exists():
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

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
