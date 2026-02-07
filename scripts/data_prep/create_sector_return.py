
import os
import numpy as np
import pandas as pd
import glob
import mlflow
import gc
from scipy.special import erfinv
from tqdm import tqdm
import pyarrow.parquet as pq
from pathlib import Path
from src.data_loader.loader import DataLoader
from src.data_loader.get_sector_code_from_JQuants import get_sector_master_from_api

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / 'data/intermediate'
# J-Quants 認証情報 (環境変数推奨)
JQ_MAIL = os.environ.get('JQ_MAIL', '') 
JQ_PASS = os.environ.get('JQ_PASS', '')

def main():
    jq_mail = JQ_MAIL
    jq_pass = JQ_PASS
    loader = DataLoader()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # calc sector_return
    prices = loader.fetch_all_close_data()
    df_sector_master = get_sector_master_from_api(jq_mail, jq_pass)
    merged = prices.merge(df_sector_master[['scode', 'sector33_code']], on='scode')
    merged['ret'] = merged.groupby('scode')['close'].pct_change()
    df_sector_indices = merged.groupby(['date', 'sector33_code'])['ret'].mean().reset_index().rename(columns={'ret': 'sector_return'})

    # save sector_return
    df_sector_indices.to_parquet(OUTPUT_DIR / 'sector_return.parquet', index=False)
    print(f"Sector returns saved to {OUTPUT_DIR / 'sector_return.parquet'}")
    
if __name__ == "__main__":
    main()