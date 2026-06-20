import os
import re
import sys
import pandas as pd
from pathlib import Path
from datetime import timedelta

# Add project root to sys.path to allow importing from src
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_loader.loader import DataLoader

def main():
    """
    predictions/ 配下のCSVファイルに実績値を追加し、ファイル名をリネームするスクリプト。
    実績値 = (スコアリング実施日の5営業日後の終値) / (スコアリング実施日の翌営業日の初値)
    """
    predictions_dir = Path("predictions")
    
    # Identify target files: start with predictions_ and end with .csv
    files = list(predictions_dir.glob("predictions_*.csv"))
    
    if not files:
        print("No files found matching predictions_*.csv in the predictions directory.")
        return

    # Extract info from files
    file_tasks = []
    all_scodes = set()
    all_pred_dates = []

    print("Analyzing prediction files...")
    for f in files:
        # Match YYYYMMDD in filename
        match = re.search(r"predictions_(\d{8})", f.name)
        if not match:
            continue
            
        date_str = match.group(1)
        try:
            pred_date = pd.to_datetime(date_str)
        except ValueError:
            print(f"Skipping {f.name}: Invalid date format.")
            continue

        # Check if it's a valid prediction file and collect symbols
        try:
            # Try reading with cp932 as files contain Japanese characters
            df_temp = pd.read_csv(f, usecols=['scode'], encoding='cp932')
            all_scodes.update(df_temp['scode'].astype(str).tolist())
        except Exception as e:
            print(f"Skipping {f.name} due to error reading: {e}")
            continue

        file_tasks.append({
            'path': f,
            'pred_date': pred_date,
            'date_str': date_str
        })
        all_pred_dates.append(pred_date)

    if not file_tasks:
        print("No valid prediction tasks identified.")
        return

    # Determine data range: from earliest prediction to latest + buffer
    min_date = min(all_pred_dates)
    # 20 days buffer to ensure we cover "5 business days after" including weekends/holidays
    max_date = max(all_pred_dates) + timedelta(days=20)

    print(f"Initializing DataLoader. Fetching data from {min_date.date()} to {max_date.date()}...")
    loader = DataLoader()
    
    try:
        # Fetch trading calendar
        calendar_df = loader.fetch_topix_data(start_date=min_date.strftime('%Y-%m-%d'))
        if calendar_df.empty:
            print("Error: Could not fetch trading calendar (TOPIX data).")
            return
        
        trading_days = sorted(calendar_df['date'].unique())
        
        # Fetch OHLC data for all relevant symbols
        symbols_list = list(all_scodes)
        print(f"Fetching OHLC data for {len(symbols_list)} symbols...")
        price_df = loader.fetch_batch_data(symbols_list, start_date=min_date.strftime('%Y-%m-%d'))
        
        if price_df.empty:
            print("Error: No price data fetched from database.")
            return

        # Prepare price lookup tables
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df['scode'] = price_df['scode'].astype(str)
        
        print("Creating price lookup maps...")
        # pivot creates a table where rows are dates and columns are scodes
        open_prices = price_df.pivot(index='date', columns='scode', values='open')
        close_prices = price_df.pivot(index='date', columns='scode', values='close')

        # Process each file
        for task in file_tasks:
            f = task['path']
            pred_date = task['pred_date']
            
            # Find future trading days
            future_days = [d for d in trading_days if d > pred_date]
            
            if len(future_days) < 5:
                print(f"Skipping {f.name}: Insufficient future data (need 5 trading days after {pred_date.date()}).")
                continue
            
            entry_date = future_days[0]
            exit_date = future_days[4] # 5th trading day starting from entry_date
            
            print(f"Processing {f.name}: Entry(Open)={entry_date.date()}, Exit(Close)={exit_date.date()}")
            
            # Read full prediction file
            df = pd.read_csv(f, encoding='cp932')
            df['scode'] = df['scode'].astype(str)
            
            # Calculate results
            if entry_date in open_prices.index and exit_date in close_prices.index:
                entry_p = open_prices.loc[entry_date]
                exit_p = close_prices.loc[exit_date]
                
                # Map prices to symbols
                df['result'] = df['scode'].map(exit_p) / df['scode'].map(entry_p)
            else:
                print(f"Warning: Price data missing for required dates in {f.name}")
                df['result'] = None

            # Rename and save
            new_filename = f.name.replace("predictions_", "results_")
            new_path = f.parent / new_filename
            
            df.to_csv(new_path, index=False, encoding='cp932')
            print(f"Successfully saved to {new_filename}")
            
            # Remove original file as requested
            f.unlink()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        loader.close()
        print("DataLoader connection closed.")

if __name__ == "__main__":
    main()
