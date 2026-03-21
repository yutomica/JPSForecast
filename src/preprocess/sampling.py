import numpy as np
import pandas as pd

def apply_sampling(df, interval):
    if interval <= 1:
        return df
    print(f" [apply_sampling] Applying sampling (Date-interval): interval={interval} days")
    
    df = df.sort_values(['scode', 'date'])
    scodes = df['scode'].values
    dates = df['date'].values
    keep_mask = np.zeros(len(df), dtype=bool)
    
    last_scode = None
    last_date = np.datetime64('1900-01-01')
    interval_td = np.timedelta64(interval, 'D')
    
    for i in range(len(df)):
        if scodes[i] != last_scode or (dates[i] - last_date) >= interval_td:
            keep_mask[i] = True
            last_scode = scodes[i]
            last_date = dates[i]
            
    sampled_df = df[keep_mask].copy()
    return sampled_df