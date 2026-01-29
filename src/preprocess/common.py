
import numpy as np
import pandas as pd

# ==========================================
# 時間減衰ウェイト計算
# ==========================================
def calculate_time_decay_weights(dates, decay_rate=0.9999):
    dates = pd.to_datetime(dates)
    max_date = dates.max()
    days_diff = (max_date - dates).dt.days
    return np.power(decay_rate, days_diff).values