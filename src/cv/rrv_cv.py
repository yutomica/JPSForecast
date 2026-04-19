# src/cv/rrv_cv.py
# Representative Regime Validation
import numpy as np
import pandas as pd

class RRVPurgedCV:
    def __init__(self, samples_info_sets, purge_days=5, embargo_days=0):
        """
        samples_info: prepare_purged_cv_input が返す t1 位置の Series
        """
        self.samples_info_sets = samples_info_sets.to_numpy()
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.blocks = [
            ('2017-02-01', '2018-12-31'),
            ('2020-01-01', '2021-12-31'),
            ('2024-01-01', '2025-12-31')
        ]

    def split(self, X, y=None, groups=None):
        """
        groups: unique_dates (datetime.date の配列)
        """
        dates_s = pd.to_datetime(groups)
        indices = np.arange(len(dates_s))
        
        for s_date, e_date in self.blocks:
            mask = (dates_s >= s_date) & (dates_s <= e_date)
            block_pos = indices[mask]
            
            if len(block_pos) < 10: continue # サンプル不足はスキップ
            
            # ブロック内分割 (Train 80% / Val 20%)
            split_idx = int(len(block_pos) * 0.8)
            tr_pos = block_pos[:split_idx]
            va_pos = block_pos[split_idx:]
            
            # パージ: Trainサンプルの位置 i + purge_days が Valの開始位置より前になるように削除
            val_start_pos = va_pos[0]
            # (今回はhorizonを無視するためインデックスを直接利用)
            actual_tr_pos = [i for i in tr_pos if i + self.purge_days < val_start_pos]
            
            yield np.array(actual_tr_pos), va_pos