
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

# ---------------------------------------------------------
# サンプル重みの計算ロジック (中期モデルの「意識」を変える)
# ---------------------------------------------------------
def calculate_sample_weights(mcap_array, domain='STR'):
    if domain == 'TAC':
        # 短期: 全銘柄平等、または流動性に緩やかに比例
        # ノイズ除去のため、極端な低流動性株のウェイトを0にする手もある
        return np.ones(len(mcap_array.shape[0]))
    elif domain == 'STR':
        # 中期: 時価総額が大きいほど重視 (シグモイド関数でS字カーブを作る)
        # 例: 時価総額100億円以下はウェイトほぼ0、1000億円以上はウェイト1
        log_mcap = mcap_array.reshape(-1)
        # シグモイド関数の中心と傾きを調整
        # center: ウェイトが0.5になる時価総額の対数値 (例: 300億円 ≈ 24.1)
        # scale: カーブの緩やかさ
        center = np.log1p(30_000_000_000) 
        scale = 1.0 
        weights = 1 / (1 + np.exp(-(log_mcap - center) * scale))
        return weights