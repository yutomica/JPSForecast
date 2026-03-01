import pandas as pd
import numpy as np

class FinancialUniverseEngine:
    def __init__(self):
        # 東証Phase III 呼値テーブル (保守的設計: 非TOPIX100銘柄用を採用)
        self.tick_bins = [0, 3000, 5000, 30000, 50000, 300000, np.inf]
        self.tick_sizes = [1, 5, 10, 50, 100, 500]

    def calc_intrinsic_metrics(self, df):
        """
        Phase 1: 銘柄固有指標の計算 (standardize_raw_data.py)
        """
        # 1. 基礎集計
        df['filt_Turnover'] = (df['close'] * df['volume']).astype('float32')
        df['filt_Return'] = df.groupby('scode')['close'].pct_change().astype('float32')
        # 2. 呼値感度 (bps): 保守的に刻みが粗いテーブルを適用
        tick_size = pd.cut(df['close'], bins=self.tick_bins, labels=self.tick_sizes, right=True).astype(float)
        df['filt_Tick_Sensitivity'] = (tick_size / df['close'] * 10000).astype('float32')
        # 3. 20日売買代金中央値 (Median ADV): 仕手株のスパイクを排除
        df['filt_Median_ADV_20'] = df.groupby('scode')['volume_p'].transform(lambda x: x.rolling(20).median()).astype('float32')
        # 4. ATR Ratio: 仕様書に基づきボラティリティが一定以上の銘柄を抽出
        high_low_range = (df['high'] - df['low']) / df['close']
        df['filt_ATR_Ratio'] = high_low_range.rolling(20).mean().astype('float32')
        # 5. 当日ストップ高安判定 (リターン15%超をプロキシとする)
        df['filt_Is_Stop_Day'] = df['filt_Return'].abs() > 0.15
        return df.drop(['open','high','low'],axis=1)

    def calc_relative_metrics(self, df):
        """
        Phase 2: 市場比較と最終フラグ確定 (create_master_data.py)
        """
        # 1. ノイズスコア: 市場ボラティリティの3倍を超える異常変動の頻度 (直近60日)
        rolling_std = df.groupby('scode')['filt_Return'].transform(lambda x: x.rolling(20).std())
        is_outlier = (df['filt_Return'].abs() > (rolling_std * 3)).astype(int)
        df['filt_Noise_Score'] = is_outlier.groupby(df['scode']).transform(lambda x: x.rolling(60).sum()).astype('float32')
        # 2. 短期予測ユニバース (is_candidate_tac: 5日予測用)
        # 議論に基づき、ストップ高安の「当日」および「翌日」を排除
        df['is_candidate_tac'] = (
            (df['close'] >= 500) &                             # 価格下限 (厳格)
            (df['filt_Median_ADV_20'] >= 5e8) &                # 流動性下限 (5億円)
            (df['filt_Tick_Sensitivity'] <= 10) &              # 呼値感度 (10bps以下)
            (df['filt_ATR_Ratio'] >= 0.01) &                   # 仕様書のボラティリティ基準
            (df['filt_Noise_Score'] <= 5) &                    # ノイズ頻度制限
            (df['filt_Is_Stop_Day'] == False) &                # 当日ストップ除外
            (df.groupby('scode')['filt_Is_Stop_Day'].shift(1) == False) # 【重要】翌日除外
        )
        # 3. 長期予測ユニバース (is_candidate_str: 60日予測用)
        df['is_candidate_str'] = (
            (df['close'] >= 200) &                             # 価格下限 (緩和)
            (df['filt_Median_ADV_20'] >= 1e8) &                # 流動性下限 (1億円)
            (df['filt_Noise_Score'] <= 8)                      # ノイズ制限 (緩和)
        )
        # 4. 不要な中間カラムの削除 (メモリ節約)
        df = df.drop(columns=[x for x in df.columns if x.startswith('filt_')])
        
        return df