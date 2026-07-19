import pandas as pd
import numpy as np

class FinancialUniverseEngine:
    def __init__(self):
        # 東証Phase III 呼値テーブル (保守的設計: 非TOPIX100銘柄用を採用)
        self.tick_bins = [0, 3000, 5000, 30000, 50000, 300000, np.inf]
        self.tick_sizes = [1, 5, 10, 50, 100, 500]

    @staticmethod
    def _time_sort_positions(df):
        """Return stable scode/date order without changing the caller's row order."""
        missing = [col for col in ['scode', 'date'] if col not in df.columns]
        if missing:
            raise KeyError(f"Required time-series columns are missing: {missing}")
        if df['scode'].isna().any():
            raise ValueError("scode contains missing values.")

        dates = pd.to_datetime(df['date'], errors='coerce')
        if dates.isna().any():
            raise ValueError("date contains invalid values.")

        entity_order, _ = pd.factorize(df['scode'], sort=False)
        return (
            pd.DataFrame({
                '_entity_order': entity_order,
                '_date': dates.to_numpy(),
                '_row_position': np.arange(len(df)),
            })
            .sort_values(
                ['_entity_order', '_date', '_row_position'],
                kind='mergesort',
            )['_row_position']
            .to_numpy()
        )

    def calc_intrinsic_metrics(self, df):
        """
        Phase 1: 銘柄固有指標の計算 (standardize_raw_data.py)
        """
        sort_positions = self._time_sort_positions(df)
        ordered = df.iloc[sort_positions].reset_index(drop=True).copy()

        # 1. 基礎集計
        ordered['filt_Turnover'] = (ordered['close'] * ordered['volume']).astype('float32')
        ordered['filt_Return'] = ordered.groupby('scode', sort=False)['close'].pct_change(fill_method=None).astype('float32')
        # 2. 呼値感度 (bps): 保守的に刻みが粗いテーブルを適用
        tick_size = pd.cut(ordered['close'], bins=self.tick_bins, labels=self.tick_sizes, right=True).astype(float)
        ordered['filt_Tick_Sensitivity'] = (tick_size / ordered['close'] * 10000).astype('float32')
        # 3. 20日売買代金中央値 (Median ADV): 仕手株のスパイクを排除
        ordered['filt_Median_ADV_20'] = ordered.groupby('scode', sort=False)['volume_p'].transform(lambda x: x.rolling(20).median()).astype('float32')
        # 4. ATR Ratio: caller互換のため算出するが、候補条件には使用しない
        high_low_range = (ordered['high'] - ordered['low']) / ordered['close']
        ordered['filt_ATR_Ratio'] = high_low_range.groupby(ordered['scode'], sort=False).transform(lambda x: x.rolling(20).mean()).astype('float32')
        # 5. 当日ストップ高安判定 (リターン15%超をプロキシとする)
        ordered['filt_Is_Stop_Day'] = ordered['filt_Return'].abs() > 0.15

        for col in [
            'filt_Turnover',
            'filt_Return',
            'filt_Tick_Sensitivity',
            'filt_Median_ADV_20',
            'filt_ATR_Ratio',
            'filt_Is_Stop_Day',
        ]:
            values = np.empty(len(df), dtype=ordered[col].to_numpy().dtype)
            values[sort_positions] = ordered[col].to_numpy()
            df[col] = values
        return df.drop(['open','high','low'],axis=1)

    def calc_relative_metrics(self, df):
        """
        Phase 2: 市場比較と最終フラグ確定 (create_master_data.py)
        """
        sort_positions = self._time_sort_positions(df)
        ordered = df.iloc[sort_positions].reset_index(drop=True).copy()

        # 1. ノイズスコア: 3σを超える異常変動の直近60日回数
        rolling_std = ordered.groupby('scode', sort=False)['filt_Return'].transform(lambda x: x.rolling(20).std())
        is_outlier = (ordered['filt_Return'].abs() > (rolling_std * 3)).astype('int8')
        ordered['filt_Noise_Score'] = is_outlier.groupby(ordered['scode'], sort=False).transform(lambda x: x.rolling(60).sum()).astype('float32')
        previous_not_stop = ordered.groupby('scode', sort=False)['filt_Is_Stop_Day'].shift(1).eq(False)
        common = (
            (ordered['close'] >= 200) &
            (ordered['filt_Noise_Score'] <= 8) &
            ordered['filt_Is_Stop_Day'].eq(False)
        )

        ordered['is_candidate_5d'] = (
            common & previous_not_stop &
            (ordered['filt_Median_ADV_20'] >= 3e8) &
            (ordered['filt_Tick_Sensitivity'] <= 20)
        )
        ordered['is_candidate_10d'] = (
            common & previous_not_stop &
            (ordered['filt_Median_ADV_20'] >= 2e8) &
            (ordered['filt_Tick_Sensitivity'] <= 25)
        )
        ordered['is_candidate_20d'] = (
            common &
            (ordered['filt_Median_ADV_20'] >= 1e8) &
            (ordered['filt_Tick_Sensitivity'] <= 30)
        )
        ordered['is_candidate_40d'] = (
            common &
            (ordered['filt_Median_ADV_20'] >= 1e8) &
            (ordered['filt_Tick_Sensitivity'] <= 40)
        )
        ordered['is_candidate_60d'] = (
            common &
            (ordered['filt_Median_ADV_20'] >= 1e8) &
            (ordered['filt_Tick_Sensitivity'] <= 50)
        )

        candidate_cols = [
            'is_candidate_5d',
            'is_candidate_10d',
            'is_candidate_20d',
            'is_candidate_40d',
            'is_candidate_60d',
        ]
        for col in candidate_cols:
            values = np.empty(len(df), dtype=bool)
            values[sort_positions] = ordered[col].to_numpy(dtype=bool)
            df[col] = values

        if not all((~df[left] | df[right]).all() for left, right in zip(candidate_cols, candidate_cols[1:])):
            raise RuntimeError("Candidate universe nesting invariant was violated.")

        # create_master_data.py / train.py の現行インターフェースを維持
        df['is_candidate_tac'] = df['is_candidate_5d']
        df['is_candidate_str'] = df['is_candidate_60d']
        # 4. 不要な中間カラムの削除 (メモリ節約)
        df = df.drop(columns=[x for x in df.columns if x.startswith('filt_')])
        
        return df
