import pandas as pd
import numpy as np
import pandas_ta_classic as ta
from tqdm import tqdm
import talib
from functools import wraps
from scipy.special import erfinv
import gc
from typing import List, Optional, Dict
import warnings

# メモリ最適化（逐次代入）によるDataFrame断片化警告を抑制
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.new_cols = list()
        self.horizon_tac = 5    # 予測期間日数：戦術モデル
        self.horizon_str = 60   # 予測期間日数：戦略モデル
    
    def _calc_rci(self, series, period):
        time_ranks = np.arange(1, period + 1)
        def rci_func(window):
            price_ranks = pd.Series(window).rank(method='average').values
            d_squared = np.sum((time_ranks - price_ranks) ** 2)
            rci = (1 - (6 * d_squared) / (period * (period ** 2 - 1))) * 100
            return rci
        return series.rolling(window=period).apply(rci_func, raw=True)

    def _generate_name(self, cat: str, col: str, proc: str, param: Optional[str] = None) -> str:
        """命名規則に基づきカラム名を生成"""
        name = f"{cat}_{col}_{proc}"
        if param:
            name += f"_{param}"
        return name

    def _apply_ta(self, func, series, **kwargs):
        """1つの列に対するpandas_taの計算を銘柄ごとに行う安全なラッパー"""
        def safe_func(x):
            res = func(x, **kwargs)
            return res if res is not None else pd.Series(np.nan, index=x.index)
        return series.groupby(self.df['scode']).transform(safe_func)

    def _apply_ta_multi(self, func, cols, **kwargs):
        """複数の列に対するpandas_taの計算を銘柄ごとに行う安全なラッパー"""
        def _calc(group):
            args = [group[c] for c in cols]
            res = func(*args, **kwargs)
            return res.iloc[:, 0] if isinstance(res, pd.DataFrame) else (res if res is not None else pd.Series(np.nan, index=group.index))
        return self.df.groupby('scode', group_keys=False).apply(_calc)

    # --- 横断面加工 (Cross-Sectional) ---
    def cs_rank(self, cat: str, col: str, store: dict = None):
        """日別全銘柄G-Rank化"""
        new_col = col.split('_')[1]
        new_col = self._generate_name(cat, new_col, "CSR")
        def to_gaussian(x):
            n = x.count()
            if n > 0:
                r = x.rank(method='average')
                pct = (r - 0.5) / n
                return (np.sqrt(2) * erfinv(2 * pct - 1)).astype('float32')
            return x
        with np.errstate(invalid='ignore'):
            res = self.df.groupby('date')[col].transform(to_gaussian)
        if store is not None:
            store[new_col] = res
        else:
            self.df[new_col] = res
        return self

    def cs_zscore(self, cat, col, p=0.01, store: dict = None):
        """日別全銘柄Zスコア"""
        # Winsorization
        with np.errstate(invalid='ignore'):
            lower = self.df.groupby('date')[col].transform(lambda x: x.quantile(p))
            upper = self.df.groupby('date')[col].transform(lambda x: x.quantile(1-p))
            self.df[col] = self.df[col].clip(lower, upper)
            new_col = col.split('_')[1]
            new_col = self._generate_name(cat, new_col, "CSZ")
            res = self.df.groupby('date')[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
        res = res.astype('float32')
        if store is not None:
            store[new_col] = res
        else:
            self.df[new_col] = res
        return self

    def sn_zscore(self, cat: str, col: str, store: dict = None):
        """セクター別Zスコア (Sector Neutral)"""
        new_col = col.split('_')[1]
        new_col = self._generate_name(cat, new_col, "SNZ")
        with np.errstate(invalid='ignore'):
            res = self.df.groupby(['date', 'sector33_code'])[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
        res = res.astype('float32')
        if store is not None:
            store[new_col] = res
        else:
            self.df[new_col] = res
        return self

    # --- 時系列加工 (Time-Series) ---
    def ts_zscore(self, cat: str, col: str, w_window: int = 252, z_window: int = 20, p: float = 0.01, store: dict = None):
        """時系列Zスコア"""
        # Winsorization
        with np.errstate(invalid='ignore'):
            rolled = self.df.groupby('scode')[col].rolling(window=w_window, min_periods=1)
            lower = rolled.quantile(p).reset_index(level=0, drop=True)
            upper = rolled.quantile(1-p).reset_index(level=0, drop=True)
            self.df[col] = self.df[col].clip(lower, upper)
            new_col = col.split('_')[1]
            new_col = self._generate_name(cat, new_col, "TSZ", f"{z_window}D")
            res = self.df.groupby('scode')[col].transform(
                lambda x: (x - x.rolling(window=z_window, min_periods=1).mean()) / (x.rolling(window=z_window, min_periods=1).std() + 1e-8)
            )
        res = res.astype('float32')
        if store is not None:
            store[new_col] = res
        else:
            self.df[new_col] = res
        return self

    def ts_rank(self, cat, col, window=252, store: dict = None):
        """時系列G-Rank化"""
        new_col = col.split('_')[1]
        new_col = self._generate_name(cat, new_col, "TSR", f"{window}D")
        def rolling_gaussian(x):
            r = x.rolling(window, min_periods=1).rank(method='average')
            n = x.rolling(window, min_periods=1).count()
            pct = (r - 0.5) / n
            return (np.sqrt(2) * erfinv(2 * pct - 1)).astype('float32')
        with np.errstate(invalid='ignore'):
            res = self.df.groupby('scode')[col].transform(rolling_gaussian)
        
        if store is not None:
            store[new_col] = res
        else:
            self.df[new_col] = res
        return self

    # --- 高度な加工: 直交化 (Orthogonalization) ---
    def orthogonalize(self, cat: str, target_col: str, base_col: str):
        """直交化 (target_col から base_col の影響を除去)"""
        from sklearn.linear_model import LinearRegression
        
        new_col = self._generate_name(cat, target_col, "ORT", base_col)
        
        def _get_residual(group):
            if len(group) < 10: return group[target_col] * np.nan
            model = LinearRegression()
            X = group[[base_col]].values
            y = group[target_col].values
            model.fit(X, y)
            return y - model.predict(X)

        self.df[new_col] = self.df.groupby('Date').apply(
            lambda x: pd.Series(_get_residual(x), index=x.index)
        ).reset_index(level=0, drop=True)
        return self

    # --- 特徴量加工 ---
    def apply_bulk_time_series(self):
        # メモリ節約のため、辞書に貯めずに直接dfに代入する方式に変更
        columns = list(self.df.columns)
        for col in columns:
            if col.startswith("MOM_") and col.endswith("_RAW"):
                self.ts_zscore("MOM", col, store=None)
                self.ts_rank("MOM", col, store=None)
            elif col.startswith("VOL_") and col.endswith("_RAW"):
                self.ts_zscore("VOL", col, store=None)
                self.ts_rank("VOL", col, store=None)
            elif col.startswith("LIQ_") and col.endswith("_RAW"):
                self.ts_zscore("LIQ", col, store=None)
                self.ts_rank("LIQ", col, store=None)
            elif col.startswith("VAL_") and col.endswith("_RAW"):
                self.ts_zscore("VAL", col, store=None)
                self.ts_rank("VAL", col, store=None)
            elif col.startswith("QLT_") and col.endswith("_RAW"):
                self.ts_zscore("QLT", col, store=None)
                self.ts_rank("QLT", col, store=None)
            elif col.startswith("SPD_") and col.endswith("_RAW"):
                self.ts_zscore("SPD", col, store=None)
                self.ts_rank("SPD", col, store=None)
            elif col.startswith("BET_") and col.endswith("_RAW"):
                self.ts_zscore("BET", col, store=None)
                self.ts_rank("BET", col, store=None)
            elif col.startswith("CON_") and col.endswith("_RAW"):
                self.ts_zscore("CON", col, store=None)
                self.ts_rank("CON", col, store=None)
            # ループごとにGCを実行してメモリピークを抑える
            gc.collect()
        return self

    def apply_bulk_cross_sectional(self):
        """
        RAW特徴量に対して、横断面加工（cs_rank, cs_zscore, sn_zscore）を効率的に一括適用する。
        - プレフィックスごとの処理ルールを定義し、ベクトル化アプローチで高速に実行する。
        """
        print(f"Applying Cross-Sectional Transformations...")
        # 1. プレフィックスごとの変換ルールを定義
        TRANSFORM_CONFIG = {
            # prefix: [transform_type, ...]
            "MOM": ["cs_rank", "cs_zscore", "sn_zscore"],
            "VOL": ["cs_rank", "cs_zscore", "sn_zscore"],
            "LIQ": ["cs_rank", "cs_zscore"],
            "VAL": ["cs_rank", "cs_zscore", "sn_zscore"],
            "QLT": ["cs_rank", "cs_zscore", "sn_zscore"],
            "SIZ": ["cs_rank", "cs_zscore"],
            "SPD": ["cs_rank", "cs_zscore"],
            "BET": ["cs_rank", "cs_zscore", "sn_zscore"],
            "EVT": ["cs_rank", "cs_zscore", "sn_zscore"],
            "CON": ["cs_rank", "cs_zscore", "sn_zscore"],
            "GOV": ["cs_rank", "cs_zscore", "sn_zscore"],
        }
        # 2. 変換対象となるRAW特徴量を特定し、ルールに基づいて各変換リストに振り分ける
        raw_cols = [c for c in self.df.columns if c.endswith("_RAW")]
        cols_for_cs_rank = []
        cols_for_cs_zscore = []
        cols_for_sn_zscore = []
        for col in raw_cols:
            prefix = col.split('_')[0]
            if prefix in TRANSFORM_CONFIG:
                # 特殊ケースの除外: GOV_Sector33Code_RAW は変換対象外
                if col == "GOV_Sector33Code_RAW":
                    continue
                transforms = TRANSFORM_CONFIG[prefix]
                if "cs_rank" in transforms:
                    cols_for_cs_rank.append(col)
                if "cs_zscore" in transforms:
                    cols_for_cs_zscore.append(col)
                if "sn_zscore" in transforms:
                    cols_for_sn_zscore.append(col)
        new_cols_data = {}
        # 3. 横断面加工を一括で実行
        # 3-1. cs_rank (日次Gauss Rank)
        def to_gaussian_series(s):
            n = s.count()
            if n == 0: return s
            r = s.rank(method='average')
            pct = (r - 0.5) / n
            return (np.sqrt(2) * erfinv(2 * pct - 1)).astype('float32')
        if cols_for_cs_rank:
            print(f" - Applying cs_rank to {len(cols_for_cs_rank)} columns...")
            with np.errstate(invalid='ignore'):
                ranked_df = self.df.groupby('date')[cols_for_cs_rank].transform(to_gaussian_series)
            for col in ranked_df.columns:
                prefix, feature_name, _ = col.split('_', 2)
                new_cols_data[self._generate_name(prefix, feature_name, "CSR")] = ranked_df[col]
        # 3-2. cs_zscore (日次Z-Score)
        def zscore_series(s, p=0.01):
            lower = s.quantile(p)
            upper = s.quantile(1 - p)
            s_clipped = s.clip(lower, upper)
            return ((s_clipped - s_clipped.mean()) / (s_clipped.std() + 1e-8)).astype('float32')
        if cols_for_cs_zscore:
            print(f" - Applying cs_zscore to {len(cols_for_cs_zscore)} columns...")
            with np.errstate(invalid='ignore'):
                zscored_df = self.df.groupby('date')[cols_for_cs_zscore].transform(zscore_series)
            for col in zscored_df.columns:
                prefix, feature_name, _ = col.split('_', 2)
                new_cols_data[self._generate_name(prefix, feature_name, "CSZ")] = zscored_df[col]
        # 3-3. sn_zscore (セクター内Z-Score)
        if cols_for_sn_zscore:
            print(f" - Applying sn_zscore to {len(cols_for_sn_zscore)} columns...")
            with np.errstate(invalid='ignore'):
                sn_zscored_df = self.df.groupby(['date', 'sector33_code'])[cols_for_sn_zscore].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-8)
                )
            sn_zscored_df = sn_zscored_df.astype('float32')
            for col in sn_zscored_df.columns:
                prefix, feature_name, _ = col.split('_', 2)
                new_cols_data[self._generate_name(prefix, feature_name, "SNZ")] = sn_zscored_df[col]
        # 4. 新しい特徴量をDataFrameに結合
        if new_cols_data:
            print(" - Concatenating new features...")
            self.df = pd.concat([self.df, pd.DataFrame(new_cols_data, index=self.df.index)], axis=1)
        gc.collect()
        return self
    
    # --- RAW特徴量作成 ---
    def apply_momentum_block(self):
        grouped = self.df.groupby('scode')
        grouped_close = grouped['close']
        grouped_high = grouped['high']
        grouped_low = grouped['low']

        col_name = self._generate_name("MOM", "ADX14", "RAW")
        self.df[col_name] = self._apply_ta_multi(ta.adx, ['high', 'low', 'close'], length=14)
        self.new_cols.append(col_name)

        sma5 = self._apply_ta(ta.sma, self.df['close'], length=5)
        sma25 = self._apply_ta(ta.sma, self.df['close'], length=25)
        sma75 = self._apply_ta(ta.sma, self.df['close'], length=75)

        for w, sma in zip([5, 25, 75], [sma5, sma25, sma75]):
            col_name = self._generate_name("MOM", f"DistSMA{w}", "RAW")
            self.df[col_name] = (self.df['close'] / sma) - 1
            self.new_cols.append(col_name)

        col_name = self._generate_name("MOM", "DistSMA5-25", "RAW")
        self.df[col_name] = (sma5 / sma25) - 1
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "DistSMA25-75", "RAW")
        self.df[col_name] = (sma25 / sma75) - 1
        self.new_cols.append(col_name)

        col_name = self._generate_name("MOM", "EfficiencyRatio10", "RAW")
        self.df[col_name] = self._apply_ta(ta.er, self.df['close'], length=10)
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "LinRegSlope10", "RAW")
        self.df[col_name] = self._apply_ta(ta.slope, self.df['close'], length=10)
        self.new_cols.append(col_name)

        def get_macd_norm(g):
            macd = ta.macd(g, fast=12, slow=26, signal=9)
            return macd['MACDh_12_26_9'] if macd is not None else pd.Series(np.nan, index=g.index)
        macd_hist = grouped_close.apply(get_macd_norm).reset_index(level=0, drop=True) / self.df['close']
        col_name = self._generate_name("MOM", "MACDHistNorm", "RAW")
        self.df[col_name] = macd_hist
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "MACDHistDiff", "RAW")
        self.df[col_name] = macd_hist.groupby(self.df['scode']).diff(1)
        self.new_cols.append(col_name)

        col_name = self._generate_name("MOM", "DistHigh60", "RAW")
        self.df[col_name] = self.df['close'] / grouped_high.rolling(60).max().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "DistHigh250", "RAW")
        self.df[col_name] = self.df['close'] / grouped_high.rolling(250).max().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)

        high_26 = grouped_high.rolling(26).max().reset_index(level=0, drop=True)
        low_26 = grouped_low.rolling(26).min().reset_index(level=0, drop=True)
        kijun_sen = (high_26 + low_26) / 2
        col_name = self._generate_name("MOM", "DistKijun", "RAW")
        self.df[col_name] = (self.df['close'] - kijun_sen) / kijun_sen
        self.new_cols.append(col_name)

        past_3d_high = grouped_high.shift(1).groupby(self.df['scode']).rolling(3).max().reset_index(level=0, drop=True)
        col_name = self._generate_name("MOM", "NewHighFlag3", "RAW")
        self.df[col_name] = (self.df['close'] > past_3d_high).astype(int)
        self.new_cols.append(col_name)

        roll_120 = grouped_close.rolling(120)
        max_120 = roll_120.max().reset_index(level=0, drop=True)
        min_120 = roll_120.min().reset_index(level=0, drop=True)
        col_name = self._generate_name("MOM", "PricePos120", "RAW")
        self.df[col_name] = (self.df['close'] - min_120) / (max_120 - min_120)
        self.new_cols.append(col_name)

        high_52 = grouped_high.rolling(52).max().reset_index(level=0, drop=True)
        low_52 = grouped_low.rolling(52).min().reset_index(level=0, drop=True)
        span_b = (high_52 + low_52) / 2
        span_b_curr = span_b.groupby(self.df['scode']).shift(26)
        col_name = self._generate_name("MOM", "IchimokuDist", "RAW")
        self.df[col_name] = (self.df['close'] - span_b_curr) / span_b_curr
        self.new_cols.append(col_name)

        col_name = self._generate_name("MOM", "LogReturn", "RAW")
        log_return = np.log(self.df['close'] / grouped_close.shift(1))
        self.df[col_name] = log_return
        self.new_cols.append(col_name)

        for w in [3,5,10,20]:
            col_name = self._generate_name("MOM", f"Return{w}d", "RAW")
            self.df[col_name] = grouped_close.pct_change(w)
            self.new_cols.append(col_name)

        for w in [1,2]:
            col_name = self._generate_name("MOM", f"Return1dLag{w}", "RAW")
            self.df[col_name] = log_return.groupby(self.df['scode']).shift(w)
            self.new_cols.append(col_name)

        col_name = self._generate_name("MOM", "RSI9", "RAW")
        rsi9 = self._apply_ta(ta.rsi, self.df['close'], length=9)
        self.df[col_name] = rsi9
        self.new_cols.append(col_name)

        col_name = self._generate_name("MOM", "RSI9-14Diff", "RAW")
        rsi14 = self._apply_ta(ta.rsi, self.df['close'], length=14)
        self.df[col_name] = rsi9 - rsi14
        self.new_cols.append(col_name)

        def calc_bb_pct(x):
            bb = ta.bbands(x, length=20, std=2)
            if bb is not None:
                bb_p_col = [c for c in bb.columns if c.startswith('BBP')][0]
                return bb[bb_p_col]
            return pd.Series(np.nan, index=x.index)
        col_name = self._generate_name("MOM", "BBPercentB", "RAW")
        self.df[col_name] = grouped_close.apply(calc_bb_pct).reset_index(level=0, drop=True)
        self.new_cols.append(col_name)

        range_len = self.df['high'] - self.df['low']
        body_size = np.abs(self.df['close'] - self.df['open'])
        upper_shadow = self.df['high'] - self.df[['close', 'open']].max(axis=1)
        lower_shadow = self.df[['close', 'open']].min(axis=1) - self.df['low']
        with np.errstate(divide='ignore', invalid='ignore'):
            col_name = self._generate_name("MOM", "BodyRatio", "RAW")
            self.df[col_name] = body_size / range_len
            self.new_cols.append(col_name)
            col_name = self._generate_name("MOM", "UpperShadowRatio", "RAW")
            self.df[col_name] = upper_shadow / range_len
            self.new_cols.append(col_name)
            col_name = self._generate_name("MOM", "LowerShadowRatio", "RAW")
            lower_shadow_ratio = lower_shadow / range_len
            self.df[col_name] = lower_shadow_ratio
            self.new_cols.append(col_name)
            col_name = self._generate_name("MOM", "IntradayStrength", "RAW")
            self.df[col_name] = (self.df['close'] - self.df['open']) / range_len
            self.new_cols.append(col_name)
            col_name = self._generate_name("MOM", "LowerShadowMA5", "RAW")
            self.df[col_name] = lower_shadow_ratio.groupby(self.df['scode']).rolling(5).mean().reset_index(level=0, drop=True)

        diff = grouped_close.diff()
        sign = np.sign(diff).fillna(0)
        is_change = sign != sign.groupby(self.df['scode']).shift(1)
        group_id = is_change.groupby(self.df['scode']).cumsum()
        count = self.df.groupby(['scode', group_id]).cumcount() + 1
        col_name = self._generate_name("MOM", "Streak", "RAW")
        self.df[col_name] = np.where(sign > 0, count, np.where(sign < 0, -count, 0))
        self.new_cols.append(col_name)

        col_name = self._generate_name("MOM", "BullishRatio20", "RAW")
        self.df[col_name] = (sign > 0).groupby(self.df['scode']).rolling(20).mean().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)

        col_name = self._generate_name("MOM", "ClosePosition", "RAW")
        self.df[col_name] = (self.df['close'] - grouped_low.rolling(20).min().reset_index(level=0, drop=True)) / (grouped_high.rolling(20).max().reset_index(level=0, drop=True) - grouped_low.rolling(20).min().reset_index(level=0, drop=True))
        self.new_cols.append(col_name)

        prev_close = grouped_close.shift(1)
        col_name = self._generate_name("MOM", "GapRate", "RAW")
        self.df[col_name] = (self.df['open'] / prev_close) - 1.0
        self.new_cols.append(col_name)

        col_name = self._generate_name("MOM", "MaxGain5", "RAW")
        self.df[col_name] = (grouped_high.rolling(5).max().reset_index(level=0, drop=True) / self.df['close']) - 1.0
        self.new_cols.append(col_name)

        col_name = self._generate_name("MOM", "RCI9", "RAW")
        rci_9 = grouped_close.apply(lambda x: self._calc_rci(x, 9)).reset_index(level=0, drop=True)
        self.df[col_name] = rci_9
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "RCI9Diff", "RAW")
        self.df[col_name] = rci_9.groupby(self.df['scode']).diff(1)
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "RCI26", "RAW")
        rci_26 = grouped_close.apply(lambda x: self._calc_rci(x, 26)).reset_index(level=0, drop=True)
        self.df[col_name] = rci_26
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "RCI26Diff", "RAW")
        self.df[col_name] = rci_26.groupby(self.df['scode']).diff(1)
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "RCI52", "RAW")
        self.df[col_name] = grouped_close.apply(lambda x: self._calc_rci(x, 52)).reset_index(level=0, drop=True)
        self.new_cols.append(col_name)

        col_name = self._generate_name("MOM", "Momentum12-1", "RAW")
        self.df[col_name] = grouped_close.shift(20) / grouped_close.shift(260) - 1
        self.new_cols.append(col_name)

        col_name = self._generate_name("MOM", "RetIntraday", "RAW")
        self.df[col_name] = (self.df['close'] / self.df['open']) - 1.0
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "WinRate10d", "RAW")
        self.df[col_name] = (log_return > 0).groupby(self.df['scode']).rolling(10).mean().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)

        for window in [25, 75, 200]:
            ma = grouped_close.rolling(window=window).mean().reset_index(level=0, drop=True)
            col_name = self._generate_name("MOM", f"MADev{window}", "RAW")
            self.df[col_name] = (self.df['close'] / ma) - 1
            self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "Return20d", "RAW")
        self.df[col_name] = grouped_close.pct_change(20)
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "Return6m", "RAW")
        self.df[col_name] = grouped_close.pct_change(120)
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "Return12m", "RAW")
        self.df[col_name] = grouped_close.pct_change(240)
        self.new_cols.append(col_name)
        max_52w = grouped_close.rolling(window=240).max().reset_index(level=0, drop=True)
        col_name = self._generate_name("MOM", "High52wDist", "RAW")
        self.df[col_name] = (self.df['close'] / max_52w) - 1
        self.new_cols.append(col_name)
        return self

    def apply_volatility_block(self):
        grouped = self.df.groupby('scode')
        grouped_close = grouped['close']
        grouped_low = grouped['low']
        grouped_high = grouped['high']

        log_return = np.log(self.df['close'] / grouped_close.shift(1))
        return_1d = grouped_close.pct_change(1)

        col_name = self._generate_name("VOL", "MAE5", "RAW")
        self.df[col_name] = (grouped_low.rolling(5).min().reset_index(level=0, drop=True) / self.df['close']) - 1.0
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "MAE10", "RAW")
        self.df[col_name] = (grouped_low.rolling(10).min().reset_index(level=0, drop=True) / self.df['close']) - 1.0
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "DownsideRun", "RAW")
        self.df[col_name] = log_return.clip(lower=0).groupby(self.df['scode']).rolling(5).sum().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "ReturnSkewnessDiff", "RAW")
        self.df[col_name] = log_return.groupby(self.df['scode']).rolling(20).skew().reset_index(level=0, drop=True).groupby(self.df['scode']).diff()
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "ReturnKurtosis", "RAW")
        self.df[col_name] = log_return.groupby(self.df['scode']).rolling(20).kurt().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)

        def calc_bb_width(x):
            bb = ta.bbands(x, length=20, std=2)
            if bb is not None:
                bb_w_cols = [c for c in bb.columns if c.startswith('BBB')]
                if bb_w_cols: return bb[bb_w_cols[0]]
            return pd.Series(np.nan, index=x.index)
        col_name = self._generate_name("VOL", "BBWidth", "RAW")
        self.df[col_name] = grouped_close.apply(calc_bb_width).reset_index(level=0, drop=True)
        self.new_cols.append(col_name)

        gap_rate = (self.df['open'] / grouped_close.shift(1)) - 1.0
        col_name = self._generate_name("VOL", "GapAbs", "RAW")
        self.df[col_name] = gap_rate.abs()
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "LargeMoveCount", "RAW")
        self.df[col_name] = (return_1d.abs() > 0.03).groupby(self.df['scode']).rolling(20).sum().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)
        range_len = self.df['high'] - self.df['low']
        col_name = self._generate_name("VOL", "RangeRatioLong", "RAW")
        range_grouped = range_len.groupby(self.df['scode'])
        self.df[col_name] = range_grouped.rolling(5).mean().reset_index(level=0, drop=True) / range_grouped.rolling(20).mean().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)

        def calc_atr(group, length):
            res = ta.atr(group['high'], group['low'], group['close'], length=length)
            return res if res is not None else pd.Series(np.nan, index=group.index)

        atr = grouped.apply(lambda g: calc_atr(g, 14)).reset_index(level=0, drop=True)
        col_name = self._generate_name("VOL", "ATRRatio", "RAW")
        self.df[col_name] = atr / self.df['close']
        self.new_cols.append(col_name)

        atr_mid = grouped.apply(lambda g: calc_atr(g, 20)).reset_index(level=0, drop=True)
        atr_short = grouped.apply(lambda g: calc_atr(g, 5)).reset_index(level=0, drop=True)
        col_name = self._generate_name("VOL", "ATRSqueeze", "RAW")
        self.df[col_name] = atr_short / atr_mid
        self.new_cols.append(col_name)

        def calc_atr_squeeze_bb(x):
            bb = ta.bbands(x, length=20, std=2)
            if bb is not None:
                return (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / (bb['BBM_20_2.0'] + 1e-9)
            return pd.Series(np.nan, index=x.index)
        col_name = self._generate_name("VOL", "ATRSqueezeBB", "RAW")
        self.df[col_name] = grouped_close.apply(calc_atr_squeeze_bb).reset_index(level=0, drop=True)
        self.new_cols.append(col_name)

        col_name = self._generate_name("VOL", "HistVol20", "RAW")
        self.df[col_name] = log_return.groupby(self.df['scode']).rolling(20).std().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "HV30", "RAW")
        hv30 = log_return.groupby(self.df['scode']).rolling(30).std().reset_index(level=0, drop=True) * np.sqrt(250)
        self.df[col_name] = hv30
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "HVSlope", "RAW")
        self.df[col_name] = hv30.groupby(self.df['scode']).diff(5)
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "RealizedSkew20", "RAW")
        self.df[col_name] = log_return.groupby(self.df['scode']).rolling(20).skew().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "UlcerIndex14", "RAW")
        self.df[col_name] = self._apply_ta(ta.ui, self.df['close'], length=14)
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "ATRChgRate", "RAW")
        self.df[col_name] = atr.groupby(self.df['scode']).pct_change()
        self.new_cols.append(col_name)
        with np.errstate(divide='ignore', invalid='ignore'):
            hl_log_sq = np.log(self.df['high'] / self.df['low']) ** 2
            const_factor = 4 * np.log(2)
            col_name = self._generate_name("VOL", "VolatilityParkinson", "RAW")
            self.df[col_name] = np.sqrt(hl_log_sq.groupby(self.df['scode']).rolling(14).mean().reset_index(level=0, drop=True) / const_factor)
            self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "ReturnVolatility", "RAW")
        self.df[col_name] = log_return.groupby(self.df['scode']).rolling(10).std().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)
        hv5 = log_return.groupby(self.df['scode']).rolling(5).std().reset_index(level=0, drop=True) * np.sqrt(250)
        col_name = self._generate_name("VOL", "VolRatioHV", "RAW")
        self.df[col_name] = hv5 / hv30
        self.new_cols.append(col_name)

        def calc_downside_std(x, window=60):
            neg_ret = x.where(x < 0, 0)
            return neg_ret.rolling(window).std()
        col_name = self._generate_name("VOL", "DownsideDev60", "RAW")
        self.df[col_name] = log_return.groupby(self.df['scode']).apply(lambda x: calc_downside_std(x)).reset_index(level=0, drop=True)
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "Volatility60", "RAW")
        self.df[col_name] = grouped_close.pct_change().groupby(self.df['scode']).rolling(60).std().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)
        return self
        
    def apply_liquidity_block(self):
        grouped = self.df.groupby('scode')
        grouped_vol = grouped['volume']
        grouped_vol_p = grouped['volume_p']
        grouped_close = grouped['close']

        volume_log = np.log(self.df['volume'] + 1)
        log_return = np.log(self.df['close'] / grouped_close.shift(1))
        col_name = self._generate_name("LIQ", "VolumeLog", "RAW")
        self.df[col_name] = volume_log
        self.new_cols.append(col_name)
        col_name = self._generate_name("LIQ", "AbnormalVolume", "RAW")
        self.df[col_name] = self.df['volume_p'] / grouped_vol_p.rolling(20).mean().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)
        col_name = self._generate_name("LIQ", "VolumeChange", "RAW")
        self.df[col_name] = grouped_vol.pct_change()
        self.new_cols.append(col_name)
        col_name = self._generate_name("LIQ", "VolumeSlope5", "RAW")
        self.df[col_name] = volume_log.groupby(self.df['scode']).transform(lambda x: ta.slope(x, length=5) if ta.slope(x, length=5) is not None else pd.Series(np.nan, index=x.index))
        self.new_cols.append(col_name)
        col_name = self._generate_name("LIQ", "LogTradingCap", "RAW")
        self.df[col_name] = np.log(self.df['close'] * self.df['volume'] + 1)
        self.new_cols.append(col_name)

        def calc_mfi(group):
            res = ta.mfi(group['high'], group['low'], group['close'], group['volume'], length=14)
            return res if res is not None else pd.Series(np.nan, index=group.index)
        col_name = self._generate_name("LIQ", "MFI14", "RAW")
        self.df[col_name] = grouped.apply(calc_mfi).reset_index(level=0, drop=True)
        self.new_cols.append(col_name)

        vol_ma5 = grouped_vol.rolling(5).mean().reset_index(level=0, drop=True)
        col_name = self._generate_name("LIQ", "VolRatio5d", "RAW")
        self.df[col_name] = self.df['volume'] / vol_ma5.replace(0, np.nan)
        self.new_cols.append(col_name)
        vol_median_20 = grouped_vol.rolling(20).median().reset_index(level=0, drop=True)
        is_spike = (self.df['volume'] > (vol_median_20 * 3)).astype(int)
        col_name = self._generate_name("LIQ", "VolSpikeCount20", "RAW")
        self.df[col_name] = is_spike.groupby(self.df['scode']).rolling(20).sum().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)
        is_up = (self.df['close'] > self.df['open']).astype(int)
        vol_up = (self.df['volume'] * is_up).groupby(self.df['scode']).rolling(20).sum().reset_index(level=0, drop=True)
        vol_down = (self.df['volume'] * (1 - is_up)).groupby(self.df['scode']).rolling(20).sum().reset_index(level=0, drop=True)
        col_name = self._generate_name("LIQ", "VolUpDownRatio", "RAW")
        self.df[col_name] = vol_up / (vol_down + 1e-9)
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "VolumeMA25", "RAW")
        self.df[col_name] = volume_log.groupby(self.df['scode']).rolling(25).mean().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)
        col_name = self._generate_name("LIQ", "TurnoverRatio", "RAW")
        self.df[col_name] = self.df['volume'] / self.df['shares_outstanding'].replace(0, np.nan)
        self.new_cols.append(col_name)
        col_name = self._generate_name("LIQ", "AmihudIlliq", "RAW")
        self.df[col_name] = log_return.abs() / (self.df['close'] * self.df['volume'] + 1e-9)
        self.new_cols.append(col_name)
        return self
    
    def apply_value_block(self):
        epsilon = 1e-6
        denom_shares = self.df['shares_outstanding'].abs() + epsilon
        actual_bps = self.df['equity'] / denom_shares
        actual_eps = self.df['net_income'] / denom_shares
        filled_eps = self.df['eps'].combine_first(actual_eps)
        col_name = self._generate_name("VAL", "LogPBR", "RAW")
        self.df[col_name] = np.log(self.df['close'] / (actual_bps.clip(lower=0.01))) 
        self.new_cols.append(col_name)
        col_name = self._generate_name("VAL", "EarningsYield", "RAW")
        self.df[col_name] = filled_eps / (self.df['close'] + epsilon)
        self.new_cols.append(col_name)
        return self
    
    def apply_quality_block(self):
        epsilon = 1e-6
        LAG_YEAR = 240
        col_name = self._generate_name("QLT", "Accruals", "RAW")
        self.df[col_name] = (self.df['net_income'] - self.df['operating_cf']) / (self.df['total_assets'].abs() + epsilon)
        self.new_cols.append(col_name)
        col_name = self._generate_name("QLT", "EquityRatio", "RAW")
        self.df[col_name] = self.df['equity'] / (self.df['total_assets'].abs() + epsilon)
        self.new_cols.append(col_name)
        col_name = self._generate_name("QLT", "OPMargin", "RAW")
        self.df[col_name] = self.df['operating_profit'] / (self.df['sales'].abs() + epsilon)
        self.new_cols.append(col_name)
        col_name = self._generate_name("QLT", "ROA", "RAW")
        self.df[col_name] = self.df['net_income'] / (self.df['total_assets'].abs() + epsilon)
        self.new_cols.append(col_name)
        col_name = self._generate_name("QLT", "ROE", "RAW")
        self.df[col_name] = self.df['net_income'] / (self.df['equity'].abs() + epsilon)
        self.new_cols.append(col_name) 
        # 成長率計算
        fund_cols = ['operating_profit', 'sales', 'eps']
        grouped = self.df.groupby('scode')
        for col in fund_cols:
            v_t = grouped[col].ffill()
            v_prev = grouped[col].shift(LAG_YEAR).ffill()
            growth = (v_t - v_prev) / (0.5 * (v_t.abs() + v_prev.abs()) + epsilon)
            self.df[f'{col}_growth_yoy'] = growth.clip(-3.0, 3.0)
        col_name = self._generate_name("QLT", "OperatingProfitGrowthYOY", "RAW")
        self.df[col_name] = self.df['operating_profit_growth_yoy']
        self.new_cols.append(col_name)
        col_name = self._generate_name("QLT", "SalesGrowthYOY", "RAW")
        self.df[col_name] = self.df['sales_growth_yoy']
        self.new_cols.append(col_name)
        col_name = self._generate_name("QLT", "EPSGrowthYOY", "RAW")
        self.df[col_name] = self.df['eps_growth_yoy']
        self.new_cols.append(col_name)
        return self
    
    def apply_size_block(self):
        col_name = self._generate_name("SIZ", "LogMarketCap", "RAW")
        self.df[col_name] = np.log(self.df['close'] * self.df['shares_outstanding'])
        self.new_cols.append(col_name)
        return self
    
    def apply_supplydemand_bloc(self):
        grouped_vol = self.df.groupby('scode')['volume']

        typ_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        pv_sum = (typ_price * self.df['volume']).groupby(self.df['scode']).rolling(5).sum().reset_index(level=0, drop=True)
        v_sum = grouped_vol.rolling(5).sum().reset_index(level=0, drop=True)
        rolling_vwap = pv_sum / v_sum
        col_name = self._generate_name("SPD", "DistVWAP5", "RAW")
        dist_vwap_5 = (self.df['close'] - rolling_vwap) / rolling_vwap
        self.df[col_name] = dist_vwap_5
        self.new_cols.append(col_name)
        col_name = self._generate_name("SPD", "DistVWAPSlope", "RAW")
        self.df[col_name] = dist_vwap_5.groupby(self.df['scode']).diff()
        self.new_cols.append(col_name)
        vwap = self.df['volume_p'] / self.df['volume'].replace(0, 1)
        col_name = self._generate_name("SPD", "VWAPDev", "RAW")
        self.df[col_name] = (self.df['close'] / vwap) - 1
        self.new_cols.append(col_name)
        avg_vol_60 = grouped_vol.rolling(60).mean().reset_index(level=0, drop=True)
        col_name = self._generate_name("SPD", "MarginBuyImpact", "RAW")
        self.df[col_name] = (
            self.df['long_margin_trade_balance_share'] / avg_vol_60.replace(0, np.nan)
        )
        self.new_cols.append(col_name)
        margin_ratio = np.log(
            (self.df['long_margin_trade_balance_share'] + 1) / 
            (self.df['short_margin_trade_balance_share'] + 1)
        )
        col_name = self._generate_name("SPD", "MarginRatio", "RAW")
        self.df[col_name] = margin_ratio
        self.new_cols.append(col_name)
        col_name = self._generate_name("SPD", "MarginRatioDelta4w", "RAW")
        self.df[col_name] = margin_ratio.groupby(self.df['scode']).diff(20)
        self.new_cols.append(col_name)
        col_name = self._generate_name("SPD", "MarketForeignBuy", "RAW")
        self.df[col_name] = self.df['Foreign_Net_Buy']
        self.new_cols.append(col_name)
        col_name = self._generate_name("SPD", "MarketIndividualBuy", "RAW")
        self.df[col_name] = self.df['Individual_Net_Buy']
        self.new_cols.append(col_name)
        col_name = self._generate_name("SPD", "OverseaFlowTrend", "RAW")
        flow = self.df['Foreign_Net_Buy'].groupby(self.df['scode']).rolling(20).mean().reset_index(level=0, drop=True)
        self.df[col_name] = flow
        self.new_cols.append(col_name)
        col_name = self._generate_name("SPD", "FlowAccel", "RAW")
        self.df[col_name] = flow - flow.groupby(self.df['scode']).rolling(5).mean().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)
        return self
    
    def apply_beta_block(self):
        grouped = self.df.groupby('scode')
        grouped_close = grouped['close']
        
        ret_market = self.df['Market_Return']
        def calc_cov(g):
            r_s = g['close'].pct_change()
            return r_s.rolling(60).cov(g['Market_Return'])
        rolling_cov = grouped.apply(calc_cov).reset_index(level=0, drop=True)
        rolling_var = grouped['Market_Return'].rolling(60).var().reset_index(level=0, drop=True)
        
        log_return = np.log(self.df['close'] / grouped_close.shift(1))
        col_name = self._generate_name("BET", "Beta60", "RAW")
        self.df[col_name] = rolling_cov / rolling_var
        self.new_cols.append(col_name)
        col_name = self._generate_name("BET", "RS25", "RAW")
        self.df[col_name] = grouped_close.pct_change(25) - grouped['close_mkt'].pct_change(25)
        self.new_cols.append(col_name)
        col_name = self._generate_name("BET", "MarketReturn", "RAW")
        self.df[col_name] = ret_market
        self.new_cols.append(col_name)
        col_name = self._generate_name("BET", "MarketTrendIdx", "RAW")
        self.df[col_name] = self.df['Market_Trend_Idx']
        self.new_cols.append(col_name)
        col_name = self._generate_name("BET", "MarketHV20", "RAW")
        self.df[col_name] = self.df['Market_HV_20']
        self.new_cols.append(col_name)
        col_name = self._generate_name("BET", "MarketVolChange", "RAW")
        self.df[col_name] = self.df['market_vol_change']
        self.new_cols.append(col_name)
        col_name = self._generate_name("BET", "SectorMomentum5d", "RAW")
        self.df[col_name] = grouped['sector_return'].rolling(5).mean().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)
        col_name = self._generate_name("BET", "SectorReturn", "RAW")
        self.df[col_name] = self.df['sector_return']
        self.new_cols.append(col_name)
        col_name = self._generate_name("BET", "SectorRel", "RAW")
        self.df[col_name] = self.df['close'] / self.df['sector_return']
        self.new_cols.append(col_name)
        col_name = self._generate_name("BET", "RelSectorReturn1d", "RAW")
        self.df[col_name] = log_return - self.df['sector_return']
        self.new_cols.append(col_name)
        col_name = self._generate_name("BET", "RelSectorReturn5d", "RAW")
        self.df[col_name] = log_return.groupby(self.df['scode']).rolling(5).sum().reset_index(level=0, drop=True) - grouped['sector_return'].rolling(5).sum().reset_index(level=0, drop=True)
        self.new_cols.append(col_name)
        sector_ret_60 = (1 + self.df['sector_return']).groupby(self.df['scode']).rolling(60).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1
        market_ret_60 = (1 + self.df['Market_Return']).groupby(self.df['scode']).rolling(60).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1
        col_name = self._generate_name("BET", "SectorRelStrength60", "RAW")
        self.df[col_name] = sector_ret_60 - market_ret_60
        self.new_cols.append(col_name)
        return self
    
    def apply_seasonality_block(self):
        day_num = self.df['date'].dt.day
        month = self.df['date'].dt.month
        col_name = self._generate_name("SEA", "MonthInQuarter", "RAW")
        self.df[col_name] = self.df['date'].dt.month % 3
        self.new_cols.append(col_name)
        col_name = self._generate_name("SEA", "DayOfMonth", "RAW")
        self.df[col_name] = self.df['date'].dt.day
        self.new_cols.append(col_name)
        col_name = self._generate_name("SEA", "DayOfWeek", "RAW")
        day_of_week = self.df['date'].dt.dayofweek
        self.df[col_name] = day_of_week
        self.new_cols.append(col_name)
        col_name = self._generate_name("SEA", "SinDayOfWeek", "RAW")
        self.df[col_name] = np.sin(2 * np.pi * day_of_week / 6)
        self.new_cols.append(col_name)
        col_name = self._generate_name("SEA", "CosDayOfWeek", "RAW")
        self.df[col_name] = np.cos(2 * np.pi * day_of_week / 6)
        self.new_cols.append(col_name)
        col_name = self._generate_name("SEA", "IsGotobi", "RAW")
        self.df[col_name] = ((day_num % 5 == 0) | (day_num == 31)).astype(int)
        self.new_cols.append(col_name)
        col_name = self._generate_name("SEA", "IsMonthEnd", "RAW")
        self.df[col_name] = self.df['date'].dt.is_month_end.astype(int)
        self.new_cols.append(col_name)
        col_name = self._generate_name("SEA", "Quarter", "RAW")
        self.df[col_name] = self.df['date'].dt.quarter
        self.new_cols.append(col_name)
        col_name = self._generate_name("SEA", "IsQuarterEnd", "RAW")
        self.df[col_name] = month.isin([3, 6, 9, 12]).astype(int) # 簡易判定
        self.new_cols.append(col_name)
        col_name = self._generate_name("SEA", "TimeIdx", "RAW")
        self.df[col_name] = (self.df['date'] - self.df['date'].min()).dt.days
        self.new_cols.append(col_name)
        return self
    
    def apply_event_block(self):
        col_name = self._generate_name("EVT", "EPSActual", "RAW")
        self.df[col_name] = self.df.groupby('scode')['eps'].ffill()
        self.new_cols.append(col_name)
        col_name = self._generate_name("EVT", "IsMissingEPS", "RAW")
        self.df[col_name] = self.df['eps'].isna().astype(int)
        self.new_cols.append(col_name)
        col_name = self._generate_name("EVT", "LogDaysSincePub", "RAW")
        self.df[col_name] = np.log1p((self.df['date'] - self.df['published_date']).dt.days).fillna(0)
        self.new_cols.append(col_name)
        return self
    
    def apply_consensus_block(self):
        LAG_YEAR = 240 
        epsilon = 1e-6
        grouped_scode = self.df.groupby('scode')
        v_t = self.df['eps']
        v_prev = grouped_scode['eps'].shift(20) # 1ヶ月前
        col_name = self._generate_name("CON", "RevisionRate", "RAW")
        self.df[col_name] = (v_t - v_prev) / (0.5 * (v_t.abs() + v_prev.abs()) + epsilon)
        self.new_cols.append(col_name)
        col_name = self._generate_name("CON", "ProgressRate", "RAW")
        self.df[col_name] = self.df['operating_profit'] / (self.df['operating_profit_forecast'].abs() + epsilon)
        self.new_cols.append(col_name)
        return self
    
    def apply_governance_block(self):
        """マーケット指標の一括作成"""
        def map_market_segment(market_name):
            """
            市場名称を3つの主要セグメントに統合する関数
            Args:
                market_name (str): J-Quants等の市場名称（例: "東証1部", "プライム", "マザーズ"）
            Returns:
                int: グループID
                    1: Prime_Class   (Large/Liquid: プライム, 1部)
                    2: Standard_Class(Mid/Stable: スタンダード, 2部, JQ)
                    3: Growth_Class  (Emerging/Volatile: グロース, マザーズ)
                    0: Others        (その他)
            """
            # 欠損値や非文字列は 0 (Others) とする
            if pd.isna(market_name) or not isinstance(market_name, str):
                return 0
            # 正規化: 前後の空白削除、全角スペース除去、半角スペース除去
            # これにより "東証 1部" や "J Q" といった表記ゆれを吸収
            m = market_name.strip().replace("　", "").replace(" ", "")
            # --- 1. Prime / Large Class (ID: 1) ---
            # プライム、東証一部などを統合
            if m in ['プライム', '東証PR', '東証1部', '東証一部']:
                return 1
            # --- 3. Growth / Emerging Class (ID: 3) ---
            # グロース、マザーズ、JQグロースなどを統合
            # ※Standard判定より先に記述することで、"JQグロース"が"JQ"として判定されるのを防ぐ
            growth_keywords = ['グロース', '東証GR', 'マザーズ', '東証マザ', 'JQG', 'JQグロース', 'HCグロース']
            if any(keyword in m for keyword in growth_keywords):
                return 3
            # --- 2. Standard / Mid Class (ID: 2) ---
            # スタンダード、二部、JASDAQ(Standard)などを統合
            # "東証" 単体の場合も、性質的にここが最も近い
            standard_keywords = ['スタンダード', '東証ST', '東証2部', '東証二部', 'JQ', 'JQS', 'JQスタンダード', '東証']
            if any(keyword in m for keyword in standard_keywords):
                return 2
            # --- 0. Others (ID: 0) ---
            # TOKYO PRO MARKET や その他
            return 0
        col_name = self._generate_name("GOV", "MarketSegment", "RAW")
        self.df[col_name] = self.df['market'].apply(map_market_segment)
        self.new_cols.append(col_name)
        col_name = self._generate_name("GOV", "Sector33Code", "RAW")
        self.df[col_name] = self.df['sector33_code']
        self.new_cols.append(col_name)
        return self
    
    def apply_tempfeat(self):
        grouped = self.df.groupby('scode')
        log_return = np.log(self.df['close'] / grouped['close'].shift(1))
        self.df['Vol_20d'] = log_return.groupby(self.df['scode']).rolling(20).std().reset_index(level=0, drop=True)
        self.df['volume_p_MA5'] = grouped['volume_p'].rolling(5).mean().reset_index(level=0, drop=True)
        self.df['log_market_cap'] = np.log(self.df['close'] * self.df['shares_outstanding'])
        
        def _fwd_sum(s, w):
            return s.iloc[::-1].rolling(w, min_periods=1).sum().iloc[::-1].shift(-1)
            
        self.df['Market_Return_Future'] = grouped['Market_Return'].apply(lambda x: _fwd_sum(x, self.horizon_tac)).reset_index(level=0, drop=True)
        self.df['Sector_Return_Future'] = grouped['sector_return'].apply(lambda x: _fwd_sum(x, self.horizon_tac)).reset_index(level=0, drop=True)
        return self
    
    # --- ターゲット作成 ---
    def apply_timeseries_targets(self):
        grouped = self.df.groupby('scode')
        self.df = self.df.sort_values(["scode", "date"]).reset_index(drop=True)
        grouped = self.df.groupby('scode', sort=False)
        def _fwd_max(s, w):
            return s.iloc[::-1].rolling(w, min_periods=1).max().iloc[::-1].shift(-1)
        def _fwd_min(s, w):
            return s.iloc[::-1].rolling(w, min_periods=1).min().iloc[::-1].shift(-1)
        def _fwd_sum(s, w):
            return s.iloc[::-1].rolling(w, min_periods=1).sum().iloc[::-1].shift(-1)
        entry_price = grouped['open'].shift(-1)
        future_high_tac = grouped['high'].apply(lambda x: _fwd_max(x, self.horizon_tac)).reset_index(level=0, drop=True)
        future_low_tac = grouped['low'].apply(lambda x: _fwd_min(x, self.horizon_tac)).reset_index(level=0, drop=True)
        future_close_tac = grouped['close'].shift(-self.horizon_tac) 
        # 基本情報格納
        self.df['Entry_Price'] = entry_price
        self.df['Future_High_Tac'] = future_high_tac
        self.df['Future_Low_Tac'] = future_low_tac
        self.df['Future_Close_Tac'] = future_close_tac
        self.df['target_ret_5'] = (future_close_tac / entry_price.replace(0, np.nan)) - 1.0
        # --- プロダクション仕様: TAC 攻めターゲット (Volatility-Scaled Asymmetric Return 対数残差版) ---
        log_market_ret = self.df['Market_Return'].fillna(0)
        market_ret_future_log = log_market_ret.groupby(self.df['scode']).apply(lambda x: _fwd_sum(x, self.horizon_tac)).reset_index(level=0, drop=True)
        log_ret_5 = np.log(future_close_tac / entry_price.replace(0, np.nan))
        residual_ret_log = log_ret_5 - (self.df['BET_Beta60_RAW'] * market_ret_future_log)
        hv_floor = self.df.groupby('date')['Vol_20d'].transform(lambda x: x.quantile(0.10))
        vol_scaled_denom = np.maximum(self.df['Vol_20d'], hv_floor) * np.sqrt(self.horizon_tac)
        clip_lower, clip_upper = -1.5, 2.0 # ※本来は学習データの1%~99.5%点等を動的に計算して適用
        self.df['target_tac_vol_scaled_asym_return'] = residual_ret_log / (vol_scaled_denom + 1e-6)
        self.df['target_tac_vol_scaled_asym_return_clipped'] = np.clip(residual_ret_log / (vol_scaled_denom + 1e-6), clip_lower, clip_upper)
        # --- プロダクション仕様: TAC 守りターゲット (Max Negative Path Exposure 対数版) ---
        self.df['target_tac_max_neg_path'] = np.log(future_low_tac / entry_price.replace(0, np.nan))
        self.df['target_tac_risk'] = np.log(future_low_tac / entry_price.replace(0, np.nan)) + np.log(future_high_tac / entry_price.replace(0, np.nan)) 
        # --- プロダクション仕様: Metaモデル ターゲット (Survival Return Raw 動的ペナルティ版) ---
        R_i = log_ret_5
        dynamic_threshold = -1.5 * self.df['Vol_20d'] # 銘柄ごとのHVに連動したペナルティ閾値
        meta_y = np.where(R_i >= 0, np.minimum(R_i, 0.15),
                 np.where(R_i >= dynamic_threshold, R_i * 2.0,
                 np.where(pd.notna(R_i), -0.5, np.nan)))
        self.df['target_meta_survival_return_raw'] = meta_y
        # --- ターゲット作成：戦略モデル ---
        future_high_str = grouped['high'].apply(lambda x: _fwd_max(x, self.horizon_str)).reset_index(level=0, drop=True)
        future_low_str = grouped['low'].apply(lambda x: _fwd_min(x, self.horizon_str)).reset_index(level=0, drop=True)
        future_close_str = grouped['close'].shift(-self.horizon_str)
        self.df['Future_High_Str'] = future_high_str
        self.df['Future_Low_Str'] = future_low_str
        self.df['Future_Close_Str'] = future_close_str
        self.df['target_ret_60'] = (self.df['close'].shift(-self.horizon_str) / entry_price.replace(0, np.nan)) - 1.0
        # --- プロダクション仕様: STR 攻めターゲット (Sharpe Adjusted 60d 対数残差版) ---
        market_ret_60_log = log_market_ret.groupby(self.df['scode']).apply(
            lambda x: _fwd_sum(x, self.horizon_str)
        ).reset_index(level=0, drop=True)
        log_ret_60 = np.log(future_close_str / entry_price.replace(0, np.nan))
        residual_log_60 = log_ret_60 - (self.df['BET_Beta60_RAW'] * market_ret_60_log)
        if 'VOL_Volatility60_RAW' in self.df.columns:
            hv_60 = self.df['VOL_Volatility60_RAW']
            hv_floor_60 = self.df.groupby('date')['VOL_Volatility60_RAW'].transform(lambda x: x.quantile(0.10))
        else:
            hv_60 = self.df['Vol_20d'] * np.sqrt(60 / 20)
            hv_floor_60 = self.df.groupby('date')['Vol_20d'].transform(lambda x: x.quantile(0.10)) 
        vol_scaled_denom_daily = np.maximum(hv_60, hv_floor_60)
        vol_scaled_denom_60 = vol_scaled_denom_daily * np.sqrt(self.horizon_str)
        raw_target_str = residual_log_60 / (vol_scaled_denom_60 + 1e-6)
        clip_lower, clip_upper = -3.0, 3.0
        self.df['target_str_sharpe_adj'] = np.clip(raw_target_str, clip_lower, clip_upper)
        # Triple Barrier Method 
        # 3値分類ラベル: 1(利確), -1(損切), 0(時間切れ)
        # バリア幅の設定: ボラティリティベース (De Prado流)
        # 上値(PT) = 期間ボラティリティ * 1.0
        # 下値(SL) = 期間ボラティリティ * 1.0 (損益比率1:1の設定)
        vol_horizon = self.df['Vol_20d'] * np.sqrt(self.horizon_str)
        pt_width = vol_horizon * 1.0
        sl_width = vol_horizon * 1.0
        labels = np.full(len(self.df), np.nan)
        mdd_labels = np.full(len(self.df), np.nan)
        for _, idx in grouped.indices.items():
            idx = np.asarray(idx)
            highs = self.df.loc[idx, 'high'].to_numpy(dtype=float)
            lows = self.df.loc[idx, 'low'].to_numpy(dtype=float)
            entries = entry_price.loc[idx].to_numpy(dtype=float)
            pts = pt_width.loc[idx].to_numpy(dtype=float)
            sls = sl_width.loc[idx].to_numpy(dtype=float)
            n = len(idx)
            for pos in range(n):
                if pos + 1 >= n or np.isnan(entries[pos]) or np.isnan(pts[pos]) or np.isnan(sls[pos]):
                    continue
                end = min(pos + 1 + self.horizon_str, n)
                window_high = highs[pos + 1:end]
                window_low = lows[pos + 1:end]
                if len(window_low) == 0:
                    continue
                entry = entries[pos]
                upper_barrier = entry * np.exp(pts[pos])
                lower_barrier = entry * np.exp(-sls[pos])
                wl_log = np.log(np.maximum(window_low, 1e-9))
                mdd_labels[idx[pos]] = np.max(np.log(entry) - wl_log)
                hit_upper = np.where(window_high >= upper_barrier)[0]
                hit_lower = np.where(window_low <= lower_barrier)[0]
                first_upper = hit_upper[0] if len(hit_upper) > 0 else np.inf
                first_lower = hit_lower[0] if len(hit_lower) > 0 else np.inf
                if np.isinf(first_upper) and np.isinf(first_lower):
                    labels[idx[pos]] = 0.0
                elif first_upper < first_lower:
                    labels[idx[pos]] = 1.0
                else:
                    labels[idx[pos]] = -1.0        
        self.df['target_str_triple_barrier'] = labels
        self.df['target_str_mdd'] = mdd_labels
        # --- プロダクション仕様: STR 守りターゲット (Volatility-Scaled MDD) ---
        # MDDを20日HVで正規化し、レジーム間のボラティリティの差異を吸収する。
        # ゼロ除算および過剰な感度を防ぐため、日次のクロスセクション10%タイルでフロアリングを実施。
        hv_floor_20 = self.df.groupby('date')['Vol_20d'].transform(lambda x: x.quantile(0.10))
        vol_scaled_denom_mdd = np.maximum(self.df['Vol_20d'], hv_floor_20)
        self.df['target_str_vol_scaled_mdd'] = self.df['target_str_mdd'] / (vol_scaled_denom_mdd + 1e-6)

        return self

    def apply_crosssectional_targets(self):
        """クロスセクションターゲットの追加"""
        new_cols = {}
        # フラグによるフィルタリング（事前スキャン時やフラグ未計算時のフォールバック付）
        tac_mask = self.df['is_candidate_tac'] == True if 'is_candidate_tac' in self.df.columns else pd.Series(True, index=self.df.index)
        str_mask = self.df['is_candidate_str'] == True if 'is_candidate_str' in self.df.columns else pd.Series(True, index=self.df.index)

        # --- 1. Era-wise Rank (Category A) ---
        # 単純なRank (0.0 ~ 1.0)
        tac_rank = self.df.loc[tac_mask].groupby('date')['target_ret_5'].rank(pct=True, method='average')
        new_cols['target_tac_rank'] = tac_rank
        # 既存: Gauss Rank (正規分布化)
        epsilon = 1e-6
        rank_clipped = tac_rank * (1 - 2 * epsilon) + epsilon
        new_cols['target_tac_gauss_rank'] = (erfinv(2 * rank_clipped - 1)).clip(-3.0, 3.0)

        # --- 2. Linear Residual (Category C) ---
        indexer_sec = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon_tac)
        sec_ret_fut = self.df['sector_return'].shift(-1).rolling(window=indexer_sec).sum()
        mkt_ret_fut = self.df['Market_Return'].shift(-1).rolling(window=indexer_sec).sum()
        
        # 候補レコードのみに対して残差と相対フラグを計算
        new_cols['target_tac_linear_residual'] = self.df.loc[tac_mask, 'target_ret_5'] - (0.5 * mkt_ret_fut.loc[tac_mask] + 0.5 * sec_ret_fut.loc[tac_mask])
        new_cols['target_tac_sector_relative'] = (self.df.loc[tac_mask, 'target_ret_5'] > sec_ret_fut.loc[tac_mask]).astype(float)

        # --- 戦略モデル用クロスセクション ---
        # Relative Rank Change (60d)
        str_rank = self.df.loc[str_mask].groupby('date')['target_ret_60'].rank(pct=True, method='average')
        new_cols['target_str_rank'] = str_rank
        str_rank_clipped = str_rank * (1 - 2 * epsilon) + epsilon
        new_cols['target_str_gauss_rank'] = (erfinv(2 * str_rank_clipped - 1)).clip(-3.0, 3.0)
        
        # Peer Group Neutralized Alpha (60d)
        sector_mean_60 = self.df.loc[str_mask].groupby(['date', 'sector33_code'])['target_ret_60'].transform('mean')
        new_cols['target_str_peer_alpha'] = self.df.loc[str_mask, 'target_ret_60'] - sector_mean_60

        # 一括結合（Indexアライメントにより非候補レコードは自動的にNaNになる）
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self



    def get_df(self):
        return self.df 