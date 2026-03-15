import pandas as pd
import numpy as np
import pandas_ta as ta
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
        # self.initial_cols = [
        #     'scode', 'sector33_code', 'date', 'volume_p', 'open', 'high', 'low', 'close', 'volume', 'shares_outstanding',
        #     # -- maket系
        #     'Market_Return', 'Market_Trend_Idx', 'Market_HV_20', 'market_vol_change',
        #     'Market_Return_GR_126', 'Market_Trend_Idx_GR_126', 'Market_HV_20_GR_126', 'market_vol_change_GR_126',
        #     'Market_Return_GR_252', 'Market_Trend_Idx_GR_252', 'Market_HV_20_GR_252', 'market_vol_change_GR_252',
        #     'Market_Foreign_GR_63', 'Market_Individual_GR_63', 'Market_Foreign_GR_252', 'Market_Individual_GR_252',
        #     'Market_Foreign_Diff', 'overseas_flow_trend', 'flow_accel', 
        #     # -- セクター別空売り比率
        #     'selling_volume_ratio',
        #     'selling_volume_ratio_GR_126', 'selling_volume_ratio_GR_252',
        # ]
        # # 辞書のキーとして格納（Python 3.7+ では挿入順が保持されます）
        # self._feature_registry = dict()
        # self.meta_cols = [
        #     'scode', 'date', 'close',
        #     # 検証用
        #     'Entry_Price','Future_High_Tac','Future_Low_Tac','Future_Close_Tac',
        #     'Future_High_Str','Future_Low_Str','Future_Close_Str'
        # ]
        # self.target_cols = [
        #     # --- 戦術モデル用推奨ターゲット (5日先) ---
        #     # 1. Ranking系
        #     'target_tac_rank',          # Era-wise Rank (0~1)
        #     'target_tac_gauss_rank',    # Gauss Rank
        #     # 2. Risk調整系
        #     'target_tac_vol_scaled_residual', # Beta調整後 & Vol調整後
        #     # 3. 実執行・Alpha系
        #     'target_tac_smoothed_return',     # VWAP基準
        #     'target_tac_linear_residual',     # 線形モデル残差
        #     # 4. Triple Barrier (Dynamic)
        #     'target_tac_tb_strategy_a',       # A: Balance (1.0σ / 1.0σ)
        #     'target_tac_tb_strategy_b',       # B: Trend (1.5σ / 0.75σ)
        #     'target_tac_tb_strategy_c',       # C: Reversion (0.5σ / 1.0σ)
        #     # 戦略モデル用ターゲット
        #     'target_str_risk_adj','target_str_consistency','target_str_vol_scale','target_str_triple_barrier',
        #     'target_str_rank','target_str_peer_alpha',
        #     # 戦略モデル用ターゲット、別スクリプトで生成
        #     # 'target_reg', 'target_cls',
        #     # 比較用
        #     'target_ret_5'#, 'target_ret_60'
        # ]
    
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
        # メモリ節約のため、辞書に貯めずに直接dfに代入する方式に変更
        print(f"Applying Cross-Sectional Transformations to {len(self.df.columns)} columns...")
        columns = list(self.df.columns)
        for col in tqdm(columns):
            if col.startswith("MOM_") and col.endswith("_RAW"):
                self.cs_rank("MOM", col, store=None)
                self.cs_zscore("MOM", col, store=None)
                self.sn_zscore("MOM", col, store=None)
            elif col.startswith("VOL_") and col.endswith("_RAW"):
                self.cs_rank("VOL", col, store=None)
                self.cs_zscore("VOL", col, store=None)
                self.sn_zscore("VOL", col, store=None)
            elif col.startswith("LIQ_") and col.endswith("_RAW"):
                self.cs_rank("LIQ", col, store=None)
                self.cs_zscore("LIQ", col, store=None)
            elif col.startswith("VAL_") and col.endswith("_RAW"):
                self.cs_rank("VAL", col, store=None)
                self.cs_zscore("VAL", col, store=None)
                self.sn_zscore("VAL", col, store=None)
            elif col.startswith("QLT_") and col.endswith("_RAW"):
                self.cs_rank("QLT", col, store=None)
                self.cs_zscore("QLT", col, store=None)
                self.sn_zscore("QLT", col, store=None)
            elif col.startswith("SIZ_") and col.endswith("_RAW"):
                self.cs_rank("SIZ", col, store=None)
                self.cs_zscore("SIZ", col, store=None)
            elif col.startswith("SPD_") and col.endswith("_RAW"):
                self.cs_rank("SPD", col, store=None)
                self.cs_zscore("SPD", col, store=None)
            elif col.startswith("BET_") and col.endswith("_RAW"):
                self.cs_rank("BET", col, store=None)
                self.cs_zscore("BET", col, store=None)
                self.sn_zscore("BET", col, store=None)
            elif col.startswith("EVT_") and col.endswith("_RAW"):
                self.cs_rank("EVT", col, store=None)
                self.cs_zscore("EVT", col, store=None)
                self.sn_zscore("EVT", col, store=None)
            elif col.startswith("CON_") and col.endswith("_RAW"):
                self.cs_rank("CON", col, store=None)
                self.cs_zscore("CON", col, store=None)
                self.sn_zscore("CON", col, store=None)
            elif col.startswith("GOV_") and col.endswith("_RAW"):
                if "Sector33Code" in col:
                    continue
                self.cs_rank("GOV", col, store=None)
                self.cs_zscore("GOV", col, store=None)
                self.sn_zscore("GOV", col, store=None)
            # ループごとにGCを実行してメモリピークを抑える
            gc.collect()
        return self

    
    # --- RAW特徴量作成 ---
    def apply_momentum_block(self):
        col_name = self._generate_name("MOM", "ADX14", "RAW")
        adx = ta.adx(self.df['high'], self.df['low'], self.df['close'], length=14)
        self.df[col_name] = adx.iloc[:, 0].values if adx is not None else np.nan
        self.new_cols.append(col_name)
        for w in [5, 25, 75]:
            col_name = self._generate_name("MOM", f"DistSMA{w}", "RAW")
            sma = ta.sma(self.df['close'], length=w)
            self.df[col_name] = (self.df['close'].values / sma.values) - 1 if sma is not None else np.nan
            self.new_cols.append(col_name)
        sma5 = ta.sma(self.df['close'], length=5)
        sma25 = ta.sma(self.df['close'], length=25)
        sma75 = ta.sma(self.df['close'], length=75)
        col_name = self._generate_name("MOM", "DistSMA5-25", "RAW")
        self.df[col_name] = (sma5.values / sma25.values) - 1 if (sma5 is not None and sma25 is not None) else np.nan
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "DistSMA25-75", "RAW")
        self.df[col_name] = (sma25.values / sma75.values) - 1 if (sma25 is not None and sma75 is not None) else np.nan
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "EfficiencyRatio10", "RAW")
        er = ta.er(self.df['close'], length=10)
        self.df[col_name] = er.values if er is not None else np.nan
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "LinRegSlope10", "RAW")
        slope = ta.slope(self.df['close'], length=10)
        self.df[col_name] = slope.values if slope is not None else np.nan
        self.new_cols.append(col_name)
        macd = ta.macd(self.df['close'], fast=12, slow=26, signal=9)
        col_name = self._generate_name("MOM", "MACDHistNorm", "RAW")
        if macd is not None:
            macd_hist = macd['MACDh_12_26_9'].values / self.df['close'].values
        else:
            macd_hist = np.full(len(self.df), np.nan)
        self.df[col_name] = macd_hist
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "MACDHistDiff", "RAW")
        self.df[col_name] = pd.Series(macd_hist).diff(1).values
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "DistHigh60", "RAW")
        self.df[col_name] = self.df['close'] / self.df['high'].rolling(60).max()
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "DistHigh250", "RAW")
        self.df[col_name] = self.df['close'] / self.df['high'].rolling(250).max()
        self.new_cols.append(col_name)
        high_26 = self.df['high'].rolling(26).max()
        low_26 = self.df['low'].rolling(26).min()
        kijun_sen = (high_26 + low_26) / 2
        col_name = self._generate_name("MOM", "DistKijun", "RAW")
        self.df[col_name] = (self.df['close'] - kijun_sen) / kijun_sen
        self.new_cols.append(col_name)
        past_3d_high = self.df['high'].shift(1).rolling(3).max()
        col_name = self._generate_name("MOM", "NewHighFlag3", "RAW")
        self.df[col_name] = (self.df['close'] > past_3d_high).astype(int)
        self.new_cols.append(col_name)
        roll_120 = self.df['close'].rolling(120)
        max_120 = roll_120.max()
        min_120 = roll_120.min()
        col_name = self._generate_name("MOM", "PricePos120", "RAW")
        self.df[col_name] = (self.df['close'] - min_120) / (max_120 - min_120)
        self.new_cols.append(col_name)
        high_9 = self.df['high'].rolling(9).max()
        low_9 = self.df['low'].rolling(9).min()
        tenkan_sen = (high_9 + low_9) / 2
        high_26 = self.df['high'].rolling(26).max()
        low_26 = self.df['low'].rolling(26).min()
        kijun_sen = (high_26 + low_26) / 2
        high_52 = self.df['high'].rolling(52).max()
        low_52 = self.df['low'].rolling(52).min()
        span_b = (high_52 + low_52) / 2
        span_b_curr = span_b.shift(26)
        col_name = self._generate_name("MOM", "IchimokuDist", "RAW")
        self.df[col_name] = (self.df['close'] - span_b_curr) / span_b_curr
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "LogReturn", "RAW")
        self.df[col_name] = np.log(self.df['close'] / self.df['close'].shift(1))
        self.new_cols.append(col_name)
        for w in [3,5,10,20]:
            col_name = self._generate_name("MOM", f"Return{w}d", "RAW")
            self.df[col_name] = self.df['close'].pct_change(w)
            self.new_cols.append(col_name)
        log_return = np.log(self.df['close'] / self.df['close'].shift(1))
        for w in [1,2]:
            col_name = self._generate_name("MOM", f"Return1dLag{w}", "RAW")
            self.df[col_name] = log_return.shift(w)
            self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "RSI9", "RAW")
        rsi9 = ta.rsi(self.df['close'], length=9)
        self.df[col_name] = rsi9.values if rsi9 is not None else np.nan
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "RSI9-14Diff", "RAW")
        rsi14 = ta.rsi(self.df['close'], length=14)
        self.df[col_name] = (rsi9.values - rsi14.values) if (rsi9 is not None and rsi14 is not None) else np.nan
        self.new_cols.append(col_name)
        bb = ta.bbands(self.df['close'], length=20, std=2)
        col_name = self._generate_name("MOM", "BBPercentB", "RAW")
        if bb is not None:
            bb_p_col = [c for c in bb.columns if c.startswith('BBP')][0]
            self.df[col_name] = bb[bb_p_col].values
        else:
            self.df[col_name] = np.nan
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
            self.df[col_name] = lower_shadow_ratio.rolling(5).mean()
        diff = self.df['close'].diff()
        sign = np.sign(diff).fillna(0)
        is_change = sign != sign.shift(1)
        group_id = is_change.cumsum()
        count = self.df.groupby(group_id).cumcount() + 1
        col_name = self._generate_name("MOM", "Streak", "RAW")
        self.df[col_name] = np.where(sign > 0, count, np.where(sign < 0, -count, 0))
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "BullishRatio20", "RAW")
        self.df[col_name] = (sign > 0).rolling(20).mean()
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "ClosePosition", "RAW")
        self.df[col_name] = (self.df['close'] - self.df['low'].rolling(20).min()) / (self.df['high'].rolling(20).max() - self.df['low'].rolling(20).min())
        self.new_cols.append(col_name)
        prev_close = self.df['close'].shift(1)
        col_name = self._generate_name("MOM", "GapRate", "RAW")
        self.df[col_name] = (self.df['open'] / prev_close) - 1.0
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "MaxGain5", "RAW")
        self.df[col_name] = (self.df['high'].rolling(5).max() / self.df['close']) - 1.0
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "RCI9", "RAW")
        rci_9 = self._calc_rci(self.df['close'], 9)
        self.df[col_name] = rci_9
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "RCI9Diff", "RAW")
        self.df[col_name] = rci_9.diff(1)
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "RCI26", "RAW")
        rci_26 = self._calc_rci(self.df['close'], 26)
        self.df[col_name] = rci_26
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "RCI26Diff", "RAW")
        self.df[col_name] = rci_26.diff(1)
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "RCI52", "RAW")
        self.df[col_name] = self._calc_rci(self.df['close'], 52)
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "Momentum12-1", "RAW")
        self.df[col_name] = self.df['close'].shift(20) / self.df['close'].shift(260) - 1
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "RetIntraday", "RAW")
        self.df[col_name] = (self.df['close'] / self.df['open']) - 1.0
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "WinRate10d", "RAW")
        self.df[col_name] = (log_return > 0).rolling(10).mean()
        self.new_cols.append(col_name)
        for window in [25, 75, 200]:
            ma = self.df['close'].rolling(window=window).mean()
            col_name = self._generate_name("MOM", f"MADev{window}", "RAW")
            self.df[col_name] = (self.df['close'] / ma) - 1
            self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "Return20d", "RAW")
        self.df[col_name] = self.df['close'].pct_change(20)
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "Return6m", "RAW")
        self.df[col_name] = self.df['close'].pct_change(120)
        self.new_cols.append(col_name)
        col_name = self._generate_name("MOM", "Return12m", "RAW")
        self.df[col_name] = self.df['close'].pct_change(240)
        self.new_cols.append(col_name)
        max_52w = self.df['close'].rolling(window=240).max()
        col_name = self._generate_name("MOM", "High52wDist", "RAW")
        self.df[col_name] = (self.df['close'] / max_52w) - 1
        self.new_cols.append(col_name)
        return self

    def apply_volatility_block(self):
        log_return = np.log(self.df['close'] / self.df['close'].shift(1))
        return_1d = self.df['close'].pct_change(1)
        col_name = self._generate_name("VOL", "MAE5", "RAW")
        self.df[col_name] = (self.df['low'].rolling(5).min() / self.df['close']) - 1.0
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "MAE10", "RAW")
        self.df[col_name] = (self.df['low'].rolling(10).min() / self.df['close']) - 1.0
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "DownsideRun", "RAW")
        self.df[col_name] = log_return.clip(lower=0).rolling(5).sum()
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "ReturnSkewnessDiff", "RAW")
        self.df[col_name] = log_return.rolling(20).skew().diff()
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "ReturnKurtosis", "RAW")
        self.df[col_name] = log_return.rolling(20).kurt()
        self.new_cols.append(col_name)
        bb = ta.bbands(self.df['close'], length=20, std=2)
        if bb is not None:
            # pandas_taのBandwidthは通常'BBB'プレフィックス
            bb_w_cols = [c for c in bb.columns if c.startswith('BBB')]
            if bb_w_cols:
                col_name = self._generate_name("VOL", "BBWidth", "RAW")
                self.df[col_name] = bb[bb_w_cols[0]].values
                self.new_cols.append(col_name)
        gap_rate = (self.df['open'] / self.df['close'].shift(1)) - 1.0
        col_name = self._generate_name("VOL", "GapAbs", "RAW")
        self.df[col_name] = gap_rate.abs()
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "LargeMoveCount", "RAW")
        self.df[col_name] = (return_1d.abs() > 0.03).rolling(20).sum()
        self.new_cols.append(col_name)
        range_len = self.df['high'] - self.df['low']
        col_name = self._generate_name("VOL", "RangeRatioLong", "RAW")
        self.df[col_name] = range_len.rolling(5).mean() / range_len.rolling(20).mean()
        self.new_cols.append(col_name)
        atr = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=14)
        col_name = self._generate_name("VOL", "ATRRatio", "RAW")
        self.df[col_name] = (atr.values / self.df['close'].values) if atr is not None else np.nan
        self.new_cols.append(col_name)
        atr_mid = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=20)
        atr_short = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=5)
        col_name = self._generate_name("VOL", "ATRSqueeze", "RAW")
        self.df[col_name] = (atr_short.values / atr_mid.values) if (atr_short is not None and atr_mid is not None) else np.nan
        self.new_cols.append(col_name)
        bb = ta.bbands(self.df['close'], length=20, std=2)
        if bb is not None:
            col_name = self._generate_name("VOL", "ATRSqueezeBB", "RAW")
            self.df[col_name] = (bb['BBU_20_2.0'].values - bb['BBL_20_2.0'].values) / (bb['BBM_20_2.0'].values + 1e-9)
            self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "HistVol20", "RAW")
        self.df[col_name] = log_return.rolling(20).std()
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "HV30", "RAW")
        hv30 = log_return.rolling(30).std() * np.sqrt(250)
        self.df[col_name] = hv30
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "HVSlope", "RAW")
        self.df[col_name] = hv30.diff(5)
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "RealizedSkew20", "RAW")
        self.df[col_name] = log_return.rolling(20).skew()
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "UlcerIndex14", "RAW")
        ui = ta.ui(self.df['close'], length=14)
        self.df[col_name] = ui.values if ui is not None else np.nan
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "ATRChgRate", "RAW")
        self.df[col_name] = atr.pct_change().values if atr is not None else np.nan
        self.new_cols.append(col_name)
        with np.errstate(divide='ignore', invalid='ignore'):
            hl_log_sq = np.log(self.df['high'] / self.df['low']) ** 2
            const_factor = 4 * np.log(2)
            col_name = self._generate_name("VOL", "VolatilityParkinson", "RAW")
            self.df[col_name] = np.sqrt(hl_log_sq.rolling(14).mean() / const_factor)
            self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "ReturnVolatility", "RAW")
        self.df[col_name] = log_return.rolling(10).std()
        self.new_cols.append(col_name)
        hv5 = log_return.rolling(5).std() * np.sqrt(250)
        col_name = self._generate_name("VOL", "VolRatioHV", "RAW")
        self.df[col_name] = hv5 / hv30
        self.new_cols.append(col_name)
        self.new_cols.append(col_name)
        def calc_downside_std(x, window=60):
            neg_ret = x.where(x < 0, 0)
            return neg_ret.rolling(window).std()
        col_name = self._generate_name("VOL", "DownsideDev60", "RAW")
        self.df[col_name] = log_return.transform(lambda x: calc_downside_std(x))
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "Volatility60", "RAW")
        self.df[col_name] = self.df['close'].pct_change().rolling(60).std()
        self.new_cols.append(col_name)
        return self
        
    def apply_liquidity_block(self):
        volume_log = np.log(self.df['volume'] + 1)
        log_return = np.log(self.df['close'] / self.df['close'].shift(1))
        col_name = self._generate_name("LIQ", "VolumeLog", "RAW")
        self.df[col_name] = volume_log
        self.new_cols.append(col_name)
        col_name = self._generate_name("LIQ", "AbnormalVolume", "RAW")
        self.df[col_name] = self.df['volume_p'] / self.df['volume_p'].rolling(20).mean()
        self.new_cols.append(col_name)
        col_name = self._generate_name("LIQ", "VolumeChange", "RAW")
        self.df[col_name] = self.df['volume'].pct_change()
        self.new_cols.append(col_name)
        col_name = self._generate_name("LIQ", "VolumeSlope5", "RAW")
        slope = ta.slope(np.log(self.df['volume'] + 1), length=5)
        self.df[col_name] = slope.values if slope is not None else np.nan
        self.new_cols.append(col_name)
        col_name = self._generate_name("LIQ", "LogTradingCap", "RAW")
        self.df[col_name] = np.log(self.df['close'] * self.df['volume'] + 1)
        self.new_cols.append(col_name)
        col_name = self._generate_name("LIQ", "MFI14", "RAW")
        mfi = ta.mfi(self.df['high'], self.df['low'], self.df['close'], self.df['volume'], length=14)
        self.df[col_name] = mfi.values if mfi is not None else np.nan
        self.new_cols.append(col_name)
        vol_ma5 = self.df['volume'].rolling(5).mean()
        col_name = self._generate_name("LIQ", "VolRatio5d", "RAW")
        self.df[col_name] = self.df['volume'] / vol_ma5.replace(0, np.nan)
        self.new_cols.append(col_name)
        vol_median_20 = self.df['volume'].rolling(20).median()
        is_spike = (self.df['volume'] > (vol_median_20 * 3)).astype(int)
        col_name = self._generate_name("LIQ", "VolSpikeCount20", "RAW")
        self.df[col_name] = is_spike.rolling(20).sum()
        self.new_cols.append(col_name)
        is_up = (self.df['close'] > self.df['open']).astype(int)
        vol_up = (self.df['volume'] * is_up).rolling(20).sum()
        vol_down = (self.df['volume'] * (1 - is_up)).rolling(20).sum()
        col_name = self._generate_name("LIQ", "VolUpDownRatio", "RAW")
        self.df[col_name] = vol_up / (vol_down + 1e-9)
        self.new_cols.append(col_name)
        col_name = self._generate_name("VOL", "VolumeMA25", "RAW")
        self.df[col_name] = volume_log.rolling(25).mean()
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
        temp_df = self.df[fund_cols + ['scode']].copy()
        temp_df[fund_cols] = temp_df.groupby('scode')[fund_cols].ffill()
        grouped = temp_df.groupby('scode')
        for col in fund_cols:
            v_t = temp_df[col]
            v_prev = grouped[col].shift(LAG_YEAR)
            self.df[f'{col}_growth_yoy'] = (v_t - v_prev) / (0.5 * (v_t.abs() + v_prev.abs()) + epsilon)
            self.df[f'{col}_growth_yoy'] = self.df[f'{col}_growth_yoy'].clip(-3.0, 3.0)
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
        typ_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        pv_sum = (typ_price * self.df['volume']).rolling(5).sum()
        v_sum = self.df['volume'].rolling(5).sum()
        rolling_vwap = pv_sum / v_sum
        col_name = self._generate_name("SPD", "DistVWAP5", "RAW")
        dist_vwap_5 = (self.df['close'] - rolling_vwap) / rolling_vwap
        self.df[col_name] = dist_vwap_5
        self.new_cols.append(col_name)
        col_name = self._generate_name("SPD", "DistVWAPSlope", "RAW")
        self.df[col_name] = dist_vwap_5.diff()
        self.new_cols.append(col_name)
        vwap = self.df['volume_p'] / self.df['volume'].replace(0, 1)
        col_name = self._generate_name("SPD", "VWAPDev", "RAW")
        self.df[col_name] = (self.df['close'] / vwap) - 1
        self.new_cols.append(col_name)
        avg_vol_60 = self.df['volume'].rolling(60).mean()
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
        self.df[col_name] = margin_ratio.diff(20)
        self.new_cols.append(col_name)
        col_name = self._generate_name("SPD", "MarketForeignBuy", "RAW")
        self.df[col_name] = self.df['Foreign_Net_Buy']
        self.new_cols.append(col_name)
        col_name = self._generate_name("SPD", "MarketIndividualBuy", "RAW")
        self.df[col_name] = self.df['Individual_Net_Buy']
        self.new_cols.append(col_name)
        col_name = self._generate_name("SPD", "OverseaFlowTrend", "RAW")
        flow = self.df['Foreign_Net_Buy'].rolling(20).mean()
        self.df[col_name] = flow
        self.new_cols.append(col_name)
        col_name = self._generate_name("SPD", "FlowAccel", "RAW")
        self.df[col_name] = flow - flow.rolling(5).mean()
        self.new_cols.append(col_name)
        return self
    
    def apply_beta_block(self):
        ret_stock = self.df['close'].pct_change()
        ret_market = self.df['Market_Return']
        rolling_cov = ret_stock.rolling(60).cov(ret_market)
        rolling_var = ret_market.rolling(60).var()
        log_return = np.log(self.df['close'] / self.df['close'].shift(1))
        col_name = self._generate_name("BET", "Beta60", "RAW")
        self.df[col_name] = rolling_cov / rolling_var
        self.new_cols.append(col_name)
        col_name = self._generate_name("BET", "RS25", "RAW")
        self.df[col_name] = self.df['close'].pct_change(25) - self.df['close_mkt'].pct_change(25)
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
        self.df[col_name] = self.df['sector_return'].rolling(5).mean()
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
        self.df[col_name] = log_return.rolling(5).sum() - self.df['sector_return'].rolling(5).sum()
        self.new_cols.append(col_name)
        sector_ret_60 = (1 + self.df['sector_return']).rolling(60).apply(np.prod, raw=True) - 1
        market_ret_60 = (1 + self.df['Market_Return']).rolling(60).apply(np.prod, raw=True) - 1
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
        self.df[col_name] = self.df['eps'].ffill()
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
        # ターゲットやフィルタリングフラグ作成用一次変数
        log_return = np.log(self.df['close'] / self.df['close'].shift(1))
        self.df['Vol_20d'] = log_return.rolling(20).std()
        self.df['volume_p_MA5'] = self.df['volume_p'].rolling(5).mean()
        self.df['log_market_cap'] = np.log(self.df['close'] * self.df['shares_outstanding'])
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon_tac)
        self.df['Market_Return_Future'] = self.df['Market_Return'].shift(-1).rolling(window=indexer).sum()
        self.df['Sector_Return_Future'] = self.df['sector_return'].shift(-1).rolling(window=indexer).sum()
        return self

    
    # --- ターゲット作成 ---
    def apply_timeseries_targets(self):
        """ダーゲット作成"""
        # --- ターゲット作成：戦略モデル ---
        entry_price = self.df['open'].shift(-1) # 翌日始値エントリー
        # インデクサ作成
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon_tac)
        # 未来データの取得
        future_high_max = self.df['high'].shift(-1).rolling(window=indexer).max()
        future_low_min = self.df['low'].shift(-1).rolling(window=indexer).min()
        future_close_end = self.df['close'].shift(-self.horizon_tac)
        # 基本情報格納
        self.df['Entry_Price'] = entry_price
        self.df['Future_High_Tac'] = future_high_max
        self.df['Future_Low_Tac'] = future_low_min
        self.df['Future_Close_Tac'] = future_close_end
        # --- 1. Smoothed Target (Category C) ---
        # 翌日から5日間のVWAPを計算
        # VWAP = Sum(Volume_P) / Sum(Volume)
        future_pv_sum = self.df['volume_p'].shift(-1).rolling(window=indexer).sum()
        future_v_sum = self.df['volume'].shift(-1).rolling(window=indexer).sum()
        future_vwap = future_pv_sum / (future_v_sum + 1e-9)
        # target = VWAP_5d / Entry_Price - 1
        self.df['target_tac_smoothed_return'] = (future_vwap / entry_price) - 1.0
        # --- 2. Volatility-Scaled Residual (Category B) ---
        self.df['target_ret_5'] = (future_close_end / entry_price.replace(0, np.nan)) - 1.0
        # Beta調整 (Market_Return_Futureは _add_market_self.dfures で作成済みと仮定)
        # もし未作成なら簡易的に self.df['Market_Return'].shift(-1).rolling(window=indexer).sum() を使用
        market_ret_future = self.df['Market_Return'].shift(-1).rolling(window=indexer).sum()
        residual_ret = self.df['target_ret_5'] - (self.df['BET_Beta60_RAW'] * market_ret_future)
        # Vol調整 (日次Vol * sqrt(5) で期間Volに換算)
        vol_5d = self.df['Vol_20d'] * np.sqrt(self.horizon_tac)
        self.df['target_tac_vol_scaled_residual'] = residual_ret / (vol_5d + 1e-6)
        # --- 3. Triple Barrier Methods (Category D) ---
        # 期間ボラティリティに基づく動的閾値
        # Vectorized implementation for speed (avoid loop)
        def calc_triple_barrier(up_multiplier, down_multiplier):
            """
            ベクトル化されたトリプルバリア計算
            return: 1(利確), -1(損切), 0(時間切れ)
            """
            barrier_up = entry_price * (1 + vol_5d * up_multiplier)
            barrier_dn = entry_price * (1 - vol_5d * down_multiplier)
            # 1日後～5日後の高値・安値を取得
            h1 = self.df['high'].shift(-1); l1 = self.df['low'].shift(-1)
            h2 = self.df['high'].shift(-2); l2 = self.df['low'].shift(-2)
            h3 = self.df['high'].shift(-3); l3 = self.df['low'].shift(-3)
            h4 = self.df['high'].shift(-4); l4 = self.df['low'].shift(-4)
            h5 = self.df['high'].shift(-5); l5 = self.df['low'].shift(-5)
            # 各日のヒット判定 (利確=1, 損切=-1, なし=0)
            # 損切を優先判定（保守的）または同時なら損切とするロジック
            def check_hit(h, l, b_up, b_dn):
                # 損切ヒット
                sl = (l < b_dn)
                # 利確ヒット
                tp = (h > b_up)
                # 両方ヒットした場合(大きな足)は、損切(-1)とみなす（保守的運用）
                # 利確のみ=1, 損切のみ=-1, 両方=-1, なし=0
                res = np.where(sl, -1, np.where(tp, 1, 0))
                return res
            r1 = check_hit(h1, l1, barrier_up, barrier_dn)
            r2 = check_hit(h2, l2, barrier_up, barrier_dn)
            r3 = check_hit(h3, l3, barrier_up, barrier_dn)
            r4 = check_hit(h4, l4, barrier_up, barrier_dn)
            r5 = check_hit(h5, l5, barrier_up, barrier_dn)
            # 最初のヒットを探す (r1から順に0以外があれば採用)
            # np.select は条件の優先順位順に評価される
            conds = [r1!=0, r2!=0, r3!=0, r4!=0, r5!=0]
            choices = [r1, r2, r3, r4, r5]
            return np.select(conds, choices, default=0)
        # Strategy A: Balance (1.0σ / 1.0σ)
        self.df['target_tac_tb_strategy_a'] = calc_triple_barrier(1.0, 1.0)
        # Strategy B: Trend (1.5σ / 0.75σ) - 損小利大
        self.df['target_tac_tb_strategy_b'] = calc_triple_barrier(1.5, 0.75)
        # Strategy C: Reversion (0.5σ / 1.0σ) - 高勝率
        self.df['target_tac_tb_strategy_c'] = calc_triple_barrier(0.5, 1.0)
        
        # --- ターゲット作成：戦略モデル ---
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon_str)
        future_high_max = self.df['high'].shift(-1).rolling(window=indexer).max()
        future_low_min = self.df['low'].shift(-1).rolling(window=indexer).min()
        future_close_end = self.df['close'].shift(-self.horizon_str)
        self.df['Entry_Price'] = entry_price
        self.df['Future_High_Str'] = future_high_max
        self.df['Future_Low_Str'] = future_low_min
        self.df['Future_Close_Str'] = future_close_end
        # 60日累積リターン（基本値） RankやPeer Alphaの計算ベースとして後続のクロスセクション処理で使用
        self.df['target_ret_60'] = (self.df['close'].shift(-self.horizon_str) / entry_price.replace(0, np.nan)) - 1.0
        # Risk-Adjusted Residual Momentum (60d) ベータ調整済みリターンをボラティリティで標準化
        market_ret_60 = self.df['Market_Return'].shift(-1).rolling(window=indexer).sum()
        residual_60 = self.df['target_ret_60'] - (self.df['BET_Beta60_RAW'] * market_ret_60)
        self.df['target_str_risk_adj'] = residual_60 / (self.df['Vol_20d'] * np.sqrt(12) + 1e-6) # 20日Volを60日換算(sqrt(3)近似だが実務上Vol_20dで正規化も一般的)
        # Return Consistency Score (60d) 60日間の累積リターン曲線の直線性をR2で算出
        def _calc_consistency(window):
            if np.isnan(window).any(): return np.nan
            cum_ret = np.cumprod(1 + window)
            x = np.arange(len(cum_ret))
            return np.corrcoef(x, cum_ret)[0, 1]**2
        fwd_ret_1d = self.df['close'].pct_change().shift(-1)
        self.df['target_str_consistency'] = fwd_ret_1d.rolling(window=indexer).apply(_calc_consistency, raw=True)
        # Volatility Scaling Alpha (60d) 銘柄固有のボラティリティでスケーリング
        self.df['target_str_vol_scale'] = self.df['target_ret_60'] / (self.df['VOL_Volatility60_RAW'] + 1e-6)
        # Triple Barrier Method 
        # 3値分類ラベル: 1(利確), -1(損切), 0(時間切れ)
        # バリア幅の設定: ボラティリティベース (De Prado流)
        # 上値(PT) = 期間ボラティリティ * 1.0
        # 下値(SL) = 期間ボラティリティ * 1.0 (損益比率1:1の設定)
        vol_horizon = self.df['Vol_20d'] * np.sqrt(self.horizon_str)
        pt_width = vol_horizon * 1.0
        sl_width = vol_horizon * 1.0
        # 高速化のためのNumpy配列化
        high_vals = self.df['high'].values
        low_vals = self.df['low'].values
        entry_vals = entry_price.values
        pt_vals = pt_width.values
        sl_vals = sl_width.values
        labels = np.zeros(len(self.df)) # デフォルト0 (Time-out)
        # 60日間のウィンドウ走査（ループ処理）
        # ※PandasのRollingのみでの「First Touch」判定は困難なため、Numpyループを使用
        horizon = self.horizon_str
        n_samples = len(self.df)
        for i in range(n_samples - horizon - 1):
            if np.isnan(entry_vals[i]) or np.isnan(pt_vals[i]):
                labels[i] = np.nan
                continue
            entry = entry_vals[i]
            upper_barrier = entry * (1 + pt_vals[i])
            lower_barrier = entry * (1 - sl_vals[i])
            # 未来ウィンドウを取得 (i+1 ~ i+horizon)
            # エントリーは i の次の足(i+1)のOpenなので、高安の参照は i+1 から
            window_high = high_vals[i+1 : i+1+horizon]
            window_low = low_vals[i+1 : i+1+horizon]
            # バリアブレイク判定
            # 上抜けした最初のインデックス
            hit_upper = np.where(window_high > upper_barrier)[0]
            # 下抜けした最初のインデックス
            hit_lower = np.where(window_low < lower_barrier)[0]
            first_upper = hit_upper[0] if len(hit_upper) > 0 else horizon + 1
            first_lower = hit_lower[0] if len(hit_lower) > 0 else horizon + 1
            if first_upper == horizon + 1 and first_lower == horizon + 1:
                labels[i] = 0 # どちらにも触れず期限切れ
            elif first_upper < first_lower:
                labels[i] = 1 # 利確バリアに先に到達
            else:
                labels[i] = -1 # 損切バリアに先に到達（同時なら保守的に損切とみなす）
        self.df['target_str_triple_barrier'] = labels
        # 不要変数の削除
        return self
    
    def apply_crosssectional_targets(self):
        """クロスセクションターゲットの追加"""
        new_cols = {}
        # --- 1. Era-wise Rank (Category A) ---
        # 単純なRank (0.0 ~ 1.0)
        tac_rank = self.df.groupby('date')['target_ret_5'].rank(pct=True, method='average')
        new_cols['target_tac_rank'] = tac_rank

        # 既存: Gauss Rank (正規分布化)
        epsilon = 1e-6
        rank_clipped = tac_rank * (1 - 2 * epsilon) + epsilon
        new_cols['target_tac_gauss_rank'] = erfinv(2 * rank_clipped - 1)

        # --- 2. Linear Residual (Category C) ---
        # 簡易的な実装: リターンを「セクター平均」と「市場平均」で説明する線形モデルの残差
        # 本来はRidge回帰などが望ましいが、計算コストを考慮し
        # Target = Return - (Beta_Market * Market_Ret + Beta_Sector * Sector_Ret) の簡易版とする
        # ここではさらにシンプルに、「セクター相対リターン」の分布内偏差（Zスコア的なもの）を
        # 線形モデルで説明しきれない固有リターンとみなす
        # 手順:
        # 1. セクターリターンは _add_sector_relative_features で 'Sector_Return_Future' として計算済みと仮定
        #    (もしなければ計算する)
        indexer_sec = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon_tac)
        sec_ret_fut = self.df['sector_return'].shift(-1).rolling(window=indexer_sec).sum()
        mkt_ret_fut = self.df['Market_Return'].shift(-1).rolling(window=indexer_sec).sum()
        # 2. 残差 = Target_Return - (0.5 * Market + 0.5 * Sector) ※係数は簡易
        # より厳密には、日次で回帰係数を決めるのが良いが、ここでは
        # 「市場とセクターの影響を引いたもの」をLinear Residualの代替とする
        new_cols['target_tac_linear_residual'] = self.df['target_ret_5'] - (0.5 * mkt_ret_fut + 0.5 * sec_ret_fut)
        # セクター相対フラグ
        new_cols['target_tac_sector_relative'] = (self.df['target_ret_5'] > sec_ret_fut).astype(int)
        # --- 戦略モデル用クロスセクション ---
        # Relative Rank Change (60d)
        new_cols['target_str_rank'] = self.df.groupby('date')['target_ret_60'].rank(pct=True)
        # Peer Group Neutralized Alpha (60d)
        sector_mean_60 = self.df.groupby(['date', 'sector33_code'])['target_ret_60'].transform('mean')
        new_cols['target_str_peer_alpha'] = self.df['target_ret_60'] - sector_mean_60
        # 一括結合
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self
    
    
    def get_df(self):
        return self.df

# import numpy as np
# import pandas as pd

# def get_weights_ffd(d: float, thres: float, size: int) -> np.ndarray:
#     """
#     FFD(Fixed-Width Window FracDiff)用の重みを計算する
#     """
#     w = [1.0]
#     for k in range(1, size):
#         # 再帰的な重みの計算: w_k = -w_{k-1} * (d - k + 1) / k
#         w_curr = -w[-1] * (d - k + 1) / k
#         w.append(w_curr)
    
#     w = np.array(w[::-1]).reshape(-1, 1) # 反転させて列ベクトル化
    
#     # 閾値（thres）を下回る重みを除外（メモリのカットオフ）
#     # ただし、計算を安定させるため最小限のウィンドウサイズを確保する
#     mask = np.abs(w) > thres
#     return w[mask].reshape(-1, 1)

# def apply_frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
#     """
#     Pandas Seriesに対して分数次微分を適用する
#     """
#     # 1. 重みの取得
#     weights = get_weights_ffd(d, thres, len(series))
#     width = len(weights) - 1
    
#     # 2. 窓関数内での畳み込み演算
#     # NaNを避けるため、重みの幅が確保できる箇所から計算開始
#     res = {}
#     for i in range(width, series.shape[0]):
#         # 指定区間のデータと重みのドット積
#         res[series.index[i]] = np.dot(weights.T, series.iloc[i-width : i+1].values.reshape(-1, 1))[0, 0]
        
#     return pd.Series(res)



    # @register_block
    # def _add_lagged_targets(self, df):
    #     """ターゲット変数のラグ特徴量を追加"""
    #     targets = [c for c in self.target_cols if c in df.columns]
    #     new_cols = {}
    #     for col in targets:
    #         lag = 0
    #         if '_tac_' in col or 'target_ret_5' in col:
    #             lag = self.horizon_tac
    #         elif '_str_' in col or 'target_ret_60' in col:
    #             lag = self.horizon_str
    #         if lag > 0:
    #             new_col_name = f"{col}_Lag{lag}"
    #             if new_col_name not in df.columns:
    #                 new_cols[new_col_name] = df[col].shift(lag)
    #     if new_cols:
    #         df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    #     return df

    # def _fill_missing_values_with_sector_median(self, df):
    #     """指定カラムの欠損を業種別中央値で埋める"""
    #     target_cols = ['EPS_Actual', 'turnover_ratio', 'log_market_cap']
    #     if 'sector33_code' in df.columns:
    #         for col in target_cols:
    #             if col in df.columns:
    #                 sector_median = df.groupby(['date', 'sector33_code'])[col].transform('median')
    #                 df[col] = df[col].fillna(sector_median)
    #     return df
