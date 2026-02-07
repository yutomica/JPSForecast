import pandas as pd
import numpy as np
import pandas_ta as ta
import talib
from functools import wraps
from scipy.special import erfinv
import gc

def register_block(func):
    """ブロック内で生成された全ての新規カラムを自動登録するデコレータ"""
    @wraps(func)
    def wrapper(self, df, *args, **kwargs):
        cols_before = set(df.columns)
        df = func(self, df, *args, **kwargs)
        cols_after = set(df.columns)
        new_features = list(cols_after - cols_before)
        if new_features:
            # 異常値置換と型変換
            df[new_features] = df[new_features].replace([np.inf, -np.inf], np.nan).astype('float32')
            # 重複を排除しつつ、挿入順を維持して登録
            for col in new_features:
                if col not in self._feature_registry:
                    self._feature_registry[col] = None # 辞書のキーとして登録
        return df
    return wrapper

class FeatureEngineer:
    def __init__(self):
        self.horizon_tac = 5    # 予測期間日数：戦術モデル
        self.horizon_str = 60   # 予測期間日数：戦略モデル
        self.initial_cols = [
            'scode', 'sector33_code', 'date', 'volume_p', 'close', 'shares_outstanding',
            'Market_Return', 'Market_Trend_Idx', 'Market_HV_20', 
            'Market_Foreign_Z_60', 'Market_Individual_Z_60', 'Market_Foreign_Z_250',
            'Market_Individual_Z_250', 'Market_Foreign_Diff', 'selling_volume_ratio',
        ]
        # 辞書のキーとして格納（Python 3.7+ では挿入順が保持されます）
        self._feature_registry = dict()
        self.meta_cols = ['scode', 'date', 'volume_p', 'close', 'shares_outstanding']
        self.target_cols = [
            'target_return', 'target_residual', 'target_risk_adjusted', 
            'target_balanced', 'target_sector_relative', 'target_sharpe_filter',
            'target_aggressive', 'target_low_drawdown',
            'target_gauss_rank', 'target_top_percentile',
            # 戦略モデル用ターゲット、別スクリプトで生成
            'target_reg', 'target_cls'
        ]

    def _calc_rci(self, series, period):
        time_ranks = np.arange(1, period + 1)
        def rci_func(window):
            price_ranks = pd.Series(window).rank(method='average').values
            d_squared = np.sum((time_ranks - price_ranks) ** 2)
            rci = (1 - (6 * d_squared) / (period * (period ** 2 - 1))) * 100
            return rci
        return series.rolling(window=period).apply(rci_func, raw=True)

    def _z_score(self, x):
        """
        Z-Score計算用関数
        単一銘柄のみの場合(len<=1)や、標準偏差が0の場合は 0.0 を返す
        """
        if len(x) <= 1:
            return 0.0
        std = x.std()
        if std == 0: 
            return 0.0
        return (x - x.mean()) / std

    def calculate_sector_z_score(self, df, feature_col, group_cols=['date', 'sector33_code']):
        """指定されたカラムを日次・セクターごとにZ-Score化する"""
        # 単一銘柄の場合、group_colsでグルーピングすると各グループのサイズが1になるため
        # _z_scoreメソッド側で 0.0 を返す処理が必須となる
        z_values = df.groupby(group_cols)[feature_col].transform(self._z_score)
        return z_values.clip(-3, 3)
    
    @register_block
    def _add_trend_features(self, feat, df):
        """トレンド系指標の一括作成"""
        feat['ADX_14'] = df.ta.adx(length=14).iloc[:, 0]
        # ルールベースフィルタ用にSMAを作成
        feat['SMA_5'] = df.ta.sma(length=5)
        feat['SMA_25'] = df.ta.sma(length=25)
        feat['SMA_75'] = df.ta.sma(length=75)
        feat['Dist_SMA5'] = (df['close'] - feat['SMA_5']) / feat['SMA_5']
        feat['Dist_SMA75'] = (df['close'] - feat['SMA_75']) / feat['SMA_75']
        feat['Dist_SMA25'] = (df['close'] - feat['SMA_25']) / feat['SMA_25']
        feat['MA_Diff_5'] = df['close'] / df['close'].rolling(5).mean()
        # VWAP関連
        typ_price = (df['high'] + df['low'] + df['close']) / 3
        pv_sum = (typ_price * df['volume']).rolling(5).sum()
        v_sum = df['volume'].rolling(5).sum()
        rolling_vwap = pv_sum / v_sum
        feat['Dist_VWAP_5'] = (df['close'] - rolling_vwap) / rolling_vwap
        feat['Dist_VWAP_Slope'] = feat['Dist_VWAP_5'].diff()
        vwap = df['volume_p'] / df['volume'].replace(0, 1)
        feat['vwap_dev'] = (df['close'] / vwap) - 1
        feat['Efficiency_Ratio_10'] = df.ta.er(length=10)
        feat['LinReg_Slope_10'] = df.ta.slope(length=10)
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        if macd is not None:
            feat['MACD_Hist_Norm'] = macd['MACDh_12_26_9'] / df['close']
            feat['MACD_Hist_Diff'] = feat['MACD_Hist_Norm'].diff(1)
        feat['MAE_5'] = (df['low'].rolling(5).min() / df['close']) - 1.0
        feat['MAE_10'] = (df['low'].rolling(10).min() / df['close']) - 1.0
        feat['Dist_High_60'] = df['close'] / df['high'].rolling(60).max()
        feat['Dist_High_250'] = df['close'] / df['high'].rolling(250).max()
        high_26 = df['high'].rolling(26).max()
        low_26 = df['low'].rolling(26).min()
        kijun_sen = (high_26 + low_26) / 2
        feat['Dist_Kijun'] = (df['close'] - kijun_sen) / kijun_sen
        past_3d_high = df['high'].shift(1).rolling(3).max()
        feat['New_High_Flag_3'] = (df['close'] > past_3d_high).astype(int)
        feat['Close_log'] = np.log(df['close'])
        feat['MA_25_log'] = np.log(df['close'].rolling(25).mean())
        feat['MA_75_log'] = np.log(df['close'].rolling(75).mean())
        # セクター内相対指標用
        for window in [25, 75, 200]:
            # transformを使用して形状を維持
            ma = df['close'].transform(lambda x: x.rolling(window).mean())
            feat[f'ma_dev_{window}'] = (df['close'] / ma) - 1
        return feat
    
    @register_block
    def _add_momentnum_features(self, feat, df):
        """モメンタム系指標の一括作成"""
        feat['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
        feat['beta_proxy'] = feat['Log_Return'] - df['Market_Return']
        feat['Downside_Run'] = feat['Log_Return'].clip(upper=0).rolling(5).sum()       
        feat['Return_Skewness'] = feat['Log_Return'].rolling(20).skew()
        feat['Return_Kurtosis'] = feat['Log_Return'].rolling(20).kurt()
        feat['Return_1d'] = df['close'].pct_change(1)
        feat['Return_3d'] = df['close'].pct_change(3)
        feat['Return_5d'] = df['close'].pct_change(5)
        feat['Return_10d'] = df['close'].pct_change(10)
        feat['Return_20d'] = df['close'].pct_change(20)
        feat['Return_1d_Lag1'] = feat['Return_1d'].shift(1)
        feat['Return_1d_Lag2'] = feat['Return_1d'].shift(2)
        feat['RSI_9'] = df.ta.rsi(length=9)
        feat['RSI_14'] = df.ta.rsi(length=14)
        feat['RSI_14_norm'] = (df.ta.rsi(length=14) - 50) / 50 # Normalize -1 to 1
        bb = df.ta.bbands(length=20, std=2)
        if bb is not None:
            bb_p_col = [c for c in bb.columns if c.startswith('BBP')][0]
            bb_w_col = [c for c in bb.columns if c.startswith('BBB')][0]
            feat['BB_Percent_B'] = bb[bb_p_col]
            feat['BB_Bandwidth'] = bb[bb_w_col]
        # range_len は上で計算済み
        range_len = df['high'] - df['low']
        body_size = np.abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
        with np.errstate(divide='ignore', invalid='ignore'):
            feat['Body_Ratio'] = body_size / range_len
            feat['Upper_Shadow_Ratio'] = upper_shadow / range_len
            feat['Lower_Shadow_Ratio'] = lower_shadow / range_len
            feat['Intraday_Strength'] = (df['close'] - df['open']) / range_len
            feat['Lower_Shadow_MA5'] = feat['Lower_Shadow_Ratio'].rolling(5).mean()
            feat['Lower_Shadow_Mean'] = ((df[['open', 'close']].min(axis=1) - df['low']) / range_len).rolling(5).mean()
        diff = df['close'].diff()
        sign = np.sign(diff).fillna(0)
        is_change = sign != sign.shift(1)
        group_id = is_change.cumsum()
        count = df.groupby(group_id).cumcount() + 1
        feat['Streak'] = np.where(sign > 0, count, np.where(sign < 0, -count, 0))
        feat['Bullish_Ratio_20'] = (sign > 0).rolling(20).mean()
        feat['Close_Open_Ratio'] = (df['close'] - df['open']) / df['close']
        feat['Close_Position'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        prev_close = df['close'].shift(1)
        feat['Gap_Rate'] = (df['open'] / prev_close) - 1.0
        feat['Gap_Abs'] = feat['Gap_Rate'].abs()
        feat['Gap_Ratio'] = feat['Gap_Rate']
        feat['Large_Move_Count'] = (feat['Log_Return'].abs() > 0.03).rolling(20).sum()
        range_len = df['high'] - df['low']
        feat['Range_Ratio_Long'] = range_len.rolling(5).mean() / range_len.rolling(20).mean()
        feat['Max_Gain_5'] = (df['high'].rolling(5).max() / df['close']) - 1.0
        feat['RCI_9'] = self._calc_rci(df['close'], 9)
        feat['RCI_26'] = self._calc_rci(df['close'], 26)
        feat['RCI_52'] = self._calc_rci(df['close'], 52)
        feat['RCI_9_Diff'] = feat['RCI_9'].diff(1)
        feat['RCI_26_Diff'] = feat['RCI_26'].diff(1)
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=25, nbdevup=2.0, nbdevdn=2.0)
        feat['Upper_Band_2sig'] = np.log(upper)
        feat['Lower_Band_2sig'] = np.log(lower)
        # セクター内相対指標用
        feat['Return_6m'] = df['close'].pct_change(120)
        feat['Return_12m'] = df['close'].pct_change(240)
        return feat

    @register_block
    def _add_volatility_features(self, feat, df):
        """ボラティリティ系指標の一括作成"""
        atr = df.ta.atr(length=14)
        feat['ATR_Ratio'] = atr / df['close']
        atr_short = df.ta.atr(length=5)
        atr_mid = df.ta.atr(length=20)
        feat['ATR_Squeeze'] = atr_short / atr_mid
        bb = df.ta.bbands(length=20, std=2)
        if bb is not None:
            bb_bandwidth = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / (bb['BBM_20_2.0'] + 1e-9)
            feat['ATR_Squeeze_bb'] = bb_bandwidth / (atr + 1e-9)
        feat['hist_vol_20'] = feat['Log_Return'].rolling(20).std()
        feat['HV_30'] = feat['Log_Return'].rolling(30).std() * np.sqrt(250)
        feat['HV_Slope'] = feat['HV_30'].diff(5)
        feat['Realized_Skew_20'] = feat['Return_1d'].rolling(20).skew()
        feat['Ulcer_Index_14'] = df.ta.ui(length=14)
        # セクター内相対指標用 
        max_52w = df['close'].transform(lambda x: x.rolling(240).max())
        feat['high_52w_dist'] = (df['close'] / max_52w) - 1
        def calc_downside_std(x, window=60):
            neg_ret = x.where(x < 0, 0)
            return neg_ret.rolling(window).std()
        feat['downside_dev_60'] = feat['Log_Return'].transform(lambda x: calc_downside_std(x))
        feat['volatility_60'] = df['close'].pct_change().transform(lambda x: x.rolling(60).std())
        # ターゲット用
        feat['Vol_20d'] = feat['Log_Return'].rolling(20).std()
        return feat

    @register_block
    def _add_volume_features(self, feat, df):
        """ボリューム系指標の一括作成"""
        feat['Volume_Log'] = np.log(df['volume'] + 1)
        feat['Abnormal_Volume'] = df['volume_p'] / df['volume_p'].rolling(20).mean()
        feat['Volume_Change'] = df['volume'].pct_change()
        feat['Volume_Slope_5'] = ta.slope(np.log(df['volume'] + 1), length=5)
        feat['Log_Trading_Cap'] = np.log(df['close'] * df['volume'] + 1)
        feat['log_vol_change'] = np.log(df['volume'] / df['volume'].shift(1).replace(0, 1))
        feat['MFI_14'] = df.ta.mfi(length=14)
        with np.errstate(divide='ignore', invalid='ignore'):
            hl_log_sq = np.log(df['high'] / df['low']) ** 2
            const_factor = 4 * np.log(2)
            feat['Volatility_Parkinson'] = np.sqrt(hl_log_sq.rolling(14).mean() / const_factor)
        feat['Return_Volatility'] = feat['Log_Return'].rolling(10).std()
        vol_ma5 = df['volume'].rolling(5).mean()
        feat['Vol_Ratio_5d'] = df['volume'] / vol_ma5.replace(0, np.nan)
        hv_5 = feat['Log_Return'].rolling(5).std() * np.sqrt(250)
        feat['Vol_Ratio_HV'] = hv_5 / feat['HV_30']
        vol_median_20 = df['volume'].rolling(20).median()
        is_spike = (df['volume'] > (vol_median_20 * 3)).astype(int)
        feat['Vol_Spike_Count_20'] = is_spike.rolling(20).sum()
        is_up = (df['close'] > df['open']).astype(int)
        vol_up = (df['volume'] * is_up).rolling(20).sum()
        vol_down = (df['volume'] * (1 - is_up)).rolling(20).sum()
        feat['Vol_Up_Down_Ratio'] = vol_up / (vol_down + 1e-9)
        feat['Volume_MA_25'] = feat['Volume_Log'].rolling(25).mean() 
        # ルールベースフィルタ用
        feat['volume_p_MA5'] = df['volume_p'].rolling(5).mean()
        return feat

    @register_block
    def _add_fundamental_features(self, feat, df):
        """財務情報系指標の一括作成"""
        feat['EPS_Actual'] = df['eps'].ffill()
        feat['log_days_since_pub'] = np.log1p((df['date'] - df['published_date']).dt.days).fillna(0)
        feat['log_market_cap'] = np.log(df['close'] * df['shares_outstanding'])
        # 以下は全てセクター内相対指標用に作成
        LAG_YEAR = 240
        grouped_scode = df.groupby('scode')
        feat['accruals'] = (df['net_income'] - df['operating_cf']) / df['total_assets'].replace(0, np.nan)
        prev_eps = grouped_scode['eps'].shift(LAG_YEAR)
        feat['eps_growth_yoy'] = (df['eps'] - prev_eps) / (prev_eps.abs() + 1e-6)
        feat['equity_ratio'] = df['equity'] / df['total_assets'].replace(0, np.nan)
        actual_bps = df['equity'] / df['shares_outstanding'].replace(0, np.nan)
        actual_eps = df['net_income'] / df['shares_outstanding'].replace(0, np.nan)
        feat['log_pbr'] = np.log(df['close'] / actual_bps.replace(0, np.nan))
        filled_eps = df['eps'].combine_first(actual_eps)
        feat['log_per'] = np.log(df['close'] / filled_eps.replace(0, np.nan))
        if 'cfps' in df.columns:
            feat['log_pcfr'] = np.log(df['close'] / df['cfps'].replace(0, np.nan))
        if 'dps' in df.columns:
            feat['div_yield'] = df['dps'] / df['close']
        feat['op_growth_yoy'] = df['operating_profit'] / grouped_scode['operating_profit'].shift(LAG_YEAR).replace(0, np.nan) - 1
        feat['op_margin'] = df['operating_profit'] / df['sales'].replace(0, np.nan)
        feat['roe'] = df['net_income'] / df['equity'].replace(0, np.nan)
        feat['roa'] = df['net_income'] / df['total_assets'].replace(0, np.nan)
        feat['sales_growth_yoy'] = df['sales'] / grouped_scode['sales'].shift(LAG_YEAR).replace(0, np.nan) - 1
        feat['revision_rate'] = (df['eps'] / grouped_scode['eps'].shift(20).replace(0, np.nan)) - 1
        feat['progress_rate'] = df['operating_profit'] / df['operating_profit_forecast'].replace(0, np.nan)
        return feat

    @register_block
    def _add_calendar_feature(self, feat, df):
        """時間・カレンダー系指標の一括作成"""
        day_num = df['date'].dt.day
        feat['Month'] = df['date'].dt.month
        feat['DayOfMonth'] = df['date'].dt.day
        feat['DayOfWeek'] = df['date'].dt.dayofweek
        feat['Sin_DayOfWeek'] = np.sin(2 * np.pi * feat['DayOfWeek'] / 6)
        feat['Cos_DayOfWeek'] = np.cos(2 * np.pi * feat['DayOfWeek'] / 6)
        feat['Is_Gotobi'] = ((day_num % 5 == 0) | (day_num == 31)).astype(int)
        feat['Is_Month_End'] = df['date'].dt.is_month_end.astype(int)
        feat['Quarter'] = df['date'].dt.quarter
        feat['Is_Quarter_End'] = feat['Month'].isin([3, 6, 9, 12]).astype(int) # 簡易判定
        feat['time_idx'] = (df['date'] - df['date'].min()).dt.days
        return feat

    @register_block
    def _add_market_features(self, feat, df):
        """マーケット指標の一括作成"""
        ret_stock = df['close'].pct_change()
        ret_market = df['Market_Return']
        rolling_cov = ret_stock.rolling(60).cov(ret_market)
        rolling_var = ret_market.rolling(60).var()
        feat['Beta_60'] = rolling_cov / rolling_var
        feat['RS_25'] = df['close'].pct_change(25) - df['close_mkt'].pct_change(25)
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
        feat['market_segment'] = df['market'].apply(map_market_segment)
        # ターゲット用
        indexer_mkt = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon_tac)
        feat['Market_Return_Future'] = df['Market_Return'].shift(-1).rolling(window=indexer_mkt).sum()
        return feat

    @register_block
    def _add_margin_features(self, feat, df):
        """信用取引情報系指標の一括作成"""
        # 信用取引系
        # (1) 信用買い残インパクト倍率
        # 平均出来高 (60日)
        df['avg_vol_60'] = df.groupby('scode')['volume'].transform(
            lambda x: x.rolling(60).mean()
        )
        feat['margin_buy_impact'] = (
            df['long_margin_trade_balance_share'] / 
            df['avg_vol_60'].replace(0, np.nan)
        )
        # (2) 信用倍率の変化 (対数)
        # 0除算回避のために +1
        feat['margin_ratio'] = np.log(
            (df['long_margin_trade_balance_share'] + 1) / 
            (df['short_margin_trade_balance_share'] + 1)
        )
        # 4週間前（約20営業日）との差分
        feat['margin_ratio_delta_4w'] = feat.groupby('scode')['margin_ratio'].diff(20)
        return feat

    @register_block
    def _add_sector_relative_features(self, df):
        """セクター相対特徴量の追加"""
        df['Sector_Momentum_5d'] = df['sector_return'].rolling(5).mean()
        df['Sector_Rel'] = df['close'] / df['sector_return']
        df['Rel_Sector_Return_1d'] = df['Return_1d'] - df['sector_return']
        df['Rel_Sector_Return_5d'] = df['Return_1d'].rolling(5).sum() - df['sector_return'].rolling(5).sum()
        # ターゲット用
        indexer_sec = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon_tac)
        df['Sector_Return_Future'] = df['sector_return'].shift(-1).rolling(window=indexer_sec).sum()
        return df
    
    @register_block
    def _add_z_score_features(self, df):
        """Zスコア特徴量の追加"""
        z_targets = [
            'accruals','eps_growth_yoy','equity_ratio',
            'log_pbr','log_pcfr','log_per',
            'div_yield','op_growth_yoy','op_margin','roa','roe',
            'sales_growth_yoy','revision_rate','progress_rate',
            'ATR_Ratio','ma_dev_25','ma_dev_75','ma_dev_200',
            'Return_20d','Return_6m','Return_12m','RSI_14',
            'high_52w_dist','downside_dev_60','volatility_60',
            'margin_buy_chg','margin_ratio','margin_buy_impact',
        ]
        # Zスコア計算後もキープする特徴量
        keep_cols = [
            'Return_20d','RSI_14','ATR_Ratio','RSI_14',
            'margin_buy_impact','margin_ratio'
        ]
        for col in z_targets:
            if col in df.columns:
                target_name = f"{col}_sector_z"
                df[target_name] = self.calculate_sector_z_score(df, col)
                if col not in keep_cols:
                    # オリジナル変数の削除 (DataFrame)
                    df.drop(columns=[col], inplace=True)
                    # オリジナル変数の削除 (Registry)
                    self._feature_registry.pop(col, None)
        return df

    def _add_targets(self, feat, df):
        """ダーゲット作成"""
        entry_price = df['open'].shift(-1)
        # --- ターゲット作成：戦術モデル ---
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon_tac)
        future_high_max = df['high'].shift(-1).rolling(window=indexer).max()
        future_low_min = df['low'].shift(-1).rolling(window=indexer).min()
        future_close_end = df['close'].shift(-self.horizon_tac)
        feat['Entry_Price'] = entry_price
        feat['Future_High_Tac'] = future_high_max
        feat['Future_Low_Tac'] = future_low_min
        feat['Future_Close_Tac'] = future_close_end
        # --- 1. Return戦略ターゲット ---
        # target_return (単純リターン)
        feat['target_return'] = (future_close_end / entry_price.replace(0, np.nan)) - 1.0
        # target_residual (残差リターン)  target = Return_5d - Beta * Market_Return_5d
        feat['target_residual'] = feat['target_return'] - (feat['Beta_60'] * feat['Market_Return_Future'])
        # target_risk_adjusted (リスク調整後リターン) target = Return_5d / (Vol_20d + 1e-6)
        feat['target_risk_adjusted'] = feat['target_return'] / (feat['Vol_20d'] + 1e-6)
        # --- 2. Balanced戦略ターゲット ---
        # target_balanced (Triple Barrier) Upper: +5%, Lower: -3% 
        is_safe_bal = future_low_min > (entry_price * (1 - 0.03))
        is_hit_tp_bal = future_high_max > (entry_price * (1 + 0.05))
        is_positive_end = future_close_end > entry_price # 補完条件
        feat['target_balanced'] = (is_safe_bal & (is_hit_tp_bal | is_positive_end)).astype(int)
        # target_sharpe_filter (シャープレシオ閾値) target = (Sharpe_5d > 0.1)
        future_daily_mean = feat['Return_1d'].shift(-1).rolling(window=indexer).mean()
        future_daily_std = feat['Return_1d'].shift(-1).rolling(window=indexer).std()
        future_sharpe = future_daily_mean / (future_daily_std + 1e-9)
        feat['target_sharpe_filter'] = (future_sharpe > 0.1).astype(int)
        # --- 3. Aggressive戦略ターゲット ---
        # target_aggressive (Triple Barrier High) Upper: +10%, Lower: -5% (仕様書に合わせて修正)
        is_safe_agg = future_low_min > (entry_price * (1 - 0.05))
        is_hit_tp_agg = future_high_max > (entry_price * (1 + 0.10))
        is_strong_end = future_close_end > (entry_price * 1.03) # 補完条件
        feat['target_aggressive'] = (is_safe_agg & (is_hit_tp_agg | is_strong_end)).astype(int)
        # target_low_drawdown (低ドローダウン急騰) High_5d > 1.05 & Low_5d > 0.98
        is_high_gain = future_high_max > (entry_price * 1.05)
        is_low_dd = future_low_min > (entry_price * 0.98)
        feat['target_low_drawdown'] = (is_high_gain & is_low_dd).astype(int)
        # --- ターゲット作成：戦略モデル ---
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon_str)
        future_high_max = df['high'].shift(-1).rolling(window=indexer).max()
        future_low_min = df['low'].shift(-1).rolling(window=indexer).min()
        future_close_end = df['close'].shift(-self.horizon_str)
        feat['Entry_Price'] = entry_price
        feat['Future_High_Str'] = future_high_max
        feat['Future_Low_Str'] = future_low_min
        feat['Future_Close_Str'] = future_close_end
        # 不要変数の削除
        feat.drop(columns=['Vol_20d','Market_Return_Future'], inplace=True)
        self._feature_registry.pop('Vol_20d', None)
        self._feature_registry.pop('Market_Return_Future', None)
        return feat

    def _add_cross_sectional_target(self, df):
        """クロスセクションターゲットの追加"""
        # target_sector_relative (セクター相対) target = (Return_5d > Sector_Return_5d)
        indexer_sec = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon_tac)
        sector_return_future = df['sector_return'].shift(-1).rolling(window=indexer_sec).sum()
        df['target_sector_relative'] = (df['target_return'] > sector_return_future).astype(int)
        for date, group in df.groupby('date'):
            # Gauss Rank 0.0 < rank < 1.0
            rank = group['target_return'].rank(pct=True, method='average')
            # erfinvの入力範囲 (-1, 1) に収めるためのクリッピング
            epsilon = 1e-6
            rank = rank * (1 - 2 * epsilon) + epsilon
            group['target_gauss_rank'] = erfinv(2 * rank - 1)
            # Top Percentile 上位10%フラグ
            threshold = group['target_return'].quantile(0.90)
            group['target_top_percentile'] = (group['target_return'] >= threshold).astype(int)
            # dfに結果を反映
            df.loc[group.index, 'target_gauss_rank'] = group['target_gauss_rank']
            df.loc[group.index, 'target_top_percentile'] = group['target_top_percentile']
        # target_sector_relative (セクター相対) target = (Return_5d > Sector_Return_5d)
        df['target_sector_relative'] = (df['target_return'] > df['Sector_Return_Future']).astype(int)
        # 不要変数の削除
        df.drop(columns=['Sector_Return_Future'], inplace=True)
        self._feature_registry.pop('Sector_Return_Future', None)
        return df

    def add_time_series_features(self, df, output_target=True):
        if len(df) < 250: return pd.DataFrame()
        self._feature_registry = {k: None for k in self.initial_cols}
        # 入力データフレームの一部のカラムは保持しないため、必要なカラムをfeatにコピーする
        feat = df[self.initial_cols]
        # 特徴量作成ブロックの実行
        feat = self._add_trend_features(feat,df)
        feat = self._add_momentnum_features(feat,df)
        feat = self._add_volatility_features(feat,df)
        feat = self._add_volume_features(feat,df)
        feat = self._add_fundamental_features(feat,df)
        feat = self._add_calendar_feature(feat,df)
        feat = self._add_market_features(feat,df)
        feat = self._add_margin_features(feat,df)
        if output_target:
            df = self._add_targets(feat,df)
        gc.collect()
        return feat
    
    def add_cross_sectional_features(self, df_in, output_target=True):
        """銘柄横断特徴量の追加"""
        initial_cols = [x for x in df_in.columns if x not in self.target_cols and x not in self.meta_cols]
        self._feature_registry = {k: None for k in initial_cols}
        df = df_in.copy()
        # 以下の処理は、入力データフレームのカラムに追加するのみであり、入力データフレームにあるカラムは全て出力する
        df = self._add_sector_relative_features(df)
        df = self._add_z_score_features(df)
        if output_target:
            df = self._add_cross_sectional_target(df)
        return df 

    @property
    def feature_list(self):
        """自動登録された全特徴量リストを返す"""
        return list(self._feature_registry.keys())
    
