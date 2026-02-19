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
            'Market_Return', 'Market_Trend_Idx', 'Market_HV_20', 'market_vol_change',
            'Market_Foreign_Z_60', 'Market_Individual_Z_60', 'Market_Foreign_Z_250',
            'Market_Individual_Z_250', 'Market_Foreign_Diff', 'overseas_flow_trend', 'flow_accel', 'selling_volume_ratio',
        ]
        # 辞書のキーとして格納（Python 3.7+ では挿入順が保持されます）
        self._feature_registry = dict()
        self.meta_cols = [
            'scode', 'date', 'volume_p', 'close', 'shares_outstanding',
            'Entry_Price','Future_High_Tac','Future_Low_Tac','Future_Close_Tac',
            'Future_High_Str','Future_Low_Str','Future_Close_Str'
        ]
        self.target_cols = [
            # --- 戦術モデル用推奨ターゲット (5日先) ---
            # 1. Ranking系
            'target_tac_rank',          # Era-wise Rank (0~1)
            'target_tac_gauss_rank',    # Gauss Rank
            # 2. Risk調整系
            'target_tac_vol_scaled_residual', # Beta調整後 & Vol調整後
            # 3. 実執行・Alpha系
            'target_tac_smoothed_return',     # VWAP基準
            'target_tac_linear_residual',     # 線形モデル残差
            # 4. Triple Barrier (Dynamic)
            'target_tac_tb_strategy_a',       # A: Balance (1.0σ / 1.0σ)
            'target_tac_tb_strategy_b',       # B: Trend (1.5σ / 0.75σ)
            'target_tac_tb_strategy_c',       # C: Reversion (0.5σ / 1.0σ)
            # 戦略モデル用ターゲット
            'target_str_risk_adj','target_str_consistency','target_str_vol_scale','target_str_triple_barrier',
            'target_str_rank','target_str_peer_alpha',
            # 戦略モデル用ターゲット、別スクリプトで生成
            # 'target_reg', 'target_cls',
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
        # 長期価格位置 (120日)
        roll_120 = df['close'].rolling(120)
        max_120 = roll_120.max()
        min_120 = roll_120.min()
        feat['price_pos_120'] = (df['close'] - min_120) / (max_120 - min_120)
        # 一目均衡表 (雲) 距離
        # 先行スパンA: (転換線+基準線)/2 を 26日先にプロット
        # 先行スパンB: (52日最高値+52日最安値)/2 を 26日先にプロット
        # 当日(t)の雲の位置は、t-26日時点で計算された先行スパンA, Bの値
        high_9 = df['high'].rolling(9).max()
        low_9 = df['low'].rolling(9).min()
        tenkan = (high_9 + low_9) / 2
        high_26 = df['high'].rolling(26).max()
        low_26 = df['low'].rolling(26).min()
        kijun = (high_26 + low_26) / 2
        high_52 = df['high'].rolling(52).max()
        low_52 = df['low'].rolling(52).min()
        span_b = (high_52 + low_52) / 2
        # 26日前の値を参照して「当日の雲」とする
        # span_a_curr = ((tenkan + kijun) / 2).shift(26) 
        span_b_curr = span_b.shift(26)
        # 雲の下限（あるいは上限、ここでは強力な抵抗帯であるスパンBを採用）との距離
        feat['ichimoku_dist'] = (df['close'] - span_b_curr) / span_b_curr
        # セクター内相対指標用
        for window in [25, 75, 200]:
            # transformを使用して形状を維持
            ma = df['close'].rolling(window=window).mean()
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
        feat['momentum_12_1'] = df['close'].shift(20) / df['close'].shift(260) - 1
        feat['ret_overnight'] = (df['open'] / df['close'].shift(1)) - 1.0
        feat['ret_intraday'] = (df['close'] / df['open']) - 1.0
        feat['gap_rate'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        range_len = (df['high'] - df['low']).replace(0, np.nan)
        feat['candle_body_ratio'] = abs(df['close'] - df['open']) / range_len
        feat['up_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / range_len
        feat['win_rate_10d'] = (feat['Log_Return'] > 0).rolling(10).mean()
        flow = df['overseas_flow_trend']
        feat['flow_price_interaction'] = flow * feat['Log_Return']
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
        feat['atr_chg_rate'] = atr.pct_change()
        # セクター内相対指標用 
        max_52w = df['close'].rolling(window=240).max()
        feat['high_52w_dist'] = (df['close'] / max_52w) - 1
        def calc_downside_std(x, window=60):
            neg_ret = x.where(x < 0, 0)
            return neg_ret.rolling(window).std()
        feat['downside_dev_60'] = feat['Log_Return'].transform(lambda x: calc_downside_std(x))
        feat['volatility_60'] = df['close'].pct_change().rolling(60).std()
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
        feat['turnover_ratio'] = df['volume'] / df['shares_outstanding'].replace(0, np.nan)
        trading_value = df['close'] * df['volume']
        feat['amihud_illiq'] = feat['Log_Return'].abs() / (trading_value + 1e-9)
        # ルールベースフィルタ用
        feat['volume_p_MA5'] = df['volume_p'].rolling(5).mean()
        return feat

    @register_block
    def _add_fundamental_features(self, feat, df):
        """財務情報系指標の一括作成"""
        LAG_YEAR = 240 # サンプリング(interval)が1の場合。5の場合は 240/5 = 48 に調整が必要
        epsilon = 1e-6
        grouped_scode = df.groupby('scode')
        feat['is_missing_eps'] = df['eps'].isna().astype(int)
        feat['EPS_Actual'] = df['eps'].ffill()
        feat['log_days_since_pub'] = np.log1p((df['date'] - df['published_date']).dt.days).fillna(0)
        feat['log_market_cap'] = np.log(df['close'] * df['shares_outstanding'])
        # 以下は全てセクター内相対指標用に作成
        # 資産・資本関連（分母を絶対値 + epsilon で保護） 負の資産や資本（債務超過）も計算可能になり、異常に大きな値は後で clip する
        feat['accruals'] = (df['net_income'] - df['operating_cf']) / (df['total_assets'].abs() + epsilon)
        feat['equity_ratio'] = df['equity'] / (df['total_assets'].abs() + epsilon)
        # 1株当たり指標（分母の株式数は 0 になりにくいが念のため）
        denom_shares = df['shares_outstanding'].abs() + epsilon
        actual_bps = df['equity'] / denom_shares
        actual_eps = df['net_income'] / denom_shares
        feat['log_pbr'] = np.log(df['close'] / (actual_bps.clip(lower=0.01))) 
        filled_eps = df['eps'].combine_first(actual_eps)
        feat['earnings_yield'] = filled_eps / (df['close'] + epsilon)
        # 収益性指標
        feat['op_margin'] = df['operating_profit'] / (df['sales'].abs() + epsilon)
        feat['roe'] = df['net_income'] / (df['equity'].abs() + epsilon)
        feat['roa'] = df['net_income'] / (df['total_assets'].abs() + epsilon)
        v_t = df['eps']
        v_prev = grouped_scode['eps'].shift(20) # 1ヶ月前
        feat['revision_rate'] = (v_t - v_prev) / (0.5 * (v_t.abs() + v_prev.abs()) + epsilon)
        feat['progress_rate'] = df['operating_profit'] / (df['operating_profit_forecast'].abs() + epsilon)
        # --- 重要：最後に全特徴量をクリッピングする ---
        # 分母が極小だった場合に出る巨大な値を、ニューラルネットが壊れない範囲（例: ±5.0）に収める
        new_cols = [
            'accruals', 'equity_ratio', 'log_pbr', 'earnings_yield', 
            'op_margin', 'roe', 'roa', 'revision_rate', 'progress_rate'
        ]
        feat[new_cols] = feat[new_cols].clip(-5.0, 5.0)
        # 成長率計算
        fund_cols = ['operating_profit', 'sales', 'eps']
        temp_df = df[fund_cols + ['scode']].copy()
        temp_df[fund_cols] = temp_df.groupby('scode')[fund_cols].ffill()
        grouped = temp_df.groupby('scode')
        for col in fund_cols:
            v_t = temp_df[col]
            v_prev = grouped[col].shift(LAG_YEAR)
            feat[f'{col}_growth_yoy'] = (v_t - v_prev) / (0.5 * (v_t.abs() + v_prev.abs()) + epsilon)
            feat[f'{col}_growth_yoy'] = feat[f'{col}_growth_yoy'].clip(-3.0, 3.0)
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
        df['avg_vol_60'] = df['volume'].rolling(60).mean()
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
        feat['margin_ratio_delta_4w'] = feat['margin_ratio'].diff(20)
        return feat

    @register_block
    def _add_sector_relative_features(self, df):
        """セクター相対特徴量の追加"""
        df['Sector_Momentum_5d'] = df['sector_return'].rolling(5).mean()
        df['Sector_Rel'] = df['close'] / df['sector_return']
        df['Rel_Sector_Return_1d'] = df['Return_1d'] - df['sector_return']
        df['Rel_Sector_Return_5d'] = df['Return_1d'].rolling(5).sum() - df['sector_return'].rolling(5).sum()
        sector_ret_60 = (1 + df['sector_return']).rolling(60).apply(np.prod, raw=True) - 1
        market_ret_60 = (1 + df['Market_Return']).rolling(60).apply(np.prod, raw=True) - 1
        df['sector_rel_strength_60'] = sector_ret_60 - market_ret_60
        df['sector_short_sell_ratio'] = df['selling_volume_ratio']
        # ターゲット用
        indexer_sec = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon_tac)
        df['Sector_Return_Future'] = df['sector_return'].shift(-1).rolling(window=indexer_sec).sum()
        return df
    
    @register_block
    def _add_z_score_features(self, df):
        """Zスコア特徴量の追加"""
        z_targets = [
            'accruals','eps_growth_yoy','equity_ratio',
            'log_pbr','log_pcfr','earnings_yield',
            'div_yield','operating_profit_growth_yoy','op_margin','roa','roe',
            'sales_growth_yoy','revision_rate','progress_rate',
            'ATR_Ratio','ma_dev_25','ma_dev_75','ma_dev_200',
            'Return_20d','Return_6m','Return_12m','RSI_14',
            'high_52w_dist','downside_dev_60','volatility_60',
            'margin_buy_chg','margin_ratio','margin_buy_impact',
        ]
        # Zスコア計算後もキープする特徴量
        keep_cols = [
            'Return_20d','RSI_14','ATR_Ratio','RSI_14',
            'margin_buy_impact','margin_ratio','sales_growth_yoy'
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


    @register_block
    def _add_rank_features(self, df):
        """
        2段階ユニバースによるランク変換 (Cross-Sectional Rank)
        Tier 1 (Broad Universe) で計算し、FFillで連続性を保つ
        """
        # 1. Broad Universe (Tier 1) 定義: 計算用
        # 基準: 売買代金 1億円以上 & 株価 50円以上
        # 投資対象(Strict)が一時的に落ちてもカバーできる広さ
        tv = df['volume_p']
        mask_broad = (tv >= 100_000_000) & (df['close'] >= 50)
        # 2. ランク変換対象のカラム選定
        # 除外リスト: メタデータ、ターゲット、カテゴリ、絶対値が必要な指標
        exclude_cols = set(self.meta_cols + self.target_cols + [
            # マーケット系
            'Market_Return', 'Market_Trend_Idx', 'Market_HV_20', 'market_vol_change',
            'Market_Foreign_Z_60','Market_Individual_Z_60','Market_Foreign_Z_250','Market_Individual_Z_250','Market_Foreign_Diff',
            'market_segment',
            # 投資部門別情報
            'overseas_flow_trend', 'flow_accel',
            # カレンダー系
            'Month', 'DayOfMonth', 'DayOfWeek', 'Sin_DayOfWeek', 'Cos_DayOfWeek',
            'Is_Gotobi', 'Is_Month_End', 'Is_Quarter_End', 'Quarter', 'time_idx',
            # --- 4. セクター全体指標 [MKT] (セクター内で共通) ---
            'sector33_code',
            'Sector_Momentum_5d',   # セクター勢い [No.123]
            'sector_return',        # セクターリターン [No.124]
            'sector_short_sell_ratio', # セクター別空売り比率 [No.130]
            # --- 6. 絶対水準・規模・ショック [ABS] (ランク化で情報消失する項目) ---
            'Volume_log',           # 対数出来高 (流動性の絶対サイズ) [No.81]
            'Abnormal_Volume',      # 異常出来高比率 (ショックの大きさ) [No.82]
            'Volume_Change',        # 出来高変化率 (変化のマグニチュード) [No.83]
            'Volume_Slope_5',       # 出来高トレンド (傾きの絶対値) [No.84]
            'Log_Trading_Cap',      # 対数売買代金 (経済規模) [No.85]
            'log_vol_change',       # 対数出来高変化率 [No.86]
            'Vol_Ratio_5d',         # 出来高比率(5日) (突発的な商いの大きさ) [No.90]
            'Vol_Spike_Count_20',   # 出来高急増回数 (頻度) [No.92]
            'EPS_Actual',           # EPS実数値 (株価未調整の生データ) [No.98]
            'is_missing_eps',       # EPSが欠損している
            'log_days_since_pub',   # 決算発表後経過日数 (情報の鮮度) [No.99]
            # --- 条件付け特徴量 (絶対値を維持・既存カラムを使用) ---
            'log_market_cap', 'turnover_ratio', 
        ])
        # 数値型のみ抽出
        num_cols = df.select_dtypes(include=[np.number]).columns
        rank_cols = [c for c in num_cols if c not in exclude_cols and not c.startswith('Market_')]
        # gaussランク計算 (Broad Universe内のみ、全カラム一括)
        df_masked = df.loc[mask_broad, rank_cols]
        df_ranks_broad = df_masked.groupby(df.loc[mask_broad, 'date'])[rank_cols].rank(pct=True)
        epsilon = 1e-6
        df_ranks_broad = df_ranks_broad.clip(epsilon, 1 - epsilon)
        df_ranks_broad = np.sqrt(2) * erfinv(2 * df_ranks_broad - 1)
        df_ranks = pd.DataFrame(np.nan, index=df.index, columns=rank_cols)
        df_ranks.loc[mask_broad] = df_ranks_broad
        df_ranks['scode'] = df['scode']
        df_ranks = df_ranks.groupby('scode').ffill()
        df_ranks = df_ranks.drop(columns=['scode'], errors='ignore').fillna(0.0) 
        df[rank_cols] = df_ranks
        df = df.copy()
        return df


    def _add_targets(self, feat, df):
        """ダーゲット作成"""
        # --- ターゲット作成：戦略モデル ---
        entry_price = df['open'].shift(-1) # 翌日始値エントリー
        # インデクサ作成
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon_tac)
        # 未来データの取得
        future_high_max = df['high'].shift(-1).rolling(window=indexer).max()
        future_low_min = df['low'].shift(-1).rolling(window=indexer).min()
        future_close_end = df['close'].shift(-self.horizon_tac)
        # 基本情報格納
        feat['Entry_Price'] = entry_price
        feat['Future_High_Tac'] = future_high_max
        feat['Future_Low_Tac'] = future_low_min
        feat['Future_Close_Tac'] = future_close_end
        # --- 1. Smoothed Target (Category C) ---
        # 翌日から5日間のVWAPを計算
        # VWAP = Sum(Volume_P) / Sum(Volume)
        future_pv_sum = df['volume_p'].shift(-1).rolling(window=indexer).sum()
        future_v_sum = df['volume'].shift(-1).rolling(window=indexer).sum()
        future_vwap = future_pv_sum / (future_v_sum + 1e-9)
        # target = VWAP_5d / Entry_Price - 1
        feat['target_tac_smoothed_return'] = (future_vwap / entry_price) - 1.0
        # --- 2. Volatility-Scaled Residual (Category B) ---
        feat['target_ret_5'] = (future_close_end / entry_price.replace(0, np.nan)) - 1.0
        # Beta調整 (Market_Return_Futureは _add_market_features で作成済みと仮定)
        # もし未作成なら簡易的に df['Market_Return'].shift(-1).rolling(window=indexer).sum() を使用
        market_ret_future = df['Market_Return'].shift(-1).rolling(window=indexer).sum()
        residual_ret = feat['target_ret_5'] - (feat['Beta_60'] * market_ret_future)
        # Vol調整 (日次Vol * sqrt(5) で期間Volに換算)
        vol_5d = feat['Vol_20d'] * np.sqrt(self.horizon_tac)
        feat['target_tac_vol_scaled_residual'] = residual_ret / (vol_5d + 1e-6)
        # 既存ターゲットの維持（後方互換性のため）
        feat['target_tac_residual'] = residual_ret
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
            h1 = df['high'].shift(-1); l1 = df['low'].shift(-1)
            h2 = df['high'].shift(-2); l2 = df['low'].shift(-2)
            h3 = df['high'].shift(-3); l3 = df['low'].shift(-3)
            h4 = df['high'].shift(-4); l4 = df['low'].shift(-4)
            h5 = df['high'].shift(-5); l5 = df['low'].shift(-5)
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
        feat['target_tac_tb_strategy_a'] = calc_triple_barrier(1.0, 1.0)
        # Strategy B: Trend (1.5σ / 0.75σ) - 損小利大
        feat['target_tac_tb_strategy_b'] = calc_triple_barrier(1.5, 0.75)
        # Strategy C: Reversion (0.5σ / 1.0σ) - 高勝率
        feat['target_tac_tb_strategy_c'] = calc_triple_barrier(0.5, 1.0)
        
        # --- ターゲット作成：戦略モデル ---
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon_str)
        future_high_max = df['high'].shift(-1).rolling(window=indexer).max()
        future_low_min = df['low'].shift(-1).rolling(window=indexer).min()
        future_close_end = df['close'].shift(-self.horizon_str)
        feat['Entry_Price'] = entry_price
        feat['Future_High_Str'] = future_high_max
        feat['Future_Low_Str'] = future_low_min
        feat['Future_Close_Str'] = future_close_end
        # 60日累積リターン（基本値） RankやPeer Alphaの計算ベースとして後続のクロスセクション処理で使用
        feat['target_ret_60'] = (df['close'].shift(-self.horizon_str) / entry_price.replace(0, np.nan)) - 1.0
        # Risk-Adjusted Residual Momentum (60d) ベータ調整済みリターンをボラティリティで標準化
        market_ret_60 = df['Market_Return'].shift(-1).rolling(window=indexer).sum()
        residual_60 = feat['target_ret_60'] - (feat['Beta_60'] * market_ret_60)
        feat['target_str_risk_adj'] = residual_60 / (feat['Vol_20d'] * np.sqrt(12) + 1e-6) # 20日Volを60日換算(sqrt(3)近似だが実務上Vol_20dで正規化も一般的)
        # Return Consistency Score (60d) 60日間の累積リターン曲線の直線性をR2で算出
        def _calc_consistency(window):
            if np.isnan(window).any(): return np.nan
            cum_ret = np.cumprod(1 + window)
            x = np.arange(len(cum_ret))
            return np.corrcoef(x, cum_ret)[0, 1]**2
        fwd_ret_1d = df['close'].pct_change().shift(-1)
        feat['target_str_consistency'] = fwd_ret_1d.rolling(window=indexer).apply(_calc_consistency, raw=True)
        # Volatility Scaling Alpha (60d) 銘柄固有のボラティリティでスケーリング
        feat['target_str_vol_scale'] = feat['target_ret_60'] / (feat['volatility_60'] + 1e-6)
        # Triple Barrier Method 
        # 3値分類ラベル: 1(利確), -1(損切), 0(時間切れ)
        # バリア幅の設定: ボラティリティベース (De Prado流)
        # 上値(PT) = 期間ボラティリティ * 1.0
        # 下値(SL) = 期間ボラティリティ * 1.0 (損益比率1:1の設定)
        vol_horizon = feat['Vol_20d'] * np.sqrt(self.horizon_str)
        pt_width = vol_horizon * 1.0
        sl_width = vol_horizon * 1.0
        # 高速化のためのNumpy配列化
        high_vals = df['high'].values
        low_vals = df['low'].values
        entry_vals = entry_price.values
        pt_vals = pt_width.values
        sl_vals = sl_width.values
        labels = np.zeros(len(df)) # デフォルト0 (Time-out)
        # 60日間のウィンドウ走査（ループ処理）
        # ※PandasのRollingのみでの「First Touch」判定は困難なため、Numpyループを使用
        horizon = self.horizon_str
        n_samples = len(df)
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
        feat['target_str_triple_barrier'] = labels
        # 不要変数の削除
        feat.drop(columns=['Vol_20d','Market_Return_Future'], inplace=True)
        self._feature_registry.pop('Vol_20d', None)
        self._feature_registry.pop('Market_Return_Future', None)
        return feat

    def _add_cross_sectional_target(self, df):
        """クロスセクションターゲットの追加"""
        # --- 1. Era-wise Rank (Category A) ---
        # 単純なRank (0.0 ~ 1.0)
        df['target_tac_rank'] = df.groupby('date')['target_ret_5'].rank(pct=True, method='average')
        # 既存: Gauss Rank (正規分布化)
        for date, group in df.groupby('date'):
            rank = group['target_ret_5'].rank(pct=True, method='average')
            epsilon = 1e-6
            rank = rank * (1 - 2 * epsilon) + epsilon
            df.loc[group.index, 'target_tac_gauss_rank'] = erfinv(2 * rank - 1)
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
        sec_ret_fut = df['sector_return'].shift(-1).rolling(window=indexer_sec).sum()
        mkt_ret_fut = df['Market_Return'].shift(-1).rolling(window=indexer_sec).sum()
        # 2. 残差 = Target_Return - (0.5 * Market + 0.5 * Sector) ※係数は簡易
        # より厳密には、日次で回帰係数を決めるのが良いが、ここでは
        # 「市場とセクターの影響を引いたもの」をLinear Residualの代替とする
        df['target_tac_linear_residual'] = df['target_ret_5'] - (0.5 * mkt_ret_fut + 0.5 * sec_ret_fut)
        # セクター相対フラグ
        df['target_tac_sector_relative'] = (df['target_ret_5'] > sec_ret_fut).astype(int)
        # --- 戦略モデル用クロスセクション ---
        # Relative Rank Change (60d)
        df['target_str_rank'] = df.groupby('date')['target_ret_60'].rank(pct=True)
        # Peer Group Neutralized Alpha (60d)
        sector_mean_60 = df.groupby(['date', 'sector33_code'])['target_ret_60'].transform('mean')
        df['target_str_peer_alpha'] = df['target_ret_60'] - sector_mean_60
        # 不要変数の削除
        df.drop(columns=['Sector_Return_Future'], inplace=True)
        self._feature_registry.pop('Sector_Return_Future', None)
        df.drop(columns=['target_ret_60'], inplace=True)
        self._feature_registry.pop('target_ret_60', None)
        return df

    def _fill_missing_values_with_sector_median(self, df):
        """指定カラムの欠損を業種別中央値で埋める"""
        target_cols = ['EPS_Actual', 'turnover_ratio', 'log_market_cap']
        if 'sector33_code' in df.columns:
            for col in target_cols:
                if col in df.columns:
                    sector_median = df.groupby(['date', 'sector33_code'])[col].transform('median')
                    df[col] = df[col].fillna(sector_median)
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
        df = self._fill_missing_values_with_sector_median(df)
        # 以下の処理は、入力データフレームのカラムに追加するのみであり、入力データフレームにあるカラムは全て出力する
        df = self._add_sector_relative_features(df)
        df = self._add_z_score_features(df)
        df = self._add_rank_features(df)
        if output_target:
            df = self._add_cross_sectional_target(df)
        return df 

    @property
    def feature_list(self):
        """自動登録された全特徴量リストを返す"""
        return list(self._feature_registry.keys())
    
