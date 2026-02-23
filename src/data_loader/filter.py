import pandas as pd
import numpy as np

class RuleBasedFilter:
    def apply(self, df):
        if df.empty: return df
        c1 = df['volume_p_MA5'] > 300_000_000
        c2 = df['Dist_SMA25'] > 0
        c3 = df['Dist_SMA25_75'] > 0
        c5 = df['ATR_Ratio'] > 0.01 
        df['is_candidate_tac'] = c1 & c2 & c3 & c5
        return df


class RuleBasedFilter_STR:
    """
    戦略モデル構築仕様書 5.A に基づくユニバース選定フィルタ
    Attributes:
        min_trading_value (float): 最小売買代金 (デフォルト: 1億円)
        min_market_cap (float): 最小時価総額 (デフォルト: 50億円)
    """
    def __init__(self, min_trading_value=100_000_000, min_market_cap=5_000_000_000):
        self.min_trading_value = min_trading_value
        self.min_market_cap = min_market_cap
    def apply(self, df):
        """
        データフレームに対し、流動性と時価総額によるフィルタリングを適用する。
        Args:
            df (pd.DataFrame): 特徴量エンジニアリング済みのデータ
                               (date, scode, volume_p, close, shares_outstanding を含む)
        Returns:
            pd.DataFrame: 条件を満たす行のみを抽出したデータ
        """
        if df.empty:
            return df
        # データのコピー（警告回避）
        df_filtered = df.copy()
        # ----------------------------------------------------
        # 1. 売買代金フィルタ (Trading Value)
        # ----------------------------------------------------
        # 売買代金カラムの特定
        if 'volume_p' in df_filtered.columns:
            series_val = df_filtered['volume_p']
        elif 'volume' in df_filtered.columns and 'close' in df_filtered.columns:
            series_val = df_filtered['close'] * df_filtered['volume']
        else:
            # 計算不能な場合はフィルタを通さない（全削除）か、警告を出してスルーするか
            # ここでは安全側に倒して全削除
            return df_filtered.iloc[0:0]
        # 60日移動平均売買代金を計算
        # 入力が単一銘柄(Feature_Engineerの出力)であることを想定
        # もし複数銘柄が混ざる可能性があるなら groupby('scode') が必要だが、
        # create_data_STR.pyのループ構造上、ここは単一銘柄で来るはず。
        # 念のため scode が複数あるかチェックして分岐
        if df_filtered['scode'].nunique() > 1:
            avg_trading_value = df_filtered.groupby('scode')['volume_p'].transform(
                lambda x: x.rolling(60).mean()
            )
        else:
            avg_trading_value = series_val.rolling(60).mean()
        # ----------------------------------------------------
        # 2. 時価総額フィルタ (Market Cap)
        # ----------------------------------------------------
        if 'shares_outstanding' in df_filtered.columns:
            market_cap = df_filtered['close'] * df_filtered['shares_outstanding']
        elif 'log_market_cap' in df_filtered.columns:
            # log_mcap から復元 (e^x)
            market_cap = np.exp(df_filtered['log_market_cap'])
        else:
            # データがない場合はフィルタできないのでinf扱い（通過させる）
            market_cap = pd.Series(float('inf'), index=df_filtered.index)
        # ----------------------------------------------------
        # 3. フィルタ適用
        # ----------------------------------------------------
        # 条件作成 (欠損値はFalse扱い)
        mask_liquidity = (avg_trading_value >= self.min_trading_value).fillna(False)
        mask_cap = (market_cap >= self.min_market_cap).fillna(False)
        # ターゲットチェック (学習データ作成時用)
        # ターゲット列が存在する場合のみ、NaNでないことを条件に加える
        # 最新仕様: target_reg, target_cls
        mask_target = pd.Series(True, index=df_filtered.index)
        if 'target_reg' in df_filtered.columns:
            mask_target &= df_filtered['target_reg'].notna()
        if 'target_cls' in df_filtered.columns:
            mask_target &= df_filtered['target_cls'].notna()
        # 最終フラグ
        df_filtered['is_candidate_str'] = mask_liquidity & mask_cap & mask_target
        return df_filtered
