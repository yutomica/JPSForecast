
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import gc
from scipy.special import erfinv
from pathlib import Path
from src.data_loader.loader import DataLoader
from src.features.engineer import FeatureEngineer
from src.data_loader.filter import RuleBasedFilter, RuleBasedFilter_STR
from src.data_loader.get_sector_code_from_JQuants import get_sector_master_from_api

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / 'data/intermediate'
# J-Quants 認証情報 (環境変数推奨)
JQ_MAIL = os.environ.get('JQ_MAIL', '') 
JQ_PASS = os.environ.get('JQ_PASS', '')

## 戦略モデル用のターゲット作成
def create_orthogonalized_target(df_all_prices, df_fins, df_sector_master, df_topix):
    print("=== Phase 1: Creating Targets via Cross-Sectional Regression ===")
    # 1. 回帰に必要なデータを結合 (全銘柄・全期間)
    #    Close, Financials(Shares, BPS), Sector, TopixBeta
    #    ※ df_all_prices は [date, scode, close]
    # マージ用にソート
    df_univ = df_all_prices.sort_values('date')
    df_fins_sorted = df_fins.sort_values('published_date')
    # 財務データの結合 (asof merge)
    df_univ = pd.merge_asof(
        df_univ, 
        df_fins_sorted[['scode', 'published_date', 'bps', 'shares_outstanding']], 
        left_on='date', 
        right_on='published_date', 
        by='scode', 
        direction='backward'
    )
    # セクター結合
    df_univ = df_univ.merge(df_sector_master[['scode', 'sector33_code']], on='scode', how='left')
    # 必要なパラメータ計算
    # 60日後リターン (目的変数)
    df_univ['return_60d'] = df_univ.groupby('scode')['close'].shift(-60) / df_univ['close'] - 1
    # 60日ボラティリティ (リスク調整用)
    df_univ['log_ret'] = df_univ.groupby('scode')['close'].pct_change()
    df_univ['volatility_60d'] = df_univ.groupby('scode')['log_ret'].transform(lambda x: x.rolling(60).std())
    # 時価総額 (対数)
    df_univ['log_mcap'] = np.log(df_univ['close'] * df_univ['shares_outstanding'])
    # PBR (対数)
    df_univ['log_pbr'] = np.log(df_univ['close'] / df_univ['bps'])
    # ---------------------------------------------------------
    # Pre-Phase 1: Market Beta Calculation (Using TOPIX)
    # ---------------------------------------------------------
    print("Calculating Market Beta using TOPIX...")
    # 1. TOPIXのリターンと分散を計算
    #    df_topix は create_data_STR.py 内で既にロードされている前提
    df_mkt = df_topix[['date', 'close']].copy().sort_values('date')
    df_mkt['mkt_ret'] = df_mkt['close'].pct_change()
    # Beta計算の分母となる市場分散 (Rolling 60days)
    df_mkt['mkt_var'] = df_mkt['mkt_ret'].rolling(60).var()
    # 必要な列だけにして df_univ に結合
    # df_univ は全銘柄データ
    df_univ = df_univ.merge(df_mkt[['date', 'mkt_ret', 'mkt_var']], on='date', how='left')
    # 2. 個別銘柄のリターン計算
    df_univ = df_univ.sort_values(['scode', 'date'])
    df_univ['ret'] = df_univ.groupby('scode')['close'].pct_change()
    # 3. 共分散(Cov)の計算とBeta算出
    print("  - Computing rolling covariance...")
    grouped = df_univ.groupby('scode')
    # Rolling Covariance: Cov(Stock, Market)
    # pandasのrolling().cov()を使用
    def calc_rolling_cov(x):
        # 欠損等でズレないよう、同一DataFrame内の列同士で計算
        return x['ret'].rolling(60).cov(x['mkt_ret'])
    cov_series = grouped.apply(calc_rolling_cov)
    # MultiIndexの調整 (groupby.applyの結果がMultiIndexになる場合への対応)
    if isinstance(cov_series.index, pd.MultiIndex):
        cov_series = cov_series.reset_index(level=0, drop=True)
    # 元のDataFrameに結合
    df_univ['cov_stock_mkt'] = cov_series
    # Beta = Cov / Var
    df_univ['market_beta'] = df_univ['cov_stock_mkt'] / df_univ['mkt_var']
    # 計算に使用した一時カラムの削除
    df_univ.drop(columns=['ret', 'mkt_ret', 'mkt_var', 'cov_stock_mkt'], inplace=True)
    print("Market Beta calculation complete.")
    print("Starting Cross-Sectional Orthogonalization...")
    # 欠損がある行は回帰できないので一時的に除外（または埋める）
    # 回帰に必要なカラム
    reg_cols = ['return_60d', 'market_beta', 'log_mcap', 'log_pbr']
    # create_orthogonalized_target メソッド内
    # (NOTE:以下ターゲットの欠損率を下げる必要がある場合に実行)欠損がある行を捨てる前に、平均値や固定値で埋める
    # df_univ['market_beta'] = df_univ['market_beta'].fillna(1.0) # Beta不明なら市場連動とみなす
    # df_univ['log_pbr'] = df_univ['log_pbr'].fillna(df_univ['log_pbr'].mean()) # 全体平均で埋める（荒っぽい手法）
    # df_univ['log_mcap'] = df_univ['log_mcap'].fillna(df_univ['log_mcap'].median())
    df_valid = df_univ.dropna(subset=reg_cols + ['sector33_code']).copy()
    del df_univ
    gc.collect()
    # セクターダミーの作成
    df_valid = pd.get_dummies(df_valid, columns=['sector33_code'], prefix='sec', drop_first=True)
    dummy_cols = [c for c in df_valid.columns if c.startswith('sec_')]
    # 説明変数リスト
    X_cols = ['market_beta', 'log_mcap', 'log_pbr'] + dummy_cols
    # 結果格納用リスト
    results = []
    # 日次でループ (LinearRegressionは高速なのでループでも十分速い)
    dates = df_valid['date'].unique()
    model = LinearRegression()
    for d in dates:
        # その日のデータを抽出
        day_data = df_valid[df_valid['date'] == d]
        if len(day_data) < 100: # 銘柄数が少なすぎる日はスキップ
            continue
        X = day_data[X_cols]
        y = day_data['return_60d']
        # 回帰実行
        model.fit(X, y)
        # 残差(Alpha)計算
        residuals = y - model.predict(X)
        # リスク調整 (Residual / Volatility)
        # Volatilityが極端に小さい場合のエラー回避
        vol = day_data['volatility_60d'].replace(0, np.nan)
        risk_adj_alpha = residuals / vol
        # 外れ値クリッピング (-5 ~ 5 sigma程度)
        risk_adj_alpha = risk_adj_alpha.clip(-5, 5)
        # 保存用にIDと値を保持
        res_df = pd.DataFrame({
            'date': d,
            'scode': day_data['scode'],
            'raw_resid': residuals,
            'target_reg': risk_adj_alpha
        })
        results.append(res_df)
    if not results:
        return pd.DataFrame()
    df_res = pd.concat(results, ignore_index=True)
    # ---------------------------------------------------------
    # ランク化 & サブターゲット(分類)作成
    # ---------------------------------------------------------
    # 日次でランク化 (pct=True で 0.0~1.0 に正規化)
    df_res['rank_score'] = df_res.groupby('date')['target_reg'].rank(pct=True)
    # サブターゲット: 上位20%なら1, それ以外0
    df_res['target_cls'] = (df_res['rank_score'] >= 0.8).astype(int)
    # メインターゲット: rank_score を採用するか、risk_adj_alpha(生値)を採用するか
    # 仕様書に基づき「スコア」として学習させるなら rank_score の方が安定する
    # ここでは rank_score を target_reg として上書き採用する
    df_res['target_reg'] = df_res['rank_score']
    return df_res[['date', 'scode', 'target_reg', 'target_cls']]

def main(output_dir):
    jq_mail = JQ_MAIL
    jq_pass = JQ_PASS
    loader = DataLoader()
    os.makedirs(output_dir, exist_ok=True)

    df_sector_master = get_sector_master_from_api(jq_mail, jq_pass)
    df_sector_indices = pd.DataFrame()
    df_all_prices = loader.fetch_all_close_data()
    df_merged = df_all_prices.merge(df_sector_master[['scode', 'sector33_code']], on='scode', how='inner')
    df_merged['pct_change'] = df_merged.groupby('scode')['close'].pct_change()
    df_sector_indices = df_merged.groupby(['date', 'sector33_code'])['pct_change'].mean().reset_index()
    df_sector_indices.rename(columns={'pct_change': 'sector_return'}, inplace=True)
    print(f"Sector indices created. Rows: {len(df_sector_indices)}")
    df_topix = loader.fetch_topix_data()
    df_fins = loader.fetch_financial()

    # ---------------------------------------------------------
    # ターゲットの一括生成 (回帰分析)
    # ---------------------------------------------------------
    df_targets = create_orthogonalized_target(df_all_prices,df_fins,df_sector_master,df_topix)
    print(f"Target Created. Shape: {df_targets.shape}")

    # 保存
    output_path = os.path.join(output_dir, f'orthogonalized_targets.parquet')
    df_targets.to_parquet(output_path, index=False)
    print(f"Orthogonalized targets saved to {output_path}")

if __name__ == "__main__":
    main(OUTPUT_DIR)
    
    


