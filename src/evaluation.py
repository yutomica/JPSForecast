import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from scipy.stats import spearmanr

def evaluate_metrics(y_true, y_pred, task_type='regression'):
    """基本メトリクスの算出"""
    metrics = {}
    if task_type == 'regression':
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        # 金融MLで重要なIC(ランク相関)を追加
        metrics['ic'], _ = spearmanr(y_true, y_pred)
    else:
        metrics['auc'] = roc_auc_score(y_true, y_pred)
    return metrics

def calculate_bin_stats(df_eval, score_col, target_col, task_type='regression',metadata_cols=None, n_bins=10):
    df_eval = df_eval.copy()
    """ビン分析スクリプト"""
    if task_type == 'regression':
        # 1. 回帰: 等頻度で分割
        # データの偏りがあっても各ビンに同程度のサンプル数が入る
        df_eval['bin_obj'] = pd.qcut(df_eval[score_col], n_bins, duplicates='drop')
        # 表示名を作成: "実測最小値 - 実測最大値"
        bin_ranges = df_eval.groupby('bin_obj', observed=True)[score_col].agg(['min', 'max'])
        label_map = {
            interval: f"{row['min']:.4f} - {row['max']:.4f}"
            for interval, row in bin_ranges.iterrows()
        }
        df_eval['bin_label'] = df_eval['bin_obj'].map(label_map)
    else:
        # 2. 分類: 0.0〜1.0 を等間隔（10%刻み）で分割
        # スコアの分布に依らず、固定の確率帯で評価する
        bins = np.linspace(0, 1, n_bins + 1)
        df_eval['bin_obj'] = pd.cut(df_eval[score_col], bins=bins, include_lowest=True)
        # 表示名を作成: "0.1 - 0.2" 等の固定形式
        label_map = {
            interval: f"{interval.left:.1f} - {interval.right:.1f}"
            for interval in df_eval['bin_obj'].cat.categories
        }
        df_eval['bin_label'] = df_eval['bin_obj'].map(label_map)
    # 集計処理
    # サンプル数
    stats = df_eval.groupby('bin_label', observed=True).size().to_frame(name='sample_count')
    # 表示順を元のスコア順（bin_objの順序）に合わせる
    sort_order = df_eval.groupby('bin_label', observed=True)['bin_obj'].first().sort_values().index
    stats = stats.reindex(sort_order)
    # ターゲット平均
    stats['target_mean'] = df_eval.groupby('bin_label', observed=True)[target_col].mean()
    # メタデータ集計 (Future_High/Low/Close 等)
    if metadata_cols:
        for col in metadata_cols:
            if col in df_eval.columns:
                grp = df_eval.groupby('bin_label', observed=True)[col]
                stats[f'{col}_mean'] = grp.mean()
                stats[f'{col}_std'] = grp.std()
                # 分位点算出
                for q in [0.05, 0.1, 0.5, 0.9, 0.95]:
                    stats[f'{col}_q{int(q*100)}'] = grp.quantile(q)
    return stats

def calculate_equity_curve(df_eval, date_col, score_col, target_col, top_n=50):
    """簡易バックテスト：予測上位N銘柄の累積リターン"""
    # 日付ごとに予測上位N銘柄を抽出
    daily_returns = df_eval.groupby(date_col).apply(
        lambda x: x.nlargest(top_n, score_col)[target_col].mean()
    )
    equity_curve = (1 + daily_returns).cumprod()
    return equity_curve