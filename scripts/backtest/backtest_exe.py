import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def evaluate_model_performance(df, score_col, target_col, n_bins=5, horizon=5, save_dir=None):
    """
    Rank ICの算出およびクオンタイル分析を実行する
    Args:
        df (pd.DataFrame): 'date', 'scode', score_col, target_col を含むDF
        score_col (str): モデルの予測スコア列名
        target_col (str): リターン（価格比）の列名 (例: Future_Close_Tac)
        n_bins (int): クオンタイル分割数
        horizon (int): 予測期間 (5 or 60)
        save_dir (str, optional): グラフを保存するディレクトリのパス
    Returns:
        pd.DataFrame: 日付ごとの対象件数、daily_ic、各クオンタイルの累積リターンを含む明細データフレーム
    """
    df = df[['date', 'scode', score_col, target_col]].dropna(subset=[score_col, target_col]).copy()
    # 前処理: リターンをログリターンに変換
    df['log_return'] = np.log(df[target_col])

    # --- 日別スコア上位10銘柄の評価 ---
    print(f"🏆 Calculating Top 10 Metrics for {score_col}...")
    def get_top10_metrics(group):
        top10 = group.nlargest(10, score_col)
        return pd.Series({
            'top10_min_score': top10[score_col].min(),
            'top10_mean_return': top10[target_col].mean()
        })
    top10_metrics = df.groupby('date').apply(get_top10_metrics)

    # Rank IC の算出 
    print(f"📊 Calculating Rank IC for {score_col}...")
    def calc_daily_ic(group):
        # スピアマンの順位相関係数
        ic, _ = stats.spearmanr(group[score_col], group['log_return'])
        return ic
    daily_ic = df.groupby('date').apply(calc_daily_ic)
    mean_ic = daily_ic.mean()
    icir = daily_ic.mean() / daily_ic.std()
    t_stat, p_value = stats.ttest_1samp(daily_ic.dropna(), 0)
    print(f"  Mean IC: {mean_ic:.4f}")
    print(f"  ICIR:    {icir:.4f}")
    print(f"  t-stat:  {t_stat:.4f} (p-value: {p_value:.4f})")
    
    # クオンタイル分析
    print(f"📈 Calculating Quantile Analysis (bins={n_bins})...")
    # 日次でスコアに基づきグループ分け (0: 最弱, n_bins-1: 最強)
    df['quantile'] = df.groupby('date')[score_col].transform(
        lambda x: pd.qcut(x, n_bins, labels=False, duplicates='drop')
    )
    # グループごとの日次平均ログリターン
    # 各スコアリング日時点での「その後のn日間のリターン」の平均
    daily_group_ret = df.groupby(['date', 'quantile'])['log_return'].mean().unstack()
    # 累積ログリターンの算出
    # 予測期間の重複（オーバーラップ）を考慮し、horizonで割ることで1日あたりの期待値に補正
    cum_log_ret = daily_group_ret.cumsum() / horizon
    
    # --- 4. 可視化 ---
    daily_count = df.groupby('date').size().rename('target_count')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
    # ICの推移
    daily_ic.rolling(window=20).mean().plot(ax=ax1, color='tab:blue', alpha=0.8)
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title(f"Rank IC (20-day Rolling Mean) - {score_col}")
    ax1.set_ylabel("IC")
    # 累積リターンの推移
    for i in range(n_bins):
        label = f"Q{i+1} (Top)" if i == n_bins-1 else f"Q{i+1}"
        cum_log_ret[i].plot(ax=ax2, label=label, linewidth=2)
    ax2.set_title(f"Cumulative Log Returns by Quantile (Scaled by {horizon}d)")
    ax2.set_ylabel("Cumulative Log Return")
    ax2.legend(loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    # 対象件数と生ICの推移
    ax3.bar(daily_count.index, daily_count.values, color='tab:gray', alpha=0.3, width=1.0, label='Target Count')
    ax3.set_ylabel("Target Count")
    ax3.set_title(f"Raw Rank IC and Target Count - {score_col}")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(daily_ic.index, daily_ic.values, color='tab:red', alpha=0.6, linewidth=1, label='Raw Rank IC')
    ax3_twin.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3_twin.set_ylabel("Raw Rank IC")
    
    lines_1, labels_1 = ax3.get_legend_handles_labels()
    lines_2, labels_2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # 上位10銘柄の推移
    ax4.plot(top10_metrics.index, top10_metrics['top10_min_score'], color='tab:purple', label='Top 10 Min Score')
    ax4.set_ylabel("Min Score (Top 10)")
    ax4.set_title(f"Top 10 Min Score and Mean Return - {score_col}")
    ax4_twin = ax4.twinx()
    ax4_twin.plot(top10_metrics.index, top10_metrics['top10_mean_return'], color='tab:orange', alpha=0.8, label='Top 10 Mean Return')
    ax4_twin.set_ylabel(f"Mean Return ({target_col})")
    
    lines_3, labels_3 = ax4.get_legend_handles_labels()
    lines_4, labels_4 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper left')

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"evaluation_{score_col}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Plot saved to {save_path}")
    plt.show()

    # --- 5. 結果の結合と返却 ---
    daily_ic.name = 'daily_ic'
    cum_log_ret_renamed = cum_log_ret.copy()
    cum_log_ret_renamed.columns = [f'cum_log_ret_Q{i+1}' for i in range(n_bins)]
    result_df = pd.concat([daily_count, daily_ic, cum_log_ret_renamed, top10_metrics], axis=1).reset_index()
    return result_df

# --- 実行例 ---
# df = pd.read_parquet("your_data.parquet")
# result_df = evaluate_model_performance(
#     df, 
#     score_col='score_target_tac_rank_LGBM', 
#     target_col='Future_Close_Tac', 
#     horizon=5
# )