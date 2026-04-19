import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, ndcg_score, average_precision_score
from scipy.stats import spearmanr

def evaluate_metrics(y_true, y_pred, y_ret=None, task_type='regression', target_col=None, dates=None, ndcg_k=10):
    """基本メトリクスの算出"""
    # --- 入力からNaNデータを除外 ---
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid_mask = pd.notna(y_true) & pd.notna(y_pred)
    
    if y_ret is not None:
        y_ret = np.asarray(y_ret)
        valid_mask &= pd.notna(y_ret)
        
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    if y_ret is not None:
        y_ret = y_ret[valid_mask]
    if dates is not None:
        dates = np.asarray(dates)[valid_mask]
        
    if len(y_true) == 0:
        return {'ic': np.nan, 'rmse': np.nan, 'logloss': np.nan, 'auc': np.nan}

    metrics = {}
    # 金融MLで重要なIC(ランク相関)をタスクに関わらず追加
    # y_ret が指定されている場合は予測スコアと生リターン(y_ret)の相関を計算する
    if target_col == 'target_tac_vol_scaled_residual' and y_ret is not None:
        target_ic, _ = spearmanr(y_true, y_pred)
        raw_return_ic, _ = spearmanr(y_ret, y_pred)
        metrics['ic'] = 0.3 * target_ic + 0.7 * raw_return_ic
    else:
        target_for_ic = y_ret if y_ret is not None else y_true
        metrics['ic'], _ = spearmanr(target_for_ic, y_pred)
    
    if task_type == 'classification':
        metrics['auc'] = roc_auc_score(y_true, y_pred)
    elif task_type == 'multiclass':
        try:
            # スコア化（負の値を含む）されている場合はエラーを回避
            metrics['logloss'] = log_loss(y_true, y_pred)
        except ValueError:
            metrics['logloss'] = np.nan
    else:
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
    if y_ret is not None:
        # Severe Drawdown Recall の計算
        actual_threshold = -0.05
        pred_threshold = np.percentile(y_pred, 20)
        
        actual_severe = (y_ret <= actual_threshold)
        pred_alert = (y_pred <= pred_threshold)
        
        tp = np.sum(actual_severe & pred_alert)
        fn = np.sum(actual_severe & ~pred_alert)
        
        if (tp + fn) > 0:
            metrics['severe_drawdown_recall'] = float(tp / (tp + fn))
        else:
            metrics['severe_drawdown_recall'] = np.nan
            
    # AP_severe の計算 (5pt, 7pt, 10pt)
    target_for_ap = y_ret if y_ret is not None else y_true
    ap_scores = []
    for pt in [0.05, 0.07, 0.10]:
        binary_true = (target_for_ap <= -pt).astype(int)
        if np.sum(binary_true) > 0:
            # リターン予測の場合、予測値が低いほど下落になりやすいので符号を反転させる
            score_for_ap = -y_pred if task_type == 'regression' else y_pred
            ap = average_precision_score(binary_true, score_for_ap)
            ap_scores.append(ap)
    if ap_scores:
        metrics['AP_severe'] = float(np.mean(ap_scores))
    else:
        metrics['AP_severe'] = np.nan
            
    # 日付ごとの指標 (Top10-spread, NDCG@K, RankIC_reb, Recall@Gate30%)
    if dates is not None:
        df_cols = {'date': dates, 'pred': y_pred, 'true': y_true}
        if y_ret is not None:
            df_cols['ret'] = y_ret
        df_tmp = pd.DataFrame(df_cols)
        
        # groupbyのキーとrebalance_datesの型を一致させるため、日付(date)型に統一
        df_tmp['date'] = pd.to_datetime(df_tmp['date']).dt.date
        
        spreads = []
        ndcgs = []
        rank_ics_reb = []
        recalls_gate30 = []
        
        unique_dates = np.sort(df_tmp['date'].unique())
        rebalance_dates = set(unique_dates[::5])
        
        for d, grp in df_tmp.groupby('date'):
            if y_ret is not None and len(grp) >= 10:
                top10_ret = grp.nlargest(10, 'pred')['ret'].mean()
                univ_ret = grp['ret'].mean()
                spreads.append(top10_ret - univ_ret)
                
            if len(grp) >= ndcg_k:
                y_true_g = grp['true'].values
                # ndcg_scoreは関連度が非負である必要があるため最小値を引く
                rel = y_true_g - np.min(y_true_g)
                if np.max(rel) > 0:
                    try:
                        score = ndcg_score([rel], [grp['pred'].values], k=ndcg_k)
                        ndcgs.append(score)
                    except ValueError:
                        pass
                        
            # RankIC_reb: リバランス日のみ
            target_g = grp['ret'].values if y_ret is not None else grp['true'].values
            if d in rebalance_dates and len(grp) > 1:
                ic, _ = spearmanr(target_g, grp['pred'].values)
                if not np.isnan(ic):
                    rank_ics_reb.append(ic)
                    
            # Recall_Gate30pct: 重大イベント（<=-15%）の検出
            target_series = grp['ret'] if y_ret is not None else grp['true']
            mines = (target_series <= -0.15)
            num_mines = mines.sum()
            if num_mines > 0:
                if task_type == 'regression':
                    # 回帰（リターン予測）の場合、予測値が低い（昇順）＝リスクが高い
                    risk_order = grp['pred'].sort_values(ascending=True).index
                else:
                    risk_order = grp['pred'].sort_values(ascending=False).index
                    
                k = int(len(grp) * 0.3)
                if k > 0:
                    gate_indices = risk_order[:k]
                    caught_mines = mines.loc[gate_indices].sum()
                    recalls_gate30.append(float(caught_mines / num_mines))
                else:
                    recalls_gate30.append(0.0)

        if y_ret is not None:
            if spreads:
                metrics['top10_spread'] = float(np.mean(spreads))
            else:
                metrics['top10_spread'] = np.nan
                
        if ndcgs:
            metrics[f'ndcg_{ndcg_k}'] = float(np.mean(ndcgs))
        else:
            metrics[f'ndcg_{ndcg_k}'] = np.nan
            
        if rank_ics_reb:
            metrics['RankIC_reb'] = float(np.mean(rank_ics_reb))
        else:
            metrics['RankIC_reb'] = np.nan
            
        if recalls_gate30:
            metrics['Recall_Gate30pct'] = float(np.mean(recalls_gate30))
        else:
            metrics['Recall_Gate30pct'] = np.nan
                
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