import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, average_precision_score
from scipy.stats import spearmanr

def _group_by_date(y_true, y_pred, dates):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'date': dates})
    return df.groupby('date')

def _reduce_multiclass_to_score(yp):
    if yp.ndim == 2:
        # Reduce probabilities to expected class index [0, 1, 2, 3...]
        return np.dot(yp, np.arange(yp.shape[1]))
    return yp

# =========================================================================
# 汎用 NumPy 関数 (TCN, FTT 等で early_stopping_metric に指定可能)
# =========================================================================

def calc_ndcg_10(y_true, y_pred, dates=None):
    """
    ① TAC攻め用: NDCG@10
    日付ごとに予測スコア上位の質を評価し、その平均値を返します。
    """
    y_pred_1d = _reduce_multiclass_to_score(y_pred)
    
    if dates is None:
        if len(y_true) < 2: return 0.0
        rel = np.maximum(0, y_true)
        if np.max(rel) <= 0: return 0.0
        return ndcg_score([rel], [y_pred_1d], k=10)
    
    scores = []
    for _, group in _group_by_date(y_true, y_pred_1d, dates):
        if len(group) < 2: continue
        rel = np.maximum(0, group['y_true'].values)
        if np.max(rel) > 0:
            scores.append(ndcg_score([rel], [group['y_pred'].values], k=10))
    return np.mean(scores) if scores else 0.0

def calc_ap_severe(y_true, y_pred, dates=None):
    """
    ② TAC守り用: Average Precision (Severe Drop)
    5%, 7%, 10% の暴落閾値に対するAPを計算し、その平均を返します。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    ap_scores = []
    pts = [0.05, 0.07, 0.10]
    
    if y_pred.ndim == 2 and y_pred.shape[1] >= 2:
        # Multiclass probability logic
        p_5  = np.sum(y_pred[:, 1:], axis=1)
        p_7  = np.sum(y_pred[:, 2:], axis=1) if y_pred.shape[1] >= 3 else np.zeros(len(y_pred))
        p_10 = np.sum(y_pred[:, 3:], axis=1) if y_pred.shape[1] >= 4 else np.zeros(len(y_pred))
        
        # Labels
        if np.nanmin(y_true) < 0: # Raw returns
            targets = [ (y_true <= -pt).astype(int) for pt in pts ]
        else: # Class IDs [0, 1, 2, 3]
            targets = [ (y_true >= 1).astype(int), (y_true >= 2).astype(int), (y_true >= 3).astype(int) ]
            
        for yt, yp in zip(targets, [p_5, p_7, p_10]):
            if np.sum(yt) > 0 and len(np.unique(yt)) > 1:
                ap_scores.append(average_precision_score(yt, yp))
    else:
        # Scalar score
        if np.nanmin(y_true) < 0: # Raw returns
            targets = [ (y_true <= -pt).astype(int) for pt in pts ]
            score_for_ap = -y_pred # lower return is higher risk
        else: # Class IDs
            targets = [ (y_true >= 1).astype(int), (y_true >= 2).astype(int), (y_true >= 3).astype(int) ]
            score_for_ap = y_pred # higher index is higher risk
            
        for yt in targets:
            if np.sum(yt) > 0 and len(np.unique(yt)) > 1:
                ap_scores.append(average_precision_score(yt, score_for_ap))
            
    return np.mean(ap_scores) if ap_scores else 0.0

def calc_rank_ic_reb(y_true, y_pred, dates=None):
    """
    ③ STR攻め用: Rank IC (Rebalance)
    11日ごとのリバランス日に限定して全体の順位の相関を評価し、その平均値を返します。
    """
    y_pred_1d = _reduce_multiclass_to_score(y_pred)
    
    if dates is None:
        if len(y_true) < 2 or np.max(y_pred_1d) == np.min(y_pred_1d) or np.max(y_true) == np.min(y_true): return 0.0
        ic, _ = spearmanr(y_true, y_pred_1d)
        return ic if not np.isnan(ic) else 0.0
        
    df_tmp = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred_1d, 'date': dates})
    df_tmp['date'] = pd.to_datetime(df_tmp['date']).dt.date
    
    unique_dates = np.sort(df_tmp['date'].unique())
    rebalance_dates = set(unique_dates[::11])
    
    scores = []
    for d, group in df_tmp.groupby('date'):
        if d not in rebalance_dates:
            continue
        g_y_true = group['y_true'].values
        g_y_pred = group['y_pred'].values
        if len(g_y_true) < 2 or np.max(g_y_pred) == np.min(g_y_pred) or np.max(g_y_true) == np.min(g_y_true): continue
        ic, _ = spearmanr(g_y_true, g_y_pred)
        if not np.isnan(ic):
            scores.append(ic)
    return np.mean(scores) if scores else 0.0

def calc_rank_ic(y_true, y_pred, dates=None):
    y_pred_1d = _reduce_multiclass_to_score(y_pred)
    
    if dates is None:
        if len(y_true) < 2 or np.max(y_pred_1d) == np.min(y_pred_1d) or np.max(y_true) == np.min(y_true): return 0.0
        ic, _ = spearmanr(y_true, y_pred_1d)
        return ic if not np.isnan(ic) else 0.0
    df_tmp = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred_1d, 'date': dates})
    df_tmp['date'] = pd.to_datetime(df_tmp['date']).dt.date
    scores = []
    for d, group in df_tmp.groupby('date'):
        g_y_true = group['y_true'].values
        g_y_pred = group['y_pred'].values
        if len(g_y_true) < 2 or np.max(g_y_pred) == np.min(g_y_pred) or np.max(g_y_true) == np.min(g_y_true): continue
        ic, _ = spearmanr(g_y_true, g_y_pred)
        if not np.isnan(ic):
            scores.append(ic)
    return np.mean(scores) if scores else 0.0

# Alias for calc_rank_ic
mean_daily_spearman_ic = calc_rank_ic

def calc_pr_auc_30pt(y_true, y_pred, dates=None, threshold=0.30):
    """
    ④ STR守り用: PR-AUC (Average Precision for 30pt drop)
    日付ごとに下落イベントに対するランキング性能を評価し、その平均値を返します。
    """
    # STR guard metrics are often calculated on expected class index if multiclass
    y_pred_1d = _reduce_multiclass_to_score(y_pred)
    
    if dates is None:
        y_true_binary = (y_true >= threshold).astype(int)
        if len(np.unique(y_true_binary)) < 2: return 0.0
        return average_precision_score(y_true_binary, y_pred_1d)
        
    scores = []
    for _, group in _group_by_date(y_true, y_pred_1d, dates):
        y_true_binary = (group['y_true'].values >= threshold).astype(int)
        if len(np.unique(y_true_binary)) < 2: continue
        scores.append(average_precision_score(y_true_binary, group['y_pred'].values))
    return np.mean(scores) if scores else 0.0

def calc_Top30_SR(y_true, y_pred, dates=None):
    """
    ⑤ Top30 Equal-Weight Sharpe Ratio (Annualized)
    日付ごとに予測スコア上位30銘柄を選定し、その等金額ポートフォリオリターンの年率換算SRを返します。
    """
    y_pred_1d = _reduce_multiclass_to_score(y_pred)
    
    if dates is None:
        return 0.0
    if np.std(y_pred_1d) < 1e-8:
        return 0.0
        
    daily_rets = []

    for d, group in _group_by_date(y_true, y_pred_1d, dates):
        # 予測値が一意でない（タイが多い）場合を検知
        unique_preds = len(np.unique(group['y_pred']))
        if unique_preds < 30:
            # 予測値の解像度が低すぎる（Top30を選べない）場合は、その日のリターンをゼロ（ペナルティ）とする
            daily_rets.append(0.0)
        else:
            # 正常に解像度がある場合はTop30を計算
            top30_ret = group.nlargest(30, 'y_pred')['y_true'].mean()
            daily_rets.append(top30_ret)
        
    if len(daily_rets) < 2:
        return 0.0
        
    mu_p = np.mean(daily_rets)
    sigma_p = np.std(daily_rets, ddof=1)
    
    if sigma_p > 1e-8:
        return float((mu_p / sigma_p) * np.sqrt(252))
    else:
        return 0.0

def calc_rank_ic_reb_60d_multi_offset(y_true, y_pred, dates=None):
    """
    ⑥ STR攻め用 (Robust): Rank IC (Multi-offset Rebalance)
    60日ターゲット向けの頑健な評価指標。
    """
    from src.evaluation.metrics import calc_rank_ic_reb_multi_offset as calc_base
    y_pred_1d = _reduce_multiclass_to_score(y_pred)
    
    if dates is None:
        if len(y_true) < 2 or np.max(y_pred_1d) == np.min(y_pred_1d) or np.max(y_true) == np.min(y_true): return 0.0
        ic, _ = spearmanr(y_true, y_pred_1d)
        return ic if not np.isnan(ic) else 0.0
        
    df_tmp = pd.DataFrame({'y_true': y_true, 'pred': y_pred_1d, 'date': dates})
    # evaluate_metrics と同様に date型に変換
    df_tmp['date'] = pd.to_datetime(df_tmp['date']).dt.date
    
    res = calc_base(df_tmp, pred_col='pred', target_col='y_true', date_col='date', interval=60)
    # MDAで使用されるエイリアス(mean - std)を返す
    val = res.get('rank_ic_reb_60d_multi_offset', 0.0)
    return val if not np.isnan(val) else 0.0

# =========================================================================
# LightGBM 用のカスタム評価関数 (LGBMの early_stopping_metric に指定可能)
# =========================================================================

def create_lgbm_evaluator(metric_name, metric_func, train_dates, valid_dates, is_higher_better=True):
    """
    LightGBMの feval 向けに、クロージャを使って日付配列(dates)を注入するファクトリ関数
    """
    def lgbm_eval(preds, data):
        y_true = data.get_label()
        dates = None
        if train_dates is not None and len(y_true) == len(train_dates):
            dates = train_dates
        elif valid_dates is not None and len(y_true) == len(valid_dates):
            dates = valid_dates

        score = metric_func(y_true, preds, dates=dates)
        return metric_name, score, is_higher_better
    return lgbm_eval
