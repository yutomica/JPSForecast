import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, average_precision_score
from scipy.stats import spearmanr

def _group_by_date(y_true, y_pred, dates):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'date': dates})
    return df.groupby('date')

# =========================================================================
# 汎用 NumPy 関数 (TCN, FTT 等で early_stopping_metric に指定可能)
# =========================================================================

def calc_ndcg_10(y_true, y_pred, dates=None):
    """
    ① TAC攻め用: NDCG@10
    日付ごとに予測スコア上位の質を評価し、その平均値を返します。
    """
    if dates is None:
        if len(y_true) < 2: return 0.0
        rel = np.maximum(0, y_true)
        if np.max(rel) <= 0: return 0.0
        return ndcg_score([rel], [y_pred], k=10)
    
    scores = []
    for _, group in _group_by_date(y_true, y_pred, dates):
        if len(group) < 2: continue
        rel = np.maximum(0, group['y_true'].values)
        if np.max(rel) > 0:
            scores.append(ndcg_score([rel], [group['y_pred'].values], k=10))
    return np.mean(scores) if scores else 0.0

def calc_ap_severe(y_true, y_pred, dates=None):
    """
    ② TAC守り用: Average Precision (Severe Drop)
    evaluation.py と同様に、5%, 7%, 10% の暴落閾値に対するAPを全期間で計算し、その平均を返します。
    ※ 暴落イベントは稀少であるため、日別(Era別)ではなく期間全体で評価します。
    """
    thresholds = [-0.05, -0.07, -0.10]
    ap_scores = []
    
    for th in thresholds:
        y_true_binary = (y_true <= th).astype(int)
        if len(np.unique(y_true_binary)) > 1:
            ap_scores.append(average_precision_score(y_true_binary, -y_pred))
            
    return np.mean(ap_scores) if ap_scores else 0.0

def calc_rank_ic_reb(y_true, y_pred, dates=None):
    """
    ③ STR攻め用: Rank IC (Rebalance)
    evaluation.py に合わせ、11日ごとのリバランス日に限定して全体の順位の相関を評価し、その平均値を返します。
    """
    if dates is None:
        if len(y_true) < 2 or np.max(y_pred) == np.min(y_pred) or np.max(y_true) == np.min(y_true): return 0.0
        ic, _ = spearmanr(y_true, y_pred)
        return ic if not np.isnan(ic) else 0.0
        
    df_tmp = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'date': dates})
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

def calc_pr_auc_30pt(y_true, y_pred, dates=None, threshold=0.30):
    """
    ④ STR守り用: PR-AUC (Average Precision for 30pt drop)
    日付ごとに下落イベントに対するランキング性能を評価し、その平均値を返します。
    """
    if dates is None:
        y_true_binary = (y_true >= threshold).astype(int)
        if len(np.unique(y_true_binary)) < 2: return 0.0
        return average_precision_score(y_true_binary, y_pred)
        
    scores = []
    for _, group in _group_by_date(y_true, y_pred, dates):
        y_true_binary = (group['y_true'].values >= threshold).astype(int)
        if len(np.unique(y_true_binary)) < 2: continue
        scores.append(average_precision_score(y_true_binary, group['y_pred'].values))
    return np.mean(scores) if scores else 0.0

# =========================================================================
# LightGBM 用のカスタム評価関数 (LGBMの early_stopping_metric に指定可能)
# =========================================================================

def create_lgbm_evaluator(metric_name, metric_func, train_dates, valid_dates):
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
        return metric_name, score, True
    return lgbm_eval