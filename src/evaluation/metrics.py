import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    ndcg_score,
    precision_recall_curve,
    roc_auc_score,
)
from scipy.stats import spearmanr

EXTRA_BIN_META_COLS = ('Future_High', 'Future_Low', 'Future_Close')
EXTRA_BIN_METRIC_PREFIXES = (
    'top_bin_Future_',
    'bot_bin_Future_',
    'top10_Future_',
    'bot10_Future_',
)

def is_extra_bin_metric_key(key):
    return key.startswith(EXTRA_BIN_METRIC_PREFIXES)

def _safe_spearmanr(a, b):
    if len(a) < 2 or np.max(a) == np.min(a) or np.max(b) == np.min(b):
        return np.nan, np.nan
    return spearmanr(a, b)

def _resolve_cost(df_or_series, cost_col=0.005):
    """
    cost_col が列名ならその列を返す。
    cost_col がスカラーならfloatとして返す。
    文字列なのに列が存在しない場合は明示的にエラーにする。
    """
    if isinstance(cost_col, str):
        if cost_col in df_or_series.columns:
            return df_or_series[cost_col].astype(float)
        raise ValueError(f"cost_col='{cost_col}' not found in dataframe.")
    return float(cost_col)

def _scale_active_mean(x):
    return float(0.02 * np.clip(x / 0.01, -1.0, 1.0))

def _safe_ap(y_true, y_score):
    """
    Average Precisionの計算。
    y_trueに正例がない場合や、y_scoreに不備がある場合にNaNを返す安全版。
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if len(y_true) == 0:
        return np.nan
    if y_true.sum() == 0:
        return np.nan
    return float(average_precision_score(y_true, y_score))

def _safe_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))

def _as_binary_score(y_pred):
    """Return a 1D score where larger means higher probability of class 1."""
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.ndim == 2:
        if y_pred.shape[1] >= 2:
            return y_pred[:, 1]
        return y_pred[:, 0]
    return y_pred

def _as_probability(score):
    score = np.asarray(score, dtype=float)
    if len(score) == 0:
        return score
    finite = score[np.isfinite(score)]
    if len(finite) and np.nanmin(finite) >= 0.0 and np.nanmax(finite) <= 1.0:
        return np.clip(score, 1e-15, 1.0 - 1e-15)
    return 1.0 / (1.0 + np.exp(-np.clip(score, -50.0, 50.0)))

def _is_tac_tb_binary_target(task_type, target_col, y_true):
    target_name = str(target_col or "").lower()
    if "target_tac_tb" not in target_name and "tac_tb" not in target_name:
        return False
    if str(task_type).lower() not in {"binary", "classification"}:
        return False
    y = pd.Series(y_true).dropna().unique()
    if len(y) == 0:
        return False
    return set(np.asarray(y, dtype=float)).issubset({0.0, 1.0})

def calc_tac_tb_binary_metrics(df, y_pred=None, top_ks=(10, 20, 30), ndcg_ks=(10, 20, 30), date_col='date'):
    """
    Metrics for TAC triple-barrier binary targets.

    y_true=1 means the take-profit barrier was hit before the stop-loss barrier.
    These metrics evaluate event concentration in the top-ranked names, not
    subsequent close-to-close return.
    """
    if df.empty:
        return {}

    work = df[[date_col, 'pred', 'y_true']].copy()
    if y_pred is not None:
        work['score'] = _as_binary_score(y_pred)
    else:
        work['score'] = work['pred'].to_numpy(dtype=float)

    work['y_true'] = work['y_true'].astype(float)
    work = work[np.isfinite(work['score']) & work['y_true'].isin([0.0, 1.0])]
    if work.empty:
        return {}

    y = work['y_true'].astype(int).to_numpy()
    score = work['score'].to_numpy(dtype=float)
    proba = _as_probability(score)
    event_rate = float(np.mean(y))

    metrics = {
        'tb_event_rate': event_rate,
        'tb_ap': _safe_ap(y, score),
        'tb_auc': _safe_auc(y, score),
        'tb_recall_at_precision_20': _recall_at_precision(y, score, min_precision=0.20),
        'tb_recall_at_precision_30': _recall_at_precision(y, score, min_precision=0.30),
    }
    if len(np.unique(y)) >= 2:
        metrics['tb_logloss'] = float(log_loss(y, proba, labels=[0, 1]))
        metrics['tb_brier'] = float(brier_score_loss(y, proba))
    else:
        metrics['tb_logloss'] = np.nan
        metrics['tb_brier'] = np.nan

    daily_rows = []
    for d, grp in work.groupby(date_col):
        grp = grp.dropna(subset=['score', 'y_true'])
        n = len(grp)
        if n == 0:
            continue
        positives = float(grp['y_true'].sum())
        base_rate = positives / n
        row = {'date': d, 'base_rate': base_rate, 'positives': positives}

        for k in top_ks:
            if n < k:
                continue
            topk = grp.nlargest(k, 'score')
            hit_rate = float(topk['y_true'].mean())
            row[f'top{k}_hit_rate'] = hit_rate
            row[f'top{k}_expected_hits'] = float(topk['y_true'].sum())
            row[f'top{k}_lift'] = hit_rate / base_rate if base_rate > 0 else np.nan
            row[f'top{k}_capture'] = float(topk['y_true'].sum() / positives) if positives > 0 else np.nan

        for k in ndcg_ks:
            if n < k or positives <= 0:
                continue
            try:
                row[f'ndcg_{k}'] = float(ndcg_score([grp['y_true'].to_numpy()], [grp['score'].to_numpy()], k=k))
            except ValueError:
                row[f'ndcg_{k}'] = np.nan

        daily_rows.append(row)

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return metrics

    for k in top_ks:
        for suffix in ['hit_rate', 'expected_hits', 'lift', 'capture']:
            col = f'top{k}_{suffix}'
            metrics[f'tb_{col}'] = float(daily[col].mean()) if col in daily else np.nan
    for k in ndcg_ks:
        col = f'ndcg_{k}'
        metrics[f'tb_{col}'] = float(daily[col].mean()) if col in daily else np.nan

    metrics['tb_positive_day_ratio'] = float((daily['positives'] > 0).mean()) if 'positives' in daily else np.nan
    return metrics

def _recall_at_precision(y_true, y_score, min_precision=0.80):
    """
    指定したPrecisionを維持できるスコア閾値の中で、最大のRecallを計算する。
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if len(y_true) == 0 or y_true.sum() == 0:
        return np.nan
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    valid = precision >= min_precision
    if not np.any(valid):
        return 0.0
    return float(np.max(recall[valid]))

def calc_tac_risk_class_metrics(y_class, proba):
    """
    tac_risk_class (4クラス) 用の評価指標を算出。
    proba: (N, 4) の確率行列
    """
    y_class = np.asarray(y_class).astype(int)
    proba = np.asarray(proba, dtype=float)
    # 累積危険確率
    # class 0: <5%, class 1: 5-7%, class 2: 7-10%, class 3: >=10%
    p_5 = proba[:, 1] + proba[:, 2] + proba[:, 3]
    p_7 = proba[:, 2] + proba[:, 3]
    p_10 = proba[:, 3]
    # 正解ラベルの累積化
    y_5 = (y_class >= 1).astype(int)
    y_7 = (y_class >= 2).astype(int)
    y_10 = (y_class >= 3).astype(int)
    ap_5 = _safe_ap(y_5, p_5)
    ap_7 = _safe_ap(y_7, p_7)
    ap_10 = _safe_ap(y_10, p_10)
    aps = np.array([ap_5, ap_7, ap_10], dtype=float)
    weights = np.array([0.25, 0.35, 0.40], dtype=float)
    valid = np.isfinite(aps)
    if valid.sum() == 0:
        ap_severe_weighted = np.nan
        ap_severe_mean = np.nan
    else:
        ap_severe_weighted = float(
            np.sum(aps[valid] * weights[valid]) / np.sum(weights[valid])
        )
        ap_severe_mean = float(np.nanmean(aps))
    # recall_at_precision_80 用の risk_score (重み付け確率)
    risk_score = 0.25 * p_5 + 0.35 * p_7 + 0.40 * p_10
    recall_p80 = _recall_at_precision(
        y_true=y_7,
        y_score=risk_score,
        min_precision=0.80,
    )
    return {
        "ap_5": ap_5,
        "ap_7": ap_7,
        "ap_10": ap_10,
        "ap_severe": ap_severe_mean,
        "ap_severe_weighted": ap_severe_weighted,
        "recall_at_precision_80": recall_p80,
    }

def _threshold_to_class_idx(threshold, max_idx=None):
    idx = 1 if threshold == 0.05 else (2 if threshold == 0.07 else 3)
    if threshold >= 0.15:
        idx = 3
    return min(idx, max_idx) if max_idx is not None else idx

def _calc_multi_threshold_ap(y, yp, thresholds, task_type):
    is_proba = yp.ndim == 2
    target_is_return = np.nanmin(y) < 0
    ap_scores = []
    for pt in thresholds:
        if target_is_return:
            binary_true = (y <= -pt).astype(int)
        else:
            idx = _threshold_to_class_idx(pt)
            binary_true = (y >= idx).astype(int)
        if np.sum(binary_true) == 0 or len(np.unique(binary_true)) < 2:
            continue
        if is_proba:
            idx = _threshold_to_class_idx(pt, max_idx=yp.shape[1] - 1)
            score = np.sum(yp[:, idx:], axis=1)
        else:
            score = -yp if task_type == 'regression' else yp
        ap_scores.append(average_precision_score(binary_true, score))
    return float(np.mean(ap_scores)) if ap_scores else np.nan

def calc_daily_rankic_series(df, pred_col, target_col, date_col='date', min_names=10):
    """dateごとにSpearman(pred, target)を計算"""
    def _rankic(grp):
        if len(grp) < min_names:
            return np.nan
        ic, _ = _safe_spearmanr(grp[pred_col], grp[target_col])
        return ic
    return df.groupby(date_col).apply(_rankic, include_groups=False)

def calc_positive_day_ratio(daily_rankic_series):
    """daily_rankic > 0 の割合を計算"""
    valid_ics = daily_rankic_series.dropna()
    if len(valid_ics) == 0:
        return np.nan, np.nan
    ratio = (valid_ics > 0).sum() / len(valid_ics)
    scaled = 0.02 * np.clip((ratio - 0.50) / 0.20, -1.0, 1.0)
    return ratio, scaled

def calc_topk_active_mean(df, k, pred_col='pred', ret_col='ret', cost_col=0.005, date_col='date'):
    """TopK active returnを gross / net に分けて計算する。"""
    daily_rows = []
    for d, grp in df.groupby(date_col):
        cols_to_use = [pred_col, ret_col]
        if isinstance(cost_col, str) and cost_col in grp.columns:
            cols_to_use.append(cost_col)
        grp = grp[cols_to_use].dropna()

        if len(grp) < k:
            continue

        topk = grp.nlargest(k, pred_col)
        cost = _resolve_cost(topk, cost_col)

        univ_ret = grp[ret_col].mean()
        gross_active = topk[ret_col].mean() - univ_ret
        net_active = (topk[ret_col] - cost).mean() - univ_ret

        daily_rows.append({
            'date': d,
            'gross_active': gross_active,
            'net_active': net_active,
        })

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return {
            'gross_mean': np.nan, 'gross_scaled': np.nan,
            'net_mean': np.nan, 'net_scaled': np.nan,
            'std': np.nan, 'hit_rate': np.nan,
            'worst_day': np.nan, 'cvar_5pct': np.nan,
            'utility_raw': np.nan, 'utility_scaled': np.nan,
        }

    gross = daily['gross_active'].dropna()
    net = daily['net_active'].dropna()

    gross_mean = float(gross.mean()) if len(gross) else np.nan
    net_mean = float(net.mean()) if len(net) else np.nan
    net_std = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    hit_rate = float((net > 0).mean()) if len(net) else np.nan
    worst_day = float(net.min()) if len(net) else np.nan

    if len(net):
        n_tail = max(1, int(len(net) * 0.05))
        cvar_5pct = float(np.sort(net.values)[:n_tail].mean())
    else:
        cvar_5pct = np.nan

    utility_raw = float(net_mean - net_std) if pd.notna(net_mean) and pd.notna(net_std) else np.nan
    utility_scaled = _scale_active_mean(utility_raw) if pd.notna(utility_raw) else np.nan

    return {
        'gross_mean': gross_mean, 'gross_scaled': _scale_active_mean(gross_mean) if pd.notna(gross_mean) else np.nan,
        'net_mean': net_mean, 'net_scaled': _scale_active_mean(net_mean) if pd.notna(net_mean) else np.nan,
        'std': net_std, 'hit_rate': hit_rate, 'worst_day': worst_day,
        'cvar_5pct': cvar_5pct, 'utility_raw': utility_raw, 'utility_scaled': utility_scaled,
    }

def calc_top_quintile_spread(df, pred_col='pred', ret_col='ret', date_col='date'):
    """pred上位20%と下位20%のリターン平均差を計算"""
    def _spread(grp):
        if len(grp) < 10:
            return np.nan
        k_q = max(1, int(len(grp) * 0.2))
        top = grp.nlargest(k_q, pred_col)[ret_col].mean()
        bot = grp.nsmallest(k_q, pred_col)[ret_col].mean()
        return top - bot
    
    daily_spread = df.groupby(date_col).apply(_spread, include_groups=False)
    raw_mean = np.nanmean(daily_spread)
    scaled = 0.02 * np.clip(raw_mean / 0.02, -1.0, 1.0)
    return raw_mean, scaled

def calc_top30_rankic_alpha(df, pred_col='pred', ret_col='ret', cost_col=0.005, date_col='date'):
    """Top30銘柄内でのpredとcost-adjusted returnのランク相関を計算"""
    def _rankic_alpha(grp):
        if len(grp) < 30:
            return np.nan
        cols = [pred_col, ret_col]
        if isinstance(cost_col, str) and cost_col in grp.columns:
            cols.append(cost_col)
        top30 = grp[cols].dropna().nlargest(30, pred_col)
        if len(top30) < 30:
            return np.nan
        cost = _resolve_cost(top30, cost_col)
        ic, _ = _safe_spearmanr(top30[pred_col], top30[ret_col] - cost)
        return ic
    daily_ic = df.groupby(date_col).apply(_rankic_alpha, include_groups=False)
    raw_mean = np.nanmean(daily_ic)
    scaled = 0.02 * np.clip(raw_mean / 0.05, -1.0, 1.0)
    return raw_mean, scaled

def _calc_additional_groupby_metrics(df, task_type='regression', ndcg_k=10):
    """
    evaluate_metrics固有の日次ループ処理指標を算出する内部ヘルパー
    """
    ndcgs, rank_ics_reb, recalls_gate30, recalls_gate30_severe, spreads, top30_returns = [], [], [], [], [], []
    
    unique_dates = np.sort(df['date'].unique())
    rebalance_dates = set(unique_dates[::11])
    
    for d, grp in df.groupby('date'):
        n = len(grp)
        # NDCG
        if n >= ndcg_k:
            rel = np.maximum(0, grp['y_ret'].values)
            if np.max(rel) > 0:
                try: ndcgs.append(ndcg_score([rel], [grp['pred'].values], k=ndcg_k))
                except: pass
        
        # Spreads (Top10 - Univ)
        if n >= 10:
            spreads.append(grp.nlargest(10, 'pred')['y_ret'].mean() - grp['y_ret'].mean())
            
        # Top30 Returns (for SR)
        if n >= 30:
            top30_returns.append(grp.nlargest(30, 'pred')['y_ret'].mean())
            
        # Rebalance RankIC
        if d in rebalance_dates and n >= 2:
            ic, _ = _safe_spearmanr(grp['y_true'].values, grp['pred'].values)
            if not np.isnan(ic):
                rank_ics_reb.append(ic)

        # Risk Gate Recall
        mines = (grp['y_ret'] <= -0.15)
        mines_severe = (grp['y_ret'] <= -0.25)
        if mines.any() or mines_severe.any():
            risk_order = grp['pred'].sort_values(ascending=(task_type == 'regression')).index
            k_gate = int(n * 0.3)
            if k_gate > 0:
                gate_idx = risk_order[:k_gate]
                if mines.sum() > 0: recalls_gate30.append(float(mines.loc[gate_idx].sum() / mines.sum()))
                if mines_severe.sum() > 0: recalls_gate30_severe.append(float(mines_severe.loc[gate_idx].sum() / mines_severe.sum()))

    results = {}
    results[f'ndcg_{ndcg_k}'] = float(np.mean(ndcgs)) if ndcgs else np.nan
    results['top10_spread'] = float(np.mean(spreads)) if spreads else np.nan
    
    if top30_returns:
        mu_p = np.mean(top30_returns)
        sigma_p = np.std(top30_returns, ddof=1)
        results['Top30_SR'] = float((mu_p / sigma_p) * np.sqrt(252)) if sigma_p > 1e-8 else 0.0
    else:
        results['Top30_SR'] = np.nan
        
    results['RankIC_reb'] = float(np.mean(rank_ics_reb)) if rank_ics_reb else np.nan
    results['Recall_Gate30pct'] = float(np.mean(recalls_gate30)) if recalls_gate30 else np.nan
    results['Recall_Gate30pct_severe'] = float(np.mean(recalls_gate30_severe)) if recalls_gate30_severe else np.nan
    
    return results

def calc_daily_rankic_icir(df, pred_col='pred', target_col='y_true', date_col='date', target_dates=None, min_names=2):
    daily_ics = []
    target_dates = set(target_dates) if target_dates is not None else None

    for d, grp in df.groupby(date_col):
        if target_dates is not None and d not in target_dates:
            continue
        if len(grp) < min_names:
            continue
        ic, _ = _safe_spearmanr(grp[target_col].values, grp[pred_col].values)
        if not np.isnan(ic):
            daily_ics.append(ic)

    if not daily_ics:
        return np.nan

    ic_mean = np.mean(daily_ics)
    ic_std = np.std(daily_ics)
    return float(ic_mean / (ic_std + 1e-8))

def _add_daily_icir_metrics(metrics, df_eval):
    unique_dates = np.sort(df_eval['date'].unique())
    metrics['daily_icir'] = calc_daily_rankic_icir(df_eval, target_dates=None)
    metrics['daily_icir_reb'] = calc_daily_rankic_icir(df_eval, target_dates=unique_dates[::11])

def calc_rank_ic_reb_multi_offset(
    df: pd.DataFrame,
    pred_col: str = "pred",
    target_col: str = "y_true",
    date_col: str = "date",
    interval: int = 60,
    offsets: list[int] | None = None,
    min_names: int = 30,
    return_detail: bool = False,
) -> dict:
    if offsets is None:
        offsets = [0, 10, 20, 30, 40, 50]
        
    unique_dates = np.sort(df[date_col].unique())
    
    offset_results = []
    
    for offset in offsets:
        target_dates = unique_dates[offset::interval]
        target_dates_set = set(target_dates)
        
        # Filter dataframe for these dates
        df_offset = df[df[date_col].isin(target_dates_set)]
        
        ic_list = []
        for _, grp in df_offset.groupby(date_col):
            if len(grp) < min_names:
                continue
            ic, _ = _safe_spearmanr(grp[pred_col].values, grp[target_col].values)
            if not np.isnan(ic):
                ic_list.append(ic)
                
        if len(ic_list) > 0:
            mean_ic = np.mean(ic_list)
            std_ic = np.std(ic_list, ddof=1) if len(ic_list) > 1 else 0.0
            icir = mean_ic / std_ic if std_ic > 0 and len(ic_list) >= 2 else np.nan
            offset_results.append({
                'offset': offset,
                'mean': float(mean_ic),
                'std': float(std_ic),
                'icir': float(icir),
                'n_dates': len(ic_list)
            })
            
    if not offset_results:
        return {
            'rank_ic_reb_60d_multi_offset_mean': np.nan,
            'rank_ic_reb_60d_multi_offset_std': np.nan,
            'rank_ic_reb_60d_multi_offset_icir_mean': np.nan,
            'rank_ic_reb_60d_multi_offset_icir_worst': np.nan,
            'rank_ic_reb_60d_multi_offset_mean_minus_std': np.nan,
            'rank_ic_reb_60d_multi_offset_worst_offset_mean': np.nan,
            'rank_ic_reb_60d_multi_offset_positive_offset_ratio': np.nan,
            'rank_ic_reb_60d_multi_offset_n_dates_total': 0.0,
            'rank_ic_reb_60d_multi_offset': np.nan,
        }
        
    means = [r['mean'] for r in offset_results]
    stds = [r['std'] for r in offset_results]
    icirs = [r['icir'] for r in offset_results if not np.isnan(r['icir'])]
    n_dates = [r['n_dates'] for r in offset_results]
    
    mean_of_means = float(np.mean(means))
    std_of_means = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
    icir_mean = float(np.mean(icirs)) if len(icirs) > 0 else np.nan
    icir_worst = float(np.min(icirs)) if len(icirs) > 0 else np.nan
    mean_minus_std = float(mean_of_means - std_of_means)
    worst_mean = float(np.min(means))
    pos_ratio = float(np.mean([1 if m > 0 else 0 for m in means]))
    total_dates = float(np.sum(n_dates))
    
    res = {
        'rank_ic_reb_60d_multi_offset_mean': mean_of_means,
        'rank_ic_reb_60d_multi_offset_std': std_of_means,
        'rank_ic_reb_60d_multi_offset_icir_mean': icir_mean,
        'rank_ic_reb_60d_multi_offset_icir_worst': icir_worst,
        'rank_ic_reb_60d_multi_offset_mean_minus_std': mean_minus_std,
        'rank_ic_reb_60d_multi_offset_worst_offset_mean': worst_mean,
        'rank_ic_reb_60d_multi_offset_positive_offset_ratio': pos_ratio,
        'rank_ic_reb_60d_multi_offset_n_dates_total': total_dates,
        'rank_ic_reb_60d_multi_offset': mean_minus_std,
    }
    
    if return_detail:
        res['details'] = offset_results
        
    return res

def _normalize_metric_inputs(y_true, y_pred=None, y_ret=None, dates=None, df=None):
    """evaluate_metrics の入力を配列と評価用DataFrameに正規化する。"""
    source_df = df
    if isinstance(y_true, pd.DataFrame):
        source_df = y_true
        y_true = source_df['y_true'].values
        y_ret = source_df['y_ret'].values
        dates = source_df['date'].values
        if y_pred is None:
            y_pred = source_df['pred_1d'].values if 'pred_1d' in source_df.columns else source_df['pred'].values

    if y_ret is None or dates is None or y_pred is None:
        return None

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_ret = np.asarray(y_ret)
    dates = np.asarray(dates)

    valid_mask = pd.notna(y_true) & pd.notna(y_ret) & pd.notna(dates)
    if y_pred.ndim == 2:
        valid_mask &= pd.notna(y_pred).all(axis=1)
    else:
        valid_mask &= pd.notna(y_pred)

    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    y_ret = y_ret[valid_mask]
    dates = dates[valid_mask]
    if len(y_true) == 0:
        return None

    if y_pred.ndim == 2:
        y_pred_1d = np.dot(y_pred, np.arange(y_pred.shape[1]))
    else:
        y_pred_1d = y_pred

    df_eval = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'pred': y_pred_1d,
        'y_true': y_true,
        'y_ret': y_ret
    })
    df_eval['date'] = df_eval['date'].dt.date

    if source_df is not None:
        source_valid = source_df.loc[valid_mask] if len(source_df) == len(valid_mask) else source_df
        for col in EXTRA_BIN_META_COLS:
            if col in source_valid.columns:
                df_eval[col] = source_valid[col].values

    return y_true, y_pred, y_pred_1d, y_ret, df_eval

def _add_rankic_metrics(metrics, rankic_series):
    metrics['RankIC'] = float(np.nanmean(rankic_series))
    metrics['mean_daily_rankic'] = metrics['RankIC']
    metrics['ic'] = metrics['RankIC']
    metrics['rank_ic'] = metrics['RankIC']

    pos_raw, pos_scaled = calc_positive_day_ratio(rankic_series)
    metrics['positive_day_ratio_raw'] = pos_raw
    metrics['positive_day_ratio_scaled'] = pos_scaled
    metrics['positive_day_ratio'] = pos_scaled

def _add_topk_active_metrics(metrics, df_eval, cost_buffer):
    for k in [10, 20, 30]:
        k_metrics = calc_topk_active_mean(df_eval, k, pred_col='pred', ret_col='y_ret', cost_col=cost_buffer)
        for key, val in k_metrics.items():
            metrics[f'top{k}_{key}'] = val
        # Compatibility aliases for calc_fold_metrics / objectives.py
        metrics[f'top{k}_active_mean_raw'] = k_metrics['net_mean']
        metrics[f'top{k}_active_mean_scaled'] = k_metrics['net_scaled']
        metrics[f'top{k}_gross_active_mean_scaled'] = k_metrics['gross_scaled']
        metrics[f'top{k}_net_active_mean_scaled'] = k_metrics['net_scaled']
        metrics[f'top{k}_active_std'] = k_metrics['std']
        metrics[f'top{k}_active_hit_rate'] = k_metrics['hit_rate']
        metrics[f'top{k}_active_worst_day'] = k_metrics['worst_day']
        metrics[f'top{k}_active_cvar_5pct'] = k_metrics['cvar_5pct']
        metrics[f'top{k}_active_utility_raw'] = k_metrics['utility_raw']
        metrics[f'top{k}_active_utility_scaled'] = k_metrics['utility_scaled']

    metrics['cost_adjusted_top30_active_utility_raw'] = metrics['top30_utility_raw']
    metrics['cost_adjusted_top30_active_utility_scaled'] = metrics['top30_utility_scaled']
    metrics['cost_adjusted_top30_active_utility'] = metrics['top30_utility_scaled']

def _add_spread_alpha_metrics(metrics, df_eval, cost_buffer):
    metrics['top_quintile_spread_raw'], metrics['top_quintile_spread_scaled'] = calc_top_quintile_spread(df_eval, 'pred', 'y_ret')
    metrics['top_quintile_spread'] = metrics['top_quintile_spread_scaled']
    metrics['top30_rankic_alpha_raw'], metrics['top30_rankic_alpha_scaled'] = calc_top30_rankic_alpha(df_eval, 'pred', 'y_ret', cost_col=cost_buffer)
    metrics['top30_rankic_alpha'] = metrics['top30_rankic_alpha_scaled']

def _add_tac_risk_class_metrics_if_needed(metrics, y_true, y_pred, task_type, target_col):
    if task_type == 'multiclass' and (target_col is not None and 'tac_risk' in str(target_col)) and y_pred.ndim == 2:
        metrics.update(calc_tac_risk_class_metrics(y_true, y_pred))

def _add_tac_tb_binary_metrics_if_needed(metrics, df_eval, y_true, y_pred, task_type, target_col):
    if _is_tac_tb_binary_target(task_type, target_col, y_true):
        metrics.update(calc_tac_tb_binary_metrics(df_eval, y_pred=y_pred))

def _add_drawdown_alert_metrics(metrics, y_ret, y_pred, y_pred_1d, task_type):
    metrics['AP_severe'] = _calc_multi_threshold_ap(y_ret, y_pred, [0.05, 0.07, 0.10], task_type)
    metrics['AP_severe_STR'] = _calc_multi_threshold_ap(y_ret, y_pred, [0.15, 0.20, 0.30], task_type)
    pred_threshold = np.percentile(y_pred_1d, 20)
    actual_severe = (y_ret <= -0.05)
    pred_alert = (y_pred_1d <= pred_threshold) if task_type == 'regression' else (y_pred_1d >= np.percentile(y_pred_1d, 80))
    tp = np.sum(actual_severe & pred_alert)
    fn = np.sum(actual_severe & ~pred_alert)
    metrics['severe_drawdown_recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan

def _add_groupby_metrics(metrics, df_eval, task_type, ndcg_k):
    metrics.update(_calc_additional_groupby_metrics(df_eval, task_type=task_type, ndcg_k=ndcg_k))

def _add_multi_offset_rebalance_metrics(metrics, df_eval, reb_interval, reb_offsets, reb_min_names):
    metrics.update(calc_rank_ic_reb_multi_offset(
        df_eval, pred_col='pred', target_col='y_true', date_col='date',
        interval=reb_interval, offsets=reb_offsets, min_names=reb_min_names
    ))

def _add_extra_bin_metrics_if_needed(metrics, df_eval, include_extra_bin_metrics):
    if include_extra_bin_metrics:
        metrics.update(calculate_extra_bin_metrics(df_eval, score_col='pred'))

def _add_compatibility_aliases(metrics):
    metrics['rank_ic_reb'] = metrics.get('RankIC_reb', np.nan)
    metrics['top30_sr'] = metrics.get('Top30_SR', np.nan)
    metrics['recall_gate30pct'] = metrics.get('Recall_Gate30pct', np.nan)

def evaluate_metrics(
    y_true,
    y_pred=None,
    y_ret=None,
    task_type='regression',
    target_col=None,
    dates=None,
    ndcg_k=10,
    cost_buffer=0.005,
    df=None,
    reb_interval: int = 60,
    reb_offsets: list[int] | None = None,
    reb_min_names: int = 30,
    include_extra_bin_metrics: bool = False,
):
    """
    統一された評価指標算出関数。
    arrays (y_true, y_pred, y_ret, dates) または DataFrame (df/y_true) のいずれかを受け取る。
    """
    normalized = _normalize_metric_inputs(y_true, y_pred=y_pred, y_ret=y_ret, dates=dates, df=df)
    if normalized is None:
        return {}
    y_true, y_pred, y_pred_1d, y_ret, df_eval = normalized

    metrics = {}

    # Tac Risk Class Metrics
    _add_tac_risk_class_metrics_if_needed(metrics, y_true, y_pred, task_type, target_col)

    # TAC triple-barrier hit metrics
    _add_tac_tb_binary_metrics_if_needed(metrics, df_eval, y_true, y_pred, task_type, target_col)

    # Drawdown / AP系
    _add_drawdown_alert_metrics(metrics, y_ret, y_pred, y_pred_1d, task_type)

    # RankIC系
    rankic_series = calc_daily_rankic_series(df_eval, 'pred', 'y_true', 'date')
    _add_rankic_metrics(metrics, rankic_series)

    # TopK Active Mean (10, 20, 30)
    _add_topk_active_metrics(metrics, df_eval, cost_buffer)

    # Spread / Alpha IC
    _add_spread_alpha_metrics(metrics, df_eval, cost_buffer)

    # group-by 指標 (NDCG, SR, Gate Recall, etc.)
    _add_groupby_metrics(metrics, df_eval, task_type, ndcg_k)
    
    # Multi-offset Rebalance RankIC
    _add_multi_offset_rebalance_metrics(metrics, df_eval, reb_interval, reb_offsets, reb_min_names)
    
    _add_extra_bin_metrics_if_needed(metrics, df_eval, include_extra_bin_metrics)

    # 互換性エイリアス
    _add_compatibility_aliases(metrics)

    # Daily ICIR Metrics
    _add_daily_icir_metrics(metrics, df_eval)

    return metrics


def calculate_bin_stats(df_eval, score_col, target_col, task_type='regression', metadata_cols=None, n_bins=20, date_col='date', global_bin=False):
    df_eval = df_eval.copy()
    # 1. Bin分割
    if global_bin:
        df_eval['bin_id'] = pd.qcut(df_eval[score_col].rank(method='first'), n_bins, labels=False, duplicates='drop')
    else:
        df_eval['bin_id'] = df_eval.groupby(date_col)[score_col].transform(
            lambda x: pd.qcut(x.rank(method='first'), n_bins, labels=False, duplicates='drop')
        )
    # NaNチェック
    if df_eval['bin_id'].isna().all():
        return pd.DataFrame()
    # Binラベル作成 (0 -> "Bin 01", 1 -> "Bin 02", ...)
    df_eval['bin_label'] = df_eval['bin_id'].apply(lambda x: f"Bin {int(x)+1:02d}" if pd.notna(x) else "NaN")
    # 2. 統計量の算出
    # 各Binに含まれるサンプル数
    grouped = df_eval.groupby('bin_label', observed=True)
    stats = grouped.size().to_frame(name='sample_count')
    # Binラベルのソート順を保証 (Bin 01, Bin 02, ...)
    stats = stats.sort_index()
    # スコア平均
    stats['score_mean'] = grouped[score_col].mean()
    # ターゲット平均
    stats['target_mean'] = grouped[target_col].mean()
    # メタデータの統計量
    if metadata_cols:
        for col in metadata_cols:
            if col in df_eval.columns:
                grp = grouped[col]
                stats[f'{col}_mean'] = grp.mean()
                stats[f'{col}_std'] = grp.std()
                for q in [0.05, 0.1, 0.5, 0.9, 0.95]:
                    stats[f'{col}_q{int(q*100)}'] = grp.quantile(q)
    return stats

def calculate_extra_bin_metrics(df_eval, score_col, date_col='date'):
    """
    Calculate top/bottom bin and top/bottom 10 samples metrics.
    Required columns in df_eval: [date_col, score_col, 'Future_High', 'Future_Low', 'Future_Close']
    """
    available_meta = [c for c in EXTRA_BIN_META_COLS if c in df_eval.columns]
    if not available_meta:
        return {}

    daily_results = []
    df_eval = df_eval.copy()
    if pd.api.types.is_datetime64_any_dtype(df_eval[date_col]):
        df_eval[date_col] = df_eval[date_col].dt.date

    for d, grp in df_eval.groupby(date_col):
        n = len(grp)
        res = {'date': d}
        
        # Binning (q=20)
        if n >= 20:
            bins = pd.qcut(grp[score_col].rank(method='first'), 20, labels=False, duplicates='drop')
            # Top Bin (index 19)
            top_mask = (bins == 19)
            if top_mask.any():
                for c in available_meta:
                    res[f'top_bin_{c}'] = grp.loc[top_mask, c].mean()
            # Bottom Bin (index 0)
            bot_mask = (bins == 0)
            if bot_mask.any():
                for c in available_meta:
                    res[f'bot_bin_{c}'] = grp.loc[bot_mask, c].mean()
        
        # Top 10 samples (highest score)
        top10 = grp.nlargest(min(n, 10), score_col)
        for c in available_meta:
            res[f'top10_{c}'] = top10[c].mean()
            
        # Bottom 10 samples (lowest score)
        bot10 = grp.nsmallest(min(n, 10), score_col)
        for c in available_meta:
            res[f'bot10_{c}'] = bot10[c].mean()
            
        daily_results.append(res)
        
    if not daily_results:
        return {}
        
    daily_df = pd.DataFrame(daily_results)
    return {
        col: float(daily_df[col].mean())
        for col in daily_df.columns
        if col != 'date'
    }
