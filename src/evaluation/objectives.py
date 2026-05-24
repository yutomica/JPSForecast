import numpy as np
import pandas as pd

def calc_tac_risk_objective(fold_metrics):
    """
    Optuna目的関数の計算 (tac_risk_class用)
    fold_metrics: list of dict (each dict from calc_tac_risk_class_metrics)
    """
    ap_w = np.array(
        [m["ap_severe_weighted"] for m in fold_metrics if "ap_severe_weighted" in m],
        dtype=float,
    )
    recall_p80 = np.array(
        [m["recall_at_precision_80"] for m in fold_metrics if "recall_at_precision_80" in m],
        dtype=float,
    )
    valid_ap = np.isfinite(ap_w)
    if valid_ap.sum() == 0:
        return -1.0
    mean_ap_w = float(np.nanmean(ap_w))
    std_ap_w = float(np.nanstd(ap_w))
    worst_ap_w = float(np.nanmin(ap_w))
    valid_recall = np.isfinite(recall_p80)
    mean_recall_p80 = (
        float(np.nanmean(recall_p80[valid_recall]))
        if valid_recall.any()
        else 0.0
    )
    # 安定性重視スコア: 平均 - 0.5*標準偏差 + 最悪fold + recall
    objective = (
        0.70 * (mean_ap_w - 0.50 * std_ap_w)
        + 0.20 * worst_ap_w
        + 0.10 * mean_recall_p80
    )
    return float(objective)

def aggregate_fold_metrics(fold_metrics_list):
    """fold別metricsを集約"""
    if not fold_metrics_list:
        return {}
    
    aggregated = {}
    keys = fold_metrics_list[0].keys()
    for k in keys:
        vals = [m[k] for m in fold_metrics_list if m[k] is not None and not np.isnan(m[k])]
        if vals:
            aggregated[f'{k}_mean'] = np.mean(vals)
            aggregated[f'{k}_median'] = np.median(vals)
            aggregated[f'{k}_std'] = np.std(vals)
            aggregated[f'{k}_min'] = np.min(vals)
            aggregated[f'{k}_max'] = np.max(vals)
        else:
            aggregated[f'{k}_mean'] = np.nan
            aggregated[f'{k}_median'] = np.nan
            aggregated[f'{k}_std'] = np.nan
            aggregated[f'{k}_min'] = np.nan
            aggregated[f'{k}_max'] = np.nan

    # 指定されたエイリアスの作成
    aggregated['mean_daily_rankic_mean'] = aggregated.get('mean_daily_rankic_mean', np.nan)
    aggregated['worst_fold_rankic'] = aggregated.get('mean_daily_rankic_min', np.nan)
    
    return aggregated

def calc_objective_v2(aggregated_metrics):
    """objective_v2 の計算"""
    m = aggregated_metrics
    required = [
        'mean_daily_rankic_mean',
        'top30_gross_active_mean_scaled_mean',
        'top30_net_active_mean_scaled_mean',
        'top20_gross_active_mean_scaled_mean',
        'top20_net_active_mean_scaled_mean',
        'top_quintile_spread_scaled_mean',
        'worst_fold_rankic',
        'positive_day_ratio_scaled_mean',
    ]
    for key in required:
        if key not in m or pd.isna(m[key]):
            return -999.0, -999.0
    mean_daily_rankic = m['mean_daily_rankic_mean']
    scaled_top30_gross_active_mean = m['top30_gross_active_mean_scaled_mean']
    scaled_top30_net_active_mean = m['top30_net_active_mean_scaled_mean']
    scaled_top20_gross_active_mean = m['top20_gross_active_mean_scaled_mean']
    scaled_top20_net_active_mean = m['top20_net_active_mean_scaled_mean']
    scaled_top_quintile_spread = m['top_quintile_spread_scaled_mean']
    worst_fold_rankic = m['worst_fold_rankic']
    positive_day_ratio_scaled = m['positive_day_ratio_scaled_mean']


    # 加重和
    score = (
        0.25 * mean_daily_rankic
        + 0.15 * scaled_top30_gross_active_mean
        + 0.10 * scaled_top30_net_active_mean
        + 0.10 * scaled_top20_gross_active_mean
        + 0.10 * scaled_top20_net_active_mean
        + 0.15 * scaled_top_quintile_spread
        + 0.10 * worst_fold_rankic
        + 0.05 * positive_day_ratio_scaled
    )
    
    # ペナルティ
    penalty = 0.0
    if mean_daily_rankic <= 0:
        penalty -= 0.05
    if worst_fold_rankic < 0:
        penalty -= 0.03
    top30_net_raw = m.get('top30_net_active_mean_raw_mean', m.get('top30_active_mean_raw_mean', np.nan))
    top20_net_raw = m.get('top20_net_active_mean_raw_mean', m.get('top20_active_mean_raw_mean', np.nan))
    if pd.isna(top30_net_raw) or pd.isna(top20_net_raw):
        return -999.0, -999.0
    penalty += 0.005 * np.clip(top30_net_raw / 0.005, -1.0, 0.0)
    penalty += 0.005 * np.clip(top20_net_raw / 0.005, -1.0, 0.0)
    if m.get('positive_day_ratio_raw_mean', 0.50) < 0.50:
        penalty -= 0.01
    return float(score + penalty), float(penalty)
