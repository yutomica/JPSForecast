import numpy as np
import pandas as pd


OBJECTIVE_ALIAS_KEYS = {
    "valid_mean_daily_rankic_mean": ("mean_daily_rankic_mean", np.nan),
    "valid_worst_fold_rankic": ("worst_fold_rankic", np.nan),
    "valid_top30_active_mean_raw": ("top30_active_mean_raw_mean", np.nan),
    "valid_top20_active_mean_raw": ("top20_active_mean_raw_mean", np.nan),
    "valid_top10_active_mean_raw": ("top10_active_mean_raw_mean", np.nan),
}

OBJECTIVE_COMPONENT_KEYS = {
    "objective_component_mean_daily_rankic": ("mean_daily_rankic_mean", 0),
    "objective_component_top30_active_mean_scaled": ("top30_active_mean_scaled_mean", 0),
    "objective_component_top20_active_mean_scaled": ("top20_active_mean_scaled_mean", 0),
    "objective_component_top_quintile_spread_scaled": ("top_quintile_spread_scaled_mean", 0),
    "objective_component_top30_rankic_alpha_scaled": ("top30_rankic_alpha_scaled_mean", 0),
    "objective_component_worst_fold_rankic": ("worst_fold_rankic", 0),
    "objective_component_positive_day_ratio_scaled": ("positive_day_ratio_scaled_mean", 0),
}

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


def calc_composite_tac_objective(pooled_metrics, worst_fold_rankic):
    """Step4 Final Sweep互換のTAC統合指標を計算する。"""
    components = {
        "rank_ic": pooled_metrics.get("RankIC", pooled_metrics.get("mean_daily_rankic", 0.0)),
        "utility": pooled_metrics.get("top30_active_utility_scaled", 0.0),
        "spread": pooled_metrics.get("top_quintile_spread_scaled", 0.0),
        "alpha_ic": pooled_metrics.get("top30_rankic_alpha_scaled", 0.0),
        "pos_ratio": pooled_metrics.get("positive_day_ratio_scaled", 0.0),
        "worst_fold_rankic": worst_fold_rankic,
    }
    score = (
        0.30 * components["rank_ic"]
        + 0.30 * components["utility"]
        + 0.15 * components["spread"]
        + 0.10 * components["alpha_ic"]
        + 0.10 * components["worst_fold_rankic"]
        + 0.05 * components["pos_ratio"]
    )
    return float(score), components


def calculate_final_optimization_score(
    valid_metrics,
    train_metrics,
    fold_metrics_results,
    pooled_metrics,
    opt_metric_name,
    direction,
    fallback_metric,
):
    """Optunaへ返す最終スコアとMLflow記録用メトリクスをまとめて作成する。"""
    log_metrics = {}
    messages = []

    if not valid_metrics:
        messages.append("⚠️ WARNING: No valid metrics found in validation results.")
        return fallback_metric, log_metrics, messages

    mean_score = np.nanmean(valid_metrics)
    std_score = np.nanstd(valid_metrics)
    min_score = np.nanmin(valid_metrics)

    obj_v2 = 0.0
    penalty_v2 = 0.0
    aggregated_f_metrics = {}
    if fold_metrics_results:
        aggregated_f_metrics = aggregate_fold_metrics(fold_metrics_results)
        log_metrics.update({f"valid_{k}": v for k, v in aggregated_f_metrics.items()})
        for log_key, (metric_key, default) in OBJECTIVE_ALIAS_KEYS.items():
            log_metrics[log_key] = aggregated_f_metrics.get(metric_key, default)

        obj_v2, penalty_v2 = calc_objective_v2(aggregated_f_metrics)
        log_metrics["objective_v2"] = obj_v2
        log_metrics["objective_penalty_total"] = penalty_v2
        for log_key, (metric_key, default) in OBJECTIVE_COMPONENT_KEYS.items():
            log_metrics[log_key] = aggregated_f_metrics.get(metric_key, default)

    train_mean_ic = np.nanmean(train_metrics) if train_metrics else 0.0
    valid_mean_ic = aggregated_f_metrics.get("mean_daily_rankic_mean", 0.0)
    log_metrics["train_valid_rankic_gap"] = train_mean_ic - valid_mean_ic

    train_top30_active = (
        np.nanmean([m.get("top30_active_mean_raw", 0.0) for m in fold_metrics_results])
        if fold_metrics_results
        else 0.0
    )
    valid_top30_active = aggregated_f_metrics.get("top30_active_mean_raw_mean", 0.0)
    log_metrics["train_valid_top30_active_mean_gap"] = train_top30_active - valid_top30_active

    if opt_metric_name == "objective_v2":
        final_opt_score = obj_v2
        messages.append(f"  🔹 Objective V2: {final_opt_score:.6f} (Penalty: {penalty_v2:.4f})")
    elif opt_metric_name == "tac_risk_class_guarded_ap":
        final_opt_score = calc_tac_risk_objective(fold_metrics_results)
        messages.append(f"  🔹 TAC Risk Guarded AP Objective: {final_opt_score:.6f}")
    elif opt_metric_name == "composite_tac":
        final_opt_score, components = calc_composite_tac_objective(pooled_metrics, min_score)
        messages.extend([
            f"  🔹 Composite Objective (TAC): {final_opt_score:.6f}",
            (
                f"    - RankIC: {components['rank_ic']:.4f}, "
                f"Utility: {components['utility']:.4f}, "
                f"Spread: {components['spread']:.4f}"
            ),
            (
                f"    - AlphaIC: {components['alpha_ic']:.4f}, "
                f"WorstFoldIC: {components['worst_fold_rankic']:.4f}, "
                f"PosRatio: {components['pos_ratio']:.4f}"
            ),
        ])
    elif opt_metric_name.startswith("worst_fold_"):
        final_opt_score = min_score
    elif opt_metric_name == "daily_icir_reb":
        final_opt_score = pooled_metrics.get("daily_icir_reb", fallback_metric)
    elif opt_metric_name.startswith("pooled_oof_"):
        base_key = opt_metric_name.replace("pooled_oof_", "")
        final_opt_score = pooled_metrics.get(base_key, fallback_metric)
    elif direction == "minimize":
        final_opt_score = mean_score + std_score
    else:
        final_opt_score = mean_score - std_score

    return final_opt_score, log_metrics, messages
