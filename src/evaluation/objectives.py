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

def calc_objective_tac(aggregated_metrics):
    """objective_tac の計算"""
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


def calc_objective_tac_gr_guarded(aggregated_metrics, train_valid_rankic_gap=0.0):
    """
    tac_alpha_gr 用のガード付き目的関数。

    gauss-rank 系ターゲットのため RankIC を主軸にしつつ、TopK の
    コスト控除後 active return が悪い trial を強めに減点する。
    """
    m = aggregated_metrics
    required = [
        "mean_daily_rankic_mean",
        "mean_daily_rankic_std",
        "worst_fold_rankic",
        "top_quintile_spread_scaled_mean",
        "top30_gross_active_mean_scaled_mean",
        "top20_gross_active_mean_scaled_mean",
        "positive_day_ratio_scaled_mean",
        "top30_rankic_alpha_scaled_mean",
    ]
    for key in required:
        if key not in m or pd.isna(m[key]):
            return -999.0, {}

    top30_net_raw = m.get("top30_net_active_mean_raw_mean", m.get("top30_active_mean_raw_mean", np.nan))
    top20_net_raw = m.get("top20_net_active_mean_raw_mean", m.get("top20_active_mean_raw_mean", np.nan))
    if pd.isna(top30_net_raw) or pd.isna(top20_net_raw):
        return -999.0, {}

    mean_rankic = m["mean_daily_rankic_mean"]
    rankic_std = m["mean_daily_rankic_std"]
    rankic_stability = mean_rankic - 0.5 * rankic_std
    worst_fold_rankic = m["worst_fold_rankic"]
    spread = m["top_quintile_spread_scaled_mean"]
    top30_gross = m["top30_gross_active_mean_scaled_mean"]
    top20_gross = m["top20_gross_active_mean_scaled_mean"]
    positive_day_ratio = m["positive_day_ratio_scaled_mean"]
    top30_alpha = m["top30_rankic_alpha_scaled_mean"]

    base_score = (
        0.35 * rankic_stability
        + 0.20 * worst_fold_rankic
        + 0.15 * spread
        + 0.10 * top30_gross
        + 0.10 * top20_gross
        + 0.05 * positive_day_ratio
        + 0.05 * top30_alpha
    )

    top30_net_penalty = 0.02 * np.clip(-top30_net_raw / 0.005, 0.0, 1.0)
    top20_net_penalty = 0.01 * np.clip(-top20_net_raw / 0.005, 0.0, 1.0)
    overfit_penalty = 0.01 * np.clip((train_valid_rankic_gap - 0.08) / 0.05, 0.0, 1.0)
    rankic_guard_penalty = 0.0
    if mean_rankic <= 0:
        rankic_guard_penalty += 0.05
    if worst_fold_rankic < 0:
        rankic_guard_penalty += 0.03

    total_penalty = top30_net_penalty + top20_net_penalty + overfit_penalty + rankic_guard_penalty
    score = base_score - total_penalty
    components = {
        "objective_tac_gr_guarded_base_score": float(base_score),
        "objective_tac_gr_guarded_rankic_stability": float(rankic_stability),
        "objective_tac_gr_guarded_top30_net_penalty": float(top30_net_penalty),
        "objective_tac_gr_guarded_top20_net_penalty": float(top20_net_penalty),
        "objective_tac_gr_guarded_overfit_penalty": float(overfit_penalty),
        "objective_tac_gr_guarded_rankic_guard_penalty": float(rankic_guard_penalty),
        "objective_tac_gr_guarded_penalty_total": float(total_penalty),
    }
    return float(score), components


def calc_objective_10_gr_guarded(aggregated_metrics, train_valid_rankic_gap=0.0):
    """
    10d gauss-rank ターゲット用のガード付き目的関数。

    10営業日のリターンはTACよりノイズと重なりが大きいため、TopK raw return
    への直接報酬を抑え、RankICの安定性とnet utilityを重視する。
    """
    m = aggregated_metrics
    required = [
        "mean_daily_rankic_mean",
        "mean_daily_rankic_std",
        "worst_fold_rankic",
        "top_quintile_spread_scaled_mean",
        "top30_active_utility_scaled_mean",
        "positive_day_ratio_scaled_mean",
    ]
    for key in required:
        if key not in m or pd.isna(m[key]):
            return -999.0, {}

    top30_net_raw = m.get("top30_net_active_mean_raw_mean", m.get("top30_active_mean_raw_mean", np.nan))
    if pd.isna(top30_net_raw):
        return -999.0, {}

    mean_rankic = m["mean_daily_rankic_mean"]
    rankic_std = m["mean_daily_rankic_std"]
    rankic_stability = mean_rankic - 0.5 * rankic_std
    worst_fold_rankic = m["worst_fold_rankic"]
    spread = m["top_quintile_spread_scaled_mean"]
    top30_net_utility = m["top30_active_utility_scaled_mean"]
    positive_day_ratio = m["positive_day_ratio_scaled_mean"]

    base_score = (
        0.45 * rankic_stability
        + 0.25 * worst_fold_rankic
        + 0.15 * spread
        + 0.10 * top30_net_utility
        + 0.05 * positive_day_ratio
    )

    top30_net_penalty = 0.02 * np.clip(-top30_net_raw / 0.005, 0.0, 1.0)
    overfit_penalty = 0.01 * np.clip((train_valid_rankic_gap - 0.08) / 0.05, 0.0, 1.0)
    rankic_guard_penalty = 0.0
    if mean_rankic <= 0:
        rankic_guard_penalty += 0.05
    if worst_fold_rankic < 0:
        rankic_guard_penalty += 0.03

    total_penalty = top30_net_penalty + overfit_penalty + rankic_guard_penalty
    score = base_score - total_penalty
    components = {
        "objective_10_gr_guarded_base_score": float(base_score),
        "objective_10_gr_guarded_rankic_stability": float(rankic_stability),
        "objective_10_gr_guarded_top30_net_utility": float(top30_net_utility),
        "objective_10_gr_guarded_top30_net_penalty": float(top30_net_penalty),
        "objective_10_gr_guarded_overfit_penalty": float(overfit_penalty),
        "objective_10_gr_guarded_rankic_guard_penalty": float(rankic_guard_penalty),
        "objective_10_gr_guarded_penalty_total": float(total_penalty),
    }
    return float(score), components


def calc_objective_tac_tb_hit_guarded(aggregated_metrics, train_valid_rankic_gap=0.0):
    """
    tac_tb_* binary triple-barrier targets用のガード付き目的関数。

    目的はclose-to-close returnの最大化ではなく、日次上位候補に
    take-profit barrier hitを濃縮すること。Top30 hit rate / lift を主軸に、
    AP/AUC/RankICで全体順位品質を補助し、loglossと過学習を減点する。
    """
    m = aggregated_metrics
    required = [
        "tb_top30_hit_rate_mean",
        "tb_top30_hit_rate_std",
        "tb_top30_lift_mean",
        "tb_top30_lift_std",
        "tb_top30_capture_mean",
        "tb_top10_hit_rate_mean",
        "tb_top10_hit_rate_std",
        "tb_ndcg_30_mean",
        "tb_ap_mean",
        "tb_auc_mean",
        "tb_logloss_mean",
        "mean_daily_rankic_mean",
        "mean_daily_rankic_std",
        "worst_fold_rankic",
    ]
    for key in required:
        if key not in m or pd.isna(m[key]):
            return -999.0, {}

    hit30_stability = m["tb_top30_hit_rate_mean"] - 0.5 * m["tb_top30_hit_rate_std"]
    hit10_stability = m["tb_top10_hit_rate_mean"] - 0.5 * m["tb_top10_hit_rate_std"]
    lift30_stability = m["tb_top30_lift_mean"] - 0.5 * m["tb_top30_lift_std"]
    lift30_scaled = float(np.clip((lift30_stability - 1.0) / 3.0, -0.5, 1.0))
    capture30 = m["tb_top30_capture_mean"]
    ndcg30 = m["tb_ndcg_30_mean"]
    ap = m["tb_ap_mean"]
    auc_edge = max(0.0, m["tb_auc_mean"] - 0.5)
    rankic_stability = m["mean_daily_rankic_mean"] - 0.5 * m["mean_daily_rankic_std"]
    worst_fold_rankic = m["worst_fold_rankic"]

    base_score = (
        0.30 * hit30_stability
        + 0.12 * hit10_stability
        + 0.16 * lift30_scaled
        + 0.10 * capture30
        + 0.12 * ndcg30
        + 0.10 * ap
        + 0.05 * auc_edge
        + 0.05 * rankic_stability
    )

    logloss_penalty = 0.035 * np.clip((m["tb_logloss_mean"] - 0.38) / 0.08, 0.0, 1.0)
    lift_guard_penalty = 0.025 * np.clip((2.5 - m["tb_top30_lift_mean"]) / 1.0, 0.0, 1.0)
    overfit_penalty = 0.015 * np.clip((train_valid_rankic_gap - 0.06) / 0.05, 0.0, 1.0)
    rankic_guard_penalty = 0.0
    if m["mean_daily_rankic_mean"] <= 0:
        rankic_guard_penalty += 0.05
    if worst_fold_rankic < 0:
        rankic_guard_penalty += 0.03

    total_penalty = (
        logloss_penalty
        + lift_guard_penalty
        + overfit_penalty
        + rankic_guard_penalty
    )
    score = base_score - total_penalty
    components = {
        "objective_tac_tb_hit_guarded_base_score": float(base_score),
        "objective_tac_tb_hit_guarded_hit30_stability": float(hit30_stability),
        "objective_tac_tb_hit_guarded_hit10_stability": float(hit10_stability),
        "objective_tac_tb_hit_guarded_lift30_stability": float(lift30_stability),
        "objective_tac_tb_hit_guarded_lift30_scaled": float(lift30_scaled),
        "objective_tac_tb_hit_guarded_capture30": float(capture30),
        "objective_tac_tb_hit_guarded_ndcg30": float(ndcg30),
        "objective_tac_tb_hit_guarded_ap": float(ap),
        "objective_tac_tb_hit_guarded_auc_edge": float(auc_edge),
        "objective_tac_tb_hit_guarded_rankic_stability": float(rankic_stability),
        "objective_tac_tb_hit_guarded_logloss_penalty": float(logloss_penalty),
        "objective_tac_tb_hit_guarded_lift_guard_penalty": float(lift_guard_penalty),
        "objective_tac_tb_hit_guarded_overfit_penalty": float(overfit_penalty),
        "objective_tac_tb_hit_guarded_rankic_guard_penalty": float(rankic_guard_penalty),
        "objective_tac_tb_hit_guarded_penalty_total": float(total_penalty),
    }
    return float(score), components


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

    obj_tac = 0.0
    penalty_tac = 0.0
    obj_tac_gr_guarded = 0.0
    guarded_components = {}
    obj_10_gr_guarded = 0.0
    guarded_10_components = {}
    obj_tac_tb_hit_guarded = 0.0
    guarded_tb_components = {}
    aggregated_f_metrics = {}
    if fold_metrics_results:
        aggregated_f_metrics = aggregate_fold_metrics(fold_metrics_results)
        log_metrics.update({f"valid_{k}": v for k, v in aggregated_f_metrics.items()})
        for log_key, (metric_key, default) in OBJECTIVE_ALIAS_KEYS.items():
            log_metrics[log_key] = aggregated_f_metrics.get(metric_key, default)

        obj_tac, penalty_tac = calc_objective_tac(aggregated_f_metrics)
        log_metrics["objective_tac"] = obj_tac
        log_metrics["objective_penalty_total"] = penalty_tac
        for log_key, (metric_key, default) in OBJECTIVE_COMPONENT_KEYS.items():
            log_metrics[log_key] = aggregated_f_metrics.get(metric_key, default)

    train_mean_ic = np.nanmean(train_metrics) if train_metrics else 0.0
    valid_mean_ic = aggregated_f_metrics.get("mean_daily_rankic_mean", 0.0)
    log_metrics["train_valid_rankic_gap"] = train_mean_ic - valid_mean_ic
    if fold_metrics_results:
        obj_tac_gr_guarded, guarded_components = calc_objective_tac_gr_guarded(
            aggregated_f_metrics,
            train_valid_rankic_gap=log_metrics["train_valid_rankic_gap"],
        )
        log_metrics["objective_tac_gr_guarded"] = obj_tac_gr_guarded
        log_metrics.update(guarded_components)
        obj_10_gr_guarded, guarded_10_components = calc_objective_10_gr_guarded(
            aggregated_f_metrics,
            train_valid_rankic_gap=log_metrics["train_valid_rankic_gap"],
        )
        log_metrics["objective_10_gr_guarded"] = obj_10_gr_guarded
        log_metrics["objective_10d_gr_guarded"] = obj_10_gr_guarded
        log_metrics.update(guarded_10_components)
        obj_tac_tb_hit_guarded, guarded_tb_components = calc_objective_tac_tb_hit_guarded(
            aggregated_f_metrics,
            train_valid_rankic_gap=log_metrics["train_valid_rankic_gap"],
        )
        log_metrics["objective_tac_tb_hit_guarded"] = obj_tac_tb_hit_guarded
        log_metrics.update(guarded_tb_components)

    train_top30_active = (
        np.nanmean([m.get("top30_active_mean_raw", 0.0) for m in fold_metrics_results])
        if fold_metrics_results
        else 0.0
    )
    valid_top30_active = aggregated_f_metrics.get("top30_active_mean_raw_mean", 0.0)
    log_metrics["train_valid_top30_active_mean_gap"] = train_top30_active - valid_top30_active

    if opt_metric_name == "objective_tac":
        final_opt_score = obj_tac
        messages.append(f"  🔹 Objective TAC: {final_opt_score:.6f} (Penalty: {penalty_tac:.4f})")
    elif opt_metric_name == "objective_tac_gr_guarded":
        final_opt_score = obj_tac_gr_guarded
        penalty_total = guarded_components.get("objective_tac_gr_guarded_penalty_total", np.nan)
        messages.append(f"  🔹 Objective TAC GR Guarded: {final_opt_score:.6f} (Penalty: {penalty_total:.4f})")
    elif opt_metric_name in {"objective_10_gr_guarded", "objective_10d_gr_guarded"}:
        final_opt_score = obj_10_gr_guarded
        penalty_total = guarded_10_components.get("objective_10_gr_guarded_penalty_total", np.nan)
        messages.append(f"  🔹 Objective 10D GR Guarded: {final_opt_score:.6f} (Penalty: {penalty_total:.4f})")
    elif opt_metric_name == "objective_tac_tb_hit_guarded":
        final_opt_score = obj_tac_tb_hit_guarded
        penalty_total = guarded_tb_components.get("objective_tac_tb_hit_guarded_penalty_total", np.nan)
        messages.append(f"  🔹 Objective TAC TB Hit Guarded: {final_opt_score:.6f} (Penalty: {penalty_total:.4f})")
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
