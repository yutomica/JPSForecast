#!/usr/bin/env python3
"""Select Step4 candidates using a fixed validation window.

The script extracts top Step4 Optuna trials, re-evaluates them with
``cv=fixed`` so the period is controlled by ``config/cv/fixed.yaml``, computes
domain-specific selection scores from the fixed validation metrics, and writes
a leaderboard plus a selected-candidate manifest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.objectives import calc_objective_10_gr_guarded, calc_objective_tac_tb_hit_guarded
from src.utils.mlflow_utils import get_or_create_parent_run


@dataclass(frozen=True)
class ObjectiveSpec:
    metric: str
    direction: str
    label: str


@dataclass(frozen=True)
class Candidate:
    objective: ObjectiveSpec
    study_name: str
    trial_number: int
    optuna_value: float
    params: dict[str, Any]
    rank_in_study: int


def safe_label(value: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
    return label or "label"


def parse_objective_spec(value: str) -> ObjectiveSpec:
    parts = value.split(":")
    if len(parts) == 1:
        metric = parts[0]
        direction = "maximize"
        label = safe_label(metric)
    elif len(parts) == 2:
        metric, direction = parts
        label = safe_label(metric)
    elif len(parts) == 3:
        metric, direction, label = parts
        label = safe_label(label)
    else:
        raise ValueError(f"Invalid objective spec: {value}")
    direction = direction.lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError(f"Invalid objective direction: {value}")
    return ObjectiveSpec(metric=metric, direction=direction, label=label)


def target_name(domain: str, role: str) -> str:
    return f"{domain}_{role}"


def experiment_name(domain: str, role: str) -> str:
    return f"JPSForecast_{target_name(domain, role)}"


def default_features_name(model: str, domain: str, role: str) -> str:
    return f"features_{model}_{target_name(domain, role)}_fixed"


def study_prefix(model: str, target: str, objective_label: str) -> str:
    return f"final_sweep_{model}_{target}_{objective_label}_"


def candidate_source_label(candidate: Candidate) -> str:
    match = re.search(r"_(optimize|refine)_(\d{8}_\d{6})$", candidate.study_name)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return safe_label(candidate.study_name)[-32:]


def complete_trials(study: optuna.Study, direction: str) -> list[optuna.trial.FrozenTrial]:
    trials = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
    reverse = direction == "maximize"
    return sorted(trials, key=lambda t: float(t.value), reverse=reverse)


def collect_candidates(
    storage: str,
    model: str,
    target: str,
    objectives: list[ObjectiveSpec],
    top_n_per_study: int,
    max_candidates: int,
    study_names: list[str] | None,
) -> list[Candidate]:
    summaries = optuna.get_all_study_summaries(storage=storage)
    summary_by_name = {s.study_name: s for s in summaries}
    summary_names = set(summary_by_name)
    candidate_buckets: list[list[Candidate]] = []

    for objective in objectives:
        if study_names:
            names = [name for name in study_names if name in summary_names]
        else:
            prefix = study_prefix(model, target, objective.label)
            names = [s.study_name for s in summaries if s.study_name.startswith(prefix)]
            names = sorted(
                names,
                key=lambda name: (
                    summary_by_name[name].datetime_start or datetime.min,
                    name,
                ),
                reverse=True,
            )

        bucket: list[Candidate] = []
        for name in names:
            study = optuna.load_study(study_name=name, storage=storage)
            direction = study.directions[0].name.lower() if study.directions else objective.direction
            for rank, trial in enumerate(complete_trials(study, direction)[:top_n_per_study], start=1):
                bucket.append(
                    Candidate(
                        objective=objective,
                        study_name=name,
                        trial_number=int(trial.number),
                        optuna_value=float(trial.value),
                        params=dict(trial.params),
                        rank_in_study=rank,
                    )
                )
        candidate_buckets.append(bucket)

    deduped: list[Candidate] = []
    seen: set[str] = set()
    positions = [0 for _ in candidate_buckets]
    while len(deduped) < max_candidates:
        added = False
        for bucket_idx, bucket in enumerate(candidate_buckets):
            while positions[bucket_idx] < len(bucket):
                candidate = bucket[positions[bucket_idx]]
                positions[bucket_idx] += 1
                key = json.dumps(candidate.params, sort_keys=True, default=str)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(candidate)
                added = True
                break
            if len(deduped) >= max_candidates:
                break
        if not added:
            break
    return deduped


def hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError(f"Cannot pass non-finite Hydra value: {value}")
        return repr(value)
    return str(value)


def canonical_param_key(key: str) -> str:
    return key[1:] if key.startswith("+") else key


def find_param_key(params: dict[str, Any], canonical_key: str) -> str | None:
    for key in params:
        if canonical_param_key(key) == canonical_key:
            return key
    return None


def has_param(params: dict[str, Any], canonical_key: str) -> bool:
    return find_param_key(params, canonical_key) is not None


def effective_candidate_params(candidate: Candidate, model: str) -> dict[str, Any]:
    params = dict(candidate.params)

    # LightGBM rejects is_unbalance and scale_pos_weight when both are active.
    # Step4 sweeps may tune scale_pos_weight while keeping is_unbalance=false as
    # a fixed parameter, which is not included in Optuna trial.params.
    if model.lower() == "lgbm" and has_param(params, "hparams.scale_pos_weight"):
        existing_key = find_param_key(params, "hparams.is_unbalance")
        if existing_key is not None:
            params.pop(existing_key)
        params["hparams.is_unbalance"] = False

    return params


def candidate_hparam_overrides(candidate: Candidate, model: str) -> list[str]:
    params = effective_candidate_params(candidate, model)
    return [f"{key}={hydra_value(value)}" for key, value in params.items()]


def metric(metrics: dict[str, float], name: str, default: float = np.nan) -> float:
    for key in (
        f"fold0_valid_{name}",
        f"valid_{name}",
        f"valid_{name}_mean",
        f"pooled_oof_{name}",
        name,
    ):
        value = metrics.get(key)
        if value is not None and np.isfinite(value):
            return float(value)
    return float(default)


def metric_any(metrics: dict[str, float], names: list[str], default: float = np.nan) -> float:
    for name in names:
        value = metric(metrics, name, np.nan)
        if np.isfinite(value):
            return value
    return float(default)


def clip_component(value: float, scale: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if not np.isfinite(value) or scale == 0:
        return 0.0
    return float(np.clip(value / scale, lo, hi))


def score_tac_alpha(metrics: dict[str, float]) -> tuple[float, dict[str, float]]:
    rankic = metric(metrics, "RankIC", 0.0)
    top30_net = metric(metrics, "top30_active_mean_raw", 0.0)
    top30_utility = metric(metrics, "top30_active_utility_raw", 0.0)
    spread = metric(metrics, "top_quintile_spread_raw", 0.0)
    alpha_ic = metric(metrics, "top30_rankic_alpha_raw", 0.0)
    pos_ratio = metric(metrics, "positive_day_ratio_raw", 0.5)
    worst_day = metric(metrics, "top30_active_worst_day", 0.0)

    components = {
        "rankic_component": clip_component(rankic, 0.05),
        "top30_net_component": clip_component(top30_net, 0.01),
        "utility_component": clip_component(top30_utility, 0.01),
        "spread_component": clip_component(spread, 0.02),
        "alpha_ic_component": clip_component(alpha_ic, 0.05),
        "positive_day_component": clip_component(pos_ratio - 0.50, 0.20),
        "worst_day_penalty": min(0.0, clip_component(worst_day, 0.03)),
    }
    score = (
        0.30 * components["rankic_component"]
        + 0.25 * components["top30_net_component"]
        + 0.15 * components["utility_component"]
        + 0.10 * components["spread_component"]
        + 0.10 * components["alpha_ic_component"]
        + 0.05 * components["positive_day_component"]
        + 0.05 * components["worst_day_penalty"]
    )
    return float(score), components


def score_tac_tb_hit(metrics: dict[str, float]) -> tuple[float, dict[str, float]]:
    rankic_mean = metric_any(metrics, ["mean_daily_rankic_mean", "mean_daily_rankic", "RankIC"], 0.0)
    aggregated_metrics = {
        "tb_top30_hit_rate_mean": metric_any(metrics, ["tb_top30_hit_rate_mean", "tb_top30_hit_rate"]),
        "tb_top30_hit_rate_std": metric(metrics, "tb_top30_hit_rate_std", 0.0),
        "tb_top30_lift_mean": metric_any(metrics, ["tb_top30_lift_mean", "tb_top30_lift"]),
        "tb_top30_lift_std": metric(metrics, "tb_top30_lift_std", 0.0),
        "tb_top30_capture_mean": metric_any(metrics, ["tb_top30_capture_mean", "tb_top30_capture"]),
        "tb_top10_hit_rate_mean": metric_any(metrics, ["tb_top10_hit_rate_mean", "tb_top10_hit_rate"]),
        "tb_top10_hit_rate_std": metric(metrics, "tb_top10_hit_rate_std", 0.0),
        "tb_ndcg_30_mean": metric_any(metrics, ["tb_ndcg_30_mean", "tb_ndcg_30"]),
        "tb_ap_mean": metric_any(metrics, ["tb_ap_mean", "tb_ap"]),
        "tb_auc_mean": metric_any(metrics, ["tb_auc_mean", "tb_auc"]),
        "tb_logloss_mean": metric_any(metrics, ["tb_logloss_mean", "tb_logloss"]),
        "mean_daily_rankic_mean": rankic_mean,
        "mean_daily_rankic_std": metric(metrics, "mean_daily_rankic_std", 0.0),
        "worst_fold_rankic": metric_any(
            metrics,
            ["worst_fold_rankic", "mean_daily_rankic_min", "mean_daily_rankic", "RankIC"],
            rankic_mean,
        ),
    }
    score, components = calc_objective_tac_tb_hit_guarded(
        aggregated_metrics,
        train_valid_rankic_gap=metric(metrics, "train_valid_rankic_gap", 0.0),
    )
    components = {
        **components,
        "tb_top30_hit_rate": aggregated_metrics["tb_top30_hit_rate_mean"],
        "tb_top30_lift": aggregated_metrics["tb_top30_lift_mean"],
        "tb_top30_capture": aggregated_metrics["tb_top30_capture_mean"],
        "tb_top10_hit_rate": aggregated_metrics["tb_top10_hit_rate_mean"],
        "tb_ndcg_30": aggregated_metrics["tb_ndcg_30_mean"],
        "tb_ap": aggregated_metrics["tb_ap_mean"],
        "tb_auc": aggregated_metrics["tb_auc_mean"],
        "tb_logloss": aggregated_metrics["tb_logloss_mean"],
    }
    return float(score), components


def score_10d_alpha(metrics: dict[str, float]) -> tuple[float, dict[str, float]]:
    rankic_mean = metric_any(metrics, ["mean_daily_rankic_mean", "mean_daily_rankic", "RankIC"], 0.0)
    aggregated_metrics = {
        "mean_daily_rankic_mean": rankic_mean,
        "mean_daily_rankic_std": metric(metrics, "mean_daily_rankic_std", 0.0),
        "worst_fold_rankic": metric_any(
            metrics,
            ["worst_fold_rankic", "mean_daily_rankic_min", "mean_daily_rankic", "RankIC"],
            rankic_mean,
        ),
        "top_quintile_spread_scaled_mean": metric_any(
            metrics,
            ["top_quintile_spread_scaled_mean", "top_quintile_spread_scaled", "top_quintile_spread"],
            0.0,
        ),
        "top30_active_utility_scaled_mean": metric_any(
            metrics,
            ["top30_active_utility_scaled_mean", "top30_active_utility_scaled", "top30_active_utility"],
            0.0,
        ),
        "positive_day_ratio_scaled_mean": metric_any(
            metrics,
            ["positive_day_ratio_scaled_mean", "positive_day_ratio_scaled", "positive_day_ratio_raw"],
            0.0,
        ),
        "top30_net_active_mean_raw_mean": metric_any(
            metrics,
            ["top30_net_active_mean_raw_mean", "top30_net_active_mean_raw", "top30_active_mean_raw_mean", "top30_active_mean_raw"],
            0.0,
        ),
    }
    score, components = calc_objective_10_gr_guarded(
        aggregated_metrics,
        train_valid_rankic_gap=metric(metrics, "train_valid_rankic_gap", 0.0),
    )
    return float(score), components


def score_str_alpha(metrics: dict[str, float]) -> tuple[float, dict[str, float]]:
    rankic = metric(metrics, "rank_ic_reb", metric(metrics, "RankIC", 0.0))
    top30_net = metric(metrics, "top30_active_mean_raw", 0.0)
    top30_utility = metric(metrics, "top30_active_utility_raw", 0.0)
    top30_sr = metric(metrics, "top30_sr", 0.0)
    spread = metric(metrics, "top10_spread", metric(metrics, "top_quintile_spread_raw", 0.0))

    components = {
        "rankic_component": clip_component(rankic, 0.05),
        "top30_net_component": clip_component(top30_net, 0.02),
        "utility_component": clip_component(top30_utility, 0.02),
        "top30_sr_component": clip_component(top30_sr, 1.0),
        "spread_component": clip_component(spread, 0.03),
    }
    score = (
        0.35 * components["rankic_component"]
        + 0.25 * components["top30_net_component"]
        + 0.15 * components["utility_component"]
        + 0.15 * components["top30_sr_component"]
        + 0.10 * components["spread_component"]
    )
    return float(score), components


def score_risk(metrics: dict[str, float]) -> tuple[float, dict[str, float]]:
    ap = metric(metrics, "ap_severe_weighted", metric(metrics, "AP_severe", 0.0))
    recall = metric(metrics, "recall_at_precision_80", metric(metrics, "severe_drawdown_recall", 0.0))
    rankic = metric(metrics, "RankIC", 0.0)
    components = {
        "ap_component": clip_component(ap, 0.25),
        "recall_component": clip_component(recall, 0.50),
        "rankic_component": clip_component(rankic, 0.05),
    }
    score = (
        0.55 * components["ap_component"]
        + 0.30 * components["recall_component"]
        + 0.15 * components["rankic_component"]
    )
    return float(score), components


def score_generic(metrics: dict[str, float]) -> tuple[float, dict[str, float]]:
    rankic = metric(metrics, "RankIC", 0.0)
    utility = metric(metrics, "top30_active_utility_raw", 0.0)
    components = {
        "rankic_component": clip_component(rankic, 0.05),
        "utility_component": clip_component(utility, 0.02),
    }
    return float(0.70 * components["rankic_component"] + 0.30 * components["utility_component"]), components


SELECTION_SCORE_ALIASES = {
    "auto": "auto",
    "tac_tb": "tac_tb_hit",
    "tb_hit": "tac_tb_hit",
    "tac_tb_hit": "tac_tb_hit",
    "10d": "10d_alpha_guarded",
    "10d_alpha": "10d_alpha_guarded",
    "10d_alpha_guarded": "10d_alpha_guarded",
    "tac": "tac_alpha",
    "tac_alpha": "tac_alpha",
    "str": "str_alpha",
    "str_alpha": "str_alpha",
    "risk": "risk",
    "generic": "generic",
}


def resolve_selection_score_name(selection_score: str, domain: str, role: str) -> str:
    domain_key = domain.lower()
    role_key = role.lower()
    requested = safe_label(selection_score or "auto")
    if requested not in SELECTION_SCORE_ALIASES:
        valid = ", ".join(sorted(SELECTION_SCORE_ALIASES))
        raise ValueError(f"Invalid selection score: {selection_score}. Valid values: {valid}")

    resolved = SELECTION_SCORE_ALIASES[requested]
    if resolved != "auto":
        return resolved

    if "risk" in role_key:
        return "risk"
    if domain_key == "tac" and ("tb" in role_key or "triple_barrier" in role_key):
        return "tac_tb_hit"
    if domain_key == "10d":
        return "10d_alpha_guarded"
    if domain_key == "tac":
        return "tac_alpha"
    if domain_key in {"str", "20d", "40d"}:
        return "str_alpha"
    return "generic"


def calculate_selection_score(
    domain: str,
    role: str,
    metrics: dict[str, float],
    selection_score: str = "auto",
) -> tuple[float, dict[str, float], str]:
    score_name = resolve_selection_score_name(selection_score, domain, role)
    if score_name == "risk":
        score, components = score_risk(metrics)
        return score, components, score_name
    if score_name == "tac_tb_hit":
        score, components = score_tac_tb_hit(metrics)
        return score, components, score_name
    if score_name == "10d_alpha_guarded":
        score, components = score_10d_alpha(metrics)
        return score, components, score_name
    if score_name == "tac_alpha":
        score, components = score_tac_alpha(metrics)
        return score, components, score_name
    if score_name == "str_alpha":
        score, components = score_str_alpha(metrics)
        return score, components, score_name
    score, components = score_generic(metrics)
    return score, components, score_name


def build_eval_command(args: argparse.Namespace, candidate: Candidate, child_run_name: str) -> list[str]:
    target = target_name(args.domain, args.role)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "train.py"),
        f"domain={args.domain}",
        f"target={target}",
        f"data={args.data}",
        f"features={args.features or default_features_name(args.model, args.domain, args.role)}",
        f"model={args.model}",
        f"period={args.domain}_standard",
        "cv=fixed",
        "+mode=candidate_selection",
        f"mlflow.experiment_name={args.experiment_name or experiment_name(args.domain, args.role)}",
        f"experiment={args.model}_{target}",
        f"++mlflow.run_name={args.parent_run_name}",
        f"++mlflow.child_run_name={child_run_name}",
        f"++mlflow.tags.pipeline_stage=candidate_selection",
        f"++mlflow.tags.source_study={candidate.study_name}",
        f"++mlflow.tags.source_trial={candidate.trial_number}",
        f"++mlflow.tags.source_objective={candidate.objective.label}",
        *candidate_hparam_overrides(candidate, args.model),
        *args.extra_arg,
    ]
    if args.use_gpu:
        if args.model == "lgbm":
            command.append("++hparams.device_type=gpu")
        else:
            command.append("++hparams.device_name=auto")
    return command


def candidate_child_run_name(idx: int, candidate: Candidate) -> str:
    source = candidate_source_label(candidate)
    return f"Candidate_{idx:03d}_{candidate.objective.label}_{source}_trial{candidate.trial_number}"


def find_child_run(client: MlflowClient, experiment_id: str, parent_run_id: str, child_run_name: str):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            f"tags.mlflow.parentRunId = '{parent_run_id}' "
            f"and tags.`mlflow.runName` = '{child_run_name}'"
        ),
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    return runs[0] if runs else None


def evaluate_candidate(
    args: argparse.Namespace,
    candidate: Candidate,
    idx: int,
    total: int,
    child_run_name: str,
    env: dict[str, str],
    experiment_id: str | None,
    parent_run_id: str | None,
) -> dict[str, Any] | None:
    command = build_eval_command(args, candidate, child_run_name)

    print("=" * 80)
    print(f"Evaluating candidate {idx}/{total}")
    print(f"Study: {candidate.study_name}")
    print(f"Trial: {candidate.trial_number}")
    print(f"Objective: {candidate.objective.label}")
    print("Command:", " ".join(command))
    print("=" * 80)

    if args.dry_run:
        score, components, score_name = np.nan, {}, "dry_run"
        eval_run = None
    else:
        result = subprocess.run(command, env=env, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"Candidate failed with return code {result.returncode}. Skipping.")
            return None

        if experiment_id is not None and parent_run_id is not None:
            mlflow.set_tracking_uri(args.tracking_uri)
            client = MlflowClient()
            eval_run = find_child_run(client, experiment_id, parent_run_id, child_run_name)
            if eval_run is None:
                print(f"Could not find MLflow child run: {child_run_name}. Skipping.")
                return None
            score, components, score_name = calculate_selection_score(
                args.domain, args.role, eval_run.data.metrics, args.selection_score
            )
            client.log_metric(eval_run.info.run_id, "selection_score", score)
            for key, value in components.items():
                client.log_metric(eval_run.info.run_id, f"selection_{key}", value)
        else:
            score, components, score_name = np.nan, {}, "unscored"
            eval_run = None

    row = {
        "selection_score": score,
        "selection_score_name": score_name,
        "eval_run_id": eval_run.info.run_id if eval_run else "",
        "eval_run_name": child_run_name,
        "source_study": candidate.study_name,
        "source_trial": candidate.trial_number,
        "source_objective": candidate.objective.label,
        "rank_in_study": candidate.rank_in_study,
        "optuna_value": candidate.optuna_value,
        "params_json": json.dumps(effective_candidate_params(candidate, args.model), sort_keys=True, ensure_ascii=False),
    }
    row.update(components)
    if eval_run:
        for key, value in eval_run.data.metrics.items():
            if key.startswith("fold0_valid_") or key.startswith("fold0_test_") or key == "optimization_score":
                row[key] = value
    return row


def write_outputs(
    args: argparse.Namespace,
    parent_run_id: str | None,
    rows: list[dict[str, Any]],
    timestamp: str,
) -> tuple[Path, Path | None, Path | None]:
    out_dir = PROJECT_ROOT / args.output_dir / target_name(args.domain, args.role)
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = out_dir / f"leaderboard_{args.model}_{target_name(args.domain, args.role)}_{timestamp}.csv"
    selected_path = out_dir / f"selected_{args.model}_{target_name(args.domain, args.role)}_{timestamp}.yaml"

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("selection_score", ascending=False).reset_index(drop=True)
    df.to_csv(leaderboard_path, index=False)

    selected_manifest_path: Path | None = None
    stable_selected_path: Path | None = None
    if not df.empty:
        best = df.iloc[0].to_dict()

        params = json.loads(best["params_json"])
        manifest = {
            "selection": {
                "model": args.model,
                "domain": args.domain,
                "role": args.role,
                "target": target_name(args.domain, args.role),
                "selected_at": timestamp,
                "selection_score": float(best["selection_score"]),
                "selection_score_name": best["selection_score_name"],
                "fixed_cv_config": "config/cv/fixed.yaml",
                "mlflow_run_id": best.get("eval_run_id"),
                "mlflow_parent_run_id": parent_run_id,
                "source_study": best["source_study"],
                "source_trial": int(best["source_trial"]),
                "source_objective": best["source_objective"],
                "optuna_value": float(best["optuna_value"]),
                "params": params,
            }
        }
        with selected_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
        selected_manifest_path = selected_path
        if args.selected_output:
            stable_selected_path = PROJECT_ROOT / args.selected_output
            stable_selected_path.parent.mkdir(parents=True, exist_ok=True)
            with stable_selected_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)

    if parent_run_id and not args.dry_run:
        mlflow.set_tracking_uri(args.tracking_uri)
        with mlflow.start_run(run_id=parent_run_id):
            mlflow.log_artifact(str(leaderboard_path), artifact_path="candidate_selection")
            if selected_manifest_path:
                mlflow.log_artifact(str(selected_manifest_path), artifact_path="candidate_selection")
            if stable_selected_path:
                mlflow.log_artifact(str(stable_selected_path), artifact_path="candidate_selection")

    return leaderboard_path, selected_manifest_path, stable_selected_path


def run_selection(args: argparse.Namespace) -> int:
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be greater than zero.")

    objectives = [parse_objective_spec(v) for v in args.objective]
    target = target_name(args.domain, args.role)
    resolved_selection_score = resolve_selection_score_name(args.selection_score, args.domain, args.role)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.parent_run_name = f"Step5_CandidateSelection_{args.model}_{target}_{timestamp}"
    print(f"Selection score: {resolved_selection_score} (requested: {args.selection_score})")

    candidates = collect_candidates(
        storage=args.storage,
        model=args.model,
        target=target,
        objectives=objectives,
        top_n_per_study=args.top_n_per_study,
        max_candidates=args.max_candidates,
        study_names=args.study,
    )
    if not candidates:
        raise RuntimeError("No Step4 candidates found.")

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    experiment = args.experiment_name or experiment_name(args.domain, args.role)
    mlflow.set_experiment(experiment)
    exp = client.get_experiment_by_name(experiment)
    parent_run_id = None
    if not args.dry_run:
        parent_run_id = get_or_create_parent_run(
            tracking_uri=args.tracking_uri,
            experiment_name=experiment,
            parent_run_name=args.parent_run_name,
        )
        client.set_tag(parent_run_id, "pipeline_stage", "candidate_selection")
        client.set_tag(parent_run_id, "fixed_cv_config", "config/cv/fixed.yaml")
        client.set_tag(parent_run_id, "target", target)
        client.set_tag(parent_run_id, "model", args.model)
        client.set_tag(parent_run_id, "selection_score_name", resolved_selection_score)

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = args.tracking_uri
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("LOKY_MAX_CPU_COUNT", "1")
    if parent_run_id:
        env["MLFLOW_PARENT_RUN_ID"] = parent_run_id

    rows: list[dict[str, Any]] = []
    jobs = [
        (idx, candidate, candidate_child_run_name(idx, candidate))
        for idx, candidate in enumerate(candidates, start=1)
    ]
    experiment_id = exp.experiment_id if exp is not None else None
    n_jobs = min(args.n_jobs, len(jobs))
    if args.dry_run or n_jobs == 1:
        for idx, candidate, child_run_name in jobs:
            row = evaluate_candidate(
                args, candidate, idx, len(candidates), child_run_name, env, experiment_id, parent_run_id
            )
            if row is not None:
                rows.append(row)
    else:
        print(f"Running candidate evaluations with n_jobs={n_jobs}.")
        indexed_rows: list[tuple[int, dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_idx = {
                executor.submit(
                    evaluate_candidate,
                    args,
                    candidate,
                    idx,
                    len(candidates),
                    child_run_name,
                    env,
                    experiment_id,
                    parent_run_id,
                ): idx
                for idx, candidate, child_run_name in jobs
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                row = future.result()
                if row is not None:
                    indexed_rows.append((idx, row))
        rows.extend(row for _, row in sorted(indexed_rows, key=lambda item: item[0]))

    if args.dry_run:
        print("Dry run completed. Leaderboard and selected-candidate manifest were not written.")
        return 0

    leaderboard_path, selected_path, stable_selected_path = write_outputs(args, parent_run_id, rows, timestamp)
    print(f"Leaderboard written: {leaderboard_path}")
    if selected_path:
        print(f"Selected candidate written: {selected_path}")
    if stable_selected_path:
        print(f"Stable selected candidate written: {stable_selected_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--objective", action="append", required=True, help="metric:direction:label")
    parser.add_argument("--study", action="append", default=None, help="Explicit Optuna study name. Can be repeated.")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--storage", default="sqlite:///optuna.db")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--data", default="master")
    parser.add_argument("--features", default=None)
    parser.add_argument("--top-n-per-study", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", default="reports/candidate_selection")
    parser.add_argument("--selected-output", default=None)
    parser.add_argument(
        "--selection-score",
        default="auto",
        help=(
            "Selection score to rank fixed-CV candidates. "
            "Use auto, tac_tb_hit, 10d_alpha_guarded, tac_alpha, str_alpha, risk, or generic."
        ),
    )
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_selection(args)


if __name__ == "__main__":
    raise SystemExit(main())
