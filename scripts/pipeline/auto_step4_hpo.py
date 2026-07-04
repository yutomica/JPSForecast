#!/usr/bin/env python3
"""Review, propose, and execute Step4 final HPO rounds.

This script keeps the optimization loop deterministic:
it reads Optuna/MLflow results, applies rule-based diagnostics, writes a
generated sweep config, and can execute several optimization objectives with
the same search space for fair comparison.
"""

from __future__ import annotations

import argparse
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

from src.utils.config_utils import get_trial_count
from src.utils.mlflow_utils import get_or_create_parent_run


DEFAULT_OBJECTIVES = (
    "objective_tac_gr_guarded:maximize:guarded",
    "objective_tac:maximize:tac",
    "RankIC:maximize:rankic",
)

METRIC_KEYS = (
    "optimization_score",
    "objective_tac",
    "objective_tac_gr_guarded",
    "valid_mean_daily_rankic_mean",
    "valid_mean_daily_rankic_std",
    "valid_worst_fold_rankic",
    "valid_top30_active_mean_raw",
    "valid_top20_active_mean_raw",
    "valid_top10_active_mean_raw",
    "valid_top30_rankic_alpha_scaled_mean",
    "valid_top_quintile_spread_scaled_mean",
    "valid_positive_day_ratio_scaled_mean",
    "train_valid_rankic_gap",
    "pooled_oof_RankIC",
    "pooled_oof_top30_active_utility_scaled",
    "pooled_oof_top_quintile_spread_scaled",
    "pooled_oof_positive_day_ratio_scaled",
)


@dataclass(frozen=True)
class ObjectiveSpec:
    metric: str
    direction: str
    label: str


@dataclass(frozen=True)
class ParamSpec:
    key: str
    raw: Any
    low: float | None = None
    high: float | None = None
    log: bool = False
    tunable: bool = False


def safe_label(value: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
    return label or "metric"


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
        raise ValueError(f"Objective direction must be maximize/minimize: {value}")
    return ObjectiveSpec(metric=metric, direction=direction, label=label)


def target_name(domain: str, role: str) -> str:
    return f"{domain}_{role}"


def experiment_name(domain: str, role: str) -> str:
    return f"JPSForecast_{target_name(domain, role)}"


def default_sweep_name(model: str, domain: str, role: str) -> str:
    return f"{model}_{target_name(domain, role)}_final"


def default_features_name(model: str, domain: str, role: str) -> str:
    return f"features_{model}_{target_name(domain, role)}_fixed"


def study_prefix(model: str, domain: str, role: str, objective_label: str | None = None) -> str:
    base = f"final_sweep_{model}_{target_name(domain, role)}"
    if objective_label:
        return f"{base}_{objective_label}"
    return base


def resolve_storage(storage: str) -> str:
    return storage


def latest_study_name(storage: str, prefix: str) -> str:
    summaries = optuna.get_all_study_summaries(storage=storage)
    matches = [s for s in summaries if s.study_name.startswith(prefix)]
    if not matches:
        raise ValueError(f"No Optuna studies found with prefix: {prefix}")
    latest = max(matches, key=lambda s: (s.datetime_start or datetime.min, s.study_name))
    return latest.study_name


def study_to_frame(study: optuna.Study) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value,
        }
        row.update(trial.params)
        rows.append(row)
    return pd.DataFrame(rows)


def complete_trials_frame(study: optuna.Study, direction: str = "maximize") -> pd.DataFrame:
    df = study_to_frame(study)
    if df.empty:
        return df
    df = df[(df["state"] == "COMPLETE") & df["value"].notna()].copy()
    if df.empty:
        return df
    ascending = direction == "minimize"
    return df.sort_values("value", ascending=ascending).reset_index(drop=True)


def parse_interval(raw: Any, key: str) -> ParamSpec:
    if not isinstance(raw, str):
        return ParamSpec(key=key, raw=raw)
    text = raw.strip()
    log_match = re.fullmatch(
        r"tag\(\s*log\s*,\s*interval\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)\s*\)",
        text,
    )
    if log_match:
        return ParamSpec(
            key=key,
            raw=raw,
            low=float(log_match.group(1)),
            high=float(log_match.group(2)),
            log=True,
            tunable=True,
        )
    linear_match = re.fullmatch(r"interval\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)", text)
    if linear_match:
        return ParamSpec(
            key=key,
            raw=raw,
            low=float(linear_match.group(1)),
            high=float(linear_match.group(2)),
            log=False,
            tunable=True,
        )
    return ParamSpec(key=key, raw=raw)


def load_sweep_config(path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_output_sweep_path(output_sweep: str) -> Path:
    raw_path = Path(output_sweep)
    if raw_path.is_absolute():
        return raw_path

    text = output_sweep
    if raw_path.suffix:
        return PROJECT_ROOT / raw_path
    if text.startswith("config/sweep/"):
        return PROJECT_ROOT / f"{text}.yaml"
    if text.startswith("generated/"):
        return PROJECT_ROOT / "config/sweep" / f"{text}.yaml"
    return PROJECT_ROOT / "config/sweep/generated" / f"{text}.yaml"


def sweep_group_from_path(path: Path) -> str:
    config_root = (PROJECT_ROOT / "config/sweep").resolve()
    try:
        relative = path.resolve().relative_to(config_root)
    except ValueError as exc:
        raise ValueError(
            "refine mode requires --output-sweep to be under config/sweep "
            "so Hydra can load it as a sweep config group."
        ) from exc
    return relative.with_suffix("").as_posix()


def sweep_params(config: dict[str, Any]) -> dict[str, Any]:
    return (((config.get("hydra") or {}).get("sweeper") or {}).get("params") or {})


def param_specs(config: dict[str, Any]) -> dict[str, ParamSpec]:
    return {key: parse_interval(value, key) for key, value in sweep_params(config).items()}


def top_fraction(df: pd.DataFrame, frac: float = 0.25) -> pd.DataFrame:
    if df.empty:
        return df
    n = max(1, math.ceil(len(df) * frac))
    return df.head(n).copy()


def numeric_position(value: float, low: float, high: float, log: bool) -> float:
    if high <= low:
        return 0.5
    if log and low > 0 and value > 0:
        return (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    return (value - low) / (high - low)


def propose_one_range(spec: ParamSpec, complete_df: pd.DataFrame) -> tuple[str, str]:
    if not spec.tunable or spec.low is None or spec.high is None or spec.key not in complete_df:
        return str(spec.raw), "fixed or unsupported search expression"

    values = pd.to_numeric(complete_df[spec.key], errors="coerce").dropna()
    if values.empty:
        return str(spec.raw), "no completed values"

    top = top_fraction(complete_df, 0.25)
    top_values = pd.to_numeric(top[spec.key], errors="coerce").dropna()
    if top_values.empty:
        return str(spec.raw), "no top-quartile values"

    low, high = spec.low, spec.high
    width = high - low
    best_value = float(complete_df.iloc[0][spec.key])
    median_top = float(top_values.median())
    min_top = float(top_values.min())
    max_top = float(top_values.max())
    pos_best = numeric_position(best_value, low, high, spec.log)
    pos_median = numeric_position(median_top, low, high, spec.log)

    reason_parts = [
        f"best={best_value:.6g}",
        f"top_q_median={median_top:.6g}",
        f"position={pos_best:.2f}",
    ]

    if spec.log and low > 0:
        log_low, log_high = math.log(low), math.log(high)
        log_width = log_high - log_low
        if pos_best <= 0.20 or pos_median <= 0.30:
            new_low = math.exp(log_low - 0.60 * log_width)
            new_high = math.exp(log_low + 0.65 * log_width)
            reason_parts.append("expanded lower log-bound")
        elif pos_best >= 0.80 or pos_median >= 0.70:
            new_low = math.exp(log_high - 0.65 * log_width)
            new_high = math.exp(log_high + 0.60 * log_width)
            reason_parts.append("expanded upper log-bound")
        else:
            log_top = [math.log(v) for v in top_values if v > 0]
            center = float(np.median(log_top))
            half = max(0.35 * log_width, (max(log_top) - min(log_top)) * 0.75)
            new_low = math.exp(center - half)
            new_high = math.exp(center + half)
            reason_parts.append("tightened around top quartile")
        new_low = max(new_low, 1e-8)
    else:
        if pos_best <= 0.20 or pos_median <= 0.30:
            new_low = low - 0.50 * width
            new_high = low + 0.65 * width
            reason_parts.append("expanded lower bound")
        elif pos_best >= 0.80 or pos_median >= 0.70:
            new_low = high - 0.65 * width
            new_high = high + 0.50 * width
            reason_parts.append("expanded upper bound")
        else:
            center = median_top
            half = max(0.30 * width, (max_top - min_top) * 0.75)
            new_low = center - half
            new_high = center + half
            reason_parts.append("tightened around top quartile")

    if low >= 0:
        new_low = max(0.0, new_low)
    if new_high <= new_low:
        new_high = new_low + max(abs(new_low) * 0.1, 1e-6)

    if spec.log:
        expression = f"tag(log, interval({new_low:.8g}, {new_high:.8g}))"
    else:
        expression = f"interval({new_low:.8g}, {new_high:.8g})"
    return expression, "; ".join(reason_parts)


def propose_params(config: dict[str, Any], complete_df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, str]]:
    specs = param_specs(config)
    proposed: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    for key, spec in specs.items():
        value, reason = propose_one_range(spec, complete_df)
        proposed[key] = value
        reasons[key] = reason
    return proposed, reasons


def get_nested(config: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = config
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def write_generated_sweep(
    base_config: dict[str, Any],
    proposed_params: dict[str, Any],
    out_path: Path,
    source_study: str,
    reasons: dict[str, str],
    n_trials: int | None,
) -> None:
    config = dict(base_config)
    config = yaml.safe_load(yaml.safe_dump(config, sort_keys=False))
    set_nested(config, ("hydra", "sweeper", "params"), proposed_params)
    if n_trials is not None:
        set_nested(config, ("hydra", "sweeper", "n_trials"), int(n_trials))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# @package _global_\n")
        f.write(f"# Generated by scripts/pipeline/auto_step4_hpo.py\n")
        f.write(f"# Source study: {source_study}\n")
        for key, reason in reasons.items():
            f.write(f"# - {key}: {reason}\n")
        body = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        body = re.sub(r"^# '@package _global_'\n", "", body)
        f.write(body)


def find_parent_run(
    tracking_uri: str,
    experiment: str,
    study_name: str | None = None,
    parent_run_name: str | None = None,
):
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return None
    filters = []
    if study_name:
        filters.append(f"tags.optuna_study_name = '{study_name}'")
    if parent_run_name:
        filters.append(f"tags.`mlflow.runName` = '{parent_run_name}'")
    for filter_string in filters:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=filter_string,
            max_results=1,
            order_by=["attributes.start_time DESC"],
        )
        if runs:
            return runs[0]
    return None


def child_runs_frame(tracking_uri: str, experiment: str, parent_run_id: str) -> pd.DataFrame:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return pd.DataFrame()
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        max_results=50000,
        order_by=["attributes.start_time ASC"],
    )
    rows: list[dict[str, Any]] = []
    for run in runs:
        row = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName"),
            "status": run.info.status,
            "start_time": run.info.start_time,
        }
        for key in METRIC_KEYS:
            row[key] = run.data.metrics.get(key, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def review_result(
    study: optuna.Study,
    complete_df: pd.DataFrame,
    child_df: pd.DataFrame,
    config: dict[str, Any],
    direction: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "n_complete": int(len(complete_df)),
        "direction": direction,
        "recommendations": [],
        "param_diagnostics": {},
        "metric_summary": {},
    }
    if complete_df.empty:
        report["recommendations"].append("No complete trials found.")
        return report

    best = complete_df.iloc[0]
    values = pd.to_numeric(complete_df["value"], errors="coerce").dropna()
    report["best_trial"] = int(best["trial"])
    report["best_value"] = float(best["value"])
    report["value_summary"] = {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "max": float(values.max()),
        "std": float(values.std(ddof=0)),
    }

    specs = param_specs(config)
    top = top_fraction(complete_df, 0.25)
    for key, spec in specs.items():
        if not spec.tunable or key not in complete_df:
            continue
        top_values = pd.to_numeric(top[key], errors="coerce").dropna()
        if top_values.empty:
            continue
        best_value = float(best[key])
        pos = numeric_position(best_value, spec.low, spec.high, spec.log)  # type: ignore[arg-type]
        diagnostic = {
            "best": best_value,
            "top_quartile_min": float(top_values.min()),
            "top_quartile_median": float(top_values.median()),
            "top_quartile_max": float(top_values.max()),
            "position": float(pos),
        }
        report["param_diagnostics"][key] = diagnostic
        if pos <= 0.20:
            report["recommendations"].append(f"{key}: best is near lower bound; expand lower side.")
        elif pos >= 0.80:
            report["recommendations"].append(f"{key}: best is near upper bound; expand upper side.")

    if not child_df.empty:
        for key in METRIC_KEYS:
            series = pd.to_numeric(child_df.get(key), errors="coerce").dropna()
            if series.empty:
                continue
            report["metric_summary"][key] = {
                "min": float(series.min()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "max": float(series.max()),
            }
        for key in (
            "valid_top30_active_mean_raw",
            "valid_top20_active_mean_raw",
            "valid_top10_active_mean_raw",
        ):
            series = pd.to_numeric(child_df.get(key), errors="coerce").dropna()
            if len(series) and int((series > 0).sum()) == 0:
                report["recommendations"].append(
                    f"{key}: all completed trials are negative; prefer objective_tac_gr_guarded."
                )
        gap = pd.to_numeric(child_df.get("train_valid_rankic_gap"), errors="coerce").dropna()
        if len(gap) and float(gap.median()) > 0.08:
            report["recommendations"].append(
                "train_valid_rankic_gap median is high; keep overfit penalty or stronger regularization."
            )

    if not report["recommendations"]:
        report["recommendations"].append("No major search-space issue detected.")
    return report


def write_reports(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{report['study_name']}_{stamp}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Step4 HPO Review\n\n")
        f.write(f"- Study: `{report['study_name']}`\n")
        f.write(f"- Complete trials: {report['n_complete']} / {report['n_trials']}\n")
        if "best_trial" in report:
            f.write(f"- Best trial: {report['best_trial']}\n")
            f.write(f"- Best value: {report['best_value']:.8f}\n")
        f.write("\n## Recommendations\n\n")
        for item in report["recommendations"]:
            f.write(f"- {item}\n")
        if report.get("metric_summary"):
            f.write("\n## Metric Summary\n\n")
            for key, stats in report["metric_summary"].items():
                f.write(
                    f"- `{key}`: min={stats['min']:.6f}, "
                    f"mean={stats['mean']:.6f}, median={stats['median']:.6f}, "
                    f"max={stats['max']:.6f}\n"
                )
    return json_path, md_path


def load_review_context(args: argparse.Namespace) -> tuple[str, optuna.Study, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    storage = resolve_storage(args.storage)
    prefix = args.study_prefix or study_prefix(args.model, args.domain, args.role, args.objective_label)
    study_name = args.study_name or latest_study_name(storage, prefix)
    study = optuna.load_study(study_name=study_name, storage=storage)
    direction = args.direction or study.directions[0].name.lower()
    complete_df = complete_trials_frame(study, direction=direction)

    experiment = args.experiment_name or experiment_name(args.domain, args.role)
    parent = find_parent_run(args.tracking_uri, experiment, study_name=study_name)
    child_df = child_runs_frame(args.tracking_uri, experiment, parent.info.run_id) if parent else pd.DataFrame()

    source_sweep = args.source_sweep or default_sweep_name(args.model, args.domain, args.role)
    sweep_path = Path(source_sweep)
    if not sweep_path.is_absolute():
        sweep_path = PROJECT_ROOT / sweep_path
    if not sweep_path.exists():
        sweep_path = PROJECT_ROOT / "config/sweep" / f"{source_sweep}.yaml"
    config = load_sweep_config(sweep_path)
    return study_name, study, complete_df, child_df, config


def command_review(args: argparse.Namespace) -> int:
    _, study, complete_df, child_df, config = load_review_context(args)
    direction = args.direction or study.directions[0].name.lower()
    report = review_result(study, complete_df, child_df, config, direction)
    json_path, md_path = write_reports(report, Path(args.output_dir))
    print(f"Review written: {md_path}")
    print(f"Review JSON: {json_path}")
    print(json.dumps(report["recommendations"], indent=2, ensure_ascii=False))
    return 0


def command_propose(args: argparse.Namespace) -> int:
    study_name, study, complete_df, child_df, config = load_review_context(args)
    if complete_df.empty:
        raise RuntimeError(f"No complete trials found in study: {study_name}")

    proposed, reasons = propose_params(config, complete_df)
    out_path = resolve_output_sweep_path(args.output_sweep)
    n_trials = args.n_trials if args.n_trials is not None else get_nested(config, ("hydra", "sweeper", "n_trials"))
    write_generated_sweep(config, proposed, out_path, study_name, reasons, n_trials)

    direction = args.direction or study.directions[0].name.lower()
    report = review_result(study, complete_df, child_df, config, direction)
    report["generated_sweep"] = str(out_path)
    report["proposed_params"] = proposed
    report["proposal_reasons"] = reasons
    json_path, md_path = write_reports(report, Path(args.output_dir))
    print(f"Generated sweep: {out_path}")
    print(f"Review written: {md_path}")
    print(f"Review JSON: {json_path}")
    return 0


def command_optimize(args: argparse.Namespace) -> int:
    objectives = [parse_objective_spec(v) for v in (args.objective or DEFAULT_OBJECTIVES)]
    sweep_name = args.sweep
    if not sweep_name:
        sweep_name = default_sweep_name(args.model, args.domain, args.role)

    target = target_name(args.domain, args.role)
    experiment = args.experiment_name or experiment_name(args.domain, args.role)
    features = args.features or default_features_name(args.model, args.domain, args.role)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = args.tracking_uri
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("LOKY_MAX_CPU_COUNT", "1")

    for objective in objectives:
        study_name = (
            args.study_name
            if len(objectives) == 1 and args.study_name
            else f"final_sweep_{args.model}_{target}_{objective.label}_{timestamp}"
        )
        completed = get_trial_count(study_name, args.storage, state="COMPLETE")
        total_trials = args.n_trials or args.total_trials
        if total_trials is None:
            sweep_path = PROJECT_ROOT / "config/sweep" / f"{sweep_name}.yaml"
            total_trials = int(get_nested(load_sweep_config(sweep_path), ("hydra", "sweeper", "n_trials"), 50))
        remaining = int(total_trials) - int(completed)
        if remaining <= 0:
            print(f"Skipping {study_name}: all {total_trials} trials complete.")
            continue

        launcher_args: list[str] = []
        if args.n_jobs > 1:
            launcher_args = [
                "hydra/launcher=joblib",
                f"hydra.sweeper.n_jobs={args.n_jobs}",
                f"hydra.launcher.n_jobs={args.n_jobs}",
            ]
        else:
            launcher_args = ["hydra/launcher=basic"]

        data_arg = "data=sample" if args.sample else f"data={args.data}"
        run_env = env.copy()
        parent_run_id = None
        if not args.dry_run:
            parent_run_id = get_or_create_parent_run(
                tracking_uri=args.tracking_uri,
                experiment_name=experiment,
                study_name=study_name,
            )
            run_env["MLFLOW_PARENT_RUN_ID"] = parent_run_id
        run_env["TRIAL_OFFSET"] = str(completed)

        command = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "-m",
            *launcher_args,
            f"hydra.sweeper.direction={objective.direction}",
            f"hydra.sweeper.n_trials={remaining}",
            "++hparams.num_threads=1",
            f"domain={args.domain}",
            f"target={target}",
            f"++target.optimization_metric={objective.metric}",
            f"++target.optimization_direction={objective.direction}",
            data_arg,
            f"features={features}",
            f"model={args.model}",
            f"period={args.domain}_standard",
            "cv=anchored_walk_forward",
            f"mlflow.experiment_name={experiment}",
            f"sweep={sweep_name}",
            "+mode=final_sweep",
            f"++mlflow.run_name=Step4_Final_Sweep_{args.model}_{target}_{objective.label}",
            f"experiment={args.model}_{target}",
            f"hydra.sweeper.storage={args.storage}",
            f"hydra.sweeper.study_name={study_name}",
            *args.extra_arg,
        ]
        if args.use_gpu:
            if args.model == "lgbm":
                command.append("++hparams.device_type=gpu")
            else:
                command.append("++hparams.device_name=auto")
        if args.sample:
            command.append("++hparams.max_epochs=2")

        print("=" * 80)
        print(f"Running objective={objective.metric} direction={objective.direction} label={objective.label}")
        print(f"Study: {study_name}")
        print(f"Sweep: {sweep_name}")
        print(f"Trials: completed={completed}, remaining={remaining}, total={total_trials}")
        print("Command:", " ".join(command))
        print("=" * 80)
        if args.dry_run:
            continue
        result = subprocess.run(command, env=run_env, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            return result.returncode
    return 0


def command_refine(args: argparse.Namespace) -> int:
    if not args.output_sweep:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_sweep = f"generated/{args.model}_{target_name(args.domain, args.role)}_refined_{stamp}"
    command_propose(args)
    args.sweep = sweep_group_from_path(resolve_output_sweep_path(args.output_sweep))
    return command_optimize(args)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--domain", default="tac")
    parser.add_argument("--model", default="lgbm")
    parser.add_argument("--role", default="alpha_gr")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--storage", default="sqlite:///optuna.db")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--study-prefix", default=None)
    parser.add_argument("--objective-label", default=None)
    parser.add_argument("--source-sweep", default=None)
    parser.add_argument("--direction", default=None)
    parser.add_argument("--output-dir", default="reports/hpo")


def add_execution_args(parser: argparse.ArgumentParser, *, include_sweep: bool) -> None:
    parser.add_argument("--objective", action="append", help="metric:direction:label. Can be repeated.")
    if include_sweep:
        parser.add_argument("--sweep", default=None, help="Hydra sweep config group, e.g. lgbm_tac_alpha_gr_final.")
    parser.add_argument("--features", default=None)
    parser.add_argument("--data", default="master")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--total-trials", type=int, default=None)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    review = subparsers.add_parser("review", help="Review the latest or specified Step4 HPO result.")
    add_common_args(review)
    review.set_defaults(func=command_review)

    propose = subparsers.add_parser("propose", help="Generate a new sweep config from HPO diagnostics.")
    add_common_args(propose)
    propose.add_argument("--output-sweep", required=True)
    propose.add_argument("--n-trials", type=int, default=None)
    propose.set_defaults(func=command_propose)

    optimize = subparsers.add_parser("optimize", help="Execute Step4 HPO with the specified sweep config.")
    add_common_args(optimize)
    add_execution_args(optimize, include_sweep=True)
    optimize.set_defaults(func=command_optimize)

    refine = subparsers.add_parser(
        "refine",
        help="Generate a refined sweep config from prior results, then execute Step4 HPO.",
    )
    add_common_args(refine)
    add_execution_args(refine, include_sweep=False)
    refine.add_argument("--output-sweep", default=None)
    refine.set_defaults(func=command_refine)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
