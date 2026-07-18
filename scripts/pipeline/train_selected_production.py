#!/usr/bin/env python3
"""Train and register the Step5-selected candidate as a Production model."""

from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.mlflow_utils import get_or_create_parent_run


def hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError(f"Cannot pass non-finite Hydra value: {value}")
        return repr(value)
    if isinstance(value, (list, dict)):
        return yaml.safe_dump(value, default_flow_style=True).strip()
    return str(value)


def load_selection(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Selected candidate manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError(f"Invalid selected candidate manifest: {path}")

    required = ["model", "domain", "role", "target", "params"]
    missing = [key for key in required if key not in selection]
    if missing:
        raise ValueError(f"Missing required selection fields in {path}: {missing}")

    if not isinstance(selection["params"], dict):
        raise ValueError(f"selection.params must be a mapping in {path}")

    return selection


def canonical_param_key(key: str) -> str:
    return key[1:] if key.startswith("+") else key


def find_param_key(params: dict[str, Any], canonical_key: str) -> str | None:
    for key in params:
        if canonical_param_key(key) == canonical_key:
            return key
    return None


def selected_effective_params(params: dict[str, Any], model: str) -> dict[str, Any]:
    effective = dict(params)
    has_scale_pos_weight = find_param_key(effective, "hparams.scale_pos_weight") is not None
    if model.lower() == "lgbm" and has_scale_pos_weight:
        existing_key = find_param_key(effective, "hparams.is_unbalance")
        if existing_key is not None:
            effective.pop(existing_key)
        effective["hparams.is_unbalance"] = False
    return effective


def selected_param_overrides(params: dict[str, Any], model: str) -> list[str]:
    effective = selected_effective_params(params, model)
    return [f"{key}={hydra_value(value)}" for key, value in effective.items()]


def build_command(args: argparse.Namespace, selection: dict[str, Any], child_run_name: str) -> list[str]:
    model = str(selection["model"])
    domain = str(selection["domain"])
    role = str(selection["role"])
    target = str(selection.get("target") or f"{domain}_{role}")
    features = args.features or f"features_{model}_{target}_fixed"
    experiment = args.experiment_name or f"JPSForecast_{target}"
    run_name = args.run_name or os.environ.get("MLFLOW_RUN_NAME") or f"Step6_Production_{model}_{target}"
    variant = args.variant or str(selection.get("selected_at") or "step5_selected")

    command = [
        sys.executable,
        str(PROJECT_ROOT / "train.py"),
        f"domain={domain}",
        f"target={target}",
        f"data={args.data}",
        f"features={features}",
        f"model={model}",
        f"period={domain}_standard",
        "cv=fixed",
        "+mode=production",
        f"variant={variant}",
        f"experiment={model}_{target}",
        f"mlflow.experiment_name={experiment}",
        f"++mlflow.run_name={run_name}",
        f"++mlflow.child_run_name={child_run_name}",
        "++mlflow.tags.pipeline_stage=production_training",
        f"++mlflow.tags.step5_selected_manifest={args.selected}",
        f"++mlflow.tags.step5_source_run_id={selection.get('mlflow_run_id', '')}",
        f"++mlflow.tags.step5_source_study={selection.get('source_study', '')}",
        f"++mlflow.tags.step5_source_trial={selection.get('source_trial', '')}",
        f"++mlflow.tags.step5_source_objective={selection.get('source_objective', '')}",
        f"++mlflow.tags.step5_selection_score={selection.get('selection_score', '')}",
        f"model.ensemble_size={args.ensemble_size}",
        *selected_param_overrides(selection["params"], model),
        *args.extra_arg,
    ]
    return command


def run(args: argparse.Namespace) -> int:
    selected_path = (PROJECT_ROOT / args.selected).resolve()
    selection = load_selection(selected_path)
    model = str(selection["model"])
    target = str(selection["target"])
    domain = str(selection["domain"])
    experiment = args.experiment_name or f"JPSForecast_{target}"
    run_name = args.run_name or os.environ.get("MLFLOW_RUN_NAME") or f"Step6_Production_{model}_{target}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_trial = selection.get("source_trial", "unknown")
    child_run_name = args.child_run_name or f"Prod_{model}_trial{source_trial}_{timestamp}"

    command = build_command(args, selection, child_run_name)
    printable = " ".join(shlex.quote(part) for part in command)

    print("=" * 80)
    print("Creating Production model from Step5 selection")
    print(f"Selected manifest : {selected_path}")
    print(f"Model / target    : {model} / {target}")
    print(f"Domain            : {domain}")
    print(f"Selection score   : {selection.get('selection_score')}")
    print(f"Source study/trial: {selection.get('source_study')} / {source_trial}")
    print(f"Experiment        : {experiment}")
    print(f"Run name          : {run_name}")
    print(f"Child run         : {child_run_name}")
    print(f"Ensemble size     : {args.ensemble_size}")
    print("Command:")
    print(printable)
    print("=" * 80)

    if args.dry_run:
        return 0

    parent_run_id = get_or_create_parent_run(
        tracking_uri=args.tracking_uri,
        experiment_name=experiment,
        parent_run_name=run_name,
    )

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = args.tracking_uri
    env["MLFLOW_PARENT_RUN_ID"] = parent_run_id
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    return int(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", required=True, help="Path to config/promotion/selected_*.yaml")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--data", default="master")
    parser.add_argument("--features", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--child-run-name", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
