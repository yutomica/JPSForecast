import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from src.utils.mlflow_utils import get_or_create_parent_run


@dataclass(frozen=True)
class GridPath:
    path_index: int
    params: dict[str, Any]
    overrides: list[str]
    signature: str


def _split_top_level_csv(value: str) -> list[str]:
    parts = []
    start = 0
    depth = 0
    quote = None
    escape = False

    for idx, char in enumerate(value):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in "([{":
            depth += 1
            continue
        if char in ")]}":
            depth -= 1
            continue
        if char == "," and depth == 0:
            parts.append(value[start:idx].strip())
            start = idx + 1

    tail = value[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_choice_token(token: str) -> Any:
    token = token.strip()
    lowered = token.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(token)
    except (SyntaxError, ValueError):
        parsed = yaml.safe_load(token)
        return token if parsed is None and token else parsed


def _parse_param_values(raw_value: Any) -> list[Any]:
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped.startswith("choice(") and stripped.endswith(")"):
            inner = stripped[len("choice("):-1]
            return [_parse_choice_token(part) for part in _split_top_level_csv(inner)]
    return [raw_value]


def _to_hydra_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ",".join(_to_hydra_literal(v) for v in value) + "]"
    if isinstance(value, tuple):
        return "[" + ",".join(_to_hydra_literal(v) for v in value) + "]"
    if isinstance(value, str):
        return value
    return repr(value)


def _canonical_key(override_key: str) -> str:
    return override_key.lstrip("+")


def _canonical_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_canonical_value(v) for v in value]
    if isinstance(value, list):
        return [_canonical_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical_value(v) for k, v in sorted(value.items())}
    return value


def _signature(params: dict[str, Any]) -> str:
    payload = json.dumps(
        {k: _canonical_value(v) for k, v in sorted(params.items())},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def build_grid_paths(sweep_config_path: str | Path, max_paths: int | None = None) -> list[GridPath]:
    with open(sweep_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    explicit_paths = ((cfg.get("step2_grid") or {}).get("paths") or [])
    if explicit_paths:
        paths = []
        for idx, raw_path in enumerate(explicit_paths):
            if not isinstance(raw_path, dict):
                raise ValueError(
                    f"step2_grid.paths[{idx}] must be a mapping of Hydra override keys to values."
                )
            path_params = {
                _canonical_key(str(key)): value
                for key, value in raw_path.items()
            }
            overrides = [
                f"{key}={_to_hydra_literal(value)}"
                for key, value in raw_path.items()
            ]
            paths.append(
                GridPath(
                    path_index=idx,
                    params=path_params,
                    overrides=overrides,
                    signature=_signature(path_params),
                )
            )

        if max_paths is not None:
            return paths[:max_paths]
        return paths

    params = (((cfg.get("hydra") or {}).get("sweeper") or {}).get("params") or {})
    if not params:
        return []

    keys = list(params.keys())
    values_by_key = [_parse_param_values(params[key]) for key in keys]

    paths = []
    for idx, values in enumerate(product(*values_by_key)):
        path_params = {_canonical_key(key): value for key, value in zip(keys, values)}
        overrides = [
            f"{key}={_to_hydra_literal(value)}"
            for key, value in zip(keys, values)
        ]
        paths.append(
            GridPath(
                path_index=idx,
                params=path_params,
                overrides=overrides,
                signature=_signature(path_params),
            )
        )

    if max_paths is not None:
        return paths[:max_paths]
    return paths


def _parse_logged_param(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None


def _get_nested_value(source: dict[str, Any], dotted_key: str) -> Any:
    current: Any = source
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]
    return current


def _run_signature_from_logged_params(run, grid_keys: list[str]) -> str | None:
    params = run.data.params
    resolved_params = {
        key: _parse_logged_param(value)
        for key, value in params.items()
    }

    path_params: dict[str, Any] = {}
    for key in grid_keys:
        if "." not in key:
            if key not in resolved_params:
                return None
            path_params[key] = resolved_params[key]
            continue

        root, nested = key.split(".", 1)
        if root not in resolved_params:
            return None
        try:
            path_params[key] = _get_nested_value(resolved_params[root], nested)
        except KeyError:
            return None

    return _signature(path_params)


def get_completed_signatures(
    tracking_uri: str,
    experiment_name: str,
    parent_run_id: str,
    grid_keys: list[str],
    include_deleted: bool = False,
) -> set[str]:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return set()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        max_results=50000,
        run_view_type=ViewType.ALL if include_deleted else ViewType.ACTIVE_ONLY,
    )

    completed = set()
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        tagged_signature = run.data.tags.get("step2_grid_signature")
        if tagged_signature:
            completed.add(tagged_signature)
            continue
        inferred_signature = _run_signature_from_logged_params(run, grid_keys)
        if inferred_signature:
            completed.add(inferred_signature)
    return completed


def _model_suffix_from_base_args(base_args: list[str], explicit_suffix: str | None = None) -> str:
    if explicit_suffix:
        raw_suffix = explicit_suffix
    else:
        raw_model = "model"
        for arg in base_args:
            if arg.startswith("model="):
                raw_model = arg.split("=", 1)[1]
        raw_suffix = raw_model

    suffix = re.sub(r"[^A-Za-z0-9]+", "_", raw_suffix).strip("_").upper()
    return suffix or "MODEL"


def _parse_trial_number(run_name: str | None, model_suffix: str) -> int | None:
    if not run_name:
        return None
    match = re.fullmatch(rf"Trial_(\d+)_{re.escape(model_suffix)}", run_name)
    if not match:
        return None
    return int(match.group(1))


def get_existing_trial_numbers(
    tracking_uri: str,
    experiment_name: str,
    parent_run_id: str,
    model_suffix: str,
    include_deleted: bool = False,
) -> tuple[set[int], set[int]]:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return set(), set()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        max_results=50000,
        run_view_type=ViewType.ALL if include_deleted else ViewType.ACTIVE_ONLY,
    )

    finished_numbers = set()
    non_finished_numbers = set()
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName")
        number = _parse_trial_number(run_name, model_suffix)
        if number is not None:
            if run.info.status == "FINISHED":
                finished_numbers.add(number)
            else:
                non_finished_numbers.add(number)
    return finished_numbers, non_finished_numbers


def assign_trial_child_run_names(
    tracking_uri: str,
    experiment_name: str,
    parent_run_id: str,
    model_suffix: str,
    n_runs: int,
    include_deleted: bool = False,
    completed_in_grid: int = 0,
) -> list[str]:
    used_numbers, non_finished_numbers = get_existing_trial_numbers(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        parent_run_id=parent_run_id,
        model_suffix=model_suffix,
        include_deleted=include_deleted,
    )
    if non_finished_numbers:
        numbers = ", ".join(str(number) for number in sorted(non_finished_numbers))
        raise RuntimeError(
            "Non-FINISHED child runs using Trial_* names already exist under "
            f"parent_run_id={parent_run_id}: {numbers}. "
            "Delete or resolve those runs before resuming to keep Trial numbers consecutive."
        )
    if not used_numbers and completed_in_grid > 0:
        used_numbers = set(range(completed_in_grid))

    names = []
    candidate = 0
    while len(names) < n_runs:
        if candidate not in used_numbers:
            names.append(f"Trial_{candidate}_{model_suffix}")
            used_numbers.add(candidate)
        candidate += 1
    return names


def build_missing_plan(
    tracking_uri: str,
    experiment_name: str,
    parent_run_name: str,
    sweep_config_path: str | Path,
    max_paths: int | None,
    parent_run_id: str | None = None,
    include_deleted: bool = False,
) -> tuple[str, list[GridPath], set[str], list[GridPath]]:
    effective_include_deleted = include_deleted or parent_run_id is not None
    parent_run_id = get_or_create_parent_run(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        parent_run_name=parent_run_name,
        parent_run_id=parent_run_id,
        include_deleted=effective_include_deleted,
    )
    grid_paths = build_grid_paths(sweep_config_path, max_paths=max_paths)
    grid_keys = sorted({key for path in grid_paths for key in path.params})
    completed = get_completed_signatures(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        parent_run_id=parent_run_id,
        grid_keys=grid_keys,
        include_deleted=effective_include_deleted,
    )
    missing = [path for path in grid_paths if path.signature not in completed]
    return parent_run_id, grid_paths, completed, missing


def _run_single_path(
    train_script: str,
    base_args: list[str],
    path: GridPath,
    parent_run_id: str,
    parent_run_name: str,
    child_run_name: str,
    env: dict[str, str],
) -> int:
    command = [
        sys.executable,
        train_script,
        *base_args,
        f"++mlflow.parent_run_id={parent_run_id}",
        f"++mlflow.child_run_name={child_run_name}",
        f"++mlflow.tags.resolved_parent_run_id={parent_run_id}",
        f"++mlflow.tags.step2_grid_signature={path.signature}",
        f"++mlflow.tags.step2_grid_path_index={path.path_index}",
        f"++mlflow.tags.step2_parent_run_name={parent_run_name}",
        *path.overrides,
    ]

    run_env = env.copy()
    run_env["MLFLOW_PARENT_RUN_ID"] = parent_run_id

    print("=" * 60, flush=True)
    print(
        f"Running missing Step2 path index={path.path_index} "
        f"signature={path.signature} child_run={child_run_name}",
        flush=True,
    )
    print("Overrides: " + " ".join(path.overrides), flush=True)
    print("=" * 60, flush=True)

    completed = subprocess.run(command, env=run_env)
    return completed.returncode


def run_missing_paths(args: argparse.Namespace) -> int:
    parent_run_id, grid_paths, completed, missing = build_missing_plan(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        parent_run_name=args.parent_run_name,
        sweep_config_path=args.sweep_config,
        max_paths=args.max_paths,
        parent_run_id=args.parent_run_id,
        include_deleted=args.include_deleted_runs,
    )

    completed_in_grid = len([path for path in grid_paths if path.signature in completed])
    print(f"Parent Run Name: {args.parent_run_name}")
    print(f"Parent Run ID: {parent_run_id}")
    print(f"Total Grid Paths: {len(grid_paths)}")
    print(f"Completed Grid Paths: {completed_in_grid}")
    print(f"Missing Grid Paths: {len(missing)}")

    if not missing:
        print("All Step2 grid paths are already completed in MLflow. Skipping.")
        return 0

    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.child_run_name_style == "trial":
        model_suffix = _model_suffix_from_base_args(
            args.base_arg,
            explicit_suffix=args.child_run_model_suffix,
        )
        child_run_names = assign_trial_child_run_names(
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            parent_run_id=parent_run_id,
            model_suffix=model_suffix,
            n_runs=len(missing),
            include_deleted=args.include_deleted_runs or args.parent_run_id is not None,
            completed_in_grid=completed_in_grid,
        )
    else:
        child_run_names = [
            (
                f"{args.parent_run_name}_missing_{path.path_index:03d}_"
                f"{path.signature}_{run_label}"
            )
            for path in missing
        ]
    run_jobs = list(zip(missing, child_run_names))

    if args.dry_run:
        for path, child_run_name in run_jobs:
            print(
                f"DRY RUN missing path index={path.path_index} "
                f"signature={path.signature} child_run={child_run_name} "
                f"overrides={' '.join(path.overrides)}"
            )
        return 0

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = args.tracking_uri
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"

    max_workers = max(1, int(args.n_jobs))
    if max_workers == 1:
        for path, child_run_name in run_jobs:
            code = _run_single_path(
                args.train_script,
                args.base_arg,
                path,
                parent_run_id,
                args.parent_run_name,
                child_run_name,
                env,
            )
            if code != 0:
                return code
        return 0

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(
                _run_single_path,
                args.train_script,
                args.base_arg,
                path,
                parent_run_id,
                args.parent_run_name,
                child_run_name,
                env,
            ): path
            for path, child_run_name in run_jobs
        }
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            code = future.result()
            if code != 0:
                failures.append((path, code))

    if failures:
        for path, code in failures:
            print(
                f"ERROR: missing path index={path.path_index} "
                f"signature={path.signature} failed with exit code {code}.",
                file=sys.stderr,
            )
        return failures[0][1]
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--parent-run-name", required=True)
    parser.add_argument("--parent-run-id", default=None)
    parser.add_argument("--include-deleted-runs", action="store_true")
    parser.add_argument("--sweep-config", required=True)
    parser.add_argument("--max-paths", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--train-script", default="train.py")
    parser.add_argument("--base-arg", action="append", default=[])
    parser.add_argument(
        "--child-run-name-style",
        choices=["missing", "trial"],
        default="missing",
    )
    parser.add_argument("--child-run-model-suffix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run_missing_paths(args)


if __name__ == "__main__":
    raise SystemExit(main())
