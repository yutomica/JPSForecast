import json
import os
import tempfile
import warnings

import mlflow
from omegaconf import OmegaConf

from src.evaluation.metrics import calculate_bin_stats
from src.models.pipeline import EnsembleInferencePipeline


def _as_bool(value):
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _restore_mlflow_logger_levels(env_logging_state):
    env_logging_state["mlflow_models_logger"].setLevel(env_logging_state["prev_models_level"])
    env_logging_state["mlflow_pyfunc_logger"].setLevel(env_logging_state["prev_pyfunc_level"])


def _save_cv_summary(tmp_dir, cv_summaries):
    json_path = os.path.join(tmp_dir, "cv_splits.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cv_summaries, f, ensure_ascii=False, indent=2)
    mlflow.log_artifact(json_path, artifact_path="cv")


def _save_inference_pipeline(tmp_dir, fold_pipelines, col_indices, oof_cols, env_logging_state):
    final_pipeline = EnsembleInferencePipeline(
        fold_pipelines=fold_pipelines,
        col_indices=col_indices,
        oof_cols=oof_cols,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # mlflow.pyfunc.log_model は内部でメトリクスを再記録しようとしてUNIQUE制約エラーを起こすことがあるため、
            # save_model と log_artifacts に分割して問題を回避する。
            model_dir = os.path.join(tmp_dir, "model_dir")
            mlflow.pyfunc.save_model(
                path=model_dir,
                python_model=final_pipeline,
                code_paths=["src"],
            )
            mlflow.log_artifacts(model_dir, artifact_path="model")
    finally:
        _restore_mlflow_logger_levels(env_logging_state)


def _save_bin_analysis(tmp_dir, cfg, bin_stats, test_res):
    bin_stats_path = os.path.join(tmp_dir, "test_bin_analysis_daily.csv")
    bin_stats.to_csv(bin_stats_path)
    mlflow.log_artifact(bin_stats_path)

    bin_stats_global = calculate_bin_stats(
        test_res,
        score_col="score",
        target_col="target",
        task_type=cfg.target.task_type,
        metadata_cols=["Future_High", "Future_Low", "Future_Close"],
        date_col="date",
        n_bins=20,
        global_bin=True,
    )
    bin_stats_global_path = os.path.join(tmp_dir, "test_bin_analysis_global.csv")
    bin_stats_global.to_csv(bin_stats_global_path)
    mlflow.log_artifact(bin_stats_global_path)


def _save_hydra_config(cfg):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        OmegaConf.save(config=cfg, f=f.name)
        mlflow.log_artifact(f.name, artifact_path="config")
    os.remove(f.name)


def save_training_artifacts(
    cfg,
    cv_summaries,
    fold_pipelines,
    col_indices,
    oof_cols,
    bin_stats,
    test_res,
    env_logging_state,
):
    """学習後にMLflowへ保存する主要成果物をまとめて記録する。"""
    artifact_cfg = cfg.get("artifacts", {})
    log_model = _as_bool(artifact_cfg.get("log_model", True))

    with tempfile.TemporaryDirectory() as tmp_dir:
        _save_cv_summary(tmp_dir, cv_summaries)
        if log_model:
            _save_inference_pipeline(tmp_dir, fold_pipelines, col_indices, oof_cols, env_logging_state)
        else:
            print("  🔹 Skipping MLflow model artifact logging (artifacts.log_model=false).")
            _restore_mlflow_logger_levels(env_logging_state)
        _save_bin_analysis(tmp_dir, cfg, bin_stats, test_res)

    _save_hydra_config(cfg)
