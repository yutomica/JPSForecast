import os
import shutil
import tempfile
import pandas as pd
import contextlib
import mlflow
from mlflow.tracking import MlflowClient
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

def _sanitize_tracking_uri(uri: str) -> str:
    """SQLiteの場合、並列実行時のロック対策として timeout を付与する"""
    if uri.startswith("sqlite:///") and "timeout=" not in uri:
        separator = "&" if "?" in uri else "?"
        return f"{uri}{separator}timeout=60"
    return uri

def get_or_create_parent_run(tracking_uri: str, experiment_name: str, study_name: str = None, parent_run_name: str = None) -> str:
    """親ランのIDを取得または作成する。study_name または parent_run_name が指定された場合は再開・紐付けを試みる。"""
    tracking_uri = _sanitize_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    exp = client.get_experiment_by_name(experiment_name)
    if exp and exp.lifecycle_stage == 'deleted':
        client.restore_experiment(exp.experiment_id)
    
    mlflow.set_experiment(experiment_name)
    exp = client.get_experiment_by_name(experiment_name)
    
    # 既存のランを検索 (Study Name優先、次に Parent Run Name)
    if study_name:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.optuna_study_name = '{study_name}'",
            max_results=1
        )
        if runs:
            return str(runs[0].info.run_id)
    
    if parent_run_name:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.'mlflow.runName' = '{parent_run_name}'",
            max_results=1
        )
        if runs:
            return str(runs[0].info.run_id)
            
    # 新規作成
    default_name = f"Sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    actual_run_name = parent_run_name or default_name
    run = client.create_run(experiment_id=exp.experiment_id, run_name=actual_run_name)
    if study_name:
        client.set_tag(run.info.run_id, "optuna_study_name", study_name)
    
    return str(run.info.run_id)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["resolve_parent"], required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--parent-run-name", default=None)
    args = parser.parse_args()
    
    if args.action == "resolve_parent":
        print(get_or_create_parent_run(args.tracking_uri, args.experiment_name, args.study_name, args.parent_run_name))

def setup_mlflow_run(cfg: DictConfig) -> tuple[MlflowClient, str, str | None, contextlib.ExitStack]:
    """MLflowの初期設定、実験の復元、Runコンテキスト（親・子）の開始を行う"""
    # --- tracking_uri の設定 ---
    # Hydraは実行時にカレントディレクトリを変更するため、相対パスは実行場所によって変わってしまう。
    # これを防ぐため、configで指定された相対パスをプロジェクトルートからの絶対パスに変換する。
    # 1. 環境変数 MLFLOW_TRACKING_URI があれば最優先
    # 2. なければ、configの tracking_uri を絶対パスに変換して使用

    # このファイルの場所からプロジェクトルートを特定 (src/utils/mlflow_utils.py -> JPSForecast/)
    project_root = Path(__file__).absolute().parents[2]

    # configで指定されたパス (例: "sqlite:///mlflow.db") を絶対パスに変換
    relative_uri = cfg.mlflow.get("tracking_uri", "sqlite:///mlflow.db")
    if relative_uri.startswith("sqlite:///"):
        db_filename = relative_uri.replace("sqlite:///", "")
        absolute_uri = f"sqlite:///{project_root / db_filename}"
    else:
        absolute_uri = relative_uri # http://, file:// などの場合はそのまま

    mlflow_db_path = os.environ.get("MLFLOW_TRACKING_URI", absolute_uri)
    mlflow_db_path = _sanitize_tracking_uri(mlflow_db_path)
    mlflow.set_tracking_uri(mlflow_db_path)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    
    if experiment is None:
        artifact_location = (project_root / "mlruns").as_uri()
        try:
            client.create_experiment(
                cfg.mlflow.experiment_name,
                artifact_location=artifact_location
            )
        except Exception:
            pass  # 他の並列プロセスがほぼ同時に作成した場合はエラーを無視
        experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    elif experiment.lifecycle_stage == 'deleted':
        print(f"Restoring deleted experiment: {cfg.mlflow.experiment_name}")
        client.restore_experiment(experiment.experiment_id)
        
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    # Set後に再度取得し、確実にexperiment_idを取得する
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    experiment_id = str(experiment.experiment_id)

    parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
    model_name = cfg.model.get("name", "unknown")
    target_col = cfg.target.get("name", "unknown")
    mode = cfg.get("mode", "train")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    # run_nameが明示的に指定されている場合はそれを使用し、なければ自動生成する
    base_run_name = cfg.mlflow.get("run_name") or f"{model_name}_{target_col}_{mode}_{timestamp}"

    if HydraConfig.initialized() and "job" in HydraConfig.get():
        raw_num = int(HydraConfig.get().job.get("num", "0"))
        offset = int(os.environ.get("TRIAL_OFFSET", "0"))
        trial_num = raw_num + offset
    else:
        trial_num = "0"
        
    trial_run_name = cfg.mlflow.get("child_run_name") or f"Trial_{trial_num}_{model_name}"

    stack = contextlib.ExitStack()
    if parent_run_id:
        client.set_tag(parent_run_id, "mlflow.runName", base_run_name)
        stack.enter_context(mlflow.start_run(run_id=parent_run_id))
        stack.enter_context(mlflow.start_run(run_name=trial_run_name, nested=True))
    else:
        stack.enter_context(mlflow.start_run(run_name=base_run_name))

    return client, experiment_id, parent_run_id, stack

def check_and_promote_model(client: MlflowClient, experiment_id: str, parent_run_id: str | None, current_run_id: str, optimization_score: float, full_res_df: pd.DataFrame, cfg: DictConfig):
    """過去のRunと比較し、最高値であればStagingに昇格してOOFを保存する"""
    is_best = True
    direction = cfg.get("optimization_direction", "maximize")

    if parent_run_id:
        # 同一親ランの過去のランを取得
        past_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
        )
        fallback_score = float("inf") if direction == "minimize" else -float("inf")
        past_scores = []
        for r in past_runs:
            if r.info.run_id != current_run_id:
                score = r.data.metrics.get("optimization_score", fallback_score)
                if score is not None: # 稀にNoneが返るケースを考慮
                    past_scores.append(score)

        if past_scores:
            if direction == "minimize":
                best_past_score = min(past_scores)
                if optimization_score >= best_past_score:
                    is_best = False
            else:
                best_past_score = max(past_scores)
                if optimization_score <= best_past_score:
                    is_best = False

    if is_best:
        print(f"\n🌟 New best score ({optimization_score:.6f}) achieved! Promoting to Staging and saving OOF data.")
        # OOFデータの保存 (Stacking用)
        oof_df = full_res_df[full_res_df['phase'] == 'valid'].copy()
        with tempfile.TemporaryDirectory() as d:
            oof_filename = os.path.join(d, f"oof_predictions_{cfg.model.name}_{cfg.target.column}.csv")
            oof_df.to_csv(oof_filename, index=False)
            mlflow.log_artifact(oof_filename, artifact_path="oof_data")

        # モデルレジストリへの登録とStagingへの昇格
        registered_model_name = f"{cfg.model.name}_{cfg.target.name}"
        model_uri = f"runs:/{current_run_id}/model"
        try:
            mv = mlflow.register_model(model_uri, registered_model_name)
            # Variant管理のため archive_existing_versions=False に変更
            client.transition_model_version_stage(
                name=registered_model_name, version=mv.version, stage="Staging", archive_existing_versions=False
            )
            
            # タグの付与
            variant = cfg.get("variant", "default")
            client.set_model_version_tag(registered_model_name, mv.version, "variant", variant)
            feature_choice = HydraConfig.get().runtime.choices.get("features", "unknown")
            client.set_model_version_tag(registered_model_name, mv.version, "feature_config", feature_choice)
            
            print(f"✅ Model registered as '{registered_model_name}' (Version {mv.version}) with variant '{variant}' and transitioned to Staging.")
        except Exception as e:
            print(f"⚠️ Failed to register model to registry (Ensure model is logged as PyFunc if required): {e}")

def bundle_and_upload_artifacts(path_to_gdrive: str, domain_name: str):
    """MLflowの成果物ディレクトリ全体をZIP圧縮し、Google Driveへ移動する"""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    zip_filename = f"{current_time}_artifacts_bundle"
    artifact_uri = mlflow.get_artifact_uri()
    local_artifact_path = urlparse(artifact_uri).path

    with tempfile.TemporaryDirectory() as tmp_zip_dir:
        zip_temp_path = os.path.join(tmp_zip_dir, zip_filename)
        if os.path.exists(local_artifact_path):
            shutil.make_archive(zip_temp_path, 'zip', local_artifact_path)
            # 動的にドメイン名(TAC等)を含んだパスを生成する
            gdrive_destination = os.path.join(path_to_gdrive, f"results_{domain_name}", f"{zip_filename}.zip")
            os.makedirs(os.path.dirname(gdrive_destination), exist_ok=True)
            shutil.move(f"{zip_temp_path}.zip", gdrive_destination)
            print(f"✅ Artifacts bundled and moved to: {gdrive_destination}")
        else:
            print("⚠️ Artifact directory not found. ZIP creation skipped.")