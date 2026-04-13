import os
import shutil
import tempfile
import contextlib
import mlflow
from mlflow.tracking import MlflowClient
from urllib.parse import urlparse
from datetime import datetime
from hydra.core.hydra_config import HydraConfig

def setup_mlflow_run(cfg):
    """MLflowの初期設定、実験の復元、Runコンテキスト（親・子）の開始を行う"""
    mlflow_db_path = cfg.mlflow.get("tracking_uri")
    mlflow.set_tracking_uri(mlflow_db_path)
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    if experiment and experiment.lifecycle_stage == 'deleted':
        print(f"Restoring deleted experiment: {cfg.mlflow.experiment_name}")
        client.restore_experiment(experiment.experiment_id)
        
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    # Set後に再度取得し、確実にexperiment_idを取得する
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    experiment_id = experiment.experiment_id
    
    parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
    model_name = cfg.model.get("name", "unknown")
    target_col = cfg.target.get("name", "unknown")
    mode = cfg.get("mode", "train")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base_run_name = f"{model_name}_{target_col}_{mode}_{timestamp}"
    
    if HydraConfig.initialized() and "job" in HydraConfig.get():
        trial_num = HydraConfig.get().job.get("num", "0")
    else:
        trial_num = "0"
    trial_run_name = f"Trial_{trial_num}_{model_name}"

    stack = contextlib.ExitStack()
    if parent_run_id:
        client.set_tag(parent_run_id, "mlflow.runName", base_run_name)
        stack.enter_context(mlflow.start_run(run_id=parent_run_id))
        stack.enter_context(mlflow.start_run(run_name=trial_run_name, nested=True))
    else:
        stack.enter_context(mlflow.start_run(run_name=base_run_name))
        
    return client, experiment_id, parent_run_id, stack

def check_and_promote_model(client, experiment_id, parent_run_id, current_run_id, optimization_score, full_res_df, cfg):
    """過去のRunと比較し、最高値であればStagingに昇格してOOFを保存する"""
    is_best = True
    direction = cfg.get("optimization_direction", "maximize")
    
    if parent_run_id:
        # 同一親ランの過去のランを取得
        past_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
        )
        past_scores = [
            r.data.metrics.get("optimization_score", float("inf") if direction == "minimize" else -float("inf")) 
            for r in past_runs 
            if r.info.run_id != current_run_id
        ]
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
        oof_filename = f"oof_predictions_{cfg.model.name}_{cfg.target.column}.csv"
        oof_df.to_csv(oof_filename, index=False)
        mlflow.log_artifact(oof_filename, artifact_path="oof_data")
        if os.path.exists(oof_filename):
            os.remove(oof_filename)
            
        # モデルレジストリへの登録とStagingへの昇格
        registered_model_name = f"{cfg.model.name}_{cfg.target.column}"
        model_uri = f"runs:/{current_run_id}/model"
        try:
            mv = mlflow.register_model(model_uri, registered_model_name)
            client.transition_model_version_stage(
                name=registered_model_name, version=mv.version, stage="Staging", archive_existing_versions=True
            )
            print(f"✅ Model registered as '{registered_model_name}' (Version {mv.version}) and transitioned to Staging.")
        except Exception as e:
            print(f"⚠️ Failed to register model to registry (Ensure model is logged as PyFunc if required): {e}")

def bundle_and_upload_artifacts(path_to_gdrive, domain_name):
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