import numpy as np
from datetime import datetime
import os
import hydra
import mlflow
import pandas as pd
import joblib
import tempfile
import shutil
from urllib.parse import urlparse
import copy
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_class
from src.preprocess.common import calculate_time_decay_weights
from src.models.ensemble import EnsembleModel
from src.evaluation import evaluate_metrics, calculate_bin_stats
path_to_gdrive = os.environ.get('path_to_gdrive', '') 

@hydra.main(version_base=None, config_path="../config", config_name="main")
def train(cfg: DictConfig):
    # 1. MLflowの初期設定
    abs_path = os.path.expanduser("~/JPSForecast/mlflow_runs")
    os.makedirs(abs_path, exist_ok=True)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        # --- A. コンフィグの保存 ---
        # 全設定を辞書形式にして記録（ドメイン、ターゲット、特徴量、HParams全てが含まれる）
        params = OmegaConf.to_container(cfg, resolve=True)
        feature_cols = params['features'].pop('feature_cols', [])
        mlflow.log_params(params)
        mlflow.log_dict({"feature_cols": feature_cols}, "configs/feature_cols.json")
        
        # --- B. データの読み込み ---
        # cfg.data.path や cfg.data.type に基づいて読み込み
        df = pd.read_pickle(cfg.data.path)
        df['date'] = pd.to_datetime(df['date'])
        target_col = cfg.target.column
        
        # --- C. モデル固有の前処理実行 ---
        # 前処理の実行（学習データ、検証データの分割含む）
        test_start = pd.to_datetime(cfg.period.test_start_date)
        valid_start = pd.to_datetime(cfg.period.valid_start_date)
        embargo = pd.Timedelta(days=cfg.period.embargo_days)
        mask_test = df['date'] >= test_start
        mask_valid = (df['date'] >= valid_start) & (df['date'] < (test_start - embargo))
        mask_train = df['date'] < (valid_start - embargo)
        # 分割
        train_raw = df[mask_train].reset_index(drop=True)
        valid_raw = df[mask_valid].reset_index(drop=True)
        test_raw = df[mask_test].reset_index(drop=True)
        # ターゲット取得
        y_train = train_raw[target_col].values
        y_valid = valid_raw[target_col].values
        y_test = test_raw[target_col].values        
        # Time Decay
        if cfg.hparams.use_time_decay:
            print("Calculating time decay weights for training data...")
            w_train = calculate_time_decay_weights(train_raw['date'])
        else:
            w_train = None
        # configで指定した前処理クラス (e.g., preprocess_lgbm.py内のクラス) を動的ロード
        print(f"Executing preprocessor: {cfg.model.preprocessor_target}")
        preprocessor_class = get_class(cfg.model.preprocessor_target)
        preprocessor = preprocessor_class(save_dir="artifacts/preprocessor", feature_cols=cfg.features.feature_cols)
        preprocessor.fit(train_raw[cfg.features.feature_cols])
        preprocessor.save('scaler.joblib')
        # transformにより特徴量のみのDataFrameにする
        X_train = preprocessor.transform(train_raw[cfg.features.feature_cols])
        X_valid = preprocessor.transform(valid_raw[cfg.features.feature_cols])
        X_test = preprocessor.transform(test_raw[cfg.features.feature_cols])
        features = X_train.columns.tolist()
        print(f"Features: {len(features)}")
        print(f"Samples: Train={len(X_train)}, Valid={len(X_valid)}, Test={len(X_test)}")
        
        # --- D. モデルの学習 ---
        print(f"Training model: {cfg.model.name}")
        # 前処理オブジェクトからモデル定義に必要な情報を取得
        model_meta_params = {}
        if hasattr(preprocessor, 'cat_idx'): # TabNet
            model_meta_params['cat_idx'] = preprocessor.cat_idx
        if hasattr(preprocessor, 'cat_dims'): # TabNet
            model_meta_params['cat_dims'] = preprocessor.cat_dims
        # 静的なhparamsと動的なmeta_paramsを統合
        # 既存のcfg.hparamsを壊さないようコピーを使用
        full_params = OmegaConf.to_container(cfg.hparams, resolve=True)
        full_params.update(model_meta_params)
        models = []
        for i in range(cfg.model.n_ensembles):
            # ハイパーパラメータを渡してモデルをインスタンス化
            model_class = get_class(cfg.model.model_target)
            cfg.hparams['random_state'] = i + 42  # アンサンブルごとに異なる乱数シードを設定
            model = model_class(task_type=cfg.target.task_type, **full_params)
            print(f"Training ensemble model {i+1}/{cfg.model.n_ensembles}")
            # model_idx を渡す
            model.fit(X_train, y_train, X_valid, y_valid, sample_weight=w_train, model_idx=i)
            models.append(copy.deepcopy(model))
        ensemble_model = EnsembleModel(models)

        # --- E. 評価メトリクスの算出 ---
        datasets = [
            ('Train', X_train, y_train, train_raw),
            ('Valid', X_valid, y_valid, valid_raw),
            ('Test', X_test, y_test, test_raw)
        ]
        all_metrics = {}
        for name, X, y, raw_df in datasets:
            # アンサンブル予測 (平均)
            final_pred = ensemble_model.predict(context=None, model_input=X)
            # 指標計算
            metrics = evaluate_metrics(y, final_pred, cfg.target.task_type, raw_df, prefix=f"{name.lower()}/")
            all_metrics.update(metrics)
            # ビン分析
            # 分析用にメタデータ(日付など)と結合
            df_res = raw_df[['date', 'scode']].copy() if 'scode' in raw_df.columns else pd.DataFrame()
            if 'Entry_Price' in raw_df.columns:
                df_res = pd.concat([df_res, raw_df[['Entry_Price', 'Future_High', 'Future_Low', 'Future_Close']]], axis=1)
            df_res['target'] = y
            df_res['score'] = final_pred
            bin_stats = calculate_bin_stats(df_res, 'score', 'target', cfg.target.task_type)
            with tempfile.TemporaryDirectory() as tmpdir:
                # ExcelまたはCSVとして保存（ご希望のExcel形式の場合）
                file_name = f"{name.lower()}_bin_stats.csv"
                local_path = os.path.join(tmpdir, file_name)
                # Excel出力（openpyxlなどのライブラリが必要です）
                bin_stats.to_csv(local_path, index=False)
                # MLflowにファイルをアップロード
                mlflow.log_artifact(local_path, artifact_path="bin_analysis")
        numeric_metrics = {k: v for k, v in all_metrics.items() if isinstance(v, (int, float, np.number))}
        mlflow.log_metrics(numeric_metrics)

        # --- F. 成果物（Artifacts）の保存 ---
        # 前処理でfitしたプリプロセッサ (StandardScalerなど)
        preprocessor_path = "preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
        # 学習済みモデル
        mlflow.pyfunc.log_model(
            artifact_path="ensemble_model",
            python_model=ensemble_model,
            # 依存ライブラリ（conda_env等）が必要な場合は追加指定
        )
        # Hydraの最終的なconfigファイル自体も保存（完全な再現用）
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config=cfg, f=f.name)
            mlflow.log_artifact(f.name, artifact_path="config")
        os.remove(f.name)
        
        # --- G. MLflow成果物の一括ZIP化とGoogle Driveへの移動 ---
        # 現在の日時を取得してファイル名を作成（YYYY-MM-DD_HH-MM-SS）
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        zip_filename = f"{current_time}_artifacts_bundle"
        # 現在のRunの成果物保存場所（ローカルパス）を取得
        artifact_uri = mlflow.get_artifact_uri()
        local_artifact_path = urlparse(artifact_uri).path
        with tempfile.TemporaryDirectory() as tmp_zip_dir:
            # 圧縮用の一時パス
            zip_temp_path = os.path.join(tmp_zip_dir, zip_filename)
            # MLflowの成果物ディレクトリ全体をZIP圧縮
            # shutil.make_archive(出力先, 形式, 圧縮対象フォルダ)
            if os.path.exists(local_artifact_path):
                shutil.make_archive(zip_temp_path, 'zip', local_artifact_path)
                # 生成されたZIPファイルをGoogle Driveのパスへ移動
                # path_gdrive は cfg.path_gdrive など、適宜コンフィグから読み取ってください
                gdrive_destination = os.path.join(path_to_gdrive,"results_TAC", f"{zip_filename}.zip")
                # 移動先ディレクトリが存在することを確認
                os.makedirs(path_to_gdrive, exist_ok=True)
                # ファイルを移動（shutil.move を使用）
                shutil.move(f"{zip_temp_path}.zip", gdrive_destination)
                print(f"✅ Artifacts bundled and moved to: {gdrive_destination}")
            else:
                print("⚠️ Artifact directory not found. ZIP creation skipped.") 
            
        print("✅ All artifacts have been bundled into a ZIP file and uploaded to MLflow.")

if __name__ == "__main__":
    train()