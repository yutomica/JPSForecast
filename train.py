import numpy as np
import os
import gc
import hydra
import mlflow
import json
import pandas as pd
import joblib
import tempfile
import copy
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import optuna
from hydra.utils import instantiate, get_class
from hydra.core.hydra_config import HydraConfig
from src.cv.cv_utils import add_t1_column, prepare_purged_cv_input
from src.cv.cv_viz import log_split_info
from src.preprocess.sampling import apply_sampling
from src.preprocess.weights import calculate_time_decay_weights, calculate_sample_weights
from src.models.pipeline import FoldPipeline, EnsembleInferencePipeline
from src.models.pruning import create_pruning_callback
from src.utils.evaluation import evaluate_metrics, calculate_bin_stats
from src.utils.feature_selection import calculate_shap, calculate_mda
from src.utils.mlflow_utils import setup_mlflow_run, check_and_promote_model, bundle_and_upload_artifacts
path_to_gdrive = os.environ.get('path_to_gdrive', '') 
import logging
# alembic のロガーを取得し、ログレベルを WARNING に上げる
logging.getLogger("alembic").setLevel(logging.WARNING)
# ついでに sqlalchemy のログも抑制したい場合は以下も有効です
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
# --- MLflowの不要な警告（DeprecationやCloudPickleのセキュリティ警告）を一時的に抑制 ---
import warnings
mlflow_models_logger = logging.getLogger("mlflow.models.model")
mlflow_pyfunc_logger = logging.getLogger("mlflow.pyfunc")
prev_models_level = mlflow_models_logger.level
prev_pyfunc_level = mlflow_pyfunc_logger.level
mlflow_models_logger.setLevel(logging.ERROR)
mlflow_pyfunc_logger.setLevel(logging.ERROR)
            


def train(cfg: DictConfig) -> float:
    client, experiment_id, parent_run_id, stack = setup_mlflow_run(cfg)

    with stack:
        mlflow.log_param("model_group", cfg.model.get("group", "unknown"))
        # --- コンフィグの保存 ---
        # 全設定を辞書形式にして記録（ドメイン、ターゲット、特徴量、HParams全てが含まれる）
        params = OmegaConf.to_container(cfg, resolve=True)
        feature_cols = params['features'].pop('feature_cols', [])
        cv_summaries = []   # foldごとの期間情報を貯めて、最後にMLflow artifactにする
        mlflow.log_params(params)
        mlflow.log_dict({"feature_cols": feature_cols}, "configs/feature_cols.json")
        print("\n" + "="*60)
        print('🚀 Start training model...')
        print("="*60)

        # --- データ分割ロジックの構築 ---
        master_dir = Path(cfg.data.path)
        meta_df = pd.read_parquet(master_dir / "index_meta.parquet")
        meta_df = meta_df.reset_index(drop=True)
        # ドメイン（戦術/戦略）に応じてフィルタフラグを選択
        print(f"📊 Domain: {cfg.domain.name}")
        if cfg.domain.name == 'TAC':
            mask = meta_df['is_candidate_tac'] == True
            meta_df = meta_df.rename(columns={
                'Future_High_Tac':'Future_High',
                'Future_Low_Tac':'Future_Low',
                'Future_Close_Tac':'Future_Close',
            })
            horizon = 5  # 5日間の予測期間
            interval = cfg.get("interval", {}).get("tac", 5)
            if cfg.model.data_category == 'timeseries': interval = 20
        else:
            mask = meta_df['is_candidate_str'] == True
            meta_df = meta_df.rename(columns={
                'Future_High_Str':'Future_High',
                'Future_Low_Str':'Future_Low',
                'Future_Close_Str':'Future_Close',
            })
            horizon = 60 # 60日間の予測期間
            interval = cfg.get("interval", {}).get("str", 20)
        
        # --- サンプリング ---
        # ユニバース選定
        train_val_meta = meta_df[mask].copy()
        if train_val_meta.empty:
            print(f"⚠️ WARNING: No valid samples found for domain: {cfg.domain.name}. Skipping trial with score -999.0.")
            return -999.0
        # T1（ホライズン終了日）の追加
        train_val_meta = add_t1_column(train_val_meta, horizon)
        
        # --- データ分割・CV ---
        # エンバーゴ（Embargo）日数の設定
        cv_method = HydraConfig.get().runtime.choices.cv
        embargo_td = pd.Timedelta(days=cfg.period.embargo_days)
        print(f"🪓 Split Method: {cv_method}")
        if cv_method == "fixed":
            # 固定分割（Config指定の期間）
            test_start = pd.to_datetime(cfg.period.test_start_date)
            valid_start = pd.to_datetime(cfg.period.valid_start_date)
            test_idx = train_val_meta.index[train_val_meta['date'] >= test_start]
            valid_idx = train_val_meta.index[
                (train_val_meta['date'] >= valid_start) & 
                (train_val_meta['date'] < (test_start - embargo_td))
            ]
            train_idx = train_val_meta.index[train_val_meta['date'] < (valid_start - embargo_td)]
            # 1つの分割としてリスト化
            splits = [(train_idx, valid_idx, test_idx, None, None)]
        else:
            samples_info, date_to_indices, unique_dates = prepare_purged_cv_input(train_val_meta)
            pos_to_date = pd.Series(unique_dates, index=np.arange(len(unique_dates)))
            cv = instantiate(cfg.cv, samples_info_sets=samples_info)
            splits = []
            cv_input = np.zeros((len(unique_dates), 1)) # posベースのダミー入力
            for tr_pos, val_pos in cv.split(X=cv_input, groups=unique_dates):
                tr_dates = unique_dates[tr_pos]
                val_dates = unique_dates[val_pos]
                train_idx = pd.Index(np.concatenate([date_to_indices[pd.Timestamp(d)] for d in tr_dates]))
                valid_idx = pd.Index(np.concatenate([date_to_indices[pd.Timestamp(d)] for d in val_dates]))
                splits.append((train_idx, valid_idx, None, tr_pos, val_pos))
        
        # --- データロード＆前処理 ---
        # 全特徴量名のロード
        all_features = pd.read_json(master_dir / "feature_names.json", typ='series').tolist()
        # 特徴量の指定
        features_choice = HydraConfig.get().runtime.choices.features
        print(f"🧬 Feature select: {features_choice}")
        feature_cols = cfg.features.get('feature_cols', all_features)
        feature_cols = list(dict.fromkeys(feature_cols)) # 特徴量の重複排除 (順序保持)
        col_indices = [all_features.index(c) for c in feature_cols]
        cat_cols = cfg.features.get('cat_cols',[])
        print(f"  - Num of features: {len(feature_cols):,}")
        # print(f"  - Num of cat features: {len(cat_cols):,}")
        # 使う列の「インデックス番号」を特定 numpyのmemmapは、列番号でスライスするのが最も高速です
        features_mmap = np.memmap(
            master_dir / "features.npy", 
            dtype='float32', 
            mode='r', 
            shape=(len(meta_df), len(all_features))
        )
        
        # --- モデルの学習 ---
        print(f"🤖 Training model: {cfg.model.name}")
        target_col = cfg.target.column
        models = []
        all_results = []
        valid_metrics = []
        # スクリーニング結果格納用
        all_fold_mda_values = []
        all_fold_shap_values = []
        fold_pipelines = []
        for i, (train_idx, valid_idx, test_idx, tr_pos, val_pos) in enumerate(splits):
            print(f"\n{'-'*25} Fold {i} {'-'*25}")
            
            # --- 学習データのみ日付間引きを適用 ---
            train_meta_subset = meta_df.loc[train_idx].copy()
            train_meta_sampled = apply_sampling(train_meta_subset, interval)
            train_idx = train_meta_sampled.index
            
            # CVサマリー
            if tr_pos is not None and val_pos is not None:
                info = log_split_info(i, tr_pos, val_pos, pos_to_date)
                cv_summaries.append(info)
            # プリプロセッサのインスタンス化
            prep_params = {
                "save_dir": ".",
                "feature_cols": feature_cols,
                "cat_cols": cat_cols
            }
            if cfg.model.data_category == 'timeseries':
                prep_params['window_size'] = cfg.model.window_size.tac if cfg.domain.name == 'TAC' else cfg.model.window_size.str
            preprocessor_class = get_class(cfg.model.preprocessor_target)
            preprocessor = preprocessor_class(**prep_params)
            # fitパラメータのアップデート
            model_meta_params = {}
            if hasattr(preprocessor, 'cat_idx'): # TabNet
                model_meta_params['cat_idx'] = preprocessor.cat_idx
            if hasattr(preprocessor, 'cat_dims'): # TabNet
                model_meta_params['cat_dims'] = preprocessor.cat_dims
            full_params = OmegaConf.to_container(cfg.hparams, resolve=True)
            full_params.update(model_meta_params)
            print(f"  🔹 Fitting preprocessor (Sampling 100k)...")
            sample_data = features_mmap[:100000, col_indices]
            preprocessor.fit(pd.DataFrame(sample_data, columns=feature_cols))
            # ウェイトの計算 (weights.py のロジックを使用)
            w_train = np.ones(len(train_idx))
            if cfg.hparams.use_time_decay:
                # 学習セットの日付のみを抽出してウェイトを算出 decay_rate は config から取得 (デフォルト: 0.9999)
                decay_rate = cfg.hparams.get('time_decay_rate', 0.9999)
                w_train *= calculate_time_decay_weights(meta_df.loc[train_idx, 'date'], decay_rate=decay_rate)
            w_train *= calculate_sample_weights(meta_df.loc[train_idx, 'log_market_cap'].values, cfg.domain.name)
            # memmap から必要な行のみを読み出し
            print(f"  🔹 Transforming data...")
            X_train = preprocessor.transform(features_mmap, row_indices=train_idx, col_indices=col_indices)
            X_valid = preprocessor.transform(features_mmap, row_indices=valid_idx, col_indices=col_indices)
            y_train = meta_df.loc[train_idx, target_col].values
            y_valid = meta_df.loc[valid_idx, target_col].values
            if test_idx is None or len(test_idx) == 0:
                X_test = None
                y_test = None
                print(f"  🔹 Samples: Train={len(X_train):,}, Valid={len(X_valid):,}")
            else:
                X_test = preprocessor.transform(features_mmap, row_indices=test_idx, col_indices=col_indices)
                y_test = meta_df.loc[test_idx, target_col].values
                print(f"  🔹 Samples: Train={len(X_train):,}, Valid={len(X_valid):,}, Test={len(X_test):,}")
            # モデルのインスタンス化と学習
            model_class = get_class(cfg.model.model_target)
            model = model_class(task_type=cfg.target.task_type, **full_params)
            # --- エポック単位の枝刈り用コールバックの設定 (Sweep時) ---
            # 各Foldのエポックにおいて、それまでのFoldの確定スコアと
            # 現在のエポックスコアの平均（蓄積スコア）を計算して判定する
            # fit_kwargs = {}
            # is_sweep = HydraConfig.get().runtime.choices.get("sweep") not in [None, "null"]
            # if is_sweep:
            #     total_epochs = cfg.hparams.get("max_epochs", cfg.hparams.get("num_boost_round", 1000))
            #     fit_kwargs["epoch_callback"] = create_pruning_callback(
            #         client=client, 
        #         experiment_id=experiment_id, 
            #         parent_run_id=parent_run_id,
            #         fold_idx=i,
            #         past_fold_scores=valid_metrics.copy(),
            #         n_startup_trials=30, 
            #         warmup_ratio=0.3,  # 各Foldの3割終了時点から枝刈り開始
            #         total_epochs=total_epochs
            #     )
            print(f"  🔹 Training model...")
            try:
                # model.fit(X_train, y_train, X_valid, y_valid, sample_weight=w_train, model_idx=i, **fit_kwargs)
                model.fit(X_train, y_train, X_valid, y_valid, sample_weight=w_train, model_idx=i)
            except optuna.exceptions.TrialPruned:
                print(f"  ✂️  Trial pruned at Fold {i}. Stopping trial and returning -999.0.")
                mlflow.log_metric("avg_valid_metrics", -999.0)
                return -999.0
                
            # 予測の実行
            preds = {
                'train': model.predict(X_train),
                'valid': model.predict(X_valid),
                'test':  model.predict(X_test) if X_test is not None else None
            }

            # --- 特徴量スクリーニングロジック ---
            if cfg.get("mode") == "feature_screening":
                print(f"  🔹 [Screening] Calculating SHAP for Fold {i}...")
                abs_shap = calculate_shap(model, X_valid)
                all_fold_shap_values.append(abs_shap)

            # メトリクス算出 (Train / Valid / Test)
            valid_ic = None
            for phase in ['train', 'valid', 'test']:
                if preds[phase] is not None:
                    idx = locals()[f'{phase}_idx']
                    y_true = locals()[f'y_{phase}']
                    # 評価用ICの計算対象として生リターン（Future_Close）を取得
                    y_ret = meta_df.loc[idx, 'Future_Close'].values
                    m = evaluate_metrics(y_true, preds[phase], y_ret=y_ret, task_type=cfg.target.task_type, target_col=target_col)
                    # MLflowにフォールドごとの結果を記録
                    mlflow.log_metrics({f"fold{i}_{phase}_{k}": v for k, v in m.items()})
                    # ValidのSharpe Ratioを収集
                    if phase == 'valid':
                        valid_metrics.append(m['ic'])
                        valid_ic = m['ic']
            # 特徴量精査 (MDA) ロジックの追加
            if cfg.get("mode") == "feature_select":
                print(f"  🔹 [Selection] Calculating MDA for Fold {i}...")
                baseline_score = valid_ic
                y_ret_valid = meta_df.loc[valid_idx, 'Future_Close'].values
                dates_for_shuffle = meta_df.loc[valid_idx, 'date'].values
                fold_mda = calculate_mda(
                    model=model, X_valid=X_valid, y_valid=y_valid, y_ret_valid=y_ret_valid,
                    dates_for_shuffle=dates_for_shuffle, feature_cols=feature_cols,
                    baseline_score=baseline_score, task_type=cfg.target.task_type, target_col=target_col
                )
                all_fold_mda_values.append(fold_mda)
            # ビン分析用データの蓄積 
            # メタデータ(Future_High/Low/Close)を含めてDataFrame化
            for phase in ['valid', 'test']:
                if preds[phase] is not None:
                    idx = locals()[f'{phase}_idx'] # valid_idx or test_idx
                    res_df = pd.DataFrame({
                        'date': meta_df.loc[idx, 'date'],
                        'scode': meta_df.loc[idx, 'scode'],
                        'target': locals()[f'y_{phase}'],
                        'score': preds[phase],
                        'phase': phase,
                        'fold': i
                    }).reset_index(drop=True)
                    # 必要なメタデータを結合
                    meta_cols = ['Future_High', 'Future_Low', 'Future_Close']
                    meta_sub = meta_df.loc[idx, meta_cols].reset_index(drop=True)
                    res_df = pd.concat([res_df, meta_sub], axis=1)
                    all_results.append(res_df)
            del X_train, X_valid, X_test
            gc.collect()
            models.append(copy.deepcopy(model))
            fold_pipelines.append(FoldPipeline(preprocessor, model))
        
        # --- スクリーニング結果の集計と保存 ---
        if cfg.get("mode") == "feature_screening":
            # SHAP集計
            if all_fold_shap_values:
                shap_df = pd.DataFrame(all_fold_shap_values).T
                shap_df.columns = ['fold_'+str(i) for i in range(len(all_fold_shap_values))]
                shap_df.index = feature_cols
                output_filename = f"screening_results_{cfg.domain.name}_{cfg.target.column}.csv"
                shap_df.to_csv(output_filename)
                mlflow.log_artifact(output_filename)
                print(f"✅ Feature screening results saved to {output_filename} and uploaded to MLflow.")
        # MDA (Feature Sharpe) の集計と保存 ---
        if cfg.get("mode") == "feature_select" and all_fold_mda_values:
            mda_df = pd.DataFrame(all_fold_mda_values) # rows=folds, cols=features
            output_filename = f"feature_sharpe_{cfg.model.name}_{cfg.domain.name}_{cfg.target.name}.csv"
            mda_df.to_csv(output_filename)
            mlflow.log_artifact(output_filename)
            print(f"✅ Feature Sharpe results saved to {output_filename} (Group Threshold check needed).")
        # 最適化スコアの算出
        if valid_metrics:
            # nan を無視して平均と標準偏差を計算する
            mean_ic = np.nanmean(valid_metrics)
            std_ic = np.nanstd(valid_metrics)
            avg_valid_metrics = mean_ic - std_ic
            # すべてが nan だった場合（定数予測など）のフォールバック
            if np.isnan(avg_valid_metrics):
                avg_valid_metrics = -1.0
        else:
            print("⚠️ WARNING: No valid metrics (sharpe/corr) found in validation results.")
            avg_valid_metrics = -1.0
        mlflow.log_metric("avg_valid_metrics", avg_valid_metrics)
        

        # --- ビン分析 ---
        full_res_df = pd.concat(all_results, ignore_index=True)
        if cv_method in ["purged_kfold", "cpcv", "anchored_walk_forward"]:
            test_res = full_res_df[full_res_df['phase'] == 'valid']
        else: 
            test_res = full_res_df[full_res_df['phase'] == 'test']
        bin_stats = calculate_bin_stats(
            test_res, score_col='score', target_col='target', task_type=cfg.target.task_type,
            metadata_cols=['Future_High', 'Future_Low', 'Future_Close']
        )

        # --- 成果物（Artifacts）の保存 ---
        with tempfile.TemporaryDirectory() as d:
            # CVサマリー
            json_path = os.path.join(d, "cv_splits.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(cv_summaries, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(json_path, artifact_path="cv")
            # パイプライン
            final_pipeline = EnsembleInferencePipeline(
                fold_pipelines=fold_pipelines,
                col_indices=col_indices
            )
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # mlflow.pyfunc.log_model は内部でメトリクスを再記録しようとしてUNIQUE制約エラーを起こすことがあるため、
                    # save_model と log_artifacts に分割して問題を回避する。
                    model_dir = os.path.join(d, "model_dir")
                    mlflow.pyfunc.save_model(
                        path=model_dir,
                        python_model=final_pipeline,
                        code_paths=["src"] # 依存するコードのパス
                    )
                    mlflow.log_artifacts(model_dir, artifact_path="model")
            finally:
                mlflow_models_logger.setLevel(prev_models_level)
                mlflow_pyfunc_logger.setLevel(prev_pyfunc_level)
                
            # ビン分析
            bin_stats_path = os.path.join(d, "test_bin_analysis.csv")
            bin_stats.to_csv(bin_stats_path)
            mlflow.log_artifact(bin_stats_path)
        # Hydraの最終的なconfigファイル自体も保存（完全な再現用）
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config=cfg, f=f.name)
            mlflow.log_artifact(f.name, artifact_path="config")
        os.remove(f.name)
        
        # --- final_sweep モード時の最高値更新・Staging昇格・OOF保存 ---
        if cfg.get("mode") == "final_sweep" and avg_valid_metrics != -1.0:
            current_run_id = mlflow.active_run().info.run_id
            check_and_promote_model(
                client=client, 
            experiment_id=experiment_id, 
                parent_run_id=parent_run_id, 
                current_run_id=current_run_id, 
                avg_valid_metrics=avg_valid_metrics, 
                full_res_df=full_res_df, 
                cfg=cfg
            )

        # --- MLflow成果物の一括ZIP化とGoogle Driveへの移動 ---
        if cfg.get("output_gdrive", False):
            bundle_and_upload_artifacts(path_to_gdrive, cfg.domain.name)
            print("✅ All artifacts have been bundled into a ZIP file and uploaded to MLflow.")
        
        print("\n" + "="*60)
        print(f"🎯 Trial finished. Score: {avg_valid_metrics:.6f}")
        print("="*60 + "\n")
        return float(avg_valid_metrics)

@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig):
    return train(cfg)

if __name__ == "__main__":
    main()