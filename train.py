import os
from src.utils.env_setup import setup_environment
env_logging_state = setup_environment()

import numpy as np
import gc
import hydra
import mlflow
import pandas as pd
import tempfile
import copy
import shutil
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import optuna
from hydra.utils import instantiate, get_class
from hydra.core.hydra_config import HydraConfig
from src.cv.cv_utils import add_t1_column, prepare_purged_cv_input
from src.cv.cv_viz import log_split_info
from src.models.pipeline import FoldPipeline
from src.evaluation.metrics import evaluate_metrics, calculate_bin_stats, is_extra_bin_metric_key
from src.evaluation.objectives import calculate_final_optimization_score
from src.utils.feature_selection import calculate_shap, calculate_mda
from src.utils.mlflow_utils import setup_mlflow_run, register_and_promote_model_for_mode
from src.utils.stacking_utils import load_stacking_oof, combine_features_with_oof
from src.utils.feature_data import resolve_feature_columns, prepare_feature_memmap, fit_base_preprocessor
from src.utils.training_weights import apply_train_sampling_for_fold, calculate_fold_weights
from src.utils.production_training import train_production_fold
from src.utils.training_artifacts import save_training_artifacts


def select_bin_analysis_data(full_res_df: pd.DataFrame, cv_method: str, mode: str | None) -> tuple[str, pd.DataFrame]:
    """Return the phase used for post-training bin analysis and its rows."""
    validation_phase_modes = {"production", "candidate_selection"}
    validation_phase_cv_methods = {"purged_kfold", "cpcv", "anchored_walk_forward"}

    preferred_phase = (
        "valid"
        if cv_method in validation_phase_cv_methods or mode in validation_phase_modes
        else "test"
    )
    selected = full_res_df[full_res_df["phase"] == preferred_phase]
    if preferred_phase == "test" and selected.empty:
        fallback = full_res_df[full_res_df["phase"] == "valid"]
        if not fallback.empty:
            print(
                "⚠️ WARNING: test phase is empty for bin analysis; "
                "falling back to validation phase."
            )
            return "valid", fallback
    return preferred_phase, selected


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _validate_target_horizon(cfg: DictConfig, domain_name: str, horizon: int) -> None:
    target_horizon = cfg.target.get("prediction_horizon", None)
    if target_horizon is None:
        return
    if int(target_horizon) != horizon:
        raise ValueError(
            f"Domain/target horizon mismatch: domain={domain_name} uses {horizon}d "
            f"but target={cfg.target.name} declares prediction_horizon={target_horizon}."
        )


def _resolve_domain_metadata_spec(cfg: DictConfig) -> dict:
    domain_name = str(cfg.domain.name).upper()
    interval_cfg = cfg.get("interval", {})

    if domain_name == "5D":
        horizon = 5
        interval = interval_cfg.get("tac", 5)
        if cfg.model.data_category == "timeseries":
            interval = 20
        _validate_target_horizon(cfg, domain_name, horizon)
        return {
            "domain_name": domain_name,
            "mask_col": "is_candidate_tac",
            "future_suffix": "Tac",
            "horizon": horizon,
            "interval": interval,
        }

    if domain_name == "60D":
        horizon = 60
        _validate_target_horizon(cfg, domain_name, horizon)
        return {
            "domain_name": domain_name,
            "mask_col": "is_candidate_str",
            "future_suffix": "Str",
            "horizon": horizon,
            "interval": interval_cfg.get("str", 20),
        }

    horizon_by_domain = {"10D": 10, "20D": 20, "40D": 40}
    if domain_name in horizon_by_domain:
        horizon = horizon_by_domain[domain_name]
        _validate_target_horizon(cfg, domain_name, horizon)
        sampling_cfg = cfg.get("preprocess", {}).get("sampling", {})
        return {
            "domain_name": domain_name,
            "mask_col": "is_candidate_tac",
            "future_suffix": f"{horizon}d",
            "horizon": horizon,
            "interval": sampling_cfg.get("interval", horizon),
        }

    raise ValueError(
        f"Unsupported domain: {cfg.domain.name}. "
        "Expected one of 5D, 10D, 20D, 40D, 60D."
    )


def _attach_domain_future_columns(meta_df: pd.DataFrame, cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series, int, int]:
    spec = _resolve_domain_metadata_spec(cfg)
    future_cols = {
        "Future_High": f"Future_High_{spec['future_suffix']}",
        "Future_Low": f"Future_Low_{spec['future_suffix']}",
        "Future_Close": f"Future_Close_{spec['future_suffix']}",
    }
    required_cols = [spec["mask_col"], *future_cols.values()]
    missing_cols = [col for col in required_cols if col not in meta_df.columns]
    if missing_cols:
        raise KeyError(
            f"Required metadata columns are missing for domain={spec['domain_name']}: "
            f"{missing_cols}. Regenerate index_meta.parquet via scripts/pipeline/run_data_pipeline.sh."
        )

    meta_df = meta_df.copy()
    for target_col, source_col in future_cols.items():
        meta_df[target_col] = meta_df[source_col]

    mask = meta_df[spec["mask_col"]].eq(True)
    print(
        "  - Future metadata: "
        f"{future_cols['Future_High']}/{future_cols['Future_Low']}/{future_cols['Future_Close']} "
        "-> Future_High/Future_Low/Future_Close"
    )
    print(f"  - Candidate mask: {spec['mask_col']}, horizon={spec['horizon']}d")
    return meta_df, mask, spec["horizon"], spec["interval"]


def should_cleanup_zarr_cache(path: str, cfg: DictConfig) -> bool:
    """永続TCN sequence cacheは残し、fold単位の一時Zarrだけ削除する。"""
    if not isinstance(path, str) or not path.endswith(".zarr"):
        return False

    hparams = cfg.get("hparams", {})
    if _as_bool(hparams.get("sequence_cache_enabled", False)):
        cache_dir = hparams.get(
            "sequence_cache_dir",
            os.path.join(tempfile.gettempdir(), "jps_tcn_sequence_cache"),
        )
        try:
            cache_root = Path(str(cache_dir)).resolve()
            cache_path = Path(path).resolve()
            if cache_path.is_relative_to(cache_root):
                return False
        except OSError:
            pass

    return os.path.exists(path)


def train(cfg: DictConfig) -> float:
    client, experiment_id, parent_run_id, stack = setup_mlflow_run(cfg)

    with stack:
        # --- タグの記録 ---
        if "tags" in cfg.mlflow:
            mlflow.set_tags(OmegaConf.to_container(cfg.mlflow.tags, resolve=True))
            
        mlflow.log_param("model_group", cfg.model.get("group", "unknown"))
        # --- コンフィグの保存 ---
        # 全設定を辞書形式にして記録（ドメイン、ターゲット、特徴量、HParams全てが含まれる）
        params = OmegaConf.to_container(cfg, resolve=True)
        feature_cols = params['features'].pop('feature_cols', [])
        cv_summaries = []   # foldごとの期間情報を貯めて、最後にMLflow artifactにする
        mlflow.log_params(params)
        mlflow.log_dict({"feature_cols": feature_cols}, "configs/feature_cols.json")

        direction = cfg.target.get("optimization_direction", "maximize")
        fallback_score = 999.0 if direction == "minimize" else -999.0
        fallback_metric = 999.0 if direction == "minimize" else -1.0

        print("\n" + "="*60)
        print('🚀 Start training model...')
        print("="*60)

        # --- データ分割ロジックの構築 ---
        master_dir = Path(cfg.data.path)
        meta_df = pd.read_parquet(master_dir / "index_meta.parquet")
        meta_df = meta_df.reset_index(drop=True)
        # ドメインに応じて候補マスクと評価用Future列を選択
        print(f"📊 Domain: {cfg.domain.name}")
        meta_df, mask, horizon, interval = _attach_domain_future_columns(meta_df, cfg)

        # --- サンプリング ---
        # ユニバース選定
        initial_count = len(meta_df)
        train_val_meta = meta_df[mask].copy()
        print(f"  - Domain filtering: {initial_count:,} -> {len(train_val_meta):,} rows")
        if train_val_meta.empty:
            print(f"⚠️ WARNING: No valid samples found for domain: {cfg.domain.name}. Skipping trial with score {fallback_score}.")
            return fallback_score
        # T1（ホライズン終了日）の追加
        train_val_meta = add_t1_column(train_val_meta, horizon)
        # 目的変数と評価用リターンが欠損しているサンプルを除外
        target_col = cfg.target.column
        count_before_dropna = len(train_val_meta)
        train_val_meta = train_val_meta.dropna(subset=[target_col, 'Future_Close'])
        print(f"  - Dropping NaNs in target/return: {count_before_dropna:,} -> {len(train_val_meta):,} rows")
        if train_val_meta.empty:
            print(f"⚠️ WARNING: No valid samples left after dropping NaNs for {target_col}. Skipping.")
            return fallback_score
            
        # --- スタッキング用 OOFデータの動的ロード ---
        meta_df, train_val_meta, oof_cols = load_stacking_oof(cfg, client, meta_df, train_val_meta)
        
        # --- データ分割・CV ---
        # エンバーゴ（Embargo）日数の設定
        cv_method = HydraConfig.get().runtime.choices.cv
        embargo_td = pd.Timedelta(days=cfg.period.embargo_days)
        print(f"🪓 Split Method: {cv_method}")
        # Always prepare base CV info for visualization and index mapping
        samples_info, date_to_indices, unique_dates = prepare_purged_cv_input(train_val_meta)
        pos_to_date = pd.Series(unique_dates, index=np.arange(len(unique_dates)))
        if cv_method == "fixed":
            # 固定分割（Config指定の期間、cv側の設定を優先）
            test_start = pd.to_datetime(cfg.cv.get('test_start_date', cfg.period.get('test_start_date')))
            valid_start = pd.to_datetime(cfg.cv.get('valid_start_date', cfg.period.get('valid_start_date')))
            train_start = pd.to_datetime(cfg.cv.get('train_start_date', cfg.period.get('train_start_date', '2000-01-01')))
            test_idx = train_val_meta.index[train_val_meta['date'] >= test_start]
            valid_idx = train_val_meta.index[
                (train_val_meta['date'] >= valid_start) & 
                (train_val_meta['date'] < (test_start - embargo_td))
            ]
            train_idx = train_val_meta.index[(train_val_meta['date'] >= train_start) & (train_val_meta['date'] < (valid_start - embargo_td))]
            tr_pos = np.where(np.isin(unique_dates, train_val_meta.loc[train_idx, 'date'].unique()))[0]
            val_pos = np.where(np.isin(unique_dates, train_val_meta.loc[valid_idx, 'date'].unique()))[0]
            te_pos = np.where(np.isin(unique_dates, train_val_meta.loc[test_idx, 'date'].unique()))[0] if not test_idx.empty else None
            splits = [(train_idx, valid_idx, test_idx, tr_pos, val_pos, te_pos)]
        else:
            cv = instantiate(
                cfg.cv, 
                samples_info_sets=samples_info,
                purge_days=cfg.period.purge_days,
                embargo_days=cfg.period.embargo_days
            )
            splits = []
            cv_input = np.zeros((len(unique_dates), 1)) # posベースのダミー入力
            for tr_pos, val_pos in cv.split(X=cv_input, groups=unique_dates):
                tr_dates = unique_dates[tr_pos]
                val_dates = unique_dates[val_pos]
                train_idx = pd.Index(np.concatenate([date_to_indices[pd.Timestamp(d)] for d in tr_dates]))
                valid_idx = pd.Index(np.concatenate([date_to_indices[pd.Timestamp(d)] for d in val_dates]))
                splits.append((train_idx, valid_idx, None, tr_pos, val_pos, None))
        
        # --- データロード＆前処理 ---
        feature_cols, cat_cols = resolve_feature_columns(cfg, master_dir)
        features_array, features_mmap_path, col_indices = prepare_feature_memmap(master_dir,len(meta_df),feature_cols)
        base_preprocessor, model_meta_params = fit_base_preprocessor(cfg,features_array,col_indices,feature_cols,cat_cols)
        
        # --- ハイパーパラメータ設定 ---
        full_params = OmegaConf.to_container(cfg.hparams, resolve=True)
        # ターゲット共通の目的関数設定をマージする。hparams 側の明示設定を優先する。
        if "objective" in cfg.target:
            if "objective" not in full_params and "loss" not in full_params:
                full_params["objective"] = cfg.target.get("objective")
        full_params["early_stopping_metric"] = cfg.target.get("early_stopping_metric", "ic")
        full_params["metric_direction"] = cfg.target.get("metric_direction", "maximize")
        # productionモードでは、最適化のためにensemble_sizeを1に固定する（best_iter探索のため）
        if cfg.get("mode") == "production":
            full_params["ensemble_size"] = 1
        else:
            full_params["ensemble_size"] = cfg.model.get("ensemble_size", 1)
        full_params.update(model_meta_params)

        # --- モデルの学習 ---
        print(f"🤖 Training model: {cfg.model.name}")
        artifact_cfg = cfg.get("artifacts", {})
        log_model_artifact = _as_bool(artifact_cfg.get("log_model", True))
        models = []
        all_results = []
        valid_metrics = []
        train_metrics = []
        fold_metrics_results = []
        # スクリーニング結果格納用
        all_fold_mda_values = []
        all_fold_shap_values = []
        fold_pipelines = []
        for i, (train_idx, valid_idx, test_idx, tr_pos, val_pos, te_pos) in enumerate(splits):
            print(f"{'-'*25} Fold {i} {'-'*25}")
            
            # CVサマリー
            info = log_split_info(i, tr_pos, val_pos, pos_to_date, te_pos=te_pos)
            cv_summaries.append(info)

            # 学習用サンプルの重み付け
            train_idx, stratified_sampling_weights = apply_train_sampling_for_fold(cfg,meta_df,train_idx,target_col,interval,i)
            w_train = calculate_fold_weights(cfg,meta_df,train_idx,target_col,i,stratified_sampling_weights=stratified_sampling_weights,is_train=True)

            # メモリ上の配列から必要な行のみを読み出し
            print(f"  🔹 Transforming data...")
            # 各Foldごとに独立したインスタンスを使用するためディープコピー
            preprocessor = copy.deepcopy(base_preprocessor)
            X_train = preprocessor.transform(features_array, row_indices=train_idx, col_indices=col_indices)
            X_valid = preprocessor.transform(features_array, row_indices=valid_idx, col_indices=col_indices)
            
            # OOF特徴量のオンザフライ結合
            X_train = combine_features_with_oof(X_train, meta_df, train_idx, oof_cols)
            X_valid = combine_features_with_oof(X_valid, meta_df, valid_idx, oof_cols)
            y_train = meta_df.loc[train_idx, target_col].values
            y_valid = meta_df.loc[valid_idx, target_col].values
            
            # 日付情報の取得 (カスタム評価指標でのEra別計算用)
            train_dates = meta_df.loc[train_idx, 'date'].values
            valid_dates = meta_df.loc[valid_idx, 'date'].values

            if test_idx is None or len(test_idx) == 0:
                X_test = None
                y_test = None
                print(f"  🔹 Samples: Train={len(train_idx):,}, Valid={len(valid_idx):,}")
            else:
                X_test = preprocessor.transform(features_array, row_indices=test_idx, col_indices=col_indices)
                X_test = combine_features_with_oof(X_test, meta_df, test_idx, oof_cols)
                y_test = meta_df.loc[test_idx, target_col].values
                print(f"  🔹 Samples: Train={len(train_idx):,}, Valid={len(valid_idx):,}, Test={len(test_idx):,}")
            # モデルのインスタンス化と学習
            model_class = get_class(cfg.model.model_target)
            model = model_class(task_type=cfg.target.task_type, **full_params)
            if hasattr(model, 'device'):
                print(f"  🔹 Using device: {model.device}")

            print(f"  🔹 Training model...")
            try:
                model.fit(
                    X_train, y_train, X_valid, y_valid, sample_weight=w_train,model_idx=i,train_dates=train_dates,valid_dates=valid_dates
                )
            except optuna.exceptions.TrialPruned:
                print(f"  ✂️  Trial pruned at Fold {i}. Stopping trial and returning {fallback_score}.")
                mlflow.log_metric("avg_valid_metrics", fallback_score)
                return fallback_score
                
            # 学習終了時の Best Iteration (Epoch) の記録
            best_iter = getattr(model, 'best_epoch_', None)
            if best_iter is None and hasattr(model, 'model') and hasattr(model.model, 'best_iteration'):
                best_iter = model.model.best_iteration
            if best_iter is not None:
                mlflow.log_metric(f"fold{i}_best_iteration", float(best_iter))

            # 予測の実行
            preds_raw = {
                'train': model.predict(X_train),
                'valid': model.predict(X_valid),
                'test':  model.predict(X_test) if X_test is not None else None
            }
            
            # Production Mode: Step 2 本学習
            if cfg.get("mode") == "production":
                model = train_production_fold(
                    cfg=cfg,fold_idx=i,model_class=model_class,preprocessor=preprocessor,full_params=full_params,
                    best_iter=best_iter,train_idx=train_idx,valid_idx=valid_idx,train_val_meta=train_val_meta,meta_df=meta_df,
                    target_col=target_col,features_array=features_array,col_indices=col_indices,oof_cols=oof_cols,unique_dates=unique_dates,
                    pos_to_date=pos_to_date,stratified_sampling_weights=stratified_sampling_weights,fold_pipelines=fold_pipelines,
                )

            # DataFrame格納用に1D化 (マルチクラスの場合は期待値または代表値)
            preds_1d = {}
            for phase, p in preds_raw.items():
                if p is None:
                    preds_1d[phase] = None
                elif p.ndim == 2:
                    # Multiclass probabilities (N, C) -> Expected class index (N,)
                    preds_1d[phase] = np.dot(p, np.arange(p.shape[1]))
                else:
                    preds_1d[phase] = p

            # 特徴量スクリーニングロジック
            if cfg.get("mode") == "feature_screening":
                print(f"  🔹 [Screening] Calculating SHAP for Fold {i}...")
                abs_shap = calculate_shap(model, X_valid)
                all_fold_shap_values.append(abs_shap)

            # メトリクス算出
            opt_metric_name = cfg.target.get("optimization_metric", cfg.get("optimization_metric", 'ic'))
            eval_metric = cfg.target.get("eval_metric", 'ic')
            valid_score = None
            c_buffer = cfg.get("preprocess", {}).get("matrix_weight", {}).get("cost_buffer", 0.005)
            for phase in ['train', 'valid', 'test']:
                if preds_raw[phase] is not None:
                    idx = locals()[f'{phase}_idx']
                    y_true = locals()[f'y_{phase}']
                    eval_df = pd.DataFrame({'date': meta_df.loc[idx, 'date'].values,'pred': preds_1d[phase],'y_true': y_true,
                        'y_ret': meta_df.loc[idx, 'Future_Close'].values - 1.0,
                    })
                    for c in ['Future_High', 'Future_Low', 'Future_Close']:
                        eval_df[c] = meta_df.loc[idx, c].values
                    m = evaluate_metrics(eval_df,y_pred=preds_raw[phase],task_type=cfg.target.task_type,target_col=target_col,ndcg_k=cfg.get("ndcg_k", 10),cost_buffer=c_buffer,include_extra_bin_metrics=(phase == 'valid'))
                    if phase == 'valid':
                        fold_metrics_results.append({k: v for k, v in m.items() if not is_extra_bin_metric_key(k)})
                    # - MLflowにフォールドごとの結果を記録
                    mlflow.log_metrics({f"fold{i}_{phase}_{k}": v for k, v in m.items()})
                    # - 指定メトリクスを収集
                    if phase == 'valid':
                        score = m.get(eval_metric)
                        if score is None:
                            # 大文字小文字を区別せずに再試行
                            m_lower = {k.lower(): v for k, v in m.items()}
                            score = m_lower.get(eval_metric.lower(), np.nan)
                        valid_metrics.append(score)
                        valid_score = score
                    elif phase == 'train':
                        train_metrics.append(m.get(eval_metric, np.nan))
            
            # 特徴量精査 (MDA) ロジックの追加
            if cfg.get("mode") == "feature_select":
                print(f"  🔹 [Selection] Calculating MDA using {eval_metric} for Fold {i}...")
                baseline_score = valid_score
                y_ret_valid = meta_df.loc[valid_idx, 'Future_Close'].values - 1.0
                dates_for_shuffle = meta_df.loc[valid_idx, 'date'].values
                fold_mda = calculate_mda(
                    model=model, X_valid=X_valid, y_valid=y_valid, y_ret_valid=y_ret_valid,
                    dates_for_shuffle=dates_for_shuffle, feature_cols=feature_cols,
                    baseline_score=baseline_score, task_type=cfg.target.task_type, target_col=target_col,
                    opt_metric=eval_metric
                )
                all_fold_mda_values.append(fold_mda)
                
            # ビン分析用データの蓄積 
            # メタデータ(Future_High/Low/Close)を含めてDataFrame化
            for phase in ['valid', 'test']:
                if preds_raw[phase] is not None:
                    idx = locals()[f'{phase}_idx'] # valid_idx or test_idx
                    res_df = pd.DataFrame({'date': meta_df.loc[idx, 'date'],'scode': meta_df.loc[idx, 'scode'],'target': locals()[f'y_{phase}'],'score': preds_1d[phase],'phase': phase,'fold': i}).reset_index(drop=True)
                    # 必要なメタデータを結合
                    meta_cols = ['Future_High', 'Future_Low', 'Future_Close']
                    meta_sub = meta_df.loc[idx, meta_cols].reset_index(drop=True)
                    res_df = pd.concat([res_df, meta_sub], axis=1)
                    all_results.append(res_df)
            
            # 中間生成されたZarrキャッシュのクリーンアップ
            for x_cache in [X_train, X_valid, X_test]:
                if should_cleanup_zarr_cache(x_cache, cfg):
                    shutil.rmtree(x_cache, ignore_errors=True)
            del X_train, X_valid, X_test
            gc.collect()
            if log_model_artifact:
                models.append(copy.deepcopy(model))
                fold_pipelines.append(FoldPipeline(preprocessor, model))

        # --- スクリーニング結果の集計と保存 ---
        if cfg.get("mode") == "feature_screening":
            # SHAP集計
            if all_fold_shap_values:
                shap_df = pd.DataFrame(all_fold_shap_values).T
                shap_df.columns = ['fold_'+str(i) for i in range(len(all_fold_shap_values))]
                shap_df.index = feature_cols
                output_filename = f"screening_results_{cfg.domain.name}_{cfg.target.name}.csv"
                shap_df.to_csv(output_filename)
                mlflow.log_artifact(output_filename)
                print(f"✅ Feature screening results saved to {output_filename} and uploaded to MLflow.")
        
        # --- MDA (Feature Sharpe) の集計と保存 ---
        if cfg.get("mode") == "feature_select" and all_fold_mda_values:
            mda_df = pd.DataFrame(all_fold_mda_values) # rows=folds, cols=features
            output_filename = f"feature_sharpe_{cfg.model.name}_{cfg.domain.name}_{cfg.target.name}.csv"
            mda_df.to_csv(output_filename)
            mlflow.log_artifact(output_filename)
            print(f"✅ Feature Sharpe results saved to {output_filename} (Group Threshold check needed).")
            
        # --- ビン分析 ---
        full_res_df = pd.concat(all_results, ignore_index=True)
        bin_analysis_phase, test_res = select_bin_analysis_data(
            full_res_df=full_res_df,
            cv_method=cv_method,
            mode=cfg.get("mode"),
        )
        mlflow.log_param("bin_analysis_phase", bin_analysis_phase)
        mlflow.log_metric("bin_analysis_rows", float(len(test_res)))
        if test_res.empty:
            print("⚠️ WARNING: No rows available for bin analysis artifact.")
        if cfg.get("mode") == "target_probe":
            max_fold = test_res['fold'].max()
            metrics = ['sample_count', 'target_mean', 'Future_High_mean', 'Future_Low_mean', 'Future_Close_mean']
            combined_df = pd.DataFrame()
            for f in range(max_fold + 1):
                fold_data = test_res[test_res['fold'] == f]
                if not fold_data.empty:
                    stats = calculate_bin_stats(
                        fold_data, score_col='score', target_col='target', task_type=cfg.target.task_type,
                        metadata_cols=['Future_High', 'Future_Low', 'Future_Close'],
                        date_col='date', n_bins=20
                    )
                    for m in metrics:
                        combined_df[f'fold{f}_{m}'] = stats[m]
            ordered_cols = []
            for m in metrics:
                for f in range(max_fold + 1):
                    col = f'fold{f}_{m}'
                    if col in combined_df.columns:
                        ordered_cols.append(col)
            bin_stats = combined_df[ordered_cols]
            output_filename = f"bin_analysis_{cfg.domain.name}_{cfg.target.name}.csv"
            bin_stats.to_csv(output_filename)
            print(f"✅ Bin analysis results saved to {output_filename}")
        else:
            bin_stats = calculate_bin_stats(
                test_res, score_col='score', target_col='target', task_type=cfg.target.task_type,
                metadata_cols=['Future_High', 'Future_Low', 'Future_Close'],
                date_col='date', n_bins=20
            )
        
        # --- Pooled OOF Metric の算出 ---
        oof_df = full_res_df[full_res_df['phase'] == 'valid']
        if not oof_df.empty:
            print(f"  🔹 Calculating Pooled OOF Metrics (Specialized)...")
            # CPCV等で同一 date, scode に複数予測がある場合は重複排除
            # pred: mean, y_true: first, y_ret: first
            oof_df_clean = oof_df.groupby(['date', 'scode']).agg({'score': 'mean','target': 'first','Future_Close': 'first'}).reset_index()
            eval_df_pooled = pd.DataFrame({'date': oof_df_clean['date'],'pred': oof_df_clean['score'],'y_true': oof_df_clean['target'],'y_ret': oof_df_clean['Future_Close'].values - 1.0})
            c_buffer_pooled = cfg.get("preprocess", {}).get("matrix_weight", {}).get("cost_buffer", 0.005)
            pooled_metrics = evaluate_metrics(eval_df_pooled, cost_buffer=c_buffer_pooled)
            # MLflowにロギング
            mlflow.log_metrics({f"pooled_oof_{k}": v for k, v in pooled_metrics.items()})
        else:
            pooled_metrics = {}
            
        # --- 最適化スコア（Optunaの戻り値）算出 ---
        final_opt_score, objective_log_metrics, objective_messages = calculate_final_optimization_score(valid_metrics=valid_metrics,train_metrics=train_metrics,fold_metrics_results=fold_metrics_results,pooled_metrics=pooled_metrics,opt_metric_name=opt_metric_name,direction=direction,fallback_metric=fallback_metric)
        if objective_log_metrics:
            mlflow.log_metrics(objective_log_metrics)
        for message in objective_messages:
            print(message)
        mlflow.log_metric("optimization_score", final_opt_score)

        # --- 成果物（Artifacts）の保存 ---
        save_training_artifacts(
            cfg=cfg,cv_summaries=cv_summaries,fold_pipelines=fold_pipelines,col_indices=col_indices,oof_cols=oof_cols,bin_stats=bin_stats,
            test_res=test_res,env_logging_state=env_logging_state,
        )

        # --- OOFデータの保存 (Stacking用) ---
        if cfg.get("mode") in ["fix", "stacking_base"]:
            oof_df = full_res_df[full_res_df['phase'] == 'valid'].copy()
            with tempfile.TemporaryDirectory() as d:
                oof_filename = os.path.join(d, f"oof_predictions_{cfg.model.name}_{cfg.target.column}.csv")
                oof_df.to_csv(oof_filename, index=False)
                mlflow.log_artifact(oof_filename, artifact_path="oof_data")
        
        # --- fixモード / productionモード / stacking_baseモード：モデルレジストリへの登録と昇格 ---
        register_and_promote_model_for_mode(client, cfg)
            
        # --- キャッシュファイルのクリーンアップ ---
        if 'features_array' in locals():
            del features_array
            gc.collect()
        if features_mmap_path.exists():
            try:
                features_mmap_path.unlink()
                print(f"✅ Cleaned up mmap cache: {features_mmap_path.name}")
            except OSError as e:
                print(f"⚠️ Failed to remove mmap cache: {e}")
        
        scaled_score = final_opt_score * 100.0
        print("\n" + "="*60)
        print(f"🎯 Trial finished. Raw Score: {final_opt_score:.6f} (Scaled for Optuna: {scaled_score:.6f})")
        print("="*60 + "\n")
        return float(scaled_score)

@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig):
    return train(cfg)

if __name__ == "__main__":
    main()
