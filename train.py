import os

# 各種ライブラリを読み込む前に、パフォーマンス安定化のための環境変数・スレッド制限を設定する
from src.utils.env_setup import setup_environment
setup_environment()

import numpy as np
import gc
import hydra
import mlflow
import json
import pandas as pd
import tempfile
import copy
from pathlib import Path
import hashlib
from omegaconf import DictConfig, OmegaConf
import optuna
from hydra.utils import instantiate, get_class
from hydra.core.hydra_config import HydraConfig
from src.cv.cv_utils import add_t1_column, prepare_purged_cv_input
from src.cv.cv_viz import log_split_info
from src.preprocess.weights import calculate_time_decay_weights, calculate_sample_weights
from src.models.pipeline import FoldPipeline, EnsembleInferencePipeline
from src.models.pruning import create_pruning_callback
from src.evaluation.metrics import (
    evaluate_metrics, calculate_bin_stats,
    is_extra_bin_metric_key,
)
from src.evaluation.objectives import (
    aggregate_fold_metrics, calc_objective_v2,
    calc_tac_risk_objective
)
from src.utils.feature_selection import calculate_shap, calculate_mda, calculate_cfi
from src.utils.mlflow_utils import setup_mlflow_run, check_and_promote_model, bundle_and_upload_artifacts
from src.utils.sampling import (
    apply_sampling, apply_target_stratified_sampling, apply_2d_matrix_weight,
    make_train_fold_class_weight, make_sample_weight, apply_hard_negative_weighting
)
from src.utils.stacking_utils import load_stacking_oof, combine_features_with_oof
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

        # Parquetチャンクから必要な特徴量のみをメモリにロード
        # DL向けの並列処理や時系列ウィンドウ化を見据え、scode, date順にソートされたParquetチャンクを利用する
        features_dir = master_dir / "features"
        if not features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {features_dir}")
        print("  - Preparing shared memory map for raw features...")
        cols_hash = hashlib.md5(",".join(feature_cols).encode()).hexdigest()[:8]
        features_mmap_path = master_dir / f"features_array_{cols_hash}.npy"
        lock_path = master_dir / f"features_array_{cols_hash}.lock"

        # 最初のプロセスのみが mmap キャッシュを作成し、他プロセスはそれを待機してアタッチする
        if not features_mmap_path.exists() or lock_path.exists():
            try:
                # ロックファイルが古すぎる場合（例：10分以上前）は、前回の残骸とみなして削除する
                import time
                if lock_path.exists() and (time.time() - lock_path.stat().st_mtime > 600):
                    print("  - Found stale lock file. Removing it...")
                    lock_path.unlink(missing_ok=True)

                lock_path.touch(exist_ok=False)
                print(f"  - Building mmap cache: {features_mmap_path.name}")
                chunk_files = sorted(features_dir.glob("features_chunk_*.parquet"))
                try:
                    shape = (len(meta_df), len(feature_cols))
                    mmap_arr = np.memmap(features_mmap_path, dtype='float32', mode='w+', shape=shape)
                    
                    current_row = 0
                    for cf in chunk_files:
                        df_chunk = pd.read_parquet(cf, columns=feature_cols)
                        chunk_len = len(df_chunk)
                        mmap_arr[current_row : current_row+chunk_len] = df_chunk.values.astype('float32')
                        current_row += chunk_len
                    
                    mmap_arr.flush()
                    del mmap_arr
                    gc.collect()
                finally:
                    lock_path.unlink(missing_ok=True)
            except FileExistsError:
                print("  - Waiting for other process to finish building mmap cache...")
                import time
                while lock_path.exists():
                    time.sleep(2)
        print("  - Attaching to shared memory map...")
        features_array = np.memmap(features_mmap_path, dtype='float32', mode='r', shape=(len(meta_df), len(feature_cols)))
        # train.py内での後続処理の互換性のため、列のインデックスマッピングを更新
        # 読み込んだ時点で配列の列は `feature_cols` と同一になるため
        col_indices = list(range(len(feature_cols)))
        
        # --- プリプロセッサの初期化と事前学習 (Fold間で共通) ---
        print(f"🔹 Fitting preprocessor (Sampling 100k)...")
        prep_params = {
            "save_dir": ".",
            "feature_cols": feature_cols,
            "cat_cols": cat_cols
        }
        if cfg.model.data_category == 'timeseries': prep_params['window_size'] = cfg.hparams.get("window_size", 20)
        preprocessor_class = get_class(cfg.model.preprocessor_target)
        base_preprocessor = preprocessor_class(**prep_params)
        sample_data = features_array[:100000, col_indices]
        base_preprocessor.fit(pd.DataFrame(sample_data, columns=feature_cols))
        
        # fitパラメータのアップデート
        model_meta_params = {}
        if hasattr(base_preprocessor, 'cat_idx'): # TabNet
            model_meta_params['cat_idx'] = base_preprocessor.cat_idx
        if hasattr(base_preprocessor, 'cat_dims'): # TabNet
            model_meta_params['cat_dims'] = base_preprocessor.cat_dims
        full_params = OmegaConf.to_container(cfg.hparams, resolve=True)
        # ターゲット共通の目的関数設定をマージ (hparams側に設定がない場合のみターゲットから取得)
        if "objective" in cfg.target:
            if "objective" not in full_params and "loss" not in full_params:
                full_params["objective"] = cfg.target.get("objective")
        # 目的関数に関連するパラメータをターゲットから取得
        # hparams側が優先されるようにして、個別のチューニングを許容する
        target_keys = ["fair_c", "tweedie_variance_power", "asym_alpha", "asym_beta", "quantile", "huber_alpha", "target_transform", "custom_objective", "custom_metric"]
        for key in target_keys:
            if key == "huber_alpha" and ("alpha" in full_params or "+alpha" in full_params):
                continue
            if key in cfg.target and key not in full_params:
                full_params[key] = cfg.target[key]
        # モデルごとのパラメータ名マッピング
        obj = cfg.target.get("objective")
        if obj == "asymmetric_mse":
            # TCN/FT-Transformer は alpha/beta という名前を期待する場合がある (Wrapper実装に合わせる)
            if cfg.model.name.lower() in ["tcn", "ft_transformer"]:
                if "asym_alpha" in full_params and "alpha" not in full_params:
                    full_params["alpha"] = full_params["asym_alpha"]
                if "asym_beta" in full_params and "beta" not in full_params:
                    full_params["beta"] = full_params["asym_beta"]
        elif obj == "quantile":
            # quantile パラメータを各モデルが期待する alpha 等に変換
            if cfg.model.name.lower() in ["tcn", "ft_transformer", "lgbm"]:
                if "quantile" in full_params and "alpha" not in full_params:
                    full_params["alpha"] = full_params["quantile"]
        # early stopping
        full_params["early_stopping_metric"] = cfg.target.get("early_stopping_metric", "ic")
        full_params["metric_direction"] = cfg.target.get("metric_direction", "maximize")
        # Always store the original ensemble_size from config
        ensemble_size_orig = cfg.model.get("ensemble_size", 1)
        # For initial fold training (Step 1 or normal CV), use ensemble_size from config
        # unless in production mode, where we want to find best_iter quickly.
        if cfg.get("mode") == "production":
            full_params["ensemble_size"] = 1
        else:
            full_params["ensemble_size"] = ensemble_size_orig
            
        full_params.update(model_meta_params)

        # --- モデルの学習 ---
        print(f"🤖 Training model: {cfg.model.name}")
        models = []
        all_results = []
        valid_metrics = []
        train_metrics = []
        fold_metrics_results = []
        # スクリーニング結果格納用
        all_fold_mda_values = []
        all_fold_cfi_values = []
        all_fold_shap_values = []
        fold_pipelines = []
        for i, (train_idx, valid_idx, test_idx, tr_pos, val_pos, te_pos) in enumerate(splits):
            print(f"{'-'*25} Fold {i} {'-'*25}")
            
            # CVサマリー
            info = log_split_info(i, tr_pos, val_pos, pos_to_date, te_pos=te_pos)
            cv_summaries.append(info)
            
            # 学習データのみ Date-interval サンプリングを適用
            if cfg.get("preprocess", {}).get("sampling", {}).get("enabled", False):
                print("  🔹 Applying date-interval sampling...")
                count_before_sampling = len(train_idx)
                sampling_interval = cfg.preprocess.sampling.get("interval", interval)
                train_meta_subset = meta_df.loc[train_idx].copy()
                train_meta_processed = apply_sampling(train_meta_subset, sampling_interval)
                train_idx = train_meta_processed.index
                print(f"    - Samples reduced: {count_before_sampling:,} -> {len(train_idx):,}")

            # 学習データのみターゲット層化サンプリングを適用
            stratified_sampling_weights = None
            if cfg.get("preprocess", {}).get("target_stratified_sampling", {}).get("enabled", False):
                sampling_cfg = cfg.preprocess.target_stratified_sampling
                mode = sampling_cfg.get('mode', 'mode_1')
                print(f"  🔹 Applying target stratified sampling (mode: {mode})...")
                count_before_stratified = len(train_idx)
                train_meta_subset = meta_df.loc[train_idx].copy()
                train_meta_processed = apply_target_stratified_sampling(
                    df=train_meta_subset,
                    target_col=target_col,
                    date_col='date',
                    scode_col='scode',
                    mode=mode,
                    center_keep_ratio=sampling_cfg.get("center_keep_ratio", 0.25),
                    other_keep_ratio=sampling_cfg.get("other_keep_ratio", 1.0),
                    weight_dict=sampling_cfg.get("weight_dict", None),
                    random_state=cfg.get("seed", 42) + i
                )
                if mode in ['mode_1', 'mode_2']:
                    # サンプリングモードの場合はインデックスを更新
                    train_idx = train_meta_processed.index
                    print(f"    - Samples reduced: {count_before_stratified:,} -> {len(train_idx):,}")
                elif mode in ['mode_3', 'mode_ap_severe']:
                    # 重み付けモードの場合は、重みを後で適用するために取得
                    # train_idx は変更されない
                    stratified_sampling_weights = train_meta_processed.loc[train_idx, 'sample_weight'].values
                    print(f"    - Weighting mode enabled. Sample count remains {len(train_idx):,}.")

            # ウェイトの計算を関数化
            def calc_weights(idx, is_train=True):
                w = np.ones(len(idx))
                w *= calculate_sample_weights(meta_df.loc[idx, 'log_market_cap'].values, cfg.domain.name)
                if cfg.hparams.use_time_decay:
                    decay_rate = cfg.hparams.get('time_decay_rate', 0.9999)
                    w *= calculate_time_decay_weights(meta_df.loc[idx, 'date'], decay_rate=decay_rate)
                if is_train and stratified_sampling_weights is not None:
                    # target_stratified_samplingはtrain_idxにのみ適用されているため
                    w *= stratified_sampling_weights
                if cfg.get("preprocess", {}).get("matrix_weight", {}).get("enabled", False):
                    matrix_cfg = cfg.preprocess.matrix_weight
                    cost_buffer = matrix_cfg.get("cost_buffer", 0.003)
                    meta_subset = meta_df.loc[idx].copy()
                    w *= apply_2d_matrix_weight(meta_subset, return_col='Future_Close', cost_buffer=cost_buffer)
                if cfg.get("preprocess", {}).get("hard_negative_weighting", {}).get("enabled", False):
                    meta_subset = meta_df.loc[idx].copy()
                    w *= apply_hard_negative_weighting(meta_subset)
                if cfg.get("preprocess", {}).get("class_weight", {}).get("enabled", False):
                    cw_cfg = cfg.preprocess.class_weight
                    num_classes = cw_cfg.get("num_classes", 4)
                    clip_min = cw_cfg.get("clip_min", 1.0)
                    clip_max = cw_cfg.get("clip_max", 10.0)
                    # train fold の y のみを使って重みを計算 (Data Leakage 防止)
                    y_series = meta_df.loc[idx, target_col]
                    class_weight_dict, class_counts = make_train_fold_class_weight(
                        y_series, num_classes=num_classes, clip_min=clip_min, clip_max=clip_max
                    )
                    w *= make_sample_weight(y_series, class_weight_dict)
                    if is_train:
                        total_n = class_counts.sum()
                        for cls_idx in range(num_classes):
                            mlflow.log_metric(f"fold{i}_class_count_{cls_idx}", float(class_counts[cls_idx]))
                            mlflow.log_metric(f"fold{i}_class_weight_{cls_idx}", float(class_weight_dict[cls_idx]))
                        if num_classes >= 4:
                            pos_rate_5 = (class_counts[1] + class_counts[2] + class_counts[3]) / total_n
                            pos_rate_7 = (class_counts[2] + class_counts[3]) / total_n
                            pos_rate_10 = class_counts[3] / total_n
                            mlflow.log_metric(f"fold{i}_positive_rate_5", float(pos_rate_5))
                            mlflow.log_metric(f"fold{i}_positive_rate_7", float(pos_rate_7))
                            mlflow.log_metric(f"fold{i}_positive_rate_10", float(pos_rate_10))
                        print(f"    - Class weight mode enabled.")
                return w
            w_train = calc_weights(train_idx, is_train=True)

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
                    X_train, y_train, 
                    X_valid, y_valid, 
                    sample_weight=w_train, 
                    model_idx=i,
                    train_dates=train_dates,
                    valid_dates=valid_dates
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
                ensemble_size = cfg.model.get("ensemble_size", 1)
                print(f"\n  🌟 [Production] Step 2: Training on Train+Valid data (Ensemble Size: {ensemble_size}) using best_iter={best_iter}...")
                full_train_idx = train_idx.append(valid_idx)
                # Visualize production period
                full_train_dates = train_val_meta.loc[full_train_idx, 'date'].unique()
                tr_pos_prod = np.where(np.isin(unique_dates, full_train_dates))[0]
                log_split_info(i, tr_pos_prod, np.array([]), pos_to_date, label="PROD")
                w_full = calc_weights(full_train_idx, is_train=True)
                # Full data features
                X_full = preprocessor.transform(features_array, row_indices=full_train_idx, col_indices=col_indices)
                X_full = combine_features_with_oof(X_full, meta_df, full_train_idx, oof_cols)
                y_full = meta_df.loc[full_train_idx, target_col].values
                # params update for full training
                prod_params = copy.deepcopy(full_params)
                # Productionの個別学習時は wrapper 内の ensemble_size は 1 に固定する（ここでループ制御するため）
                prod_params["ensemble_size"] = 1
                if best_iter is not None:
                    # LGBM or NN max epochs
                    if cfg.model.name.lower() in ["lgbm", "lightgbm"]:
                        prod_params['num_boost_round'] = int(best_iter)
                        prod_params['early_stopping_rounds'] = 0 # 無効化
                    else:
                        prod_params['max_epochs'] = int(best_iter)
                        prod_params['patience'] = int(best_iter) + 1 # 無効化
                base_seed = cfg.get("seed", 42)
                for s in range(ensemble_size):
                    if ensemble_size > 1:
                        print(f"    - Training ensemble model {s+1}/{ensemble_size} with seed {base_seed + s}...")
                    curr_params = copy.deepcopy(prod_params)
                    # 各モデルのシード値を変更
                    curr_params['seed'] = base_seed + s
                    curr_params['random_state'] = base_seed + s
                    model_prod = model_class(task_type=cfg.target.task_type, **curr_params)
                    model_prod.fit(X_full, y_full, X_valid=None, y_valid=None, sample_weight=w_full, model_idx=f"{i}_s{s}")
                    if s < ensemble_size - 1:
                        # 最後の1つ以外を先にパイプラインに追加
                        fold_pipelines.append(FoldPipeline(preprocessor, model_prod))
                    else:
                        # 最後の1つ（または唯一の1つ）を model にセット
                        # この後の既存処理で fold_pipelines に追加され、artifact保存対象になる
                        model = model_prod

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

            # 最適化に使用するメトリクスをconfigから取得（デフォルトは 'ic'）
            opt_metric_name = cfg.target.get("optimization_metric", cfg.get("optimization_metric", 'ic'))
            eval_metric = cfg.target.get("eval_metric", 'ic')
                
            # メトリクス算出 (Train / Valid / Test)
            valid_score = None
            c_buffer = cfg.get("preprocess", {}).get("matrix_weight", {}).get("cost_buffer", 0.005)
            for phase in ['train', 'valid', 'test']:
                if preds_raw[phase] is not None:
                    idx = locals()[f'{phase}_idx']
                    y_true = locals()[f'y_{phase}']
                    eval_df = pd.DataFrame({
                        'date': meta_df.loc[idx, 'date'].values,
                        'pred': preds_1d[phase],
                        'y_true': y_true,
                        'y_ret': meta_df.loc[idx, 'Future_Close'].values - 1.0,
                    })
                    for c in ['Future_High', 'Future_Low', 'Future_Close']:
                        eval_df[c] = meta_df.loc[idx, c].values

                    m = evaluate_metrics(
                        eval_df,
                        y_pred=preds_raw[phase],
                        task_type=cfg.target.task_type,
                        target_col=target_col,
                        ndcg_k=cfg.get("ndcg_k", 10),
                        cost_buffer=c_buffer,
                        include_extra_bin_metrics=(phase == 'valid'),
                    )

                    if phase == 'valid':
                        fold_metrics_results.append({
                            k: v for k, v in m.items()
                            if not is_extra_bin_metric_key(k)
                        })

                    # MLflowにフォールドごとの結果を記録
                    mlflow.log_metrics({f"fold{i}_{phase}_{k}": v for k, v in m.items()})
                    # 指定メトリクスを収集
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
                    res_df = pd.DataFrame({
                        'date': meta_df.loc[idx, 'date'],
                        'scode': meta_df.loc[idx, 'scode'],
                        'target': locals()[f'y_{phase}'],
                        'score': preds_1d[phase],
                        'phase': phase,
                        'fold': i
                    }).reset_index(drop=True)
                    # 必要なメタデータを結合
                    meta_cols = ['Future_High', 'Future_Low', 'Future_Close']
                    meta_sub = meta_df.loc[idx, meta_cols].reset_index(drop=True)
                    res_df = pd.concat([res_df, meta_sub], axis=1)
                    all_results.append(res_df)
            
            # 中間生成されたZarrキャッシュのクリーンアップ
            import shutil
            for x_cache in [X_train, X_valid, X_test]:
                if isinstance(x_cache, str) and x_cache.endswith('.zarr') and os.path.exists(x_cache):
                    shutil.rmtree(x_cache, ignore_errors=True)
                    
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
        if cv_method in ["purged_kfold", "cpcv", "anchored_walk_forward"] or cfg.get("mode") == "production":
            test_res = full_res_df[full_res_df['phase'] == 'valid']
        else: 
            test_res = full_res_df[full_res_df['phase'] == 'test']
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
            oof_df_clean = oof_df.groupby(['date', 'scode']).agg({
                'score': 'mean',
                'target': 'first',
                'Future_Close': 'first'
            }).reset_index()
            eval_df_pooled = pd.DataFrame({
                'date': oof_df_clean['date'],
                'pred': oof_df_clean['score'],
                'y_true': oof_df_clean['target'],
                'y_ret': oof_df_clean['Future_Close'].values - 1.0
            })
            c_buffer_pooled = cfg.get("preprocess", {}).get("matrix_weight", {}).get("cost_buffer", 0.005)
            pooled_metrics = evaluate_metrics(eval_df_pooled, cost_buffer=c_buffer_pooled)
            # MLflowにロギング
            mlflow.log_metrics({f"pooled_oof_{k}": v for k, v in pooled_metrics.items()})
        else:
            pooled_metrics = {}
            
        # --- 最終的な最適化スコア（Optunaの戻り値）の決定 ---
        if not valid_metrics:
            final_opt_score = fallback_metric
            print("⚠️ WARNING: No valid metrics found in validation results.")
        else:
            mean_score = np.nanmean(valid_metrics)
            std_score = np.nanstd(valid_metrics)
            min_score = np.nanmin(valid_metrics)
            
            # --- 新規 Objective v2 の計算 ---
            obj_v2 = 0.0
            penalty_v2 = 0.0
            aggregated_f_metrics = {}
            if fold_metrics_results:
                aggregated_f_metrics = aggregate_fold_metrics(fold_metrics_results)
                # MLflowに集計メトリクスを記録 (valid_ prefix)
                mlflow.log_metrics({f"valid_{k}": v for k, v in aggregated_f_metrics.items()})
                
                # 特殊なエイリアスを個別に記録
                mlflow.log_metric("valid_mean_daily_rankic_mean", aggregated_f_metrics.get('mean_daily_rankic_mean', np.nan))
                mlflow.log_metric("valid_worst_fold_rankic", aggregated_f_metrics.get('worst_fold_rankic', np.nan))
                mlflow.log_metric("valid_top30_active_mean_raw", aggregated_f_metrics.get('top30_active_mean_raw_mean', np.nan))
                mlflow.log_metric("valid_top20_active_mean_raw", aggregated_f_metrics.get('top20_active_mean_raw_mean', np.nan))
                mlflow.log_metric("valid_top10_active_mean_raw", aggregated_f_metrics.get('top10_active_mean_raw_mean', np.nan))

                obj_v2, penalty_v2 = calc_objective_v2(aggregated_f_metrics)
                mlflow.log_metric("objective_v2", obj_v2)
                mlflow.log_metric("objective_penalty_total", penalty_v2)
                
                # 成分ごとの記録
                mlflow.log_metric("objective_component_mean_daily_rankic", aggregated_f_metrics.get('mean_daily_rankic_mean', 0))
                mlflow.log_metric("objective_component_top30_active_mean_scaled", aggregated_f_metrics.get('top30_active_mean_scaled_mean', 0))
                mlflow.log_metric("objective_component_top20_active_mean_scaled", aggregated_f_metrics.get('top20_active_mean_scaled_mean', 0))
                mlflow.log_metric("objective_component_top_quintile_spread_scaled", aggregated_f_metrics.get('top_quintile_spread_scaled_mean', 0))
                mlflow.log_metric("objective_component_top30_rankic_alpha_scaled", aggregated_f_metrics.get('top30_rankic_alpha_scaled_mean', 0))
                mlflow.log_metric("objective_component_worst_fold_rankic", aggregated_f_metrics.get('worst_fold_rankic', 0))
                mlflow.log_metric("objective_component_positive_day_ratio_scaled", aggregated_f_metrics.get('positive_day_ratio_scaled_mean', 0))

            # 過学習診断 gap
            train_mean_ic = np.nanmean(train_metrics) if train_metrics else 0.0
            valid_mean_ic = aggregated_f_metrics.get('mean_daily_rankic_mean', 0.0)
            mlflow.log_metric("train_valid_rankic_gap", train_mean_ic - valid_mean_ic)
            
            train_top30_active = np.nanmean([m.get('top30_active_mean_raw', 0.0) for m in fold_metrics_results]) if fold_metrics_results else 0.0
            valid_top30_active = aggregated_f_metrics.get('top30_active_mean_raw_mean', 0.0)
            mlflow.log_metric("train_valid_top30_active_mean_gap", train_top30_active - valid_top30_active)

            if opt_metric_name == "objective_v2":
                final_opt_score = obj_v2
                print(f"  🔹 Objective V2: {final_opt_score:.6f} (Penalty: {penalty_v2:.4f})")
            elif opt_metric_name == "tac_risk_class_guarded_ap":
                final_opt_score = calc_tac_risk_objective(fold_metrics_results)
                print(f"  🔹 TAC Risk Guarded AP Objective: {final_opt_score:.6f}")
            elif opt_metric_name == "composite_tac":
                # 統合指標の計算 (Step4 Final Sweep用 - 互換性のために残すが objective_v2 を推奨)
                rank_ic = pooled_metrics.get("RankIC", pooled_metrics.get("mean_daily_rankic", 0.0))
                utility = pooled_metrics.get("top30_active_utility_scaled", 0.0)
                spread = pooled_metrics.get("top_quintile_spread_scaled", 0.0)
                alpha_ic = pooled_metrics.get("top30_rankic_alpha_scaled", 0.0)
                pos_ratio = pooled_metrics.get("positive_day_ratio_scaled", 0.0)
                worst_fold_rankic = min_score # eval_metric=RankIC である前提
                
                final_opt_score = (
                    0.30 * rank_ic
                    + 0.30 * utility
                    + 0.15 * spread
                    + 0.10 * alpha_ic
                    + 0.10 * worst_fold_rankic
                    + 0.05 * pos_ratio
                )
                print(f"  🔹 Composite Objective (TAC): {final_opt_score:.6f}")
                print(f"    - RankIC: {rank_ic:.4f}, Utility: {utility:.4f}, Spread: {spread:.4f}")
                print(f"    - AlphaIC: {alpha_ic:.4f}, WorstFoldIC: {worst_fold_rankic:.4f}, PosRatio: {pos_ratio:.4f}")
            elif opt_metric_name.startswith("worst_fold_"):
                final_opt_score = min_score
            elif opt_metric_name == "daily_icir_reb":
                final_opt_score = pooled_metrics.get('daily_icir_reb', fallback_metric)
            elif opt_metric_name.startswith("pooled_oof_"):
                # pooled_metricsの中身は接頭辞なしのキーなので、接頭辞を外して取得を試みる
                base_key = opt_metric_name.replace("pooled_oof_", "")
                final_opt_score = pooled_metrics.get(base_key, fallback_metric)
            else:
                if direction == "minimize":
                    final_opt_score = mean_score + std_score
                else:
                    final_opt_score = mean_score - std_score
        mlflow.log_metric("optimization_score", final_opt_score)

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
                col_indices=col_indices,
                oof_cols=oof_cols
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
            bin_stats_path = os.path.join(d, "test_bin_analysis_daily.csv")
            bin_stats.to_csv(bin_stats_path)
            mlflow.log_artifact(bin_stats_path)
            
            # 全期間でのビン分析
            bin_stats_global = calculate_bin_stats(
                test_res, score_col='score', target_col='target', task_type=cfg.target.task_type,
                metadata_cols=['Future_High', 'Future_Low', 'Future_Close'],
                date_col='date', n_bins=20, global_bin=True
            )
            bin_stats_global_path = os.path.join(d, "test_bin_analysis_global.csv")
            bin_stats_global.to_csv(bin_stats_global_path)
            mlflow.log_artifact(bin_stats_global_path)
        # Hydraの最終的なconfigファイル自体も保存（完全な再現用）
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config=cfg, f=f.name)
            mlflow.log_artifact(f.name, artifact_path="config")
        os.remove(f.name)
        
        # --- fixモード / productionモード / stacking_baseモード：モデルレジストリへの登録と昇格 ---
        current_mode = cfg.get("mode")
        if current_mode in ["fix", "production", "stacking_base", "stacking_ensemble"]:
            
            # --- OOFデータの保存 (Stacking用) ---
            if current_mode in ["fix", "stacking_base"]:
                oof_df = full_res_df[full_res_df['phase'] == 'valid'].copy()
                with tempfile.TemporaryDirectory() as d:
                    oof_filename = os.path.join(d, f"oof_predictions_{cfg.model.name}_{cfg.target.column}.csv")
                    oof_df.to_csv(oof_filename, index=False)
                    mlflow.log_artifact(oof_filename, artifact_path="oof_data")
                    
            # --- 役割（Role）に基づくモデル名の決定 ---
            if current_mode == "stacking_base":
                registered_model_name = f"Base_{cfg.model.name}_{cfg.target.name}_OOF"
                target_stage = "None" # または "Staging"
            elif current_mode == "stacking_ensemble":
                registered_model_name = f"Stacked_{cfg.target.name}_Final"
                target_stage = "Production"
            elif current_mode == "production":
                # Stackingが有効な場合はメタモデルとして扱う
                if cfg.get("stacking", {}).get("enabled", False):
                    registered_model_name = f"Stacked_{cfg.target.name}_Final"
                else:
                    registered_model_name = f"Base_{cfg.model.name}_{cfg.target.name}_INF"
                target_stage = "Production"
            else: # fallback (fix)
                registered_model_name = f"{cfg.model.name}_{cfg.target.name}"
                target_stage = "Staging"
                
            print(f"\n🌟 Mode '{current_mode}' detected. Registering model as '{registered_model_name}' and promoting to {target_stage}.")

            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            try:
                mv = mlflow.register_model(model_uri, registered_model_name)
                # Variant管理のため archive_existing_versions=False に変更
                if target_stage != "None":
                    client.transition_model_version_stage(
                        name=registered_model_name, version=mv.version, stage=target_stage, archive_existing_versions=False
                    )
                
                # タグの付与
                variant = cfg.get("variant", "default")
                client.set_model_version_tag(registered_model_name, mv.version, "variant", variant)
                
                # 役割の記録
                if current_mode == "stacking_base":
                    client.set_model_version_tag(registered_model_name, mv.version, "nature", "base_oof_generator")
                elif current_mode == "production" and not cfg.get("stacking", {}).get("enabled", False):
                    client.set_model_version_tag(registered_model_name, mv.version, "nature", "inference_base")
                elif current_mode in ["production", "stacking_ensemble"] and cfg.get("stacking", {}).get("enabled", False):
                    client.set_model_version_tag(registered_model_name, mv.version, "nature", "stacking_meta")
                    # 依存モデルを記録
                    target_models = cfg.get("stacking", {}).get("target_models", [])
                    client.set_model_version_tag(registered_model_name, mv.version, "dependencies", json.dumps(target_models))
                
                # 特徴量構成やターゲット情報も付与しておくと後で便利
                feature_choice = HydraConfig.get().runtime.choices.get("features", "unknown")
                client.set_model_version_tag(registered_model_name, mv.version, "feature_config", feature_choice)
                
                print(f"✅ Model registered as '{registered_model_name}' (Version {mv.version}) with variant '{variant}' and transitioned to {target_stage}.")
            except Exception as e:
                print(f"⚠️ Failed to register model to registry: {e}")


        # --- MLflow成果物の一括ZIP化とGoogle Driveへの移動 ---
        if cfg.get("output_gdrive", False):
            bundle_and_upload_artifacts(path_to_gdrive, cfg.domain.name)
            print("✅ All artifacts have been bundled into a ZIP file and uploaded to MLflow.")
            
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
