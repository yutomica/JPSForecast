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
import joblib
import yaml
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
from src.evaluation.metrics import evaluate_metrics, calculate_bin_stats
from src.utils.feature_selection import calculate_shap, calculate_mda, calculate_cfi
from src.utils.mlflow_utils import setup_mlflow_run, check_and_promote_model, bundle_and_upload_artifacts
from src.utils.sampling import apply_sampling, apply_target_stratified_sampling, apply_2d_matrix_weight
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
                full_params["objective"] = cfg.target.objective
        # 目的関数に関連するパラメータをターゲットから取得
        # hparams側が優先されるようにして、個別のチューニングを許容する
        target_keys = ["fair_c", "tweedie_variance_power", "asym_alpha", "asym_beta", "quantile", "target_transform", "custom_objective", "custom_metric"]
        for key in target_keys:
            if key in cfg.target and key not in full_params:
                full_params[key] = cfg.target[key]
        # モデルごとのパラメータ名マッピング
        obj = cfg.target.objective
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
        full_params.update(model_meta_params)

        # --- モデルの学習 ---
        print(f"🤖 Training model: {cfg.model.name}")
        models = []
        all_results = []
        valid_metrics = []
        # スクリーニング結果格納用
        all_fold_mda_values = []
        all_fold_cfi_values = []
        all_fold_shap_values = []
        fold_pipelines = []
        for i, (train_idx, valid_idx, test_idx, tr_pos, val_pos) in enumerate(splits):
            print(f"\n{'-'*25} Fold {i} {'-'*25}")
            
            # CVサマリー
            if tr_pos is not None and val_pos is not None:
                info = log_split_info(i, tr_pos, val_pos, pos_to_date)
                cv_summaries.append(info)
            
            # --- 学習データのみ Date-interval サンプリングを適用 ---
            if cfg.get("preprocess", {}).get("sampling", {}).get("enabled", False):
                print("  🔹 Applying date-interval sampling...")
                count_before_sampling = len(train_idx)
                sampling_interval = cfg.preprocess.sampling.get("interval", interval)
                train_meta_subset = meta_df.loc[train_idx].copy()
                train_meta_processed = apply_sampling(train_meta_subset, sampling_interval)
                train_idx = train_meta_processed.index
                print(f"    - Samples reduced: {count_before_sampling:,} -> {len(train_idx):,}")

            # --- 学習データのみターゲット層化サンプリングを適用 ---
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

            # --- ウェイトの計算 ---
            w_train = np.ones(len(train_idx))
            # log_market_capによるウエイト（STRのウエイトを軽くする）
            w_train *= calculate_sample_weights(meta_df.loc[train_idx, 'log_market_cap'].values, cfg.domain.name)
            # Time Decay
            if cfg.hparams.use_time_decay:
                # 学習セットの日付のみを抽出してウェイトを算出 decay_rate は config から取得 (デフォルト: 0.9999)
                decay_rate = cfg.hparams.get('time_decay_rate', 0.9999)
                w_train *= calculate_time_decay_weights(meta_df.loc[train_idx, 'date'], decay_rate=decay_rate)
            # 層化サンプリングの重みを適用 (mode_3の場合)
            if stratified_sampling_weights is not None:
                w_train *= stratified_sampling_weights

            # 2D Matrix Weight (based on Future_Close)
            if cfg.get("preprocess", {}).get("matrix_weight", {}).get("enabled", False):
                matrix_cfg = cfg.preprocess.matrix_weight
                cost_buffer = matrix_cfg.get("cost_buffer", 0.003)
                train_meta_subset = meta_df.loc[train_idx].copy()
                w_train *= apply_2d_matrix_weight(train_meta_subset, return_col='Future_Close', cost_buffer=cost_buffer)

            # メモリ上の配列から必要な行のみを読み出し
            print(f"  🔹 Transforming data...")
            # 各Foldごとに独立したインスタンスを使用するためディープコピー
            preprocessor = copy.deepcopy(base_preprocessor)
            X_train = preprocessor.transform(features_array, row_indices=train_idx, col_indices=col_indices)
            X_valid = preprocessor.transform(features_array, row_indices=valid_idx, col_indices=col_indices)
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
                y_test = meta_df.loc[test_idx, target_col].values
                print(f"  🔹 Samples: Train={len(train_idx):,}, Valid={len(valid_idx):,}, Test={len(test_idx):,}")
            # モデルのインスタンス化と学習
            model_class = get_class(cfg.model.model_target)
            model = model_class(task_type=cfg.target.task_type, **full_params)
            if hasattr(model, 'device'):
                print(f"  🔹 Using device: {model.device}")

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

            # 最適化に使用するメトリクスをconfigから取得（デフォルトは 'ic'）
            opt_metric_name = cfg.target.get("optimization_metric", cfg.get("optimization_metric", 'ic'))
            eval_metric = cfg.target.get("eval_metric", 'ic')
                
            # メトリクス算出 (Train / Valid / Test)
            valid_score = None
            for phase in ['train', 'valid', 'test']:
                if preds[phase] is not None:
                    idx = locals()[f'{phase}_idx']
                    y_true = locals()[f'y_{phase}']
                    # 評価用ICの計算対象として生リターン（Future_Close）を取得
                    y_ret = meta_df.loc[idx, 'Future_Close'].values - 1.0
                    dates = meta_df.loc[idx, 'date'].values
                    # cost_buffer の取得 (configから)
                    c_buffer = cfg.get("preprocess", {}).get("matrix_weight", {}).get("cost_buffer", 0.005)
                    m = evaluate_metrics(y_true, preds[phase], y_ret=y_ret, task_type=cfg.target.task_type, target_col=target_col, dates=dates, ndcg_k=cfg.get("ndcg_k", 10), cost_buffer=c_buffer)
                    # MLflowにフォールドごとの結果を記録
                    mlflow.log_metrics({f"fold{i}_{phase}_{k}": v for k, v in m.items()})
                    # Validの指定メトリクスを収集
                    if phase == 'valid':
                        score = m.get(eval_metric)
                        if score is None:
                            # 大文字小文字を区別せずに再試行
                            m_lower = {k.lower(): v for k, v in m.items()}
                            score = m_lower.get(eval_metric.lower(), np.nan)
                        valid_metrics.append(score)
                        valid_score = score
            
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
                
                # # CFI (Clustered Feature Importance) の計算
                # feature_groups_path = cfg.get("feature_groups_path", "clustered_features.yaml")
                # if os.path.exists(feature_groups_path):
                #     print(f"  🔹 [Selection] Calculating CFI using {opt_metric_name} for Fold {i}...")
                #     with open(feature_groups_path, 'r') as f:
                #         yaml_data = yaml.safe_load(f)
                #         feature_groups = yaml_data.get("feature_groups", {})
                #     if feature_groups:
                #         fold_cfi = calculate_cfi(
                #             model=model, X_valid=X_valid, y_valid=y_valid, y_ret_valid=y_ret_valid,
                #             dates_for_shuffle=dates_for_shuffle, feature_groups=feature_groups,
                #             feature_cols=feature_cols, baseline_score=baseline_score,
                #             task_type=cfg.target.task_type, target_col=target_col, opt_metric=opt_metric_name
                #         )
                #         all_fold_cfi_values.append(fold_cfi)
            
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
        
        # MDA (Feature Sharpe) の集計と保存 ---
        if cfg.get("mode") == "feature_select" and all_fold_mda_values:
            mda_df = pd.DataFrame(all_fold_mda_values) # rows=folds, cols=features
            output_filename = f"feature_sharpe_{cfg.model.name}_{cfg.domain.name}_{cfg.target.name}.csv"
            mda_df.to_csv(output_filename)
            mlflow.log_artifact(output_filename)
            print(f"✅ Feature Sharpe results saved to {output_filename} (Group Threshold check needed).")
            
        # # CFI の集計と保存 ---
        # if cfg.get("mode") == "feature_select" and all_fold_cfi_values:
        #     cfi_df = pd.DataFrame(all_fold_cfi_values)
        #     output_filename_cfi = f"cfi_results_{cfg.model.name}_{cfg.domain.name}_{cfg.target.name}.csv"
        #     cfi_df.to_csv(output_filename_cfi)
        #     mlflow.log_artifact(output_filename_cfi)
        #     print(f"✅ CFI results saved to {output_filename_cfi}.")
        

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
        
        # --- Pooled OOF Metric の算出 ---
        oof_df = full_res_df[full_res_df['phase'] == 'valid']
        ndcg_k = cfg.get("ndcg_k", 10)
        if not oof_df.empty:
            print(f"  🔹 Calculating Pooled OOF Metrics...")
            y_ret_pooled = oof_df['Future_Close'].values - 1.0 if 'Future_Close' in oof_df.columns else None
            c_buffer = cfg.get("preprocess", {}).get("matrix_weight", {}).get("cost_buffer", 0.005)
            pooled_metrics = evaluate_metrics(
                y_true=oof_df['target'].values,
                y_pred=oof_df['score'].values,
                y_ret=y_ret_pooled,
                task_type=cfg.target.task_type,
                target_col=cfg.target.column,
                dates=oof_df['date'].values,
                ndcg_k=ndcg_k,
                cost_buffer=c_buffer
            )
            # --- 日次RankICベースのICIRを直接計算 ---
            if opt_metric_name in ["daily_icir", "daily_icir_reb"]:
                from scipy.stats import spearmanr
                daily_ics = []
                df_tmp = oof_df.copy()
                df_tmp['date'] = pd.to_datetime(df_tmp['date']).dt.date
                unique_dates = np.sort(df_tmp['date'].unique())
                if opt_metric_name == "daily_icir_reb":
                    target_dates = set(unique_dates[::11])
                else:
                    target_dates = set(unique_dates)
                for d, group in df_tmp.groupby('date'):
                    if d not in target_dates:
                        continue
                    g_y_true = group['target'].values
                    g_y_pred = group['score'].values
                    if len(g_y_true) < 2 or np.max(g_y_pred) == np.min(g_y_pred) or np.max(g_y_true) == np.min(g_y_true): 
                        continue
                    ic, _ = spearmanr(g_y_true, g_y_pred)
                    if not np.isnan(ic):
                        daily_ics.append(ic)
                if daily_ics:
                    ic_mean = np.mean(daily_ics)
                    ic_std = np.std(daily_ics)
                    pooled_metrics[opt_metric_name] = ic_mean / (ic_std + 1e-8)
                else:
                    pooled_metrics[opt_metric_name] = fallback_metric
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
            if opt_metric_name == "composite_tac":
                # 統合指標の計算 (Step4 Final Sweep用)
                rank_ic = pooled_metrics.get("RankIC", 0.0)
                utility = pooled_metrics.get("cost_adjusted_top30_active_utility_scaled", 0.0)
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
        
        # --- fixモード：Staging昇格・OOF保存 ---
        if cfg.get("mode") == "fix":
            print(f"\n🌟 Mode 'fix' detected. Promoting model to Staging and saving OOF data.")
            # OOFデータの保存 (Stacking用)
            oof_df = full_res_df[full_res_df['phase'] == 'valid'].copy()
            with tempfile.TemporaryDirectory() as d:
                oof_filename = os.path.join(d, f"oof_predictions_{cfg.model.name}_{cfg.target.column}.csv")
                oof_df.to_csv(oof_filename, index=False)
                mlflow.log_artifact(oof_filename, artifact_path="oof_data")
            # モデルレジストリへの登録とStagingへの昇格
            registered_model_name = f"{cfg.model.name}_{cfg.target.name}"
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            try:
                mv = mlflow.register_model(model_uri, registered_model_name)
                # Variant管理のため archive_existing_versions=False に変更
                client.transition_model_version_stage(
                    name=registered_model_name, version=mv.version, stage="Staging", archive_existing_versions=False
                )
                
                # タグの付与
                variant = cfg.get("variant", "default")
                client.set_model_version_tag(registered_model_name, mv.version, "variant", variant)
                # 特徴量構成やターゲット情報も付与しておくと後で便利
                feature_choice = HydraConfig.get().runtime.choices.get("features", "unknown")
                client.set_model_version_tag(registered_model_name, mv.version, "feature_config", feature_choice)
                
                print(f"✅ Model registered as '{registered_model_name}' (Version {mv.version}) with variant '{variant}' and transitioned to Staging.")
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