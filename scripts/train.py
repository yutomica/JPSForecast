import numpy as np
from datetime import datetime
import os
import gc
import hydra
import mlflow
import json
import pandas as pd
import joblib
import tempfile
import shutil
from urllib.parse import urlparse
import copy
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_class
from src.cv.purged_kfold import SimplePurgedKFold
from src.cv.cpcv import SimpleCombinatorialPurgedKFold
from src.cv.cv_viz import summarize_split_for_logging
from src.preprocess.common import calculate_time_decay_weights
from src.models.ensemble import EnsembleModel
from src.evaluation import evaluate_metrics, calculate_bin_stats
path_to_gdrive = os.environ.get('path_to_gdrive', '') 
import logging
# alembic のロガーを取得し、ログレベルを WARNING に上げる
logging.getLogger("alembic").setLevel(logging.WARNING)
# ついでに sqlalchemy のログも抑制したい場合は以下も有効です
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
import exchange_calendars as ecals

def apply_sampling(df, interval):
    if interval <= 1:
        return df
    print(f"Applying sampling (Speed-optimized): interval={interval}")
    
    # scode ごとに連番を振り、その余りが0の行だけを残す
    # これにより scode カラムが消えることはなく、実行速度も劇的に向上します
    df = df.sort_values(['scode', 'date'])
    sampled_df = df[df.groupby('scode').cumcount() % interval == 0].copy()
    
    return sampled_df

@hydra.main(version_base=None, config_path="../config", config_name="main")
def train(cfg: DictConfig):
    # 1. MLflowの初期設定
    abs_path = os.path.expanduser("~/JPSForecast/mlflow_runs")
    os.makedirs(abs_path, exist_ok=True)
    mlflow_db_path = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(mlflow_db_path)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        # --- コンフィグの保存 ---
        # 全設定を辞書形式にして記録（ドメイン、ターゲット、特徴量、HParams全てが含まれる）
        params = OmegaConf.to_container(cfg, resolve=True)
        feature_cols = params['features'].pop('feature_cols', [])
        cv_summaries = []   # foldごとの期間情報を貯めて、最後にMLflow artifactにする
        cv_method = cfg.period.method
        if cv_method in  ["purged_kfold", "cpcv"]:
            params.update({
                "cv_method": cv_method,
                "cv_n_splits": int(cfg.period.n_splits),
                "cv_pct_embargo": float(cfg.period.pct_embargo),
                "cv_n_test_chunks": int(getattr(cfg.period, "n_test_chunks", 0)) if hasattr(cfg.period, "n_test_chunks") else 0,
            })
        mlflow.log_params(params)
        mlflow.log_dict({"feature_cols": feature_cols}, "configs/feature_cols.json")
        print('Start training model ...')
        print('Domain: '+cfg.domain.name)
        print('Model: '+cfg.model.name)
        print('Split: '+cfg.period.method)

        # --- データ分割ロジックの構築 ---
        master_dir = Path(cfg.data.path)
        meta_df = pd.read_parquet(master_dir / "index_meta.parquet")
        meta_df = meta_df.reset_index(drop=True)
        # ドメイン（戦術/戦略）に応じてフィルタフラグを選択
        if cfg.domain.name == 'TAC':
            mask = meta_df['is_candidate_tac'] == True
            meta_df = meta_df.rename(columns={
                'Future_High_Tac':'Future_High',
                'Future_Low_Tac':'Future_Low',
                'Future_Close_Tac':'Future_Close',
            })
            horizon = 5  # 5日間の予測期間
        else:
            mask = meta_df['is_candidate_str'] == True
            meta_df = meta_df.rename(columns={
                'Future_High_Str':'Future_High',
                'Future_Low_Str':'Future_Low',
                'Future_Close_Str':'Future_Close',
            })
            horizon = 60 # 60日間の予測期間
        
        # --- Embargo期間の設定 ---
        # 必要な期間の取引日を生成
        cal = ecals.get_calendar("XTKS")
        start = pd.to_datetime(meta_df['date']).min()
        end   = pd.to_datetime(meta_df['date']).max()
        sessions = cal.sessions_in_range(start, end)  # DatetimeIndex (tz awareのことが多い)
        if sessions.tz is not None:
            sessions = sessions.tz_convert(None)
        sessions = sessions.normalize()
        date_to_pos = {d: i for i, d in enumerate(sessions)}
        train_val_meta = meta_df[mask].copy()
        train_val_meta['date'] = pd.to_datetime(train_val_meta['date']).dt.normalize()
        pos = train_val_meta['date'].map(date_to_pos)
        t1_pos = pos + horizon
        valid = t1_pos < len(sessions)
        train_val_meta = train_val_meta.loc[valid].copy()
        train_val_meta['t1'] = sessions[t1_pos.loc[valid].astype(int).to_numpy()]
        train_val_meta = train_val_meta.dropna(subset=['t1']).copy()
        # サンプリングの実行 (configから interval を取得)
        sampling_interval = cfg.model.get('sampling_interval', 1)
        train_val_meta = apply_sampling(train_val_meta, sampling_interval)
        # エンバーゴ（Embargo）日数の設定
        embargo_td = pd.Timedelta(days=cfg.period.embargo_days)
        
        # --- データ分割・CV ---
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
            splits = [(train_idx, valid_idx, test_idx)]
        elif cv_method in ["purged_kfold", "cpcv"]:
            train_val_meta["date_floor"] = pd.to_datetime(train_val_meta["date"]).dt.normalize()
            train_val_meta["t1_floor"]   = pd.to_datetime(train_val_meta["t1"]).dt.normalize()
            t1_per_date = (
                train_val_meta
                .groupby("date_floor")["t1_floor"]
                .max()
                .sort_index()
            )
            dates = t1_per_date.index.to_numpy()     # 日付軸（ユニーク日）
            t1_vals = t1_per_date.to_numpy()         # その日のt1（ユニーク日上に無い可能性あり）
            start_pos = np.arange(len(dates), dtype=np.int64)
            end_pos = np.searchsorted(dates, t1_vals, side="right") - 1
            end_pos = np.clip(end_pos, 0, len(dates) - 1)
            end_pos = np.maximum(end_pos, start_pos)
            samples_info = pd.Series(end_pos, index=start_pos, name="t1")
            pos_to_date  = pd.Series(dates, index=start_pos)  # pos->date
            date_to_indices = train_val_meta.groupby("date_floor").groups
            if cv_method == "purged_kfold":
                cv = SimplePurgedKFold(
                    n_splits=int(cfg.period.n_splits),
                    samples_info_sets=samples_info,
                    pct_embargo=float(cfg.period.pct_embargo),
                )
            else:
                cv = SimpleCombinatorialPurgedKFold(
                    n_splits=int(cfg.period.n_splits),
                    n_test_splits=int(cfg.period.n_test_chunks),
                    samples_info_sets=samples_info,
                    pct_embargo=float(cfg.period.pct_embargo),
                )
            cv_input = np.zeros((len(start_pos), 1), dtype=np.float32)
            splits = []
            for tr_pos, val_pos in cv.split(X=cv_input):
                tr_dates  = pd.to_datetime(pos_to_date.iloc[tr_pos]).tolist()
                val_dates = pd.to_datetime(pos_to_date.iloc[val_pos]).tolist()
                train_idx = pd.Index(np.concatenate([date_to_indices[d] for d in tr_dates]))
                valid_idx = pd.Index(np.concatenate([date_to_indices[d] for d in val_dates]))
                splits.append((train_idx, valid_idx, None, tr_pos, val_pos))
        
        # --- データロード＆前処理 ---
        # 全特徴量名のロード
        all_features = pd.read_json(master_dir / "feature_names.json", typ='series').tolist()
        # コンフィグから「今回使う特徴量」を取得 (指定がなければ全件)
        feature_cols = cfg.features.get('feature_cols', all_features)
        print(f"Features: {len(feature_cols)}")
        cat_cols = cfg.features.get('cat_cols',[])
        # 使う列の「インデックス番号」を特定 numpyのmemmapは、列番号でスライスするのが最も高速です
        col_indices = [all_features.index(c) for c in feature_cols]
        features_mmap = np.memmap(
            master_dir / "features.npy", 
            dtype='float32', 
            mode='r', 
            shape=(len(meta_df), len(all_features))
        )
        # プリプロセッサに「選定後の列名」を渡してインスタンス化
        print(f"Executing preprocessor: {cfg.model.preprocessor_target}")
        preprocessor_class = get_class(cfg.model.preprocessor_target)
        preprocessor = preprocessor_class(save_dir="artifacts/preprocessor", feature_cols=feature_cols, cat_cols=cat_cols)
        
        # --- モデルの学習 ---
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
        target_col = cfg.target.column
        models = []
        all_results = []
        for i, (train_idx, valid_idx, test_idx, tr_pos, val_pos) in enumerate(splits):
            print(f"Starting Fold {i}...")
            if tr_pos is not None and val_pos is not None:
                info = summarize_split_for_logging(
                    fold=i,
                    tr_pos=tr_pos,
                    va_pos=val_pos,
                    pos_to_date=pos_to_date,
                    timeline_width=100,   # 好みで 80/120/160 など
                )
                print(
                    f"[CV] fold={i} | "
                    f"TRAIN days={info['train_days']} segs={info['train_segs']} ({info['train_start']}..{info['train_end']}) | "
                    f"VALID days={info['valid_days']} segs={info['valid_segs']} ({info['valid_start']}..{info['valid_end']})"
                )
                print(f"      {info['train_segments_str']}")
                print(f"      {info['valid_segments_str']}")
                print(f"      {info['timeline']}")
                # MLflow metrics
                mlflow.log_metric("cv_train_days", info["train_days"], step=i)
                mlflow.log_metric("cv_valid_days", info["valid_days"], step=i)
                mlflow.log_metric("cv_train_segs", info["train_segs"], step=i)
                mlflow.log_metric("cv_valid_segs", info["valid_segs"], step=i)
                # MLflow artifact（後でまとめて保存するために貯める）
                cv_summaries.append({
                    "fold": info["fold"],
                    "train_days": info["train_days"],
                    "valid_days": info["valid_days"],
                    "train_segs": info["train_segs"],
                    "valid_segs": info["valid_segs"],
                    "train_start": info["train_start"],
                    "train_end": info["train_end"],
                    "valid_start": info["valid_start"],
                    "valid_end": info["valid_end"],
                    "timeline": info["timeline"],
                })
            sample_data = features_mmap[:1000, col_indices]
            preprocessor.fit(pd.DataFrame(sample_data, columns=feature_cols))
            # 時間減衰ウェイトの計算 (common.py のロジックを使用)
            if cfg.hparams.use_time_decay:
                print(f"Calculating time decay weights for Fold {i}...")
                # 学習セットの日付のみを抽出してウェイトを算出
                # decay_rate は config から取得 (デフォルト: 0.9999)
                decay_rate = cfg.hparams.get('time_decay_rate', 0.9999)
                w_train = calculate_time_decay_weights(
                    meta_df.loc[train_idx, 'date'], 
                    decay_rate=decay_rate
                )
            else:
                w_train = None
            # memmap から必要な行のみを読み出し
            X_train = preprocessor.transform(features_mmap, row_indices=train_idx, col_indices=col_indices)
            X_valid = preprocessor.transform(features_mmap, row_indices=valid_idx, col_indices=col_indices)
            y_train = meta_df.loc[train_idx, target_col].values
            y_valid = meta_df.loc[valid_idx, target_col].values
            if test_idx is None or len(test_idx) == 0:
                X_test = None
                y_test = None
                print(f"Samples: Train={len(X_train)}, Valid={len(X_valid)}")
            else:
                X_test = preprocessor.transform(features_mmap, row_indices=test_idx, col_indices=col_indices)
                y_test = meta_df.loc[test_idx, target_col].values
                print(f"Samples: Train={len(X_train)}, Valid={len(X_valid)}, Test={len(X_test)}")
            # モデルのインスタンス化と学習
            model_class = get_class(cfg.model.model_target)
            cfg.hparams['random_state'] = i + 42  # アンサンブルごとに異なる乱数シードを設定
            model = model_class(task_type=cfg.target.task_type, **full_params)
            print(f"Training ensemble model {i+1}/{len(splits)}")
            model.fit(X_train, y_train, X_valid, y_valid, sample_weight=w_train, model_idx=i)
            # 予測の実行
            preds = {
                'train': model.predict(X_train),
                'valid': model.predict(X_valid),
                'test':  model.predict(X_test) if X_test is not None else None
            }
            # メトリクス算出 (Train / Valid / Test)
            for phase in ['train', 'valid', 'test']:
                if preds[phase] is not None:
                    y_true = locals()[f'y_{phase}']
                    m = evaluate_metrics(y_true, preds[phase], task_type=cfg.target.task_type)
                    # MLflowにフォールドごとの結果を記録
                    mlflow.log_metrics({f"fold{i}_{phase}_{k}": v for k, v in m.items()})
            # ビン分析用データの蓄積 
            # メタデータ(Future_High/Low/Close)を含めてDataFrame化
            for phase in ['valid', 'test']:
                if preds[phase] is not None:
                    idx = locals()[f'{phase}_idx'] # valid_idx or test_idx
                    res_df = pd.DataFrame({
                        'date': meta_df.loc[idx, 'date'],
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
        ensemble_model = EnsembleModel(models)
        
        # --- 評価メトリクスの算出 ---
        if all_results:
            full_res_df = pd.concat(all_results, ignore_index=True)
            # テストデータ全体でのビン分析
            if cv_method in ["purged_kfold", "cpcv"]:
                test_res = full_res_df[full_res_df['phase'] == 'valid']
            else: 
                test_res = full_res_df[full_res_df['phase'] == 'test']
            if not test_res.empty:
                bin_stats = calculate_bin_stats(
                    test_res, score_col='score', target_col='target', task_type=cfg.target.task_type,
                    metadata_cols=['Future_High', 'Future_Low', 'Future_Close']
                )
                # 成果物として保存
                bin_stats.to_csv("test_bin_analysis.csv")
                mlflow.log_artifact("test_bin_analysis.csv")

        # --- CVサマリをMLflow artifactとして保存 ---
        with tempfile.TemporaryDirectory() as d:
            json_path = os.path.join(d, "cv_splits.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(cv_summaries, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(json_path, artifact_path="cv")
            csv_path = os.path.join(d, "cv_splits.csv")
            pd.DataFrame(cv_summaries).to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path, artifact_path="cv")

        # --- 成果物（Artifacts）の保存 ---
        # 前処理でfitしたプリプロセッサ (StandardScalerなど)
        preprocessor_path = "preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
        # 学習済みモデル
        mlflow.pyfunc.log_model(
            name="ensemble_model",
            python_model=ensemble_model,
            # 依存ライブラリ（conda_env等）が必要な場合は追加指定
        )
        # Hydraの最終的なconfigファイル自体も保存（完全な再現用）
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config=cfg, f=f.name)
            mlflow.log_artifact(f.name, artifact_path="config")
        os.remove(f.name)
        
        # --- MLflow成果物の一括ZIP化とGoogle Driveへの移動 ---
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