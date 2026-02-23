import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import warnings
import json
import hydra
from pathlib import Path
from omegaconf import DictConfig
import mlflow
import matplotlib.pyplot as plt
from src.cv.purged_kfold import SimplePurgedKFold, add_t1_column, prepare_purged_cv_input
from src.cv.cv_viz import summarize_split_for_logging

# 警告の抑制
warnings.filterwarnings('ignore')

# プロジェクトルートへのパスを追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    print(f"Project Root: {project_root}")
    master_dir = Path(project_root) / "data/master"
    
    # --- MLflow Setup ---
    abs_path = os.path.expanduser("~/JPSForecast/mlflow_runs")
    os.makedirs(abs_path, exist_ok=True)
    mlflow_db_path = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(mlflow_db_path)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="Feature_Screening"):
        # --- 1. データ読み込み ---
        print(f"Loading data from {master_dir}...")
        # メタデータ
        meta_path = master_dir / "index_meta.parquet"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found. Please run create_master_data.py first.")
        meta_df = pd.read_parquet(meta_path).reset_index(drop=True)
        # 特徴量名
        names_path = master_dir / "feature_names.json"
        with open(names_path, 'r') as f:
            feature_names = json.load(f)
            if isinstance(feature_names, dict):
                feature_names = list(feature_names.values())
        # 特徴量 (memmap)
        features_path = master_dir / "features.npy"
        if not features_path.exists():
            raise FileNotFoundError(f"{features_path} not found.")
        X_mmap = np.memmap(
            features_path, 
            dtype='float32', 
            mode='r', 
            shape=(len(meta_df), len(feature_names))
        )
        # ターゲット取得
        target_col = cfg.target.column
        if target_col not in meta_df.columns:
            print(f"Warning: Target column '{target_col}' not found. Trying 'target_reg'...")
            if 'target_reg' in meta_df.columns:
                target_col = 'target_reg'
            else:
                raise ValueError(f"Target column '{target_col}' or 'target_reg' not found in metadata.")
        y_all = meta_df[target_col]
        # 欠損ターゲットの除外
        valid_mask = y_all.notna()
        mlflow.log_param("target_col", target_col)
        print(f"Target: {target_col}, Total rows: {len(meta_df)}, Valid rows: {valid_mask.sum()}")
        if valid_mask.sum() == 0:
            raise ValueError("No valid target data found.")
        # 有効なデータのみ抽出 (メモリ効率のため)
        # 注意: データ量が非常に大きい場合、ここでメモリ不足になる可能性があります。
        subset_idx = meta_df.index[valid_mask].to_numpy()
        # 特徴量をメモリにロード
        X = X_mmap[subset_idx]
        y = y_all.iloc[subset_idx].to_numpy()
        meta_subset = meta_df.iloc[subset_idx].copy().reset_index(drop=True)
        
        # --- 2. Purged CVの準備 ---
        print("Setting up Purged CV...")
        # Horizon設定 (ドメイン依存)
        if cfg.domain.name == 'TAC':
            horizon = 5
        else:
            horizon = 60
        print('Horizon: '+str(horizon))
        mlflow.log_param("horizon", horizon)
        # カレンダー取得とT1計算
        # 範囲外データが削除される可能性があるため、X, yも同期してフィルタリングする必要がある
        # add_t1_column はフィルタ後のDataFrameを返すため、インデックスを使ってX, yを再スライスする
        meta_subset_with_t1 = add_t1_column(meta_subset, horizon)
        # フィルタリングされたインデックスを取得
        valid_indices = meta_subset_with_t1.index
        X = X[valid_indices]
        y = y[valid_indices]
        meta_subset = meta_subset_with_t1.reset_index(drop=True)
        # CV用入力データの作成
        samples_info, date_to_indices, dates = prepare_purged_cv_input(meta_subset)
        
        # ログ出力用に pos -> date のマッピングを作成
        pos_to_date = pd.Series(dates)

        # CV設定
        n_splits = int(cfg.period.n_splits)
        pct_embargo = float(cfg.period.pct_embargo)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("pct_embargo", pct_embargo)
        cv = SimplePurgedKFold(
            n_splits=n_splits,
            samples_info_sets=samples_info,
            pct_embargo=pct_embargo
        )
        # CV入力用ダミー配列 (日付数分)
        cv_input = np.zeros((len(dates), 1), dtype=np.float32)
        
        # --- 3. LGBM学習 & SHAP算出 ---
        lgbm_params = {
            'objective': 'regression', 
            'boosting_type': 'gbdt',
            'n_jobs': -1,
            'verbosity': -1,
            'random_state': 42,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth':8,
            'min_child_samples':100,
            'colsample_bytree':0.5,
            'subsample':0.8,
            'n_estimators': 10000,
        }
        fold_importance_df = pd.DataFrame(index=feature_names)
        print(f"Starting Feature Screening (n_splits={n_splits})...")
        for fold, (tr_pos, val_pos) in enumerate(cv.split(cv_input)):
            print(f"Processing Fold {fold + 1}...")
            
            # 期間情報の表示
            info = summarize_split_for_logging(
                fold=fold,
                tr_pos=tr_pos,
                va_pos=val_pos,
                pos_to_date=pos_to_date,
                timeline_width=60
            )
            print(f"  TRAIN: {info['train_start']} ~ {info['train_end']} ({info['train_days']} days)")
            print(f"  VALID: {info['valid_start']} ~ {info['valid_end']} ({info['valid_days']} days)")

            # 日付インデックスからサンプルインデックスへ変換
            tr_dates = dates[tr_pos]
            val_dates = dates[val_pos]
            train_idx = np.concatenate([date_to_indices[pd.Timestamp(d)] for d in tr_dates])
            val_idx = np.concatenate([date_to_indices[pd.Timestamp(d)] for d in val_dates])
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            print('  Train: ', X_train.shape, y_train.shape)
            print('  Valid: ', X_val.shape, y_val.shape)
            # モデル学習
            print('  Training LGBM...')
            model = lgb.LGBMRegressor(**lgbm_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            # SHAP値の算出
            # 計算時間短縮のため、Validationデータが多い場合はサンプリング
            print('  Calculating SHAP values...')
            MAX_SHAP_SAMPLES = 300000
            if len(X_val) > MAX_SHAP_SAMPLES:
                val_sample_idx = np.random.choice(len(X_val), MAX_SHAP_SAMPLES, replace=False)
                X_val_shap = X_val[val_sample_idx]
            else:
                X_val_shap = X_val
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_val_shap)
            # Binary Classificationの場合の対応
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            # Mean Absolute SHAP
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            fold_importance_df[f'fold_{fold+1}'] = mean_abs_shap

        # --- 4. 結果の集計と保存 ---
        fold_importance_df['average_importance'] = fold_importance_df.mean(axis=1)
        fold_importance_df['CV'] = fold_importance_df.mean(axis=1) / fold_importance_df.std(axis=1)
        result = fold_importance_df.sort_values(by='average_importance', ascending=False)
        print("\n=== Feature Importance (Mean Absolute SHAP) ===")
        print(result[['average_importance']].head(10))
        OUTPUT_FILE = Path(project_root) / f'feature_importance_{cfg.domain.name}.csv'
        result.to_csv(OUTPUT_FILE)
        mlflow.log_artifact(OUTPUT_FILE)
        print(f"\nFull results saved to: {OUTPUT_FILE}")

        # --- 5. プロット作成と保存 ---
        plt.figure(figsize=(12, 10))
        # 上位30個を表示
        top_features = result.head(30).sort_values(by='average_importance', ascending=True)
        plt.barh(top_features.index, top_features['average_importance'])
        plt.xlabel("Mean Absolute SHAP Value")
        plt.title(f"Feature Importance (Target: {target_col})")
        plt.tight_layout()
        plot_path = Path(project_root) / f"feature_importance_plot_{cfg.domain.name}.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

if __name__ == "__main__":
    main()
