import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import pyarrow as pa
import pyarrow.ipc as ipc
import zarr
from .base import BaseModelWrapper
from .pruning import execute_epoch_pruning, log_epoch_metrics
import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.pretraining import TabNetPretrainer

class MLflowAndPruningCallback(Callback):
    def __init__(self, wrapper, X_val, y_val, cb, m_idx):
        super().__init__()
        self.wrapper = wrapper
        self.X_val = X_val
        self.y_val = y_val
        self.cb = cb
        self.m_idx = m_idx
    def on_epoch_end(self, epoch, logs=None):
        if self.cb is not None and self.X_val is not None:
            preds = self.wrapper.predict(self.X_val)
            execute_epoch_pruning(self.cb, epoch, preds, self.y_val)
        if logs is not None:
            metrics = {}
            if 'loss' in logs: metrics['train_loss'] = logs['loss']
            for k, v in logs.items():
                if k.startswith('valid_'):
                    metrics[k] = v
            if metrics:
                log_epoch_metrics(self.m_idx, epoch, metrics)

    def __getstate__(self):
        state = self.__dict__.copy()
        # cb (epoch_callback) はローカル関数のため、pickle化できずエラーになるのを防ぐ
        state['cb'] = None
        return state

class TabNetWrapper(BaseModelWrapper):
    def __init__(self, task_type="regression", **params):
        self.task_type = task_type
        # TabNet固有のネットワーク設定を抽出
        self.cat_idxs = params.pop("cat_idx", [])
        self.cat_dims = params.pop("cat_dim", []) # preprocessor側と名称を合わせる
        self.use_pretrain = params.pop("use_pretrain", False)
        
        self.device_name = params.pop("device_name", "auto")
        if self.device_name == "auto":
            if torch.cuda.is_available():
                self.device_name = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device_name = "mps"
            else:
                self.device_name = "cpu"
                
        self.device = torch.device(self.device_name)
        # 残りのハイパーパラメータを保持
        self.params = params
        self.model = None

    def _from_ipc_handle(self, X):
        """IPCバッファハンドルを受け取ってDataFrameに復元する"""
        if isinstance(X, pa.Buffer):
            with ipc.open_stream(X) as reader:
                table = reader.read_all()
            return table.to_pandas()
        return X

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0, epoch_callback=None):
        if isinstance(X_train, str) and X_train.endswith('.zarr'):
            X_train = zarr.open(X_train, mode='r')[:]
        else:
            X_train = self._from_ipc_handle(X_train)
            
        if isinstance(X_valid, str) and X_valid.endswith('.zarr'):
            X_valid = zarr.open(X_valid, mode='r')[:]
        elif X_valid is not None:
            X_valid = self._from_ipc_handle(X_valid)
            
        # --- データクレンジング (ターゲットのNaN除去とウェイトの正値化) ---
        train_mask = ~np.isnan(y_train) & ~np.isinf(y_train)
        if sample_weight is not None:
            # ウェイトの NaN や Inf を 0 にし、さらに負の値を 0 にクリップ
            sample_weight = np.nan_to_num(sample_weight, nan=0.0, posinf=1.0, neginf=0.0)
            sample_weight = np.clip(sample_weight, 0.0, None)
            train_mask &= (sample_weight > 0)
        
        dropped_train = len(y_train) - np.sum(train_mask)
        if dropped_train > 0:
            print(f"  ⚠️ Dropped {dropped_train:,} training samples due to NaN target or zero weights.")
            
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        if sample_weight is not None:
            sample_weight = sample_weight[train_mask]
        if X_valid is not None and y_valid is not None:
            valid_mask = ~np.isnan(y_valid) & ~np.isinf(y_valid)
            dropped_valid = len(y_valid) - np.sum(valid_mask)
            if dropped_valid > 0:
                print(f"  ⚠️ Dropped {dropped_valid:,} validation samples due to NaN target.")
            X_valid = X_valid[valid_mask]
            y_valid = y_valid[valid_mask]

        # モデル初期化
        # cat_idxs と cat_dims を idx の昇順で並び替える
        sorted_cats = sorted(zip(self.cat_idxs, self.cat_dims))
        self.cat_idxs = [x[0] for x in sorted_cats]
        self.cat_dims = [x[1] for x in sorted_cats]
        common_params = {
            'n_d': self.params['n_d'],
            'n_a': self.params['n_a'],
            'n_steps': self.params['n_steps'],
            'gamma': self.params['gamma'],
            'lambda_sparse': self.params['lambda_sparse'],
            'cat_idxs': self.cat_idxs,
            'cat_dims': self.cat_dims,
            'optimizer_params': dict(lr=self.params['optimizer_params']['lr']),
            'mask_type': self.params.get('mask_type', 'entmax'),
            'seed': self.params.get('random_state', 42),
            'device_name': self.device_name,
            'verbose': 1
        }
        if self.task_type == "classification":
            model = TabNetClassifier(**common_params)
            metric = ['auc','logloss'] # TabNetの内部評価用
            # 分類の場合は 1次元 (n,) が必須
            y_train_fit = y_train.flatten()
            y_valid_fit = y_valid.flatten() if y_valid is not None else None
        else:
            model = TabNetRegressor(**common_params)
            metric = ['rmse','mse']
            # 回帰の場合は 2次元 (n, 1) が必須
            y_train_fit = y_train.reshape(-1, 1)
            y_valid_fit = y_valid.reshape(-1, 1) if y_valid is not None else None
            
        self.model = model

        # --- Epoch Callback (Pruning等) と MLflow Logging の準備 ---
        callbacks = []
        callbacks.append(MLflowAndPruningCallback(self, X_valid, y_valid, epoch_callback, model_idx))

        # --- 事前学習 (Pretraining) ---
        if self.use_pretrain:
            print("  Pretraining...")
            pretrainer = TabNetPretrainer(
                n_d=common_params['n_d'], n_a=common_params['n_a'], n_steps=common_params['n_steps'],
                cat_idxs=self.cat_idxs, cat_dims=self.cat_dims,
                optimizer_params=common_params['optimizer_params'],
                mask_type=common_params['mask_type'],
                seed=common_params['seed'],
                verbose=1,
                device_name=common_params['device_name']
            )
            pretrainer.fit(
                X_train=X_train.values,
                eval_set=[X_valid.values],
                max_epochs=self.params.get('max_epochs', 100),
                patience=self.params.get('patience', 10),
                batch_size=self.params.get('batch_size', 1024),
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False
            )
            # 事前学習の重みを適用
            model = TabNetClassifier(**common_params) if self.task_type == "classification" else TabNetRegressor(**common_params)
            self.model = model
            model.fit(
                X_train=X_train.values, y_train=y_train_fit,
                eval_set=[(X_valid.values, y_valid_fit)],
                eval_name=['valid'],
                eval_metric=metric,
                max_epochs=self.params.get('max_epochs', 100),
                patience=self.params.get('patience', 10),
                batch_size=self.params.get('batch_size', 1024),
                virtual_batch_size=128,
                num_workers=0,
                weights=sample_weight.flatten() if sample_weight is not None else 0,
                drop_last=False,
                from_unsupervised=pretrainer,
                callbacks=callbacks
            )
        else:
            # 通常学習
            model.fit(
                X_train=X_train.values, y_train=y_train_fit,
                eval_set=[(X_valid.values, y_valid_fit)],
                eval_name=['valid'],
                eval_metric=metric,
                max_epochs=self.params.get('max_epochs', 100),
                patience=self.params.get('patience', 10),
                batch_size=self.params.get('batch_size', 1024),
                virtual_batch_size=128,
                num_workers=0,
                weights=sample_weight.flatten() if sample_weight is not None else 0,
                drop_last=False,
                callbacks=callbacks
            )

        # 学習曲線のロギング
        self._log_learning_curve(model_idx)
        # 特徴量重要度のロギング
        # TabNetの重要度を計算するためにX_trainの列名が必要
        feature_names = X_train.columns.tolist()
        self._log_feature_importance(model_idx, feature_names)
        # 重要度のDataFrame作成
        self._create_feature_importance_df(feature_names)

    def _log_learning_curve(self, model_idx):
        """学習曲線をMLflowに保存する"""
        # history が空、または必要なキーがない場合のガード
        if self.model is None or not hasattr(self.model, 'history'):
            return
        # TabNetのHistoryオブジェクトから生の辞書データを取得
        # ライブラリの内部実装により、history.history に実際のデータが入っています
        history_dict = self.model.history.history
        # 存在確認を history_dict に対して行う
        if 'loss' not in history_dict:
            return
        plt.figure(figsize=(10, 6))
        # 1. トレーニングロスのプロット
        plt.plot(history_dict['loss'], label='train_loss')
        # 2. 検証データのメトリクスをプロット
        # ログにある 'valid_auc' や 'valid_logloss' を自動で拾います
        for key in history_dict.keys():
            if key.startswith('valid_'):
                plt.plot(history_dict[key], label=key)
        plt.title(f'TabNet Learning Curve (Model {model_idx})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        temp_path = f"learning_curve_m{model_idx}.png"
        plt.savefig(temp_path)
        plt.close()
        if mlflow.active_run():
            mlflow.log_artifact(temp_path, artifact_path="plots/learning_curves")
        os.remove(temp_path)

    def _log_feature_importance(self, model_idx, feature_names):
        """特徴量重要度をMLflowに保存する"""
        if self.model is None:
            return
        # TabNetの重要度を取得
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values(by='importance', ascending=False)
        # 上位30項目をプロット
        top_n = 30
        plot_df = importance_df.head(top_n)
        plt.figure(figsize=(10, 8))
        plt.barh(plot_df['feature'], plot_df['importance'])
        plt.xlabel('Importance')
        plt.title(f'TabNet Feature Importance (Model {model_idx})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        temp_path = f"feature_importance_m{model_idx}.png"
        plt.savefig(temp_path)
        plt.close()
        if mlflow.active_run():
            mlflow.log_artifact(temp_path, artifact_path="plots/importance")
            # 重要度のCSVも保存しておくとGeminiでの分析に役立ちます
            csv_path = f"feature_importance_m{model_idx}.csv"
            importance_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path, artifact_path="importance_data")
            os.remove(csv_path)
        os.remove(temp_path)       

    def _create_feature_importance_df(self, feature_names):
        """重要度をデータフレーム形式で作成して属性に保持する"""
        if self.model is not None:
            # TabNetの重要度をDataFrame化。LGBMWrapperとカラム名を合わせる
            self.feature_importances_ = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values(by='importance', ascending=False)
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        if isinstance(X, str) and X.endswith('.zarr'):
            X = zarr.open(X, mode='r')[:]
        else:
            X = self._from_ipc_handle(X)
        # 入力を NumPy 配列に変換
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        if self.task_type == "regression":
            # 回帰の場合は通常の予測
            preds = self.model.predict(X_values)
        else:
            # 分類の場合は「確率」を返すように統一する
            # predict_proba は [クラス0の確率, クラス1の確率] を返すので、
            # 陽性クラス（通常はインデックス1）の確率を抽出する
            probs = self.model.predict_proba(X_values)
            preds = probs[:, 1] # (n_samples,) の形式になる
        # アンサンブル集計のために 1次元配列に整形して返す
        return preds.flatten()
