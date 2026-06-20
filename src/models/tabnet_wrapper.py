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
    def __init__(self, wrapper, X_val, y_val, cb, m_idx, max_epochs):
        super().__init__()
        self.wrapper = wrapper
        self.X_val = X_val
        self.y_val = y_val
        self.cb = cb
        self.m_idx = m_idx
        self.max_epochs = max_epochs
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
                if epoch % 10 == 0 or epoch == self.max_epochs - 1:
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
        self.ensemble_size = int(params.pop("ensemble_size", 1))
        self.models = []
        self.model = None

    def _from_ipc_handle(self, X):
        """IPCバッファハンドルを受け取ってDataFrameに復元する"""
        if isinstance(X, pa.Buffer):
            with ipc.open_stream(X) as reader:
                table = reader.read_all()
            return table.to_pandas()
        return X

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0, epoch_callback=None, train_dates=None, valid_dates=None):
        if isinstance(X_train, str) and X_train.endswith('.zarr'): X_train = zarr.open(X_train, mode='r')[:]
        else: X_train = self._from_ipc_handle(X_train)
        if isinstance(X_valid, str) and X_valid.endswith('.zarr'): X_valid = zarr.open(X_valid, mode='r')[:]
        elif X_valid is not None: X_valid = self._from_ipc_handle(X_valid)
        train_mask = ~np.isnan(y_train) & ~np.isinf(y_train)
        if sample_weight is not None:
            sample_weight = np.nan_to_num(sample_weight, nan=0.0, posinf=1.0, neginf=0.0)
            train_mask &= (np.clip(sample_weight, 0.0, None) > 0)
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        if sample_weight is not None: sample_weight = sample_weight[train_mask]
        if X_valid is not None and y_valid is not None:
            valid_mask = ~np.isnan(y_valid) & ~np.isinf(y_valid)
            X_valid, y_valid = X_valid[valid_mask], y_valid[valid_mask]
        sorted_cats = sorted(zip(self.cat_idxs, self.cat_dims))
        self.cat_idxs, self.cat_dims = [x[0] for x in sorted_cats], [x[1] for x in sorted_cats]
        
        self.models = []; all_feature_importances, base_seed = [], int(self.params.get("random_state", 42))
        for s_idx in range(self.ensemble_size):
            current_seed = base_seed + s_idx
            if self.ensemble_size > 1: print(f"
🚀 Training Ensemble Model {s_idx+1}/{self.ensemble_size} (seed={current_seed})...")
            common_params = {'n_d': self.params['n_d'], 'n_a': self.params['n_a'], 'n_steps': self.params['n_steps'], 'gamma': self.params['gamma'], 'lambda_sparse': self.params['lambda_sparse'], 'cat_idxs': self.cat_idxs, 'cat_dims': self.cat_dims, 'optimizer_params': dict(lr=self.params['optimizer_params']['lr']), 'mask_type': self.params.get('mask_type', 'entmax'), 'seed': current_seed, 'device_name': self.device_name, 'verbose': 1}
            if self.task_type == "classification": model = TabNetClassifier(**common_params); metric = ['auc','logloss']; y_train_fit, y_valid_fit = y_train.flatten(), (y_valid.flatten() if y_valid is not None else None)
            else: model = TabNetRegressor(**common_params); metric = ['rmse','mse']; y_train_fit, y_valid_fit = y_train.reshape(-1, 1), (y_valid.reshape(-1, 1) if y_valid is not None else None)
            self.model = model; callbacks = [MLflowAndPruningCallback(self, X_valid, y_valid, epoch_callback if s_idx == 0 else None, model_idx, self.params.get('max_epochs', 100))]
            if self.use_pretrain:
                pretrainer = TabNetPretrainer(**{k: v for k, v in common_params.items() if k in ['n_d', 'n_a', 'n_steps', 'cat_idxs', 'cat_dims', 'optimizer_params', 'mask_type', 'seed', 'verbose', 'device_name']})
                pretrainer.fit(X_train=X_train.values if hasattr(X_train, 'values') else X_train, eval_set=[X_valid.values if hasattr(X_valid, 'values') else X_valid] if X_valid is not None else [], max_epochs=self.params.get('max_epochs', 100), patience=self.params.get('patience', 10), batch_size=self.params.get('batch_size', 1024), virtual_batch_size=128, num_workers=0, drop_last=False)
                model = TabNetClassifier(**common_params) if self.task_type == "classification" else TabNetRegressor(**common_params); self.model = model
                model.fit(X_train=X_train.values if hasattr(X_train, 'values') else X_train, y_train=y_train_fit, eval_set=[(X_valid.values if hasattr(X_valid, 'values') else X_valid, y_valid_fit)] if X_valid is not None else [], eval_name=['valid'] if X_valid is not None else [], eval_metric=metric, max_epochs=self.params.get('max_epochs', 100), patience=self.params.get('patience', 10), batch_size=self.params.get('batch_size', 1024), virtual_batch_size=128, num_workers=0, weights=sample_weight.flatten() if sample_weight is not None else 0, drop_last=False, from_unsupervised=pretrainer, callbacks=callbacks)
            else:
                model.fit(X_train=X_train.values if hasattr(X_train, 'values') else X_train, y_train=y_train_fit, eval_set=[(X_valid.values if hasattr(X_valid, 'values') else X_valid, y_valid_fit)] if X_valid is not None else [], eval_name=['valid'] if X_valid is not None else [], eval_metric=metric, max_epochs=self.params.get('max_epochs', 100), patience=self.params.get('patience', 10), batch_size=self.params.get('batch_size', 1024), virtual_batch_size=128, num_workers=0, weights=sample_weight.flatten() if sample_weight is not None else 0, drop_last=False, callbacks=callbacks)
            self.models.append(copy.deepcopy(model))
            if s_idx == 0: self._log_learning_curve(model_idx); self.best_epoch_ = model.best_epoch if hasattr(model, 'best_epoch') else self.params.get('max_epochs', 100) - 1
            all_feature_importances.append(model.feature_importances_)
            gc.collect()
        self.feature_importances_ = pd.DataFrame({'feature': X_train.columns.tolist() if hasattr(X_train, 'columns') else [f"f{i}" for i in range(X_train.shape[1])], 'importance': np.mean(all_feature_importances, axis=0)}).sort_values(by='importance', ascending=False)
        self.model = self.models[0]

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

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, f"learning_curve_m{model_idx}.png")
            plt.savefig(temp_path)
            plt.close()
            if mlflow.active_run():
                mlflow.log_artifact(temp_path, artifact_path="plots/learning_curves")

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

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, f"feature_importance_m{model_idx}.png")
            plt.savefig(temp_path)
            plt.close()
            if mlflow.active_run():
                mlflow.log_artifact(temp_path, artifact_path="plots/importance")
                # 重要度のCSVも保存しておくとGeminiでの分析に役立ちます
                csv_path = os.path.join(tmpdir, f"feature_importance_m{model_idx}.csv")
                importance_df.to_csv(csv_path, index=False)
                mlflow.log_artifact(csv_path, artifact_path="importance_data")

    def _create_feature_importance_df(self, feature_names):
        """重要度をデータフレーム形式で作成して属性に保持する"""
        if self.model is not None:
            # TabNetの重要度をDataFrame化。LGBMWrapperとカラム名を合わせる
            self.feature_importances_ = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values(by='importance', ascending=False)
    
    def predict(self, X):
        if not self.models: raise ValueError("Model has not been trained yet.")
        if isinstance(X, str) and X.endswith('.zarr'): X = zarr.open(X, mode='r')[:]
        else: X = self._from_ipc_handle(X)
        X_values = X.values if hasattr(X, 'values') else X
        all_ensemble_preds = []
        for model in self.models:
            if self.task_type == "regression": preds = model.predict(X_values)
            else: probs = model.predict_proba(X_values); preds = probs[:, 1]
            all_ensemble_preds.append(preds.flatten())
        return np.mean(all_ensemble_preds, axis=0)