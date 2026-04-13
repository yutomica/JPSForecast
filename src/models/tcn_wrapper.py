import os
import io
import torch
import torch.nn as nn
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseModelWrapper
from .pruning import execute_epoch_pruning, log_epoch_metrics
from .networks.tcn import TCN

class TCNWrapper(BaseModelWrapper):
    def __init__(self, task_type="regression", **params):
        self.task_type = task_type
        self.params = params
        device_name = self.params.pop("device_name", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(
            "cuda" if (device_name == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        self.model = None
        self.history = {'train_loss': [], 'valid_loss': []}

    def _compute_loss(self, logits, y, sample_weight=None):
        loss_type = self.params.get("objective", "mse")
        
        if self.task_type == "classification":
            loss_each = nn.BCEWithLogitsLoss(reduction='none')(logits.view(-1), y.float().view(-1))
        elif loss_type == "quantile":
            alpha = self.params.get("alpha", 0.5)
            diff = y.float().view(-1) - logits.view(-1)
            loss_each = torch.max(alpha * diff, (alpha - 1) * diff)
        elif loss_type == "fair":
            c = self.params.get("fair_c", 1.0)
            abs_diff = torch.abs(y.float().view(-1) - logits.view(-1))
            loss_each = c * (abs_diff - c * torch.log1p(abs_diff / c))
        elif loss_type == "tweedie":
            p = self.params.get("tweedie_variance_power", 1.5)
            # Tweedieは予測値が正である必要があるため最小値を制限
            mu = torch.clamp(logits.view(-1), min=1e-6)
            y_c = torch.max(y.float().view(-1), torch.zeros_like(y.float().view(-1)))
            loss_each = (mu ** (2 - p)) / (2 - p) - y_c * (mu ** (1 - p)) / (1 - p)
        elif loss_type == "asymmetric_mse":
            alpha = self.params.get("alpha", 3.0)
            beta = self.params.get("beta", 1.0)
            diff = y.float().view(-1) - logits.view(-1)
            loss_each = torch.where(diff > 0, alpha * (diff ** 2), beta * (diff ** 2))
        else:
            loss_each = nn.MSELoss(reduction='none')(logits.view(-1), y.float().view(-1))
            
        if sample_weight is not None:
            loss_each = loss_each * sample_weight.view(-1)
            
        return loss_each.mean()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0, epoch_callback=None):
        # --- [1. 初期化] Early Stopping 用の変数を準備 ---
        patience = self.params.get('early_stopping_patience', 5)     # 何エポック改善がなければ止めるか
        min_delta = self.params.get('min_delta', 0.0) # どの程度の改善を「進歩」とみなすか
        best_v_loss = float('inf')
        counter = 0
        input_size = X_train.shape[2]
        self.params['input_size'] = input_size
        self.model = TCN(
            input_size=input_size,
            output_size=1 if self.task_type == "regression" else 1, # 分類も1出力(BCE)
            n_layers=self.params.get('n_layers', 2),
            num_channel=self.params.get('num_channel', 64),
            kernel_size=self.params.get('kernel_size', 3),
            dropout=self.params.get('dropout', 0.2)
        ).to(self.device)
        best_model_wts = copy.deepcopy(self.model.state_dict())
        # パラメータ数のサマリー表示
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n--- TCN Model Summary (Fold {model_idx}) ---")
        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("-" * 40)

        # --- データクレンジング (NaN / Inf の確実な除去) ---
        # ターゲット、特徴量(3次元)、ウェイトのいずれかに NaN または Inf が含まれるレコードを除外
        train_mask = ~np.isnan(y_train) & ~np.isinf(y_train)
        train_mask &= ~np.isnan(X_train).any(axis=(1, 2)) & ~np.isinf(X_train).any(axis=(1, 2))
        if sample_weight is not None:
            train_mask &= ~np.isnan(sample_weight) & ~np.isinf(sample_weight)
        dropped_train = len(y_train) - train_mask.sum()
        if dropped_train > 0:
            print(f"⚠️ Dropped {dropped_train:,} training samples due to NaN/Inf in features, target, or weights.")
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        if sample_weight is not None:
            sample_weight = sample_weight[train_mask]
        if X_valid is not None and y_valid is not None:
            valid_mask = ~np.isnan(y_valid) & ~np.isinf(y_valid)
            valid_mask &= ~np.isnan(X_valid).any(axis=(1, 2)) & ~np.isinf(X_valid).any(axis=(1, 2))
            dropped_valid = len(y_valid) - valid_mask.sum()
            if dropped_valid > 0:
                print(f"⚠️ Dropped {dropped_valid:,} validation samples due to NaN/Inf in features or target.")
            X_valid = X_valid[valid_mask]
            y_valid = y_valid[valid_mask]

        # データの準備
        if sample_weight is None:
            sample_weight = np.ones(len(y_train), dtype=np.float32)
        train_ds = TensorDataset(
            torch.from_numpy(X_train), 
            torch.from_numpy(y_train).float(),
            torch.from_numpy(sample_weight).float() # 重みをセット
        )
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.params.get('batch_size', 32), 
            shuffle=True,
            pin_memory=(self.device.type == "cuda")
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.params.get('lr', self.params.get('learning_rate', 0.001)),
            weight_decay=self.params.get('weight_decay', 0.0)
        )
        max_epochs = self.params.get('max_epochs', self.params.get('max_epochs', 10))
        
        loss_name = self.params.get("objective", "mse") if self.task_type != "classification" else "bce"

        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{max_epochs}]", unit="batch", leave=False)
            for batch_x, batch_y, batch_w in pbar:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                batch_w = batch_w.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                output = self.model(batch_x)
                weighted_loss = self._compute_loss(output, batch_y, batch_w)
                weighted_loss.backward()
                
                grad_clip = self.params.get('gradient_clip_val', 0.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    
                optimizer.step()
                epoch_loss += weighted_loss.item()
                pbar.set_postfix({"loss": f"{weighted_loss.item():.8f}"})
            avg_train_loss = epoch_loss / len(train_loader)
            self.history['train_loss'].append(epoch_loss / len(train_loader))
            progress_msg = f"Epoch [{epoch+1:03d}/{max_epochs:03d}] - train_loss: {avg_train_loss:.8f}"
            # tqdm.write(progress_msg)
            if X_valid is not None:
                self.model.eval()
                with torch.no_grad():
                    v_input = torch.from_numpy(X_valid).to(self.device)
                    v_target = torch.from_numpy(y_valid).to(self.device).float()
                    v_out = self.model(v_input)
                    v_loss = self._compute_loss(v_out, v_target)
                    avg_v_loss = v_loss.item()
                    self.history['valid_loss'].append(avg_v_loss)
                    progress_msg += f" - valid_{loss_name}: {avg_v_loss:.8f}"
                    # --- [4. Early Stopping 判定] ---
                    if avg_v_loss < best_v_loss - min_delta:
                        best_v_loss = avg_v_loss
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        counter = 0 # 改善したのでカウンターリセット
                    else:
                        counter += 1 # 改善しなかったのでカウントアップ
            tqdm.write(progress_msg)
            
            # --- MLflow Logging ---
            metrics_to_log = {"train_loss": avg_train_loss}
            if X_valid is not None:
                metrics_to_log["valid_loss"] = avg_v_loss
            log_epoch_metrics(model_idx, epoch, metrics_to_log)
            
            # --- [5. Epoch Callback (Pruning等) の実行] ---
            if epoch_callback is not None and X_valid is not None:
                valid_preds = self.predict(X_valid)
                execute_epoch_pruning(epoch_callback, epoch, valid_preds, y_valid)
            
            # 早期終了の実行
            if counter >= patience:
                tqdm.write(f"Early stopping triggered at epoch {epoch+1}")
                break
        self._log_learning_curve(model_idx)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X).to(self.device)
            logits = self.model(inputs).cpu().numpy().flatten()
            if self.task_type == "classification":
                # 分類の場合は確率を返す
                return torch.sigmoid(torch.from_numpy(logits)).numpy()
            return logits

    def _log_learning_curve(self, model_idx):
        """学習曲線ロギング"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='train_loss')
        if self.history['valid_loss']:
            plt.plot(self.history['valid_loss'], label='valid_loss')
        plt.title(f'TCN Learning Curve (Model {model_idx})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        temp_path = f"learning_curve_tcn_m{model_idx}.png"
        plt.savefig(temp_path)
        plt.close()
        if mlflow.active_run():
            mlflow.log_artifact(temp_path, artifact_path="plots/learning_curves")
        os.remove(temp_path)

    def __getstate__(self):
        """保存時の挙動を定義"""
        state = self.__dict__.copy()
        # model オブジェクトそのものは pickle できないため除外
        if "model" in state and state["model"] is not None:
            # 重み（state_dict）をバイナリ形式で保持
            buffer = io.BytesIO()
            torch.save(self.model.state_dict(), buffer)
            state["model_state_dict"] = buffer.getvalue()
            del state["model"]
        return state

    def __setstate__(self, state):
        """ロード時の挙動を定義"""
        self.__dict__.update(state)
        if "model_state_dict" in state:
            # 1. モデルの再構築 (保存されている params を使用)
            input_size = self.params.get("input_size") # 保存時に保持しておく必要あり
            if input_size is None:
                raise ValueError("input_size not found in params during __setstate__")
            self.model = TCN(
                input_size=input_size,
                output_size=1, # 課題に合わせて調整
                n_layers=self.params.get('n_layers', 2),
                num_channel=self.params.get('num_channel', 64),
                kernel_size=self.params.get('kernel_size', 3),
                dropout=self.params.get('dropout', 0.2)
            ).to(self.device)
            # 2. 重みの復元
            buffer = io.BytesIO(state["model_state_dict"])
            self.model.load_state_dict(torch.load(buffer, map_location=self.device))
            del self.__dict__["model_state_dict"]