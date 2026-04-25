import copy
import os
import io
from typing import Dict, Optional

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .base import BaseModelWrapper
from .pruning import execute_epoch_pruning, log_epoch_metrics
from .nbeats_model import NBeatsClassifier, NBeatsRegressor


class NBeatsWrapper(BaseModelWrapper):
    def __init__(self, task_type="regression", **params):
        self.task_type = task_type
        self.params = copy.deepcopy(params)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.params.get("device", "auto") != "cpu" else "cpu"
        )
        self.model = None
        self.best_state_dict = None
        self.history = {"train_loss": [], "valid_loss": []}
        self.input_shape_ = None
        self.feature_importances_ = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0, epoch_callback=None, train_dates=None, valid_dates=None):
        X_train_np = self._to_numpy(X_train).astype(np.float32)
        X_valid_np = self._to_numpy(X_valid).astype(np.float32) if X_valid is not None else None
        y_train_np = np.asarray(y_train, dtype=np.float32).reshape(-1)
        y_valid_np = np.asarray(y_valid, dtype=np.float32).reshape(-1) if y_valid is not None else None

        self.input_shape_ = X_train_np.shape[1:]
        input_size = int(np.prod(self.input_shape_))

        model_kwargs = {
            "input_size": input_size,
            "forecast_size": 1,
            "stack_type": self.params.get("stack_type", "generic"),
            "n_stacks": int(self.params.get("n_stacks", 3)),
            "n_blocks_per_stack": int(self.params.get("n_blocks_per_stack", 3)),
            "n_layers": int(self.params.get("n_layers", 4)),
            "hidden_size": int(self.params.get("hidden_size", 256)),
            "theta_size": int(self.params.get("theta_size", 64)),
            "dropout": float(self.params.get("dropout", 0.0)),
            "activation": self.params.get("activation", "relu"),
            "trend_degree": int(self.params.get("trend_degree", 2)),
            "n_harmonics": int(self.params.get("n_harmonics", 8)),
            "share_weights_in_stack": bool(self.params.get("share_weights_in_stack", False)),
        }

        if self.task_type == "classification":
            self.model = NBeatsClassifier(**model_kwargs).to(self.device)
            criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.model = NBeatsRegressor(**model_kwargs).to(self.device)
            loss_name = self.params.get("loss", "mse").lower()
            if loss_name == "huber":
                criterion = nn.HuberLoss(reduction="none", delta=float(self.params.get("huber_delta", 1.0)))
            elif loss_name == "mae":
                criterion = nn.L1Loss(reduction="none")
            else:
                criterion = nn.MSELoss(reduction="none")

        optimizer_name = self.params.get("optimizer", "adamw").lower()
        lr = float(self.params.get("lr", self.params.get("learning_rate", 1e-3)))
        weight_decay = float(self.params.get("weight_decay", 1e-5))
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = None
        scheduler_name = self.params.get("scheduler", "none").lower()
        if scheduler_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(self.params.get("scheduler_factor", 0.5)),
                patience=int(self.params.get("scheduler_patience", 3)),
            )

        batch_size = int(self.params.get("batch_size", 512))
        max_epochs = int(self.params.get("max_epochs", 100))
        patience = int(self.params.get("patience", 10))
        grad_clip_norm = float(self.params.get("grad_clip_norm", 1.0))

        train_loader = self._build_dataloader(X_train_np, y_train_np, sample_weight, batch_size, shuffle=True)
        valid_loader = None
        if X_valid_np is not None and y_valid_np is not None:
            valid_loader = self._build_dataloader(X_valid_np, y_valid_np, None, batch_size, shuffle=False)

        best_metric = np.inf
        best_epoch = -1
        epochs_without_improve = 0
        ema_metric = None

        for epoch in range(max_epochs):
            train_loss = self._run_epoch(train_loader, criterion, optimizer, grad_clip_norm)
            self.history["train_loss"].append(train_loss)

            if valid_loader is not None:
                valid_loss = self._evaluate(valid_loader, criterion)
                self.history["valid_loss"].append(valid_loss)
                metric_to_monitor = valid_loss
            else:
                metric_to_monitor = train_loss
                self.history["valid_loss"].append(np.nan)

            # Calculate EMA of metric_to_monitor to prevent stopping on noisy spikes
            if ema_metric is None:
                ema_metric = metric_to_monitor
            else:
                alpha = float(self.params.get("early_stopping_ema_alpha", 1.0))
                ema_metric = alpha * metric_to_monitor + (1.0 - alpha) * ema_metric

            if scheduler is not None:
                if scheduler_name == "reduce_on_plateau":
                    scheduler.step(ema_metric)
                else:
                    scheduler.step()

            # --- MLflow Logging ---
            metrics_to_log = {"train_loss": train_loss}
            if valid_loader is not None:
                metrics_to_log["valid_loss"] = valid_loss
            log_epoch_metrics(model_idx, epoch, metrics_to_log)

            # --- Epoch Callback (Pruning等) の実行 ---
            if epoch_callback is not None and X_valid is not None:
                valid_preds = self.predict(X_valid)
                execute_epoch_pruning(epoch_callback, epoch, valid_preds, y_valid_np)

            if ema_metric < best_metric:
                best_metric = ema_metric
                best_epoch = epoch
                epochs_without_improve = 0
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= patience:
                break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        self._log_learning_curve(model_idx)
        self._create_feature_importance_df(X_train)
        print(
            f"N-BEATS training finished. best_epoch={best_epoch}, best_valid_loss={best_metric:.6f}, device={self.device}"
        )

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        X_np = self._to_numpy(X).astype(np.float32)
        loader = self._build_dataloader(X_np, np.zeros(len(X_np), dtype=np.float32), None, self.params.get("batch_size", 1024), shuffle=False)
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                if self.task_type == "classification":
                    pred = torch.sigmoid(logits)
                else:
                    pred = logits
                outputs.append(pred.detach().cpu().numpy())
        preds = np.concatenate(outputs, axis=0).reshape(-1)
        return preds

    def save(self, save_path):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        state = {
            "task_type": self.task_type,
            "params": self.params,
            "input_shape": self.input_shape_,
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
        }
        torch.save(state, save_path)

    def load(self, save_path):
        state = torch.load(save_path, map_location=self.device)
        self.task_type = state["task_type"]
        self.params = state["params"]
        self.input_shape_ = tuple(state["input_shape"])
        input_size = int(np.prod(self.input_shape_))
        model_kwargs = {
            "input_size": input_size,
            "forecast_size": 1,
            "stack_type": self.params.get("stack_type", "generic"),
            "n_stacks": int(self.params.get("n_stacks", 3)),
            "n_blocks_per_stack": int(self.params.get("n_blocks_per_stack", 3)),
            "n_layers": int(self.params.get("n_layers", 4)),
            "hidden_size": int(self.params.get("hidden_size", 256)),
            "theta_size": int(self.params.get("theta_size", 64)),
            "dropout": float(self.params.get("dropout", 0.0)),
            "activation": self.params.get("activation", "relu"),
            "trend_degree": int(self.params.get("trend_degree", 2)),
            "n_harmonics": int(self.params.get("n_harmonics", 8)),
            "share_weights_in_stack": bool(self.params.get("share_weights_in_stack", False)),
        }
        if self.task_type == "classification":
            self.model = NBeatsClassifier(**model_kwargs).to(self.device)
        else:
            self.model = NBeatsRegressor(**model_kwargs).to(self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.history = state.get("history", {"train_loss": [], "valid_loss": []})

    def __getstate__(self):
        state = self.__dict__.copy()
        if "model" in state and state["model"] is not None:
            buffer = io.BytesIO()
            torch.save(self.model.state_dict(), buffer)
            state["model_state_dict"] = buffer.getvalue()
            del state["model"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "model_state_dict" in state:
            input_size = int(np.prod(self.input_shape_))
            model_kwargs = {
                "input_size": input_size,
                "forecast_size": 1,
                "stack_type": self.params.get("stack_type", "generic"),
                "n_stacks": int(self.params.get("n_stacks", 3)),
                "n_blocks_per_stack": int(self.params.get("n_blocks_per_stack", 3)),
                "n_layers": int(self.params.get("n_layers", 4)),
                "hidden_size": int(self.params.get("hidden_size", 256)),
                "theta_size": int(self.params.get("theta_size", 64)),
                "dropout": float(self.params.get("dropout", 0.0)),
                "activation": self.params.get("activation", "relu"),
                "trend_degree": int(self.params.get("trend_degree", 2)),
                "n_harmonics": int(self.params.get("n_harmonics", 8)),
                "share_weights_in_stack": bool(self.params.get("share_weights_in_stack", False)),
            }
            if self.task_type == "classification":
                self.model = NBeatsClassifier(**model_kwargs).to(self.device)
            else:
                self.model = NBeatsRegressor(**model_kwargs).to(self.device)
                
            buffer = io.BytesIO(state["model_state_dict"])
            self.model.load_state_dict(torch.load(buffer, map_location=self.device))
            del self.__dict__["model_state_dict"]

    def _build_dataloader(self, X, y, sample_weight, batch_size, shuffle=False):
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32))
        dataset = TensorDataset(X_tensor, y_tensor)
        if sample_weight is not None and shuffle:
            weights = torch.as_tensor(np.asarray(sample_weight, dtype=np.float32))
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def _run_epoch(self, loader, criterion, optimizer, grad_clip_norm):
        self.model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            optimizer.zero_grad()
            pred = self.model(xb)
            loss_vec = criterion(pred, yb)
            loss = loss_vec.mean()
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
            optimizer.step()
            losses.append(loss.detach().cpu().item())
        return float(np.mean(losses)) if losses else np.nan

    def _evaluate(self, loader, criterion):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)
                loss_vec = criterion(pred, yb)
                loss = loss_vec.mean()
                losses.append(loss.detach().cpu().item())
        return float(np.mean(losses)) if losses else np.nan

    def _to_numpy(self, X):
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)

    def _log_learning_curve(self, model_idx):
        if not self.history["train_loss"]:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.history["train_loss"], label="train_loss")
        if len(self.history["valid_loss"]) > 0:
            plt.plot(self.history["valid_loss"], label="valid_loss")
        plt.title(f"N-BEATS Learning Curve (Model {model_idx})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        temp_path = f"nbeats_learning_curve_m{model_idx}.png"
        plt.savefig(temp_path)
        plt.close()
        if mlflow.active_run():
            mlflow.log_artifact(temp_path, artifact_path="plots/learning_curves")
        os.remove(temp_path)

    def _create_feature_importance_df(self, X_train):
        # N-BEATS は tree model のような素直な feature importance を持たないため、
        # pipeline 互換のために空の DataFrame を保持する。
        feature_names = []
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns.tolist()
        elif hasattr(X_train, "shape") and len(X_train.shape) == 3:
            time_steps = X_train.shape[1]
            n_features = X_train.shape[2]
            feature_names = [f"t{t}_f{f}" for t in range(time_steps) for f in range(n_features)]
        elif hasattr(X_train, "shape") and len(X_train.shape) == 2:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        self.feature_importances_ = pd.DataFrame({"feature": feature_names, "importance": np.nan})
