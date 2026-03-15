import copy
import os
import random
from collections.abc import Iterable
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import mlflow
from .base import BaseModelWrapper
from .networks.gandalf import GANDALFNet


class GANDALFWrapper(BaseModelWrapper):
    """
    train.py 互換の GANDALF ラッパー。

    想定タスク:
    - regression
    - classification (binary only)

    主要仕様:
    - fit(X_train, y_train, X_valid, y_valid, sample_weight, model_idx)
    - predict(X)
    - feature_importances_ を DataFrame で保持
    - joblib 保存に備えた state_dict ベースのシリアライズ対応
    """

    def __init__(self, task_type: str = "regression", **params):
        self.task_type = task_type
        self.params = copy.deepcopy(params)

        # train.py / 他モデル互換で流れてくる未使用メタ情報は吸収しておく
        self.cat_idx = self.params.pop("cat_idx", self.params.pop("cat_idxs", []))
        self.cat_dims = self.params.pop("cat_dims", self.params.pop("cat_dim", []))

        self.model: Optional[GANDALFNet] = None
        self.model_init_kwargs: Optional[Dict[str, Any]] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history: Dict[str, list] = {"train_loss": [], "valid_loss": []}
        self.feature_importances_ = None
        self.feature_names_ = None
        self.is_binary_classification = task_type == "classification"

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0):
        X_train_np, train_feature_names = self._to_numpy_features(X_train)
        if X_train_np is None or len(X_train_np) == 0:
            raise ValueError("X_train is empty.")

        X_valid_np, _ = self._to_numpy_features(X_valid) if X_valid is not None else (None, None)
        y_train_np = np.asarray(y_train)
        y_valid_np = np.asarray(y_valid) if y_valid is not None else None

        self.feature_names_ = train_feature_names
        self._set_seed(int(self.params.get("random_state", 42)))

        batch_size = int(self.params.get("batch_size", 1024))
        max_epochs = int(self.params.get("max_epochs", 100))
        patience = int(self.params.get("patience", 10))
        learning_rate = float(
            self.params.get(
                "learning_rate",
                self.params.get("lr", self.params.get("optimizer_params", {}).get("lr", 1e-3)),
            )
        )
        weight_decay = float(self.params.get("weight_decay", 1e-5))
        gradient_clip_val = float(self.params.get("gradient_clip_val", 1.0))
        num_workers = int(self.params.get("num_workers", 0))

        gflu_stages = int(self.params.get("gflu_stages", 6))
        gflu_dropout = float(self.params.get("gflu_dropout", 0.0))
        feature_init_sparsity = float(
            self.params.get(
                "gflu_feature_init_sparsity",
                self.params.get("feature_init_sparsity", 0.3),
            )
        )
        learnable_sparsity = bool(self.params.get("learnable_sparsity", True))
        head_hidden_dims = self._normalize_hidden_dims(
            self.params.get("head_hidden_dims", self.params.get("head_dims", [128, 64]))
        )
        head_dropout = float(self.params.get("head_dropout", self.params.get("dropout", 0.1)))

        input_dim = int(X_train_np.shape[1])
        if input_dim <= 0:
            raise ValueError("GANDALF received zero input features after preprocessing.")

        if self.task_type == "classification":
            unique_vals = np.unique(y_train_np[~pd.isna(y_train_np)])
            if len(unique_vals) > 2:
                raise ValueError(
                    "This wrapper currently supports binary classification only. "
                    f"Detected classes: {unique_vals.tolist()}"
                )
            y_train_np = self._to_binary_target(y_train_np)
            y_valid_np = self._to_binary_target(y_valid_np) if y_valid_np is not None else None
            output_dim = 1
            target_bias = self._initial_binary_bias(y_train_np)
            self.is_binary_classification = True
        else:
            y_train_np = np.asarray(y_train_np, dtype=np.float32).reshape(-1)
            y_valid_np = np.asarray(y_valid_np, dtype=np.float32).reshape(-1) if y_valid_np is not None else None
            output_dim = 1
            target_bias = float(np.nanmean(y_train_np)) if len(y_train_np) else 0.0
            self.is_binary_classification = False

        self.model_init_kwargs = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "gflu_stages": gflu_stages,
            "gflu_dropout": gflu_dropout,
            "feature_init_sparsity": feature_init_sparsity,
            "learnable_sparsity": learnable_sparsity,
            "head_hidden_dims": head_hidden_dims,
            "head_dropout": head_dropout,
            "target_bias": target_bias,
        }
        self.model = GANDALFNet(**self.model_init_kwargs).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n--- GANDALF Model Summary (Fold {model_idx}) ---")
        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("-" * 40)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.params.get("lr_decay_factor", 0.5)),
            patience=max(1, int(self.params.get("lr_patience", 3))),
            min_lr=float(self.params.get("min_lr", 1e-6)),
        )

        train_dataset = self._make_dataset(X_train_np, y_train_np, sample_weight)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
        )

        best_metric = float("inf")
        best_state = None
        bad_epochs = 0
        self.history = {"train_loss": [], "valid_loss": []}

        for epoch in range(max_epochs):
            train_loss = self._train_one_epoch(train_loader, optimizer, gradient_clip_val, epoch, max_epochs)
            valid_loss = self._evaluate_loss(X_valid_np, y_valid_np) if X_valid_np is not None else train_loss
            scheduler.step(valid_loss)

            self.history["train_loss"].append(train_loss)
            self.history["valid_loss"].append(valid_loss)

            tqdm.write(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f}")

            if valid_loss < best_metric - 1e-8:
                best_metric = valid_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    tqdm.write(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

        self._create_feature_importance_df(self.feature_names_)
        self._log_learning_curve(model_idx)
        self._log_feature_importance(model_idx)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        X_np, _ = self._to_numpy_features(X)
        if X_np is None or len(X_np) == 0:
            return np.array([], dtype=np.float32)

        self.model.eval()
        preds = []
        batch_size = int(self.params.get("predict_batch_size", self.params.get("batch_size", 4096)))

        with torch.no_grad():
            for start in range(0, len(X_np), batch_size):
                xb = torch.from_numpy(X_np[start : start + batch_size]).to(self.device)
                out = self.model(xb).squeeze(-1)
                if self.is_binary_classification:
                    out = torch.sigmoid(out)
                preds.append(out.detach().cpu().numpy())

        if not preds:
            return np.array([], dtype=np.float32)
        return np.concatenate(preds, axis=0).astype(np.float32, copy=False).reshape(-1)

    def _train_one_epoch(self, train_loader, optimizer, gradient_clip_val: float, epoch: int, max_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_weight = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False) as pbar:
            for xb, yb, wb in pbar:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                wb = wb.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                out = self.model(xb).squeeze(-1)
                loss_vec = self._loss_vector(out, yb)
                loss = (loss_vec * wb).sum() / wb.sum().clamp_min(1e-8)
                loss.backward()

                if gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
                optimizer.step()

                batch_weight = float(wb.sum().detach().cpu())
                total_loss += float(loss.detach().cpu()) * batch_weight
                total_weight += batch_weight
                
                pbar.set_postfix({"train_loss": f"{total_loss / max(total_weight, 1e-8):.6f}"})

        return total_loss / max(total_weight, 1e-8)

    def _evaluate_loss(self, X_np, y_np) -> float:
        if X_np is None or y_np is None or len(X_np) == 0:
            return float("nan")

        self.model.eval()
        batch_size = int(self.params.get("eval_batch_size", self.params.get("batch_size", 4096)))
        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for start in range(0, len(X_np), batch_size):
                xb = torch.from_numpy(X_np[start : start + batch_size]).to(self.device)
                yb = torch.from_numpy(np.asarray(y_np[start : start + batch_size], dtype=np.float32)).to(self.device)
                out = self.model(xb).squeeze(-1)
                loss_vec = self._loss_vector(out, yb)
                total_loss += float(loss_vec.sum().detach().cpu())
                total_count += int(loss_vec.numel())

        return total_loss / max(total_count, 1)

    def _loss_vector(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.is_binary_classification:
            return F.binary_cross_entropy_with_logits(out, y.float(), reduction="none")
        return F.mse_loss(out, y.float(), reduction="none")

    def _make_dataset(self, X_np, y_np, sample_weight=None):
        X_tensor = torch.from_numpy(np.asarray(X_np, dtype=np.float32))
        y_tensor = torch.from_numpy(np.asarray(y_np, dtype=np.float32).reshape(-1))

        if sample_weight is None:
            w_np = np.ones(len(X_np), dtype=np.float32)
        else:
            w_np = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
            if len(w_np) != len(X_np):
                raise ValueError("sample_weight length must match X_train length")
        w_tensor = torch.from_numpy(w_np)

        return TensorDataset(X_tensor, y_tensor, w_tensor)

    def _to_numpy_features(self, X):
        if X is None:
            return None, None

        if isinstance(X, pd.DataFrame):
            values = X.values.astype(np.float32, copy=False)
            cols = X.columns.tolist()
            return values, cols

        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape={values.shape}")
        cols = [f"feature_{i}" for i in range(values.shape[1])]
        return values, cols

    def _to_binary_target(self, y):
        if y is None:
            return None

        y = np.asarray(y).reshape(-1)
        valid = y[~pd.isna(y)]
        if len(valid) == 0:
            return y.astype(np.float32)

        unique_vals = np.unique(valid)
        positive_class = unique_vals[-1]
        return (y == positive_class).astype(np.float32)

    def _initial_binary_bias(self, y_binary: np.ndarray) -> float:
        p = float(np.clip(np.mean(y_binary), 1e-5, 1.0 - 1e-5))
        return float(np.log(p / (1.0 - p)))

    def _normalize_hidden_dims(self, hidden_dims: Any):
        if hidden_dims is None:
            return [128, 64]
        if isinstance(hidden_dims, int):
            return [hidden_dims]
        if isinstance(hidden_dims, str):
            parts = [p.strip() for p in hidden_dims.split(",") if p.strip()]
            return [int(p) for p in parts]
        if isinstance(hidden_dims, Iterable):
            return [int(x) for x in hidden_dims]
        raise ValueError(f"Unsupported head_hidden_dims type: {type(hidden_dims)}")

    def _create_feature_importance_df(self, feature_names):
        if self.model is None:
            return
        with torch.no_grad():
            importance = self.model.get_feature_importance(normalize=True).detach().cpu().numpy()
        self.feature_importances_ = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values(by="importance", ascending=False)

    def _log_learning_curve(self, model_idx):
        if not self.history or len(self.history.get("train_loss", [])) == 0:
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.history["train_loss"], label="train_loss")
        if len(self.history.get("valid_loss", [])) > 0:
            plt.plot(self.history["valid_loss"], label="valid_loss")
        plt.title(f"GANDALF Learning Curve (Model {model_idx})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        temp_path = f"gandalf_learning_curve_m{model_idx}.png"
        plt.savefig(temp_path)
        plt.close()

        if mlflow is not None and mlflow.active_run():
            mlflow.log_artifact(temp_path, artifact_path="plots/learning_curves")
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def _log_feature_importance(self, model_idx):
        if self.feature_importances_ is None or self.feature_importances_.empty:
            return

        importance_df = self.feature_importances_.copy()
        top_n = min(30, len(importance_df))
        plot_df = importance_df.head(top_n)

        plt.figure(figsize=(10, 8))
        plt.barh(plot_df["feature"], plot_df["importance"])
        plt.xlabel("Importance")
        plt.title(f"GANDALF Feature Importance (Model {model_idx})")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        temp_path = f"gandalf_feature_importance_m{model_idx}.png"
        plt.savefig(temp_path)
        plt.close()

        if mlflow is not None and mlflow.active_run():
            mlflow.log_artifact(temp_path, artifact_path="plots/importance")
            csv_path = f"gandalf_feature_importance_m{model_idx}.csv"
            importance_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path, artifact_path="importance_data")
            if os.path.exists(csv_path):
                os.remove(csv_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.get("model")
        if model is not None:
            state["_serialized_model_state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            state["model"] = None
        return state

    def __setstate__(self, state):
        serialized = state.pop("_serialized_model_state", None)
        self.__dict__.update(state)
        self.device = torch.device("cpu")
        if serialized is not None and self.model_init_kwargs is not None:
            self.model = GANDALFNet(**self.model_init_kwargs)
            self.model.load_state_dict(serialized)
            self.model.to(self.device)
            self.model.eval()
