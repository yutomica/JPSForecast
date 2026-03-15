import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .base import BaseModelWrapper
from .ft_transformer import FTTransformer


class _TabularDataset(Dataset):
    def __init__(self, x_num, x_cat, y=None):
        self.x_num = torch.from_numpy(x_num.astype(np.float32, copy=False))
        self.x_cat = torch.from_numpy(x_cat.astype(np.int64, copy=False))
        self.y = None if y is None else torch.from_numpy(np.asarray(y))

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x_num[idx], self.x_cat[idx]
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


class FTTransformerWrapper(BaseModelWrapper):
    """
    train.py 互換の FT-Transformer wrapper
    - fit(X_train, y_train, X_valid, y_valid, sample_weight, model_idx)
    - predict(X)
    - X は DataFrame を想定
    """

    def __init__(self, task_type="regression", **params):
        self.task_type = task_type

        # train.py から渡される可能性のある別名も吸収
        self.cat_idxs = params.pop("cat_idx", params.pop("cat_idxs", []))
        self.cat_dims = params.pop("cat_dims", params.pop("cat_dim", []))

        # モデル設定
        self.d_token = params.pop("d_token", 192)
        self.n_blocks = params.pop("n_blocks", 3)
        self.attention_n_heads = params.pop("attention_n_heads", 8)
        self.attention_dropout = params.pop("attention_dropout", 0.1)
        self.ffn_hidden_multiplier = params.pop("ffn_hidden_multiplier", 4.0)
        self.ffn_dropout = params.pop("ffn_dropout", 0.1)
        self.residual_dropout = params.pop("residual_dropout", 0.0)
        self.head_dropout = params.pop("head_dropout", 0.0)

        # 学習設定
        self.max_epochs = params.pop("max_epochs", 100)
        self.patience = params.pop("patience", 10)
        self.batch_size = params.pop("batch_size", 512)
        self.lr = params.pop("lr", 1e-3)
        self.weight_decay = params.pop("weight_decay", 1e-5)
        self.grad_clip_norm = params.pop("grad_clip_norm", 1.0)
        self.random_state = params.pop("random_state", 42)
        self.device_name = params.pop(
            "device_name",
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.device = torch.device(
            "cuda" if (self.device_name == "cuda" and torch.cuda.is_available()) else "cpu"
        )

        self.model = None
        self.history = {"train_loss": [], "valid_loss": []}
        self.feature_importances_ = None
        self.best_epoch_ = None

        # fit時に確定
        self._num_indices = None
        self._cat_indices = None
        self._feature_names = None

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

    def _split_num_cat(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self._feature_names = X.columns.tolist()

        cat_set = set(self.cat_idxs)
        self._cat_indices = list(self.cat_idxs)
        self._num_indices = [i for i in range(X.shape[1]) if i not in cat_set]

        if self._num_indices:
            x_num = X.iloc[:, self._num_indices].to_numpy(dtype=np.float32, copy=False)
        else:
            x_num = np.zeros((len(X), 0), dtype=np.float32)

        if self._cat_indices:
            x_cat = X.iloc[:, self._cat_indices].to_numpy(dtype=np.int64, copy=False)
        else:
            x_cat = np.zeros((len(X), 0), dtype=np.int64)

        return x_num, x_cat

    def _build_model(self, X):
        x_num, _ = self._split_num_cat(X)

        self.model = FTTransformer(
            n_num_features=x_num.shape[1],
            cat_cardinalities=self.cat_dims,
            d_token=self.d_token,
            n_blocks=self.n_blocks,
            attention_n_heads=self.attention_n_heads,
            attention_dropout=self.attention_dropout,
            ffn_hidden_multiplier=self.ffn_hidden_multiplier,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
            output_dim=1,
            head_dropout=self.head_dropout,
        ).to(self.device)

    def _make_loader(self, X, y=None, shuffle=False):
        x_num, x_cat = self._split_num_cat(X)
        ds = _TabularDataset(x_num, x_cat, y)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

    def _compute_loss(self, logits, y, sample_weight=None):
        if self.task_type == "classification":
            # train.py の task_type は binary classification 前提とみなす
            y = y.float().view(-1)
            loss_each = nn.functional.binary_cross_entropy_with_logits(
                logits.view(-1), y, reduction="none"
            )
        else:
            y = y.float().view(-1)
            loss_each = (logits.view(-1) - y) ** 2

        if sample_weight is not None:
            sw = sample_weight.view(-1)
            loss_each = loss_each * sw

        return loss_each.mean()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0):
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        if X_valid is not None and not isinstance(X_valid, pd.DataFrame):
            X_valid = pd.DataFrame(X_valid)

        self._build_model(X_train)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        valid_loader = self._make_loader(X_valid, y_valid, shuffle=False) if X_valid is not None else None

        sample_weight_tensor = None
        if sample_weight is not None:
            sample_weight_tensor = torch.from_numpy(
                np.asarray(sample_weight, dtype=np.float32)
            )

        best_state = copy.deepcopy(self.model.state_dict())
        best_val_loss = float("inf")
        wait = 0

        for epoch in range(self.max_epochs):
            # ---- train ----
            self.model.train()
            train_total = 0.0
            train_count = 0

            for batch_idx, batch in enumerate(train_loader):
                x_num, x_cat, y = batch
                x_num = x_num.to(self.device, non_blocking=True)
                x_cat = x_cat.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if sample_weight_tensor is not None:
                    start = batch_idx * self.batch_size
                    end = start + x_num.shape[0]
                    sw = sample_weight_tensor[start:end].to(self.device, non_blocking=True)
                else:
                    sw = None

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(x_num, x_cat)
                loss = self._compute_loss(logits, y, sw)
                loss.backward()

                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                optimizer.step()

                train_total += loss.item() * x_num.shape[0]
                train_count += x_num.shape[0]

            train_loss = train_total / max(train_count, 1)
            self.history["train_loss"].append(train_loss)

            # ---- valid ----
            if valid_loader is not None:
                self.model.eval()
                valid_total = 0.0
                valid_count = 0

                with torch.no_grad():
                    for x_num, x_cat, y in valid_loader:
                        x_num = x_num.to(self.device, non_blocking=True)
                        x_cat = x_cat.to(self.device, non_blocking=True)
                        y = y.to(self.device, non_blocking=True)

                        logits = self.model(x_num, x_cat)
                        loss = self._compute_loss(logits, y, sample_weight=None)

                        valid_total += loss.item() * x_num.shape[0]
                        valid_count += x_num.shape[0]

                valid_loss = valid_total / max(valid_count, 1)
            else:
                valid_loss = train_loss

            self.history["valid_loss"].append(valid_loss)

            # ---- early stopping ----
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch_ = epoch
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()

        self._log_learning_curve(model_idx)
        self._create_feature_importance_df()

    def _log_learning_curve(self, model_idx):
        if not self.history["train_loss"]:
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.history["train_loss"], label="train_loss")
        if self.history["valid_loss"]:
            plt.plot(self.history["valid_loss"], label="valid_loss")
        plt.title(f"FT-Transformer Learning Curve (Model {model_idx})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        temp_path = f"ft_transformer_learning_curve_m{model_idx}.png"
        plt.savefig(temp_path)
        plt.close()

        if mlflow.active_run():
            mlflow.log_artifact(temp_path, artifact_path="plots/learning_curves")

        os.remove(temp_path)

    def _create_feature_importance_df(self):
        """
        真のSHAPではなく、tokenizer重みノルムベースの簡易 proxy。
        参考指標としてのみ使用。
        """
        if self.model is None or self._feature_names is None:
            return

        importances = np.zeros(len(self._feature_names), dtype=np.float32)

        # 数値特徴量: tokenizer の列別重みノルム
        if self._num_indices:
            num_weight = self.model.tokenizer.num_tokenizer.weight.detach().cpu().numpy()
            num_imp = np.linalg.norm(num_weight, axis=1)
            for local_idx, col_idx in enumerate(self._num_indices):
                importances[col_idx] = num_imp[local_idx]

        # カテゴリ特徴量: bias ノルムを proxy として使用
        if self._cat_indices and len(self.cat_dims) > 0:
            cat_bias = self.model.tokenizer.cat_tokenizer.bias.detach().cpu().numpy()
            cat_imp = np.linalg.norm(cat_bias, axis=1)
            for local_idx, col_idx in enumerate(self._cat_indices):
                if local_idx < len(cat_imp):
                    importances[col_idx] = cat_imp[local_idx]

        self.feature_importances_ = pd.DataFrame({
            "feature": self._feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        loader = self._make_loader(X, y=None, shuffle=False)

        outputs = []
        self.model.eval()

        with torch.no_grad():
            for x_num, x_cat in loader:
                x_num = x_num.to(self.device, non_blocking=True)
                x_cat = x_cat.to(self.device, non_blocking=True)

                logits = self.model(x_num, x_cat)

                if self.task_type == "classification":
                    preds = torch.sigmoid(logits.view(-1))
                else:
                    preds = logits.view(-1)

                outputs.append(preds.detach().cpu().numpy())

        return np.concatenate(outputs, axis=0).flatten()