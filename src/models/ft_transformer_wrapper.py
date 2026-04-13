import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .base import BaseModelWrapper
from .pruning import execute_epoch_pruning, log_epoch_metrics
from .networks.ft_transformer import FTTransformer

class FTTransformerWrapper(BaseModelWrapper):
    """
    train.py 互換の FT-Transformer wrapper
    - fit(X_train, y_train, X_valid, y_valid, sample_weight, model_idx)
    - predict(X)
    - X は DataFrame を想定
    """

    def __init__(self, task_type="regression", **params):
        self.task_type = task_type

        # paramsがpopされて消える前に、ハイパーパラメータ全体を保持
        self.params = copy.deepcopy(params)
        # train.py から渡される可能性のある別名も吸収
        self.cat_idxs = params.pop("cat_idx", params.pop("cat_idxs", []))
        self.cat_dims = params.pop("cat_dims", params.pop("cat_dim", []))

        # モデル設定
        self.d_token = params.pop("d_token", 64)
        self.n_blocks = params.pop("n_blocks", params.pop("n_layers", 2))
        self.attention_n_heads = params.pop("attention_n_heads", 4)
        self.attention_dropout = params.pop("attention_dropout", 0.1)
        self.ffn_hidden_multiplier = params.pop("ffn_hidden_multiplier", 2.0)
        self.ffn_dropout = params.pop("ffn_dropout", 0.1)
        self.residual_dropout = params.pop("residual_dropout", 0.0)
        self.head_dropout = params.pop("head_dropout", 0.0)

        # 学習設定
        self.max_epochs = params.pop("max_epochs", 100)
        self.patience = params.pop("patience", 10)
        self.batch_size = params.pop("batch_size", 512)
        self.lr = params.pop("lr", params.pop("learning_rate", 1e-3))
        self.weight_decay = params.pop("weight_decay", 1e-5)
        self.grad_clip_norm = params.pop("grad_clip_norm", params.pop("gradient_clip_val", 1.0))
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

    def _make_loader(self, X, y=None, sample_weight=None, shuffle=False):
        x_num, x_cat = self._split_num_cat(X)
        
        x_num_t = torch.from_numpy(x_num.astype(np.float32, copy=False))
        x_cat_t = torch.from_numpy(x_cat.astype(np.int64, copy=False))
        
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)) if y is not None else torch.zeros(len(x_num), dtype=torch.float32)
        
        if sample_weight is not None:
            w_t = torch.from_numpy(np.asarray(sample_weight, dtype=np.float32))
        else:
            w_t = torch.ones(len(x_num), dtype=torch.float32)
            
        ds = TensorDataset(x_num_t, x_cat_t, y_t, w_t)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

    def _compute_loss(self, logits, y, sample_weight=None):
        loss_type = self.params.get("objective", "mse")

        if self.task_type == "classification":
            # train.py の task_type は binary classification 前提とみなす
            y = y.float().view(-1)
            loss_each = nn.functional.binary_cross_entropy_with_logits(
                logits.view(-1), y, reduction="none"
            )
        elif loss_type == "quantile":
            y = y.float().view(-1)
            alpha = self.params.get("alpha", 0.5)
            diff = y - logits.view(-1)
            loss_each = torch.max(alpha * diff, (alpha - 1) * diff)
        elif loss_type in ["fair", "fair_loss"]:
            y = y.float().view(-1)
            c = self.params.get("fair_c", 1.0)
            abs_diff = torch.abs(y - logits.view(-1))
            loss_each = c * (abs_diff - c * torch.log1p(abs_diff / c))
        elif loss_type == "tweedie":
            y = torch.max(y.float().view(-1), torch.zeros_like(y.float().view(-1)))
            p = self.params.get("tweedie_variance_power", 1.5)
            mu = torch.clamp(logits.view(-1), min=1e-6)
            loss_each = (mu ** (2 - p)) / (2 - p) - y * (mu ** (1 - p)) / (1 - p)
        elif loss_type == "asymmetric_mse":
            y = y.float().view(-1)
            alpha = self.params.get("alpha", 3.0)
            beta = self.params.get("beta", 1.0)
            diff = y - logits.view(-1)
            loss_each = torch.where(diff > 0, alpha * (diff ** 2), beta * (diff ** 2))
        else:
            y = y.float().view(-1)
            loss_each = (logits.view(-1) - y) ** 2

        if sample_weight is not None:
            sw = sample_weight.view(-1)
            loss_each = loss_each * sw

        return loss_each.mean()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0, epoch_callback=None):
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        if X_valid is not None and not isinstance(X_valid, pd.DataFrame):
            X_valid = pd.DataFrame(X_valid)

        y_train_np = np.asarray(y_train)

        # --- データクレンジング (NaN / Inf の確実な除去とウェイトの正値化) ---
        train_mask = ~np.isnan(y_train_np) & ~np.isinf(y_train_np)
        train_mask &= ~X_train.isna().any(axis=1).values
        X_train_num = X_train.select_dtypes(include=[np.number]).to_numpy(copy=False)
        if X_train_num.shape[1] > 0:
            train_mask &= ~np.isinf(X_train_num).any(axis=1)

        if sample_weight is not None:
            sample_weight = np.nan_to_num(sample_weight, nan=0.0, posinf=1.0, neginf=0.0)
            sample_weight = np.clip(sample_weight, 0.0, None)
            train_mask &= (sample_weight > 0)

        dropped_train = len(y_train_np) - np.sum(train_mask)
        if dropped_train > 0:
            print(f"  ⚠️ Dropped {dropped_train:,} training samples due to NaN/Inf or zero weights.")

        X_train = X_train.iloc[train_mask]
        y_train = y_train_np[train_mask]
        if sample_weight is not None:
            sample_weight = sample_weight[train_mask]

        if X_valid is not None and y_valid is not None:
            y_valid_np = np.asarray(y_valid)
            valid_mask = ~np.isnan(y_valid_np) & ~np.isinf(y_valid_np)
            valid_mask &= ~X_valid.isna().any(axis=1).values
            X_valid_num = X_valid.select_dtypes(include=[np.number]).to_numpy(copy=False)
            if X_valid_num.shape[1] > 0:
                valid_mask &= ~np.isinf(X_valid_num).any(axis=1)
            
            dropped_valid = len(y_valid_np) - np.sum(valid_mask)
            if dropped_valid > 0:
                print(f"  ⚠️ Dropped {dropped_valid:,} validation samples due to NaN/Inf.")
            
            X_valid = X_valid.iloc[valid_mask]
            y_valid = y_valid_np[valid_mask]

        self._build_model(X_train)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n--- FT-Transformer Model Summary (Fold {model_idx}) ---")
        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("-" * 40)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        train_loader = self._make_loader(X_train, y_train, sample_weight=sample_weight, shuffle=True)
        valid_loader = self._make_loader(X_valid, y_valid, sample_weight=None, shuffle=False) if X_valid is not None else None

        best_state = copy.deepcopy(self.model.state_dict())
        best_val_loss = float("inf")
        wait = 0
        
        loss_name = self.params.get("objective", "mse") if self.task_type != "classification" else "bce"

        for epoch in range(self.max_epochs):
            # ---- train ----
            self.model.train()
            train_total = 0.0
            train_count = 0

            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}", leave=False) as pbar:
                for x_num, x_cat, y, sw in pbar:
                    x_num = x_num.to(self.device, non_blocking=True)
                    x_cat = x_cat.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    sw = sw.to(self.device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    logits = self.model(x_num, x_cat)
                    loss = self._compute_loss(logits, y, sw)
                    loss.backward()

                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                    optimizer.step()

                    train_total += loss.item() * x_num.shape[0]
                    train_count += x_num.shape[0]
                    pbar.set_postfix({f"train_{loss_name}": f"{train_total / max(train_count, 1):.6f}"})

            train_loss = train_total / max(train_count, 1)
            self.history["train_loss"].append(train_loss)

            # ---- valid ----
            if valid_loader is not None:
                self.model.eval()
                valid_total = 0.0
                valid_count = 0

                with torch.no_grad():
                    for x_num, x_cat, y, sw in valid_loader:
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

            tqdm.write(f"Epoch {epoch+1}/{self.max_epochs} | Train {loss_name}: {train_loss:.6f} | Valid {loss_name}: {valid_loss:.6f}")

            # --- MLflow Logging ---
            metrics_to_log = {"train_loss": train_loss}
            if valid_loader is not None:
                metrics_to_log["valid_loss"] = valid_loss
            log_epoch_metrics(model_idx, epoch, metrics_to_log)

            # --- Epoch Callback (Pruning等) の実行 ---
            if epoch_callback is not None and X_valid is not None:
                valid_preds = self.predict(X_valid)
                execute_epoch_pruning(epoch_callback, epoch, valid_preds, y_valid)

            # ---- early stopping ----
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch_ = epoch
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    tqdm.write(f"Early stopping triggered at epoch {epoch + 1}")
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
            for x_num, x_cat, _, _ in loader:
                x_num = x_num.to(self.device, non_blocking=True)
                x_cat = x_cat.to(self.device, non_blocking=True)

                logits = self.model(x_num, x_cat)

                if self.task_type == "classification":
                    preds = torch.sigmoid(logits.view(-1))
                else:
                    preds = logits.view(-1)

                outputs.append(preds.detach().cpu().numpy())

        return np.concatenate(outputs, axis=0).flatten()

    def __getstate__(self):
        """joblib/pickleで保存する際に呼ばれる"""
        state = self.__dict__.copy()
        # シリアライズできないモデルオブジェクトを削除し、代わりにstate_dictを保存
        if "model" in state and state["model"] is not None:
            state["_model_state_dict"] = {k: v.cpu() for k, v in state["model"].state_dict().items()}
            del state["model"]
        return state

    def __setstate__(self, state):
        """joblib/pickleで読み込む際に呼ばれる"""
        model_state = state.pop("_model_state_dict", None)
        self.__dict__.update(state)
        
        # モデルの再構築
        if model_state is not None:
            # _build_modelを呼び出すためにダミーのDataFrameを作成
            # 特徴量名とカテゴリカルインデックスが復元されていることが前提
            dummy_df = pd.DataFrame(
                np.zeros((1, len(self._feature_names))), 
                columns=self._feature_names
            )
            self._build_model(dummy_df)
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()
