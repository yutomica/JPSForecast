
import copy
import os
import gc
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base import BaseModelWrapper
from .networks.ft_transformer import FTTransformer
from .pruning import execute_epoch_pruning, log_epoch_metrics


class FTTransformerWrapper(BaseModelWrapper):
    """
    train.py 互換 FT-Transformer wrapper

    必須メソッド:
      - fit(X_train, y_train, X_valid, y_valid, sample_weight, model_idx)
      - predict(X)

    期待入力:
      - X: np.ndarray [N, F]
      - cat_idx / cat_dims は preprocessor から train.py 経由で注入
    """

    def __init__(self, task_type: str = "regression", **params):
        self.task_type = task_type
        self.params = copy.deepcopy(params)

        # feature metadata from preprocessor
        self.cat_idx = sorted([int(x) for x in params.pop("cat_idx", [])])
        self.cat_dims = [int(x) for x in params.pop("cat_dims", [])]

        # model params
        self.d_token = int(params.pop("d_token", 192))
        self.n_blocks = int(params.pop("n_blocks", 3))
        self.attention_n_heads = int(params.pop("attention_n_heads", 8))
        self.attention_dropout = float(params.pop("attention_dropout", 0.2))
        self.ffn_d_hidden = params.pop("ffn_d_hidden", None)
        self.ffn_dropout = float(params.pop("ffn_dropout", 0.1))
        self.residual_dropout = float(params.pop("residual_dropout", 0.0))
        self.activation = params.pop("activation", "gelu")
        self.head_hidden_dim = int(params.pop("head_hidden_dim", 0))
        self.head_dropout = float(params.pop("head_dropout", 0.0))

        # training params
        self.max_epochs = int(params.pop("max_epochs", 100))
        self.patience = int(params.pop("patience", 10))
        self.batch_size = int(params.pop("batch_size", 1024))
        self.lr = float(params.pop("lr", params.pop("learning_rate", 1e-3)))
        self.weight_decay = float(params.pop("weight_decay", 1e-5))
        self.grad_clip_norm = params.pop("grad_clip_norm", params.pop("gradient_clip_val", 1.0))
        self.num_workers = int(params.pop("num_workers", 0))
        self.random_state = int(params.pop("random_state", 42))
        self.device_name = params.pop("device_name", "auto")
        self.optimizer_name = params.pop("optimizer", "adamw")
        self.use_compile = bool(params.pop("use_compile", False))
        self.compile_mode = params.pop("compile_mode", "reduce-overhead")
        self.use_tf32 = bool(params.pop("use_tf32", True))

        if self.device_name == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.device_name)

        if self.device.type == "mps" and self.num_workers > 0:
            print("  ⚠️ num_workers>0 is unstable on MPS in many environments. Forcing num_workers=0.")
            self.num_workers = 0

        if hasattr(torch, "set_float32_matmul_precision"):
            # CUDA/CPU 向け最適化だが害は小さいため共通で有効化
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        if self.device.type != "cuda":
            self.use_tf32 = False

        self.model = None
        self.history = {"train_loss": [], "valid_loss": []}
        self.feature_importances_ = None
        self.best_epoch_ = None
        self.n_features_ = None
        self.num_idx_ = None
        self.n_num_features_ = None

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

    def _split_feature_indices(self, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
        cat_idx = np.array(self.cat_idx, dtype=np.int64)
        if len(cat_idx) == 0:
            num_idx = np.arange(n_features, dtype=np.int64)
            return num_idx, cat_idx

        if np.any(cat_idx < 0) or np.any(cat_idx >= n_features):
            raise ValueError(f"cat_idx contains out-of-range indices: cat_idx={self.cat_idx}, n_features={n_features}")

        mask = np.ones(n_features, dtype=bool)
        mask[cat_idx] = False
        num_idx = np.arange(n_features, dtype=np.int64)[mask]
        return num_idx, cat_idx

    def _build_model(self, X: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"FTTransformerWrapper expects 2D array [N, F], got {X.shape}")

        self.n_features_ = int(X.shape[1])
        self.num_idx_, cat_idx = self._split_feature_indices(self.n_features_)
        self.n_num_features_ = int(len(self.num_idx_))

        if len(cat_idx) != len(self.cat_dims):
            raise ValueError(
                f"Mismatch between cat_idx and cat_dims: len(cat_idx)={len(cat_idx)}, len(cat_dims)={len(self.cat_dims)}"
            )

        self.model = FTTransformer(
            n_num_features=self.n_num_features_,
            cat_cardinalities=self.cat_dims,
            d_token=self.d_token,
            n_blocks=self.n_blocks,
            attention_n_heads=self.attention_n_heads,
            attention_dropout=self.attention_dropout,
            ffn_d_hidden=self.ffn_d_hidden,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
            activation=self.activation,
            output_dim=1,
            head_hidden_dim=self.head_hidden_dim,
            head_dropout=self.head_dropout,
        ).to(self.device)

        if self.use_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode=self.compile_mode)
                print(f"  🔹 torch.compile enabled (mode={self.compile_mode})")
            except Exception as e:
                print(f"  ⚠️ torch.compile failed; continuing in eager mode. reason={e}")

    def _split_arrays(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        x_num = X[:, self.num_idx_].astype(np.float32, copy=False) if len(self.num_idx_) > 0 else None
        x_cat = X[:, self.cat_idx].astype(np.int64, copy=False) if len(self.cat_idx) > 0 else None
        return x_num, x_cat

    def _compute_loss(self, logits, y, sample_weight=None):
        loss_type = self.params.get("objective", "mse")

        if self.task_type == "classification":
            y = y.float().view(-1)
            loss_each = nn.functional.binary_cross_entropy_with_logits(
                logits.view(-1), y, reduction="none"
            )
        elif loss_type == "quantile":
            y = y.float().view(-1)
            alpha = float(self.params.get("alpha", 0.5))
            diff = y - logits.view(-1)
            loss_each = torch.maximum(alpha * diff, (alpha - 1.0) * diff)
        elif loss_type in ["fair", "fair_loss"]:
            y = y.float().view(-1)
            c = float(self.params.get("fair_c", 1.0))
            abs_diff = torch.abs(y - logits.view(-1))
            loss_each = c * (abs_diff - c * torch.log1p(abs_diff / c))
        elif loss_type == "tweedie":
            y = torch.clamp(y.float().view(-1), min=0.0)
            p = float(self.params.get("tweedie_variance_power", 1.5))
            mu = torch.clamp(logits.view(-1), min=1e-6)
            loss_each = (mu ** (2.0 - p)) / (2.0 - p) - y * (mu ** (1.0 - p)) / (1.0 - p)
        elif loss_type == "asymmetric_mse":
            y = y.float().view(-1)
            alpha = float(self.params.get("alpha", 3.0))
            beta = float(self.params.get("beta", 1.0))
            diff = y - logits.view(-1)
            mask = (diff > 0).float()
            loss_each = (mask * alpha + (1.0 - mask) * beta) * (diff ** 2)
        elif loss_type in ["huber", "smooth_l1"]:
            y = y.float().view(-1)
            beta = float(self.params.get("huber_beta", 1.0))
            loss_each = nn.functional.smooth_l1_loss(logits.view(-1), y, reduction="none", beta=beta)
        else:
            y = y.float().view(-1)
            loss_each = (logits.view(-1) - y) ** 2

        if sample_weight is not None:
            sw = sample_weight.view(-1)
            return (loss_each * sw).mean()
        return loss_each.mean()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0, epoch_callback=None):
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train_np = np.asarray(y_train)

        if X_train.ndim != 2:
            raise ValueError(f"X_train must be 2D [N, F], got {X_train.shape}")

        self._build_model(X_train)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n--- FT-Transformer Model Summary (Fold {model_idx}) ---")
        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Input Shape:          [N, {self.n_features_}]")
        print(f"Num Features:         {self.n_num_features_}")
        print(f"Cat Features:         {len(self.cat_idx)}")
        print(f"Cat Idx:              {self.cat_idx}")
        print(f"Cat Cardinalities:    {self.cat_dims}")
        print("-" * 40)
        
        # --- Validation Loader の準備とメモリ解放 ---
        valid_loader = None
        if X_valid is not None and y_valid is not None:
            X_valid = np.asarray(X_valid, dtype=np.float32)
            y_valid_np = np.asarray(y_valid)
            valid_mask = ~np.isnan(y_valid_np) & ~np.isinf(y_valid_np)
            valid_mask &= np.isfinite(X_valid).all(axis=1)

            dropped_valid = len(y_valid_np) - int(np.sum(valid_mask))
            if dropped_valid > 0:
                print(f"  ⚠️ Dropped {dropped_valid:,} validation samples due to NaN/Inf.")

            X_valid_filtered = X_valid[valid_mask]
            y_valid_filtered = y_valid_np[valid_mask]

            x_num_np, x_cat_np = self._split_arrays(X_valid_filtered)
            x_num_t = torch.from_numpy(np.ascontiguousarray(x_num_np)) if x_num_np is not None else torch.zeros((len(X_valid_filtered), 0), dtype=torch.float32)
            x_cat_t = torch.from_numpy(np.ascontiguousarray(x_cat_np)) if x_cat_np is not None else torch.zeros((len(X_valid_filtered), 0), dtype=torch.int64)

            ds = TensorDataset(
                x_num_t, x_cat_t,
                torch.from_numpy(y_valid_filtered.astype(np.float32)),
                torch.ones(len(X_valid_filtered), dtype=torch.float32)
            )
            valid_loader = DataLoader(
                ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                pin_memory=(self.device.type == "cuda"), drop_last=False,
            )
            del X_valid, y_valid, y_valid_np, valid_mask, X_valid_filtered, y_valid_filtered, x_num_np, x_cat_np, ds
            gc.collect()
            
        # --- Training Loader の準備とメモリ解放 ---
        train_mask = ~np.isnan(y_train_np) & ~np.isinf(y_train_np)
        train_mask &= np.isfinite(X_train).all(axis=1)

        if sample_weight is not None:
            sample_weight = np.nan_to_num(sample_weight, nan=0.0, posinf=1.0, neginf=0.0)
            sample_weight = np.clip(sample_weight, 0.0, None)
            train_mask &= (sample_weight > 0)

        dropped_train = len(y_train_np) - int(np.sum(train_mask))
        if dropped_train > 0:
            print(f"  ⚠️ Dropped {dropped_train:,} training samples due to NaN/Inf or zero weights.")

        X_train_filtered = X_train[train_mask]
        y_train_filtered = y_train_np[train_mask]

        if sample_weight is not None:
            sample_weight_filtered = sample_weight[train_mask]
            if len(sample_weight_filtered) > 0:
                p99 = np.percentile(sample_weight_filtered, 99)
                sample_weight_filtered = np.clip(sample_weight_filtered, 0.0, max(p99 * 10.0, 1.0))
                if sample_weight_filtered.mean() > 0:
                    sample_weight_filtered = sample_weight_filtered / sample_weight_filtered.mean()
            w_t = torch.from_numpy(sample_weight_filtered.astype(np.float32))
        else:
            w_t = torch.ones(len(X_train_filtered), dtype=torch.float32)
            
        x_num_np, x_cat_np = self._split_arrays(X_train_filtered)
        x_num_t = torch.from_numpy(np.ascontiguousarray(x_num_np)) if x_num_np is not None else torch.zeros((len(X_train_filtered), 0), dtype=torch.float32)
        x_cat_t = torch.from_numpy(np.ascontiguousarray(x_cat_np)) if x_cat_np is not None else torch.zeros((len(X_train_filtered), 0), dtype=torch.int64)

        ds = TensorDataset(x_num_t, x_cat_t, torch.from_numpy(y_train_filtered.astype(np.float32)), w_t)
        train_loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"), drop_last=False,
        )
        del X_train, y_train, y_train_np, train_mask, sample_weight, X_train_filtered, y_train_filtered, x_num_np, x_cat_np, ds, w_t
        if 'sample_weight_filtered' in locals(): del sample_weight_filtered
        gc.collect()

        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_state = copy.deepcopy(self.model.state_dict())
        best_val_loss = float("inf")
        wait = 0

        amp_device = "cuda" if self.device.type == "cuda" else "mps" if self.device.type == "mps" else "cpu"
        amp_enabled = (self.device.type in ["cuda", "mps"])
        # GradScalerは主にCUDA専用のため、MPSの場合は無効化して通常のbackwardを使用します
        use_scaler = (self.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=True) if use_scaler and hasattr(torch, "amp") else None

        loss_name = self.params.get("objective", "mse") if self.task_type != "classification" else "bce"

        for epoch in range(self.max_epochs):
            self.model.train()
            train_total = 0.0
            train_count = 0

            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}", leave=False) as pbar:
                for x_num, x_cat, y, sw in pbar:
                    x_num = x_num.to(self.device, non_blocking=(self.device.type == "cuda"))
                    x_cat = x_cat.to(self.device, non_blocking=(self.device.type == "cuda"))
                    y = y.to(self.device, non_blocking=(self.device.type == "cuda"))
                    sw = sw.to(self.device, non_blocking=(self.device.type == "cuda"))

                    optimizer.zero_grad(set_to_none=True)

                    if amp_enabled:
                        with torch.amp.autocast(device_type=amp_device, enabled=True):
                            logits = self.model(x_num if x_num.shape[1] > 0 else None, x_cat if x_cat.shape[1] > 0 else None)
                            loss = self._compute_loss(logits, y, sw)
                        
                        if use_scaler and scaler is not None:
                            scaler.scale(loss).backward()
                            if self.grad_clip_norm is not None:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            if self.grad_clip_norm is not None:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                            optimizer.step()
                    else:
                        logits = self.model(x_num if x_num.shape[1] > 0 else None, x_cat if x_cat.shape[1] > 0 else None)
                        loss = self._compute_loss(logits, y, sw)
                        loss.backward()
                        if self.grad_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                        optimizer.step()

                    train_total += loss.item() * y.shape[0]
                    train_count += y.shape[0]
                    pbar.set_postfix({f"train_{loss_name}": f"{train_total / max(train_count, 1):.6f}"})

            train_loss = train_total / max(train_count, 1)
            self.history["train_loss"].append(train_loss)

            if valid_loader is not None:
                self.model.eval()
                valid_total = 0.0
                valid_count = 0
                with torch.no_grad():
                    for x_num, x_cat, y, _ in valid_loader:
                        x_num = x_num.to(self.device, non_blocking=(self.device.type == "cuda"))
                        x_cat = x_cat.to(self.device, non_blocking=(self.device.type == "cuda"))
                        y = y.to(self.device, non_blocking=(self.device.type == "cuda"))
                        logits = self.model(x_num if x_num.shape[1] > 0 else None, x_cat if x_cat.shape[1] > 0 else None)
                        loss = self._compute_loss(logits, y, sample_weight=None)
                        valid_total += loss.item() * y.shape[0]
                        valid_count += y.shape[0]
                valid_loss = valid_total / max(valid_count, 1)
            else:
                valid_loss = train_loss

            self.history["valid_loss"].append(valid_loss)

            tqdm.write(
                f"Epoch {epoch+1}/{self.max_epochs} | Train {loss_name}: {train_loss:.6f} | Valid {loss_name}: {valid_loss:.6f}"
            )

            metrics_to_log = {"train_loss": train_loss}
            if valid_loader is not None:
                metrics_to_log["valid_loss"] = valid_loss
            log_epoch_metrics(model_idx, epoch, metrics_to_log)

            if epoch_callback is not None and X_valid is not None:
                valid_preds = self.predict(X_valid)
                execute_epoch_pruning(epoch_callback, epoch, valid_preds, y_valid)

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

        if self.device.type == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
                
        del train_loader, valid_loader, best_state, optimizer
        if 'scaler' in locals(): del scaler
        gc.collect()

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
        if self.model is None:
            return

        out = np.zeros(self.n_features_, dtype=np.float32)

        if self.model.tokenizer.num_tokenizer.weight is not None and len(self.num_idx_) > 0:
            num_imp = self.model.tokenizer.num_tokenizer.weight.detach().abs().mean(dim=1).cpu().numpy()
            out[self.num_idx_] = num_imp.astype(np.float32)

        if self.model.tokenizer.cat_tokenizer.embedding is not None and len(self.cat_idx) > 0:
            cat_imp = []
            start = 0
            emb = self.model.tokenizer.cat_tokenizer.embedding.weight.detach().abs().mean(dim=1).cpu().numpy()
            for size in self.cat_dims:
                cat_imp.append(float(emb[start:start + size].mean()))
                start += size
            out[np.asarray(self.cat_idx, dtype=np.int64)] = np.asarray(cat_imp, dtype=np.float32)

        self.feature_importances_ = out

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D [N, F], got {X.shape}")

        x_num_np, x_cat_np = self._split_arrays(X)
        x_num_t = torch.from_numpy(np.ascontiguousarray(x_num_np)) if x_num_np is not None else torch.zeros((X.shape[0], 0), dtype=torch.float32)
        x_cat_t = torch.from_numpy(np.ascontiguousarray(x_cat_np)) if x_cat_np is not None else torch.zeros((X.shape[0], 0), dtype=torch.int64)
        
        ds = TensorDataset(
            x_num_t, x_cat_t,
            torch.zeros(X.shape[0], dtype=torch.float32),
            torch.ones(X.shape[0], dtype=torch.float32)
        )
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"), drop_last=False,
        )

        outputs = []
        self.model.eval()

        with torch.no_grad():
            for x_num, x_cat, _, _ in tqdm(loader, desc="Predicting", leave=False):
                x_num = x_num.to(self.device, non_blocking=(self.device.type == "cuda"))
                x_cat = x_cat.to(self.device, non_blocking=(self.device.type == "cuda"))
                logits = self.model(x_num if x_num.shape[1] > 0 else None, x_cat if x_cat.shape[1] > 0 else None)
                if self.task_type == "classification":
                    preds = torch.sigmoid(logits.view(-1))
                else:
                    preds = logits.view(-1)
                outputs.append(preds.detach().cpu().numpy())

        return np.concatenate(outputs, axis=0).flatten()

    def __getstate__(self):
        state = self.__dict__.copy()
        if "model" in state and state["model"] is not None:
            state["_model_state_dict"] = {k: v.cpu() for k, v in state["model"].state_dict().items()}
            del state["model"]
        return state

    def __setstate__(self, state):
        model_state = state.pop("_model_state_dict", None)
        self.__dict__.update(state)

        if model_state is not None:
            if self.n_features_ is None:
                raise ValueError("Cannot restore FTTransformerWrapper because n_features_ is missing.")
            self.model = FTTransformer(
                n_num_features=self.n_num_features_,
                cat_cardinalities=self.cat_dims,
                d_token=self.d_token,
                n_blocks=self.n_blocks,
                attention_n_heads=self.attention_n_heads,
                attention_dropout=self.attention_dropout,
                ffn_d_hidden=self.ffn_d_hidden,
                ffn_dropout=self.ffn_dropout,
                residual_dropout=self.residual_dropout,
                activation=self.activation,
                output_dim=1,
                head_hidden_dim=self.head_hidden_dim,
                head_dropout=self.head_dropout,
            )
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()
