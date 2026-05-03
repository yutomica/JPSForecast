import copy
import os
import random
from collections.abc import Iterable
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import zarr
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import mlflow
from .base import BaseModelWrapper
from .pruning import execute_epoch_pruning, log_epoch_metrics
from .networks.gandalf import GANDALFNet


class ZarrBatchDataset(torch.utils.data.Dataset):
    def __init__(self, zarr_path, y, w, valid_indices, batch_size, shuffle=True):
        self.zarr_path = zarr_path
        self.y = y
        self.w = w
        self.valid_indices = valid_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(valid_indices)
        self.n_batches = (self.n_samples + self.batch_size - 1) // self.batch_size
        self.batch_indices = np.arange(self.n_batches)
        if self.shuffle:
            np.random.shuffle(self.batch_indices)
        self._z = None

    def _get_zarr(self):
        if self._z is None:
            try:
                import numcodecs
                numcodecs.blosc.set_nthreads(1)
                numcodecs.blosc.use_threads = False
            except ImportError:
                pass
            self._z = zarr.open(self.zarr_path, mode='r')
        return self._z

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        z = self._get_zarr()
        batch_idx = self.batch_indices[idx]
        start_logical = batch_idx * self.batch_size
        end_logical = min(start_logical + self.batch_size, self.n_samples)
        logical_batch = np.arange(start_logical, end_logical)
        physical_batch = self.valid_indices[logical_batch]

        start_idx = int(physical_batch[0])
        end_idx = int(physical_batch[-1]) + 1
        
        chunk = z[start_idx:end_idx, :]
        local_indices = physical_batch - start_idx
        X_batch = chunk[local_indices]

        y_batch = self.y[logical_batch]
        w_batch = self.w[logical_batch] if self.w is not None else np.ones(len(y_batch), dtype=np.float32)
        return torch.from_numpy(X_batch), torch.from_numpy(y_batch), torch.from_numpy(w_batch)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.batch_indices)

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
        device_name = self.params.pop("device_name", "auto")
        if device_name == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_name)
        self.history: Dict[str, list] = {"train_loss": [], "valid_loss": []}
        self.feature_importances_ = None
        self.feature_names_ = None
        self.is_binary_classification = task_type == "classification"
        self.early_stopping_metric = params.pop("early_stopping_metric", "loss")
        self.metric_direction = params.pop("metric_direction", "minimize")
        self.early_stopping_ema_alpha = float(params.pop("early_stopping_ema_alpha", 1.0))

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0, epoch_callback=None, train_dates=None, valid_dates=None):
        
        y_train_np = np.asarray(y_train)
        is_zarr_train = isinstance(X_train, str) and X_train.endswith('.zarr')
        
        train_mask = ~np.isnan(y_train_np) & ~np.isinf(y_train_np)
        if not is_zarr_train:
            X_train_np, train_feature_names = self._to_numpy_features(X_train)
            train_mask &= ~np.isnan(X_train_np).any(axis=1) & ~np.isinf(X_train_np).any(axis=1)
        else:
            input_dim = zarr.open(X_train, mode='r').shape[1]
            train_feature_names = [f"feature_{i}" for i in range(input_dim)]
            
        if sample_weight is not None:
            sample_weight = np.nan_to_num(sample_weight, nan=0.0, posinf=1.0, neginf=0.0)
            sample_weight = np.clip(sample_weight, 0.0, None)
            train_mask &= (sample_weight > 0)
        dropped_train = len(y_train_np) - np.sum(train_mask)
        if dropped_train > 0:
            print(f"  ⚠️ Dropped {dropped_train:,} training samples due to NaN/Inf or zero weights.")
            
        y_valid_np = np.asarray(y_valid) if y_valid is not None else None
        is_zarr_valid = isinstance(X_valid, str) and X_valid.endswith('.zarr')
        
        if X_valid is not None and y_valid_np is not None:
            valid_mask = ~np.isnan(y_valid_np) & ~np.isinf(y_valid_np)
            if not is_zarr_valid:
                X_valid_np, _ = self._to_numpy_features(X_valid)
                valid_mask &= ~np.isnan(X_valid_np).any(axis=1) & ~np.isinf(X_valid_np).any(axis=1)
                
            dropped_valid = len(y_valid_np) - int(np.sum(valid_mask))
            if dropped_valid > 0:
                print(f"  ⚠️ Dropped {dropped_valid:,} validation samples due to NaN/Inf.")

        self.feature_names_ = train_feature_names
        self._set_seed(int(self.params.get("random_state", 42)))

        batch_size = int(self.params.get("batch_size", 1024))
        num_workers = int(self.params.get("num_workers", 0))
        
        if is_zarr_train:
            w_np = np.asarray(sample_weight, dtype=np.float32) if sample_weight is not None else None
            train_dataset = ZarrBatchDataset(X_train, y_train_np, w_np, np.where(train_mask)[0], batch_size)
        else:
            train_dataset = self._make_dataset(X_train_np[train_mask], y_train_np[train_mask], sample_weight[train_mask] if sample_weight is not None else None)
            
        train_loader = DataLoader(train_dataset, batch_size=None if is_zarr_train else batch_size, shuffle=True if not is_zarr_train else False, num_workers=num_workers, pin_memory=self.device.type == "cuda")
        
        valid_loader = None
        if X_valid is not None:
            if is_zarr_valid:
                valid_dataset = ZarrBatchDataset(X_valid, y_valid_np, None, np.where(valid_mask)[0], batch_size, shuffle=False)
            else:
                valid_dataset = self._make_dataset(X_valid_np[valid_mask], y_valid_np[valid_mask], None)
            valid_loader = DataLoader(valid_dataset, batch_size=None if is_zarr_valid else batch_size, shuffle=False, num_workers=num_workers, pin_memory=self.device.type == "cuda")

        max_epochs = int(self.params.get("max_epochs", 100))
        patience = int(self.params.get("patience", 10))
        learning_rate = float(self.params.get("lr", self.params.get("learning_rate", 1e-3)))
        weight_decay = float(self.params.get("weight_decay", 1e-5))
        gradient_clip_val = float(self.params.get("gradient_clip_val", self.params.get("grad_clip_norm", 1.0)))
        num_workers = int(self.params.get("num_workers", 0))

        gflu_stages = int(self.params.get("gflu_stages", self.params.get("n_blocks", self.params.get("n_layers", 6))))
        gflu_dropout = float(self.params.get("gflu_dropout", 0.0))
        feature_init_sparsity = float(
            self.params.get("feature_init_sparsity", self.params.get("gflu_feature_init_sparsity", 0.3))
        )
        learnable_sparsity = bool(self.params.get("learnable_sparsity", True))
        head_hidden_dims = self._normalize_hidden_dims(
            self.params.get("head_hidden_dims", self.params.get("head_dims", self.params.get("hidden_dims", [128, 64])))
        )
        head_dropout = float(self.params.get("head_dropout", self.params.get("dropout", 0.1)))

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
            # Regression
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
        scheduler_mode = "max" if self.metric_direction == "maximize" else "min"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=float(self.params.get("lr_decay_factor", 0.5)),
            patience=max(1, int(self.params.get("lr_patience", 3))),
            min_lr=float(self.params.get("min_lr", 1e-6)),
        )

        best_metric_val = float("-inf") if self.metric_direction == "maximize" else float("inf")
        best_state = None
        bad_epochs = 0
        self.history = {"train_loss": [], "valid_loss": []}
        ema_val_metric = None

        for epoch in range(max_epochs):
            if hasattr(train_loader.dataset, "on_epoch_end"):
                train_loader.dataset.on_epoch_end()
            train_loss = self._train_one_epoch(train_loader, optimizer, gradient_clip_val, epoch, max_epochs)
            if valid_loader is not None:
                valid_loss, val_metric = self._evaluate(valid_loader)
            else:
                valid_loss = train_loss
                val_metric = train_loss
                
            scheduler.step(val_metric)

            self.history["train_loss"].append(train_loss)
            self.history["valid_loss"].append(valid_loss)

            tqdm.write(
                f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f}" +
                (f" | Valid {self.early_stopping_metric}: {val_metric:.6f}" if self.early_stopping_metric != "loss" else "")
            )
            
            # --- MLflow Logging ---
            metrics_to_log = {"train_loss": train_loss}
            if X_valid_np is not None:
                metrics_to_log["valid_loss"] = valid_loss
                if self.early_stopping_metric != "loss":
                    metrics_to_log[f"valid_{self.early_stopping_metric}"] = val_metric
            log_epoch_metrics(model_idx, epoch, metrics_to_log)
            
            # --- Epoch Callback (Pruning等) の実行 ---
            if epoch_callback is not None and X_valid is not None:
                valid_preds = self.predict(X_valid)
                execute_epoch_pruning(epoch_callback, epoch, valid_preds, y_valid_np)

            # Calculate EMA of validation metric to prevent stopping on noisy spikes
            if ema_val_metric is None:
                ema_val_metric = val_metric
            else:
                ema_val_metric = self.early_stopping_ema_alpha * val_metric + (1.0 - self.early_stopping_ema_alpha) * ema_val_metric

            is_best = False
            if self.metric_direction == "maximize":
                if ema_val_metric > best_metric_val - 1e-8:
                    is_best = True
            else:
                if ema_val_metric < best_metric_val + 1e-8:
                    is_best = True

            if is_best:
                best_metric_val = ema_val_metric
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

        self.model.eval()
        self._set_seed(int(self.params.get("random_state", 42)))
        preds = []
        batch_size = int(self.params.get("predict_batch_size", self.params.get("batch_size", 4096)))
        num_workers = int(self.params.get("num_workers", 0))

        is_zarr = isinstance(X, str) and X.endswith('.zarr')
        dummy_y = np.zeros(zarr.open(X, mode='r').shape[0] if is_zarr else len(X), dtype=np.float32)
        if is_zarr:
            ds = ZarrBatchDataset(X, dummy_y, None, np.arange(len(dummy_y)), batch_size, shuffle=False)
        else:
            X_np, _ = self._to_numpy_features(X)
            ds = self._make_dataset(X_np, dummy_y, None)
        loader = DataLoader(ds, batch_size=None if is_zarr else batch_size, shuffle=False, num_workers=num_workers, pin_memory=self.device.type == "cuda")

        with torch.no_grad():
            for xb, _, _ in loader:
                xb = xb.to(self.device)
                out = self.model(xb).squeeze(-1)
                if self.is_binary_classification:
                    out = torch.sigmoid(out)
                preds.append(out.detach().cpu().numpy())

        if not preds:
            return np.array([], dtype=np.float32)
        return np.concatenate(preds, axis=0).astype(np.float32, copy=False).reshape(-1)

    def _train_one_epoch(self, train_loader, optimizer, gradient_clip_val: float, epoch: int, max_epochs: int) -> float:
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        total_weight = torch.tensor(0.0, device=self.device)

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False) as pbar:
            for xb, yb, wb in pbar:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                wb = wb.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                out = self.model(xb).squeeze(-1)
                loss_vec = self._loss_vector(out, yb)
                # wb.sum()によるゼロ除算/Loss爆発を防ぐためmeanを使用
                loss = (loss_vec * wb).mean()
                loss.backward()

                if gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
                optimizer.step()

                batch_weight = wb.sum().detach()
                total_loss += loss.detach() * batch_weight
                total_weight += batch_weight
                # バッチ毎の .item() 同期（CPU busy-wait）を防ぐため tqdm の更新を省略

        return float(total_loss.item()) / max(float(total_weight.item()), 1e-8)

    def _evaluate(self, valid_loader):
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        total_count = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for xb, yb, _ in valid_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                out = self.model(xb).squeeze(-1)
                loss_vec = self._loss_vector(out, yb)
                total_loss += loss_vec.sum().detach()
                total_count += int(loss_vec.numel())
                
                if self.early_stopping_metric != "loss":
                    if self.is_binary_classification:
                        preds = torch.sigmoid(out)
                    else:
                        preds = out
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(yb.cpu().numpy())

        valid_loss = float(total_loss.item()) / max(total_count, 1)
        
        if self.early_stopping_metric == "ic":
            from scipy.stats import spearmanr
            preds_np = np.concatenate(all_preds)
            targets_np = np.concatenate(all_targets)
            if len(preds_np) < 2 or np.max(preds_np) == np.min(preds_np) or np.max(targets_np) == np.min(targets_np):
                val_metric = 0.0
            else:
                val_metric, _ = spearmanr(targets_np, preds_np)
                if np.isnan(val_metric):
                    val_metric = 0.0
        elif self.early_stopping_metric != "loss":
            preds_np = np.concatenate(all_preds)
            targets_np = np.concatenate(all_targets)
            try:
                from hydra.utils import get_method
                metric_func = get_method(self.early_stopping_metric)
                val_metric = metric_func(targets_np, preds_np)
            except Exception as e:
                print(f"  ⚠️ Warning: Failed to calculate custom metric '{self.early_stopping_metric}'. Error: {e}")
                val_metric = valid_loss
        else:
            val_metric = valid_loss
            
        return valid_loss, val_metric

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

        if isinstance(X, pa.Buffer):
            with ipc.open_stream(X) as reader:
                table = reader.read_all()
            X = table.to_pandas()

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

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, f"gandalf_learning_curve_m{model_idx}.png")
            plt.savefig(temp_path)
            plt.close()

            if mlflow is not None and mlflow.active_run():
                mlflow.log_artifact(temp_path, artifact_path="plots/learning_curves")

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

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, f"gandalf_feature_importance_m{model_idx}.png")
            plt.savefig(temp_path)
            plt.close()

            if mlflow is not None and mlflow.active_run():
                mlflow.log_artifact(temp_path, artifact_path="plots/importance")
                csv_path = os.path.join(tmpdir, f"gandalf_feature_importance_m{model_idx}.csv")
                importance_df.to_csv(csv_path, index=False)
                mlflow.log_artifact(csv_path, artifact_path="importance_data")

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            import os
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
