import os
import copy
import inspect
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import zarr
import torch
import torch.nn as nn, gc
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .base import BaseModelWrapper
from .pruning import execute_epoch_pruning, log_epoch_metrics
from .networks.tcn import TCN
import torch.profiler
from hydra.utils import get_method

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

        chunk = z[start_idx:end_idx, :, :]
        local_indices = physical_batch - start_idx
        X_batch = chunk[local_indices]
        y_batch = self.y[logical_batch]
        w_batch = self.w[logical_batch] if self.w is not None else np.ones(len(y_batch), dtype=np.float32)
        return torch.from_numpy(X_batch).float(), torch.from_numpy(y_batch).float(), torch.from_numpy(w_batch).float()


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.batch_indices)

class TCNWrapper(BaseModelWrapper):
    """
    train.py 互換の TCN wrapper
      - fit(X_train, y_train, X_valid, y_valid, sample_weight, model_idx)
      - predict(X)

    X:
      - np.ndarray [N, T, F]
    """

    def __init__(self, task_type="regression", **params):
        self.task_type = task_type
        self.params = copy.deepcopy(params)

        # model params
        raw_num_channels = params.pop("num_channels", None)
        if raw_num_channels is None:
            raw_num_channels = params.pop("num_channel", [64, 64, 64])
            
        if isinstance(raw_num_channels, int):
            self.num_channels = [raw_num_channels]
        elif isinstance(raw_num_channels, str):
            self.num_channels = [int(x.strip()) for x in raw_num_channels.split(",") if x.strip()]
        else:
            self.num_channels = [int(x) for x in raw_num_channels]

        self.kernel_size = int(params.pop("kernel_size", 3))
        self.dropout = float(params.pop("dropout", 0.1))
        self.activation = params.pop("activation", "gelu")
        self.use_weight_norm = bool(params.pop("use_weight_norm", False))
        self.norm_type = params.pop("norm_type", "group")
        self.pooling = params.pop("pooling", "last")
        self.head_hidden_dim = int(params.pop("head_hidden_dim", 0))
        self.head_dropout = float(params.pop("head_dropout", 0.0))

        # convenience option: n_layers + base_channels -> repeated channels
        n_layers = params.pop("n_layers", None)
        if n_layers is None:
            n_layers = params.pop("n_blocks", None)
            
        base_channels = params.pop("base_channels", None)
        if base_channels is None:
            base_channels = params.pop("base_channel", None)
            
        channel_growth = params.pop("channel_growth", None)
        if n_layers is not None:
            if base_channels is None:
                base_channels = self.num_channels[0] if self.num_channels else 64
                
            if channel_growth is None:
                self.num_channels = [int(base_channels)] * int(n_layers)
            else:
                self.num_channels = [int(base_channels * (channel_growth ** i)) for i in range(int(n_layers))]

        # training params
        self.max_epochs = int(params.pop("max_epochs", 100))
        self.patience = int(params.pop("patience", params.pop("early_stopping_rounds", params.pop("early_stopping_round", 10))))
        self.batch_size = int(params.pop("batch_size", 512))
        self.lr = float(params.pop("lr", params.pop("learning_rate", 1e-3)))
        self.weight_decay = float(params.pop("weight_decay", 1e-5))
        self.grad_clip_norm = params.pop("grad_clip_norm", params.pop("gradient_clip_val", 1.0))
        self.num_workers = int(params.pop("num_workers", 0))
        self.random_state = int(params.pop("random_state", 42))
        self.device_name = params.pop("device_name", "auto")
        self.optimizer_name = params.pop("optimizer", "adamw")
        self.early_stopping_metric = params.pop("early_stopping_metric", "loss")
        self.metric_direction = params.pop("metric_direction", "minimize")
        self.early_stopping_ema_alpha = float(params.pop("early_stopping_ema_alpha", 1.0))
        self.ensemble_size = int(params.pop("ensemble_size", 1))
        self.log_learning_curve = bool(params.pop("log_learning_curve", True))

        if self.device_name == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.device_name)

        # MPSでのマルチプロセス警告
        if self.device.type == "mps" and self.num_workers > 0:
            print("  ⚠️ Notice: Using num_workers>0 on MPS. If you experience hangs, set num_workers=0.")

        self.model = None
        self.models = []
        self.history = {"train_loss": [], "valid_loss": []}
        self.feature_importances_ = None
        self.best_epoch_ = None
        self.input_dim_ = None
        self.window_size_ = None

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

    def _build_model_from_shape(self, shape):
        if len(shape) != 3:
            raise ValueError(f"TCNWrapper expects 3D shape [N, T, F], but got {shape}")

        self.window_size_ = int(shape[1])
        self.input_dim_ = int(shape[2])
        self.model = TCN(
            input_dim=self.input_dim_,
            output_dim=1,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            activation=self.activation,
            use_weight_norm=self.use_weight_norm,
            norm_type=self.norm_type,
            pooling=self.pooling,
            head_hidden_dim=self.head_hidden_dim,
            head_dropout=self.head_dropout,
        ).to(self.device)

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

    def _build_dataloader(self, X, y_np, w_np, mask, batch_size, shuffle):
        is_zarr = isinstance(X, str) and X.endswith('.zarr')
        if is_zarr:
            valid_indices = np.where(mask)[0] if mask is not None else np.arange(len(y_np))
            y_filt = y_np[mask] if mask is not None else y_np
            w_filt = w_np[mask] if mask is not None and w_np is not None else w_np
            ds = ZarrBatchDataset(X, y_filt, w_filt, valid_indices, batch_size, shuffle=shuffle)
            return DataLoader(ds, batch_size=None, num_workers=self.num_workers, pin_memory=(self.device.type=="cuda"))
        else:
            if mask is not None:
                X_filt = X[mask]
                y_filt = y_np[mask]
                w_filt = w_np[mask] if w_np is not None else None
            else:
                X_filt = X
                y_filt = y_np
                w_filt = w_np

            tensors = [torch.from_numpy(X_filt).float(), torch.from_numpy(y_filt).float()]
            if w_filt is not None:
                tensors.append(torch.from_numpy(w_filt).float())
            else:
                tensors.append(torch.ones(len(y_filt), dtype=torch.float32))
            ds = TensorDataset(*tensors)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=(self.device.type=="cuda"))

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0, epoch_callback=None, train_dates=None, valid_dates=None):
        y_train_np = np.asarray(y_train)
        is_zarr_train = isinstance(X_train, str) and X_train.endswith('.zarr')
        if is_zarr_train:
            z = zarr.open(X_train, mode='r')
            self.window_size_, self.input_dim_ = int(z.shape[1]), int(z.shape[2])
        else:
            X_train = np.asarray(X_train, dtype=np.float32)
            self.window_size_, self.input_dim_ = int(X_train.shape[1]), int(X_train.shape[2])

        # --- Validation Loader ---
        valid_loader = None
        if X_valid is not None and y_valid is not None:
            y_valid_np = np.asarray(y_valid)
            is_zarr_valid = isinstance(X_valid, str) and X_valid.endswith('.zarr')
            valid_mask = ~np.isnan(y_valid_np) & ~np.isinf(y_valid_np)
            if not is_zarr_valid: valid_mask &= np.isfinite(X_valid).all(axis=(1, 2))
            valid_loader = self._build_dataloader(X_valid, y_valid_np, None, valid_mask, self.batch_size, shuffle=False)
            
        # --- Training Loader ---
        train_mask = ~np.isnan(y_train_np) & ~np.isinf(y_train_np)
        if not is_zarr_train: train_mask &= np.isfinite(X_train).all(axis=(1, 2))
        if sample_weight is not None:
            sample_weight = np.nan_to_num(sample_weight, nan=0.0, posinf=1.0, neginf=0.0)
            train_mask &= (np.clip(sample_weight, 0.0, None) > 0)
        w_np = None
        if sample_weight is not None:
            w_np = sample_weight.astype(np.float32)
            w_np_filt = w_np[train_mask]
            if len(w_np_filt) > 0:
                p99 = np.percentile(w_np_filt, 99)
                w_np_filt = np.clip(w_np_filt, 0.0, max(p99 * 10.0, 1.0))
                if w_np_filt.mean() > 0: w_np_filt /= w_np_filt.mean()
            w_np[train_mask] = w_np_filt
        train_loader = self._build_dataloader(X_train, y_train_np, w_np, train_mask, self.batch_size, shuffle=True)
        gc.collect()

        self.models = []
        all_feature_importances = []
        base_seed = self.random_state

        for s_idx in range(self.ensemble_size):
            current_seed = base_seed + s_idx
            if self.ensemble_size > 1: print(f"\n🚀 Training Ensemble Model {s_idx+1}/{self.ensemble_size} (seed={current_seed})...")
            torch.manual_seed(current_seed); np.random.seed(current_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(current_seed)
            self._build_model_from_shape((0, self.window_size_, self.input_dim_))
            if s_idx == 0:
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"\n--- TCN Model Summary (Fold {model_idx}) ---")
                print(f"Total Parameters: {total_params:,}")
                print("-" * 40)

            if self.optimizer_name.lower() == "adam":
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            else:
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            best_state, best_metric_val, wait, ema_val_metric = None, float("-inf") if self.metric_direction == "maximize" else float("inf"), 0, None
            amp_device = "cuda" if self.device.type == "cuda" else "mps" if self.device.type == "mps" else "cpu"
            amp_enabled = (self.device.type in ["cuda", "mps"])
            if self.device.type == "mps": amp_enabled = False
            scaler = torch.amp.GradScaler(amp_device, enabled=True) if amp_enabled and amp_device == "cuda" else None
            loss_name = self.params.get("objective", "mse") if self.task_type != "classification" else "bce"

            from hydra.utils import get_method
            import inspect
            stopping_func = None
            if self.early_stopping_metric == "ic":
                from .pruning import calculate_spearman_ic
                stopping_func = calculate_spearman_ic
            elif self.early_stopping_metric != "loss":
                try: stopping_func = get_method(self.early_stopping_metric)
                except Exception: pass

            for epoch in range(self.max_epochs):
                if hasattr(train_loader.dataset, "on_epoch_end"): train_loader.dataset.on_epoch_end()
                self.model.train(); train_total, train_count = torch.tensor(0.0, device=self.device), 0
                with tqdm(train_loader, desc=f"Model {s_idx+1} Epoch {epoch+1}/{self.max_epochs}", leave=False) as pbar:
                    for x, y, sw in pbar:
                        x, y, sw = x.to(self.device), y.to(self.device), sw.to(self.device)
                        optimizer.zero_grad(set_to_none=True)
                        if amp_enabled:
                            dtype = torch.bfloat16 if amp_device == "mps" else torch.float16
                            with torch.amp.autocast(device_type=amp_device, enabled=True, dtype=dtype):
                                logits = self.model(x); loss = self._compute_loss(logits, y, sw)
                            if scaler: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                            else: loss.backward(); optimizer.step()
                        else:
                            logits = self.model(x); loss = self._compute_loss(logits, y, sw)
                            loss.backward(); optimizer.step()
                        train_total += loss.detach() * x.shape[0]; train_count += x.shape[0]
                train_loss = float(train_total.item()) / max(train_count, 1)
                if s_idx == 0: self.history["train_loss"].append(train_loss)

                if valid_loader is not None:
                    self.model.eval(); valid_total, valid_count, all_preds, all_targets = torch.tensor(0.0, device=self.device), 0, [], []
                    with torch.no_grad():
                        for x, y, _ in valid_loader:
                            x, y = x.to(self.device), y.to(self.device)
                            logits = self.model(x); loss = self._compute_loss(logits, y, None)
                            valid_total += loss.detach() * x.shape[0]; valid_count += x.shape[0]
                            if stopping_func is not None:
                                preds = torch.sigmoid(logits.view(-1)) if self.task_type == "classification" else logits.view(-1)
                                all_preds.append(preds.float().cpu().numpy()); all_targets.append(y.float().cpu().numpy())
                    valid_loss = float(valid_total.item()) / max(valid_count, 1)
                    if stopping_func is not None:
                        preds_np, targets_np = np.concatenate(all_preds), np.concatenate(all_targets)
                        try:
                            sig = inspect.signature(stopping_func)
                            val_metric = stopping_func(targets_np, preds_np, dates=valid_dates) if "dates" in sig.parameters else stopping_func(targets_np, preds_np)
                        except Exception: val_metric = valid_loss
                    else: val_metric = valid_loss
                else:
                    valid_loss = np.nan
                    val_metric = np.nan

                if s_idx == 0: self.history["valid_loss"].append(valid_loss)
                if ema_val_metric is None or np.isnan(ema_val_metric):
                    ema_val_metric = val_metric
                elif not np.isnan(val_metric):
                    ema_val_metric = self.early_stopping_ema_alpha * val_metric + (1.0 - self.early_stopping_ema_alpha) * ema_val_metric

                # --- Logging ---
                metric_name_log = self.early_stopping_metric.split('.')[-1] if self.early_stopping_metric != "loss" else ""
                msg = f"Epoch {epoch+1}/{self.max_epochs} | Train {loss_name}: {train_loss:.6f} | Valid {loss_name}: {valid_loss:.6f}"
                if metric_name_log and not np.isnan(val_metric):
                    msg += f" | {metric_name_log}: {val_metric:.6f}"
                if self.early_stopping_ema_alpha < 1.0 and not np.isnan(ema_val_metric):
                    msg += f" | {metric_name_log}_smoothed: {ema_val_metric:.6f}"
                tqdm.write(msg)

                if epoch % 10 == 0 or epoch == self.max_epochs - 1:
                    metrics_to_log = {"train_loss": train_loss, "valid_loss": valid_loss}
                    if metric_name_log and not np.isnan(val_metric):
                        metrics_to_log[f"valid_{metric_name_log}"] = val_metric
                    if self.early_stopping_ema_alpha < 1.0 and not np.isnan(ema_val_metric):
                        metrics_to_log[f"valid_{metric_name_log}_smoothed"] = ema_val_metric
                    log_epoch_metrics(model_idx, epoch, metrics_to_log)

                is_best = False
                if not np.isnan(ema_val_metric):
                    is_best = (ema_val_metric > best_metric_val) if self.metric_direction == "maximize" else (ema_val_metric < best_metric_val)
                
                if is_best or valid_loader is None:
                    # valid_loaderがない(本番学習時)は常に最新をbestとする
                    if not np.isnan(ema_val_metric):
                        best_metric_val = ema_val_metric
                    best_state, wait = copy.deepcopy(self.model.state_dict()), 0
                    if s_idx == 0: self.best_epoch_ = epoch
                else:
                    wait += 1
                    if wait >= self.patience: break

            self.model.load_state_dict(best_state); self.model.eval(); self.models.append(copy.deepcopy(self.model))
            if s_idx == 0 and self.log_learning_curve:
                self._log_learning_curve(model_idx)
            self._create_feature_importance_df(); all_feature_importances.append(self.feature_importances_)
            del best_state, optimizer; gc.collect()

        self.feature_importances_ = np.mean(all_feature_importances, axis=0)
        self.model = self.models[0]
        del train_loader, valid_loader; gc.collect()

    def _log_learning_curve(self, model_idx):
        if not self.history["train_loss"]:
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.history["train_loss"], label="train_loss")
        if self.history["valid_loss"]:
            plt.plot(self.history["valid_loss"], label="valid_loss")
        plt.title(f"TCN Learning Curve (Model {model_idx})")
        plt.xlabel("Epochs")
        loss_name = self.params.get("objective", "mse") if self.task_type != "classification" else "bce"
        plt.ylabel(f"Loss ({loss_name})")
        plt.legend()
        plt.grid(True)

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, f"tcn_learning_curve_m{model_idx}.png")
            plt.savefig(temp_path)
            plt.close()

            if mlflow.active_run():
                mlflow.log_artifact(temp_path, artifact_path="plots/learning_curves")

    def _create_feature_importance_df(self):
        if self.model is None:
            return

        # proxy importance: first conv absolute weight aggregated over output channels and kernel axis
        first_block = self.model.backbone[0]
        w = first_block.conv1.conv.weight.detach().cpu().numpy()  # [C_out, C_in, K]
        importance = np.abs(w).sum(axis=(0, 2))
        self.feature_importances_ = importance

    def predict(self, X):
        if not self.models: raise ValueError("Model has not been trained yet.")
        is_zarr = isinstance(X, str) and X.endswith('.zarr')
        if not is_zarr:
            X = np.asarray(X, dtype=np.float32)
            if X.ndim != 3: raise ValueError(f"X must be 3D [N, T, F], but got {X.shape}")
        n_samples = zarr.open(X, mode='r').shape[0] if is_zarr else X.shape[0]
        dummy_y = np.zeros(n_samples, dtype=np.float32)
        loader = self._build_dataloader(X, dummy_y, None, None, self.batch_size, shuffle=False)
        all_ensemble_preds = []
        for m_idx, model in enumerate(self.models):
            model.eval(); outputs = []
            with torch.no_grad():
                for x, _, _ in tqdm(loader, desc=f"Predicting {m_idx+1}/{len(self.models)}", leave=False):
                    x = x.to(self.device); logits = model(x)
                    preds = torch.sigmoid(logits.view(-1)) if self.task_type == "classification" else logits.view(-1)
                    outputs.append(preds.float().detach().cpu().numpy())
            all_ensemble_preds.append(np.concatenate(outputs, axis=0).flatten())
        return np.mean(all_ensemble_preds, axis=0)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "models" in state and state["models"]:
            state["_models_state_dicts"] = [{k: v.cpu() for k, v in m.state_dict().items()} for m in state["models"]]
            del state["models"]
        if "model" in state: del state["model"]
        return state

    def __setstate__(self, state):
        models_states = state.pop("_models_state_dicts", [])
        self.__dict__.update(state)
        if models_states:
            self.models = []
            for m_state in models_states:
                m = TCN(
                    input_dim=self.input_dim_, output_dim=1, num_channels=self.num_channels,
                    kernel_size=self.kernel_size, dropout=self.dropout, activation=self.activation,
                    use_weight_norm=self.use_weight_norm, norm_type=self.norm_type, pooling=self.pooling,
                    head_hidden_dim=self.head_hidden_dim, head_dropout=self.head_dropout,
                )
                m.load_state_dict(m_state); m.to(self.device).eval()
                self.models.append(m)
            self.model = self.models[0]
