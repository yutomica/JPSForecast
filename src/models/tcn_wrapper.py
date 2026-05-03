import os
import copy
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
        return torch.from_numpy(X_batch), torch.from_numpy(y_batch), torch.from_numpy(w_batch)

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
            self._build_model_from_shape(z.shape)
        else:
            X_train = np.asarray(X_train, dtype=np.float32)
            self._build_model_from_shape(X_train.shape)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n--- TCN Model Summary (Fold {model_idx}) ---")
        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Input Shape:          [N, {self.window_size_}, {self.input_dim_}]")
        print(f"Channels:             {self.num_channels}")
        print("-" * 40)

        # --- Validation Loader の準備とメモリ解放 ---
        valid_loader = None
        if X_valid is not None and y_valid is not None:
            y_valid_np = np.asarray(y_valid)
            is_zarr_valid = isinstance(X_valid, str) and X_valid.endswith('.zarr')
            valid_mask = ~np.isnan(y_valid_np) & ~np.isinf(y_valid_np)
            if not is_zarr_valid:
                valid_mask &= np.isfinite(X_valid).all(axis=(1, 2))

            dropped_valid = len(y_valid_np) - int(np.sum(valid_mask))
            if dropped_valid > 0:
                print(f"  ⚠️ Dropped {dropped_valid:,} validation samples due to NaN/Inf.")

            valid_loader = self._build_dataloader(X_valid, y_valid_np, None, valid_mask, self.batch_size, shuffle=False)
            gc.collect()

        # --- Training Loader の準備とメモリ解放 ---
        train_mask = ~np.isnan(y_train_np) & ~np.isinf(y_train_np)
        if not is_zarr_train:
            train_mask &= np.isfinite(X_train).all(axis=(1, 2))

        if sample_weight is not None:
            sample_weight = np.nan_to_num(sample_weight, nan=0.0, posinf=1.0, neginf=0.0)
            sample_weight = np.clip(sample_weight, 0.0, None)
            train_mask &= (sample_weight > 0)

        dropped_train = len(y_train_np) - int(np.sum(train_mask))
        if dropped_train > 0:
            print(f"  ⚠️ Dropped {dropped_train:,} training samples due to NaN/Inf or zero weights.")

        w_np = None
        if sample_weight is not None:
            w_np = sample_weight.astype(np.float32)
            w_np_filt = w_np[train_mask]
            if len(w_np_filt) > 0:
                p99 = np.percentile(w_np_filt, 99)
                w_np_filt = np.clip(w_np_filt, 0.0, max(p99 * 10.0, 1.0))
                if w_np_filt.mean() > 0:
                    w_np_filt = w_np_filt / w_np_filt.mean()
            w_np[train_mask] = w_np_filt

        train_loader = self._build_dataloader(X_train, y_train_np, w_np, train_mask, self.batch_size, shuffle=True)
        gc.collect()

        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_state = copy.deepcopy(self.model.state_dict())
        best_metric_val = float("-inf") if self.metric_direction == "maximize" else float("inf")
        wait = 0
        ema_val_metric = None

        amp_device = "cuda" if self.device.type == "cuda" else "mps" if self.device.type == "mps" else "cpu"
        amp_enabled = (self.device.type in ["cuda", "mps"])
        
        scaler = None
        if amp_enabled:
            if hasattr(torch.amp, "GradScaler"):
                try:
                    scaler = torch.amp.GradScaler(amp_device, enabled=True)
                except TypeError:
                    if amp_device == "cuda":
                        scaler = torch.amp.GradScaler("cuda", enabled=True)
            elif hasattr(torch.cuda.amp, "GradScaler") and amp_device == "cuda":
                scaler = torch.cuda.amp.GradScaler(enabled=True)

        loss_name = self.params.get("objective", "mse") if self.task_type != "classification" else "bce"

        for epoch in range(self.max_epochs):
            if hasattr(train_loader.dataset, "on_epoch_end"):
                train_loader.dataset.on_epoch_end()

            self.model.train()
            train_total = torch.tensor(0.0, device=self.device)
            train_count = 0
            
            # プロファイリングの設定（初回エポックのみ実行してボトルネックを可視化）
            enable_profiler = (epoch == 0)
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./log/profiler/tcn_fold{model_idx}"),
                record_shapes=True,
                with_stack=True
            ) if enable_profiler else None

            if prof is not None:
                prof.start()

            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}", leave=False) as pbar:
                for x, y, sw in pbar:
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device)
                    sw = sw.to(self.device)

                    optimizer.zero_grad(set_to_none=True)

                    if amp_enabled:
                        # 金融データの広いダイナミックレンジを維持しつつ高速化するため、bfloat16 を明示的に指定
                        # M5チップの性能を最大限に引き出す
                        dtype = torch.bfloat16 if amp_device == "mps" else torch.float16
                        with torch.amp.autocast(device_type=amp_device, enabled=True, dtype=dtype):
                            logits = self.model(x)
                            loss = self._compute_loss(logits, y, sw)
                        
                        if scaler is not None:
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
                        logits = self.model(x)
                        loss = self._compute_loss(logits, y, sw)
                        loss.backward()
                        if self.grad_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                        optimizer.step()

                    train_total += loss.detach() * x.shape[0]
                    train_count += x.shape[0]
                    # バッチ毎の .item() 同期（CPU busy-wait）を防ぐため tqdm の更新を省略
                    
                    if prof is not None:
                        prof.step()
                        
            if prof is not None:
                prof.stop()

            train_loss = float(train_total.item()) / max(train_count, 1)
            self.history["train_loss"].append(train_loss)

            if valid_loader is not None:
                self.model.eval()
                valid_total = torch.tensor(0.0, device=self.device)
                valid_count = 0
                all_preds = []
                all_targets = []
                with torch.no_grad():
                    for x, y, sw in valid_loader:
                        x = x.to(self.device, non_blocking=(self.device.type == "cuda"))
                        y = y.to(self.device, non_blocking=(self.device.type == "cuda"))
                        
                        if amp_enabled:
                            with torch.amp.autocast(device_type=amp_device, enabled=True, dtype=dtype):
                                logits = self.model(x)
                                loss = self._compute_loss(logits, y, sample_weight=None)
                        else:
                            logits = self.model(x.float())
                            loss = self._compute_loss(logits, y, sample_weight=None)
                            
                        valid_total += loss.detach() * x.shape[0]
                        valid_count += x.shape[0]
                        
                        if self.early_stopping_metric != "loss":
                            if self.task_type == "classification":
                                preds = torch.sigmoid(logits.view(-1))
                            else:
                                preds = logits.view(-1)
                            all_preds.append(preds.float().cpu().numpy())
                            all_targets.append(y.float().cpu().numpy())
                            
                valid_loss = float(valid_total.item()) / max(valid_count, 1)
                
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
                        metric_func = get_method(self.early_stopping_metric)
                        import inspect
                        if "dates" in inspect.signature(metric_func).parameters:
                            val_metric = metric_func(targets_np, preds_np, dates=valid_dates)
                        else:
                            val_metric = metric_func(targets_np, preds_np)
                    except Exception as e:
                        print(f"  ⚠️ Warning: Failed to calculate custom metric '{self.early_stopping_metric}'. Error: {e}")
                        val_metric = valid_loss
                else:
                    val_metric = valid_loss
            else:
                valid_loss = train_loss
                val_metric = train_loss

            self.history["valid_loss"].append(valid_loss)

            metric_name_log = self.early_stopping_metric.split('.')[-1] if self.early_stopping_metric != "loss" else ""
            tqdm.write(
                f"Epoch {epoch+1}/{self.max_epochs} | Train {loss_name}: {train_loss:.6f} | Valid {loss_name}: {valid_loss:.6f}" +
                (f" | Valid {metric_name_log}: {val_metric:.6f}" if self.early_stopping_metric != "loss" else "")
            )

            # MLflow記録用のキーに具体的なコスト関数名（objective）を適用
            metrics_to_log = {f"train_{loss_name}": train_loss}
            if valid_loader is not None:
                metrics_to_log[f"valid_{loss_name}"] = valid_loss
                if self.early_stopping_metric != "loss":
                    metrics_to_log[f"valid_{metric_name_log}"] = val_metric
            log_epoch_metrics(model_idx, epoch, metrics_to_log)

            if epoch_callback is not None and X_valid is not None:
                valid_preds = self.predict(X_valid)
                execute_epoch_pruning(epoch_callback, epoch, valid_preds, y_valid)

            # Calculate EMA of validation metric to prevent stopping on noisy spikes
            if ema_val_metric is None:
                ema_val_metric = val_metric
            else:
                ema_val_metric = self.early_stopping_ema_alpha * val_metric + (1.0 - self.early_stopping_ema_alpha) * ema_val_metric

            is_best = False
            if self.metric_direction == "maximize":
                if ema_val_metric > best_metric_val:
                    is_best = True
            else:
                if ema_val_metric < best_metric_val:
                    is_best = True
                    
            if is_best:
                best_metric_val = ema_val_metric
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
            except Exception: pass

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
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        is_zarr = isinstance(X, str) and X.endswith('.zarr')
        if not is_zarr:
            X = np.asarray(X, dtype=np.float32)
            if X.ndim != 3:
                raise ValueError(f"X must be 3D [N, T, F], but got {X.shape}")

        dummy_y = np.zeros(zarr.open(X, mode='r').shape[0] if is_zarr else X.shape[0], dtype=np.float32)
        loader = self._build_dataloader(X, dummy_y, None, None, self.batch_size, shuffle=False)

        outputs = []
        self.model.eval()

        amp_device = "cuda" if self.device.type == "cuda" else "mps" if self.device.type == "mps" else "cpu"
        amp_enabled = (self.device.type in ["cuda", "mps"])
        dtype = torch.bfloat16 if amp_device == "mps" else torch.float16

        with torch.no_grad():
            for x, _, _ in tqdm(loader, desc="Predicting", leave=False):
                x = x.to(self.device, non_blocking=(self.device.type == "cuda"))
                
                if amp_enabled:
                    with torch.amp.autocast(device_type=amp_device, enabled=True, dtype=dtype):
                        logits = self.model(x)
                else:
                    logits = self.model(x.float())

                if self.task_type == "classification":
                    preds = torch.sigmoid(logits.view(-1))
                else:
                    preds = logits.view(-1)
                outputs.append(preds.float().detach().cpu().numpy())

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
            if self.input_dim_ is None:
                raise ValueError("Cannot restore TCNWrapper because input_dim_ is missing.")
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
            )
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()
