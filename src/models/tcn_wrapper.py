import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn, gc
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .base import BaseModelWrapper
from .pruning import execute_epoch_pruning, log_epoch_metrics
from .networks.tcn import TCN


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
        self.num_channels = params.pop("num_channels", [64, 64, 64])
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
        base_channels = params.pop("base_channels", None)
        channel_growth = params.pop("channel_growth", None)
        if n_layers is not None and base_channels is not None:
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

        if self.device_name == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.device_name)

        # MPS安定化: multiprocessing DataLoader は避ける
        if self.device.type == "mps" and self.num_workers > 0:
            print("  ⚠️ num_workers>0 is unstable on MPS in many environments. Forcing num_workers=0.")
            self.num_workers = 0

        self.model = None
        self.history = {"train_loss": [], "valid_loss": []}
        self.feature_importances_ = None
        self.best_epoch_ = None
        self.input_dim_ = None
        self.seq_len_ = None

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

    def _build_model(self, X):
        if X.ndim != 3:
            raise ValueError(f"TCNWrapper expects 3D array [N, T, F], but got {X.shape}")

        self.seq_len_ = int(X.shape[1])
        self.input_dim_ = int(X.shape[2])
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

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0, epoch_callback=None):
        X_train = np.asarray(X_train, dtype=np.float16)
        y_train_np = np.asarray(y_train)

        if X_train.ndim != 3:
            raise ValueError(f"X_train must be 3D [N, T, F], but got {X_train.shape}")

        self._build_model(X_train)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n--- TCN Model Summary (Fold {model_idx}) ---")
        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Input Shape:          [N, {self.seq_len_}, {self.input_dim_}]")
        print(f"Channels:             {self.num_channels}")
        print("-" * 40)

        # --- Validation Loader の準備とメモリ解放 ---
        valid_loader = None
        if X_valid is not None and y_valid is not None:
            X_valid = np.asarray(X_valid, dtype=np.float16)
            y_valid_np = np.asarray(y_valid)
            valid_mask = ~np.isnan(y_valid_np) & ~np.isinf(y_valid_np)
            valid_mask &= np.isfinite(X_valid).all(axis=(1, 2))

            dropped_valid = len(y_valid_np) - int(np.sum(valid_mask))
            if dropped_valid > 0:
                print(f"  ⚠️ Dropped {dropped_valid:,} validation samples due to NaN/Inf.")

            X_valid_filtered = X_valid[valid_mask]
            y_valid_filtered = y_valid_np[valid_mask]

            ds = TensorDataset(
                torch.from_numpy(X_valid_filtered),
                torch.from_numpy(y_valid_filtered.astype(np.float32)),
                torch.ones(len(X_valid_filtered), dtype=torch.float32)
            )
            valid_loader = DataLoader(
                ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                pin_memory=(self.device.type == "cuda"), drop_last=False,
            )
            del X_valid, y_valid, y_valid_np, valid_mask, X_valid_filtered, y_valid_filtered, ds
            gc.collect()

        # --- Training Loader の準備とメモリ解放 ---
        train_mask = ~np.isnan(y_train_np) & ~np.isinf(y_train_np)
        train_mask &= np.isfinite(X_train).all(axis=(1, 2))

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

        ds = TensorDataset(
            torch.from_numpy(X_train_filtered),
            torch.from_numpy(y_train_filtered.astype(np.float32)),
            w_t
        )
        train_loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"), drop_last=False,
        )
        del X_train, y_train, y_train_np, train_mask, sample_weight, X_train_filtered, y_train_filtered, ds, w_t
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
            self.model.train()
            train_total = 0.0
            train_count = 0

            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}", leave=False) as pbar:
                for x, y, sw in pbar:
                    x = x.to(self.device, non_blocking=(self.device.type == "cuda")).float()
                    y = y.to(self.device, non_blocking=(self.device.type == "cuda"))
                    sw = sw.to(self.device, non_blocking=(self.device.type == "cuda"))

                    optimizer.zero_grad(set_to_none=True)

                    if amp_enabled:
                        with torch.amp.autocast(device_type=amp_device, enabled=True):
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

                    train_total += loss.item() * x.shape[0]
                    train_count += x.shape[0]
                    pbar.set_postfix({f"train_{loss_name}": f"{train_total / max(train_count, 1):.6f}"})

            train_loss = train_total / max(train_count, 1)
            self.history["train_loss"].append(train_loss)

            if valid_loader is not None:
                self.model.eval()
                valid_total = 0.0
                valid_count = 0
                with torch.no_grad():
                    for x, y, sw in valid_loader:
                        x = x.to(self.device, non_blocking=(self.device.type == "cuda")).float()
                        y = y.to(self.device, non_blocking=(self.device.type == "cuda"))
                        logits = self.model(x)
                        loss = self._compute_loss(logits, y, sample_weight=None)
                        valid_total += loss.item() * x.shape[0]
                        valid_count += x.shape[0]
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
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        temp_path = f"tcn_learning_curve_m{model_idx}.png"
        plt.savefig(temp_path)
        plt.close()

        if mlflow.active_run():
            mlflow.log_artifact(temp_path, artifact_path="plots/learning_curves")

        os.remove(temp_path)

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

        X = np.asarray(X, dtype=np.float16)
        if X.ndim != 3:
            raise ValueError(f"X must be 3D [N, T, F], but got {X.shape}")

        ds = TensorDataset(
            torch.from_numpy(X),
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
            for x, _, _ in tqdm(loader, desc="Predicting", leave=False):
                x = x.to(self.device, non_blocking=(self.device.type == "cuda")).float()
                logits = self.model(x)
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
