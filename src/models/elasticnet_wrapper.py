import os
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import (
    ElasticNet,
    SGDClassifier,
    QuantileRegressor,
    SGDRegressor,
    TweedieRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, log_loss

from .base import BaseModelWrapper
from .pruning import execute_epoch_pruning, log_epoch_metrics


class AsymmetricElasticNet(BaseEstimator, RegressorMixin):
    """
    IRLS (Iteratively Reweighted Least Squares) を用いて
    非対称MSE (Asymmetric MSE) を最適化する ElasticNet のラッパー。
    """
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, max_iter=1000,
                 tol=1e-4, selection="cyclic", random_state=42, positive=False,
                 asym_alpha=3.0, asym_beta=1.0, irls_max_iter=10, irls_tol=1e-4,
                 warm_start=False):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.selection = selection
        self.random_state = random_state
        self.positive = positive
        self.asym_alpha = asym_alpha
        self.asym_beta = asym_beta
        self.irls_max_iter = irls_max_iter
        self.irls_tol = irls_tol
        self.warm_start = warm_start
        self._n_iter_total = 0

        self.model_ = ElasticNet(
            alpha=self.alpha, l1_ratio=self.l1_ratio, fit_intercept=self.fit_intercept,
            max_iter=self.max_iter, tol=self.tol, selection=self.selection,
            random_state=self.random_state, positive=self.positive, warm_start=True
        )

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        if sample_weight is None:
            base_w = np.ones_like(y, dtype=np.float32)
        else:
            base_w = np.asarray(sample_weight, dtype=np.float32).copy()

        if self.warm_start:
            if not hasattr(self, "w_"):
                self.w_ = base_w
            
            self.model_.fit(X, y, sample_weight=self.w_)
            if hasattr(self.model_, "n_iter_"):
                self._n_iter_total += int(np.max(self.model_.n_iter_))
            
            preds = self.model_.predict(X)
            residuals = y - preds
            asym_w = np.where(residuals > 0, self.asym_alpha, self.asym_beta)
            self.w_ = base_w * asym_w
        else:
            w = base_w
            prev_coef = None
            for _ in range(self.irls_max_iter):
                self.model_.fit(X, y, sample_weight=w)
                
                preds = self.model_.predict(X)
                residuals = y - preds
                asym_w = np.where(residuals > 0, self.asym_alpha, self.asym_beta)
                w = base_w * asym_w

                current_coef = np.concatenate([self.model_.coef_, [self.model_.intercept_]])
                if prev_coef is not None:
                    if np.max(np.abs(current_coef - prev_coef)) < self.irls_tol:
                        break
                prev_coef = current_coef
            if hasattr(self.model_, "n_iter_"):
                self._n_iter_total += int(np.max(self.model_.n_iter_))

        return self

    def predict(self, X):
        return self.model_.predict(X)

    @property
    def coef_(self):
        return self.model_.coef_

    @property
    def intercept_(self):
        return self.model_.intercept_

    @property
    def n_iter_(self):
        return self._n_iter_total

    @property
    def dual_gap_(self):
        return self.model_.dual_gap_


class ElasticNetWrapper(BaseModelWrapper):
    """
    train.py 互換のラッパー。

    - regression:
        sklearn.linear_model.ElasticNet を使用
    - classification:
        train.py との互換性確保のため、
        Elastic Net 正則化付き線形分類器として
        SGDClassifier(loss='log_loss', penalty='elasticnet') を使用

    備考:
    - 厳密な `ElasticNet` 推定器は回帰専用
    - 特徴量重要度は |coef_| を使用
    - sklearn の線形モデルは epoch ごとの validation metric を公開しないため、
      epoch_callback は受け取るが内部では使用しない
    - LabelEncoder 済みカテゴリ列もそのまま学習可能だが、
      線形モデルは整数値を順序付き・距離付きの連続量として扱うため、
      名義尺度カテゴリでは統計的には one-hot 等の方が自然なことが多い
    """

    def __init__(self, task_type="regression", **params):
        self.task_type = task_type
        self.params = params
        self.model = None
        self.feature_importances_ = None

    def _build_model(self):
        if self.task_type == "regression":
            loss = self.params.get("loss", "squared_error")

            if loss == "quantile":
                return QuantileRegressor(
                    quantile=self.params.get("quantile", 0.5),
                    alpha=self.params.get("alpha", 1.0),
                    fit_intercept=self.params.get("fit_intercept", True),
                    solver=self.params.get("solver", "highs"),
                )
                
            if loss == "asymmetric_mse":
                return AsymmetricElasticNet(
                    alpha=self.params.get("alpha", 1.0),
                    l1_ratio=self.params.get("l1_ratio", 0.5),
                    fit_intercept=self.params.get("fit_intercept", True),
                    max_iter=self.params.get("max_iter", 1000),
                    tol=self.params.get("tol", 1e-4),
                    selection=self.params.get("selection", "cyclic"),
                    random_state=self.params.get("random_state", 42),
                    positive=self.params.get("positive", False),
                    asym_alpha=self.params.get("asym_alpha", 3.0),
                    asym_beta=self.params.get("asym_beta", 1.0),
                    irls_max_iter=self.params.get("irls_max_iter", 10),
                    irls_tol=self.params.get("irls_tol", 1e-4),
                    warm_start=self.params.get("warm_start", False),
                )
            
            if loss in ["huber", "smooth_l1"]:
                return SGDRegressor(
                    loss="huber",
                    penalty="elasticnet",
                    alpha=self.params.get("alpha", 1e-4),
                    l1_ratio=self.params.get("l1_ratio", 0.15),
                    fit_intercept=self.params.get("fit_intercept", True),
                    max_iter=self.params.get("max_iter", 1000),
                    tol=self.params.get("tol", 1e-3),
                    random_state=self.params.get("random_state", 42),
                    epsilon=self.params.get("huber_beta", 1.0),
                    early_stopping=self.params.get("early_stopping", False),
                    validation_fraction=self.params.get("validation_fraction", 0.1),
                    n_iter_no_change=self.params.get("n_iter_no_change", 5),
                    warm_start=self.params.get("warm_start", False),
                )

            if loss == "tweedie":
                return TweedieRegressor(
                    power=self.params.get("tweedie_variance_power", 1.5),
                    alpha=self.params.get("alpha", 1.0),
                    l1_ratio=self.params.get("l1_ratio", 0.5),
                    fit_intercept=self.params.get("fit_intercept", True),
                    max_iter=self.params.get("max_iter", 1000),
                    tol=self.params.get("tol", 1e-4),
                    link=self.params.get("link", "auto"),
                    warm_start=self.params.get("warm_start", False),
                )

            if loss in ["squared_error", "mse"]:
                return ElasticNet(
                    alpha=self.params.get("alpha", 1.0),
                    l1_ratio=self.params.get("l1_ratio", 0.5),
                    fit_intercept=self.params.get("fit_intercept", True),
                    max_iter=self.params.get("max_iter", 1000),
                    tol=self.params.get("tol", 1e-4),
                    selection=self.params.get("selection", "cyclic"),
                    random_state=self.params.get("random_state", 42),
                    positive=self.params.get("positive", False),
                    warm_start=self.params.get("warm_start", False),
                )

            raise ValueError(f"Unsupported loss for regression: {loss}")

        if self.task_type == "classification":
            return SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=self.params.get("alpha", 1e-4),
                l1_ratio=self.params.get("l1_ratio", 0.15),
                fit_intercept=self.params.get("fit_intercept", True),
                max_iter=self.params.get("max_iter", 1000),
                tol=self.params.get("tol", 1e-3),
                random_state=self.params.get("random_state", 42),
                class_weight=self.params.get("class_weight", None),
                early_stopping=self.params.get("early_stopping", False),
                validation_fraction=self.params.get("validation_fraction", 0.1),
                n_iter_no_change=self.params.get("n_iter_no_change", 5),
                warm_start=self.params.get("warm_start", False),
            )

        raise ValueError(f"Unsupported task_type: {self.task_type}")

    def _ensure_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    def _compute_loss(self, preds, y, sample_weight=None):
        if self.task_type == "classification":
            return log_loss(y, preds, sample_weight=sample_weight)
        
        loss_type = self.params.get("loss", "squared_error")
        if loss_type == "quantile":
            quantile = self.params.get("quantile", 0.5)
            diff = y - preds
            loss_each = np.maximum(quantile * diff, (quantile - 1.0) * diff)
        elif loss_type == "asymmetric_mse":
            alpha = self.params.get("asym_alpha", 3.0)
            beta = self.params.get("asym_beta", 1.0)
            diff = y - preds
            mask = (diff > 0).astype(float)
            loss_each = (mask * alpha + (1.0 - mask) * beta) * (diff ** 2)
        elif loss_type in ["huber", "smooth_l1"]:
            beta = self.params.get("huber_beta", 1.0)
            diff = np.abs(y - preds)
            loss_each = np.where(diff < beta, 0.5 * (diff ** 2), beta * (diff - 0.5 * beta))
        elif loss_type == "tweedie":
            p = self.params.get("tweedie_variance_power", 1.5)
            mu = np.maximum(preds, 1e-6)
            y_safe = np.maximum(y, 0.0)
            loss_each = (mu ** (2.0 - p)) / (2.0 - p) - y_safe * (mu ** (1.0 - p)) / (1.0 - p)
        else: # squared_error, mse
            loss_each = (y - preds) ** 2
            
        if sample_weight is not None:
            return np.average(loss_each, weights=sample_weight)
        return np.mean(loss_each)

    def _to_model_array(self, X):
        X_df = self._ensure_dataframe(X)
        # LabelEncoder 済み int 列 + float 列の混在でも、常に明示的に数値配列へ変換する
        return X_df.to_numpy(dtype=np.float32, copy=False), X_df

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0, epoch_callback=None):

        train_mask = np.isfinite(y_train)
        if sample_weight is not None:
            sample_weight = np.nan_to_num(sample_weight, nan=0.0, posinf=0.0, neginf=0.0)
            sample_weight = np.clip(sample_weight, 0.0, None)
            train_mask &= (sample_weight > 0)

        dropped_train = len(y_train) - int(np.sum(train_mask))
        if dropped_train > 0:
            print(f"  ⚠️ Dropped {dropped_train:,} training samples due to NaN target or zero weights.")

        X_train_df = self._ensure_dataframe(X_train)
        X_train_df = X_train_df.loc[train_mask].copy()
        X_train_arr, X_train_df = self._to_model_array(X_train_df)
        y_train = np.asarray(y_train)[train_mask]

        if sample_weight is not None:
            sample_weight = sample_weight[train_mask]

        total_epochs = self.params.get("max_iter", 1000)
        early_stopping_rounds = self.params.get("early_stopping_rounds", self.params.get("patience", 10))
        
        use_manual_loop = self.params.get("warm_start", False) or (early_stopping_rounds > 0 and X_valid is not None)
        verbose = self.params.get("verbose", 0)

        if use_manual_loop:
            self.params["warm_start"] = True
            self.params["max_iter"] = 1

        self.model = self._build_model()

        if self.task_type == "classification":
            y_train = np.asarray(y_train).astype(int).ravel()
            unique_classes = np.unique(y_train)
            if unique_classes.size != 2:
                raise ValueError(
                    f"ElasticNetWrapper classification expects binary labels, got classes={unique_classes.tolist()}"
                )
        else:
            y_train = np.asarray(y_train).astype(float).ravel()

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
            
        loss_name = self.params.get("loss", "squared_error") if self.task_type == "regression" else "log_loss"

        if use_manual_loop:
            best_valid_loss = float("inf")
            best_model_state = None
            wait = 0
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", message=".*objective has been evaluated.*")
                
                for epoch in range(1, total_epochs + 1):
                    self.model.fit(X_train_arr, y_train, **fit_kwargs)
                    
                    train_preds = self.predict(X_train_df)
                    train_loss = self._compute_loss(train_preds, y_train, sample_weight=sample_weight)
                        
                    valid_loss = None
                    if X_valid is not None and y_valid is not None:
                        valid_preds = self.predict(X_valid)
                        valid_loss = self._compute_loss(valid_preds, y_valid)
                            
                    if verbose > 0 and (epoch % 10 == 0 or epoch == 1 or epoch == total_epochs):
                        if valid_loss is not None:
                            print(f"Iteration {epoch:4d}/{total_epochs} | Train {loss_name}: {train_loss:.6f} | Valid {loss_name}: {valid_loss:.6f}")
                        else:
                            print(f"Iteration {epoch:4d}/{total_epochs} | Train {loss_name}: {train_loss:.6f}")
                            
                    metrics_to_log = {"train_loss": train_loss}
                    if valid_loss is not None:
                        metrics_to_log["valid_loss"] = valid_loss
                    log_epoch_metrics(model_idx, epoch, metrics_to_log)
                    
                    if epoch_callback is not None and X_valid is not None:
                        execute_epoch_pruning(epoch_callback, epoch, valid_preds, y_valid)
                        
                    if valid_loss is not None:
                        if valid_loss < best_valid_loss:
                            best_valid_loss = valid_loss
                            best_model_state = copy.deepcopy(self.model)
                            wait = 0
                        else:
                            wait += 1
                            if early_stopping_rounds > 0 and wait >= early_stopping_rounds:
                                print(f"Early stopping triggered at iteration {epoch}. Best Valid Loss: {best_valid_loss:.6f}")
                                break
                                
            if best_model_state is not None:
                self.model = best_model_state
        else:
            self.model.fit(X_train_arr, y_train, **fit_kwargs)

        self._log_fit_diagnostics(model_idx)
        self._log_feature_importance(model_idx, X_train_df.columns.tolist())
        self._create_feature_importance_df(X_train_df.columns.tolist())

    def _coef_importance(self):
        if self.model is None or not hasattr(self.model, "coef_"):
            return None

        coef = np.asarray(self.model.coef_)
        if coef.ndim == 1:
            return np.abs(coef)

        return np.mean(np.abs(coef), axis=0)

    def _log_fit_diagnostics(self, model_idx):
        if self.model is None or not mlflow.active_run():
            return

        metrics = {}
        if hasattr(self.model, "n_iter_"):
            n_iter = self.model.n_iter_
            if np.isscalar(n_iter):
                metrics[f"model{model_idx}_n_iter"] = float(n_iter)
            else:
                metrics[f"model{model_idx}_n_iter_mean"] = float(np.mean(n_iter))
        if hasattr(self.model, "dual_gap_"):
            metrics[f"model{model_idx}_dual_gap"] = float(self.model.dual_gap_)

        if metrics:
            mlflow.log_metrics(metrics)

    def _log_feature_importance(self, model_idx, feature_names):
        importance = self._coef_importance()
        if importance is None:
            return

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values(by="importance", ascending=False)

        top_n = min(30, len(importance_df))
        plot_df = importance_df.head(top_n)

        plt.figure(figsize=(10, 8))
        plt.barh(plot_df["feature"], plot_df["importance"])
        plt.xlabel("|Coefficient|")
        plt.title(f"ElasticNet Feature Importance (Model {model_idx})")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        temp_path = f"elasticnet_feature_importance_m{model_idx}.png"
        plt.savefig(temp_path)
        plt.close()

        csv_path = f"elasticnet_feature_importance_m{model_idx}.csv"
        importance_df.to_csv(csv_path, index=False)

        if mlflow.active_run():
            mlflow.log_artifact(temp_path, artifact_path="plots/importance")
            mlflow.log_artifact(csv_path, artifact_path="importance_data")

        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(csv_path):
            os.remove(csv_path)

    def _create_feature_importance_df(self, feature_names):
        importance = self._coef_importance()
        if importance is None:
            self.feature_importances_ = None
            return

        self.feature_importances_ = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values(by="importance", ascending=False)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        Xv, _ = self._to_model_array(X)

        if self.task_type == "classification":
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(Xv)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    classes_ = getattr(self.model, "classes_", None)
                    if classes_ is not None and len(classes_) == 2:
                        # 可能ならクラス1の確率を返す。1 が存在しない場合は大きい方のクラスを陽性扱い。
                        if 1 in classes_:
                            pos_idx = int(np.where(classes_ == 1)[0][0])
                        else:
                            pos_idx = int(np.argmax(classes_))
                        return proba[:, pos_idx].astype(float).ravel()
                    return proba[:, 1].astype(float).ravel()
                return proba.ravel().astype(float)

            if hasattr(self.model, "decision_function"):
                score = self.model.decision_function(Xv)
                score = 1.0 / (1.0 + np.exp(-score))
                return np.asarray(score).astype(float).ravel()

        preds = self.model.predict(Xv)
        return np.asarray(preds).astype(float).ravel()
