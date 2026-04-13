import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from sklearn.linear_model import ElasticNet, SGDClassifier, QuantileRegressor

from .base import BaseModelWrapper


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
            )

        raise ValueError(f"Unsupported task_type: {self.task_type}")

    def _ensure_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    def _to_model_array(self, X):
        X_df = self._ensure_dataframe(X)
        # LabelEncoder 済み int 列 + float 列の混在でも、常に明示的に数値配列へ変換する
        return X_df.to_numpy(dtype=np.float32, copy=False), X_df

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0, epoch_callback=None):
        del X_valid, y_valid, epoch_callback  # 互換引数。ElasticNet 自体では未使用。

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
