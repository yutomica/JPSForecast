
import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

from .base import BasePreprocessor


class FTTransformerPreprocessor(BasePreprocessor):
    """
    FT-Transformer 用 tabular preprocessor

    train.py 互換インターフェース:
      - fit(data)
      - transform(data, row_indices=None, col_indices=None)

    返却形式:
      - np.ndarray, shape = [N, F], dtype=float32

    重要仕様:
      - cat_cols はラベルエンコードし、未知カテゴリは 0 に割り当てる
      - fit 後に cat_idx / cat_dims を公開するため、
        train.py はこれを model hparams に自動注入できる
      - 数値列は欠損補完し、必要に応じて scaler を適用する
    """

    def __init__(
        self,
        save_dir,
        feature_cols=None,
        cat_cols=None,
        numeric_impute_strategy="median",
        scaler_type="none",      # "none" | "robust" | "standard"
        clip_value=10.0,
    ):
        super().__init__(save_dir)
        self.feature_cols = feature_cols if feature_cols else []
        self.cat_cols = cat_cols if cat_cols else []
        self.numeric_impute_strategy = numeric_impute_strategy
        self.scaler_type = scaler_type
        self.clip_value = float(clip_value)

        self.num_cols: List[str] = []
        self.valid_cat_cols_: List[str] = []
        self.cat_maps: Dict[str, Dict[str, int]] = {}
        self.imputer = None
        self.scaler = None
        self.cat_idx: List[int] = []
        self.cat_dims: List[int] = []
        self.is_fitted = False

    def _build_scaler(self):
        if self.scaler_type == "none":
            return None
        if self.scaler_type == "robust":
            return RobustScaler()
        if self.scaler_type == "standard":
            return StandardScaler()
        raise ValueError(f"Unknown scaler_type: {self.scaler_type}")

    def _to_dataframe(self, data, row_indices=None, col_indices=None):
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            if self.feature_cols:
                df = df[self.feature_cols]
            return df

        if row_indices is None:
            extracted = data
        else:
            extracted = data[row_indices]

        if col_indices is not None:
            extracted = extracted[:, col_indices]

        return pd.DataFrame(extracted, columns=self.feature_cols)

    def fit(self, data):
        df = self._to_dataframe(data)

        if not self.feature_cols:
            self.feature_cols = df.columns.tolist()

        df = df[self.feature_cols].copy()
        self.valid_cat_cols_ = [c for c in self.cat_cols if c in df.columns]
        self.num_cols = [c for c in self.feature_cols if c not in self.valid_cat_cols_]

        self.cat_maps = {}
        self.cat_idx = [self.feature_cols.index(c) for c in self.valid_cat_cols_]
        self.cat_dims = []

        for col in self.valid_cat_cols_:
            ser = df[col].fillna("MISSING").astype(str)
            uniq = pd.Index(sorted(ser.unique()))
            mapping = {v: i + 1 for i, v in enumerate(uniq)}  # 0 is reserved for unknown category
            self.cat_maps[col] = mapping
            self.cat_dims.append(len(mapping) + 1)  # include unknown bucket

        if self.num_cols:
            x_num = df[self.num_cols].replace([np.inf, -np.inf], np.nan)
            self.imputer = SimpleImputer(
                strategy=self.numeric_impute_strategy,
                keep_empty_features=True,
            )
            x_num_imp = self.imputer.fit_transform(x_num)

            self.scaler = self._build_scaler()
            if self.scaler is not None:
                self.scaler.fit(x_num_imp)
        else:
            self.imputer = None
            self.scaler = None

        self.is_fitted = True
        print(
            f"FTTransformerPreprocessor fitted. "
            f"num_cols={len(self.num_cols)}, cat_cols={len(self.valid_cat_cols_)}, scaler={self.scaler_type}"
        )

    def _transform_2d(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].copy()

        for col in self.valid_cat_cols_:
            mapping = self.cat_maps.get(col, {})
            ser = X[col].fillna("MISSING").astype(str)
            X.loc[:, col] = ser.map(mapping).fillna(0).astype(np.float32)

        if self.num_cols:
            X_num = X[self.num_cols].replace([np.inf, -np.inf], np.nan)
            X_num_imp = self.imputer.transform(X_num)
            if self.scaler is not None:
                X_num_imp = self.scaler.transform(X_num_imp)
            X_num_imp = np.clip(X_num_imp, -self.clip_value, self.clip_value)
            X.loc[:, self.num_cols] = X_num_imp.astype(np.float32)

        arr = X.to_numpy(dtype=np.float32, copy=False)
        return np.ascontiguousarray(arr)

    def transform(self, data, row_indices=None, col_indices=None):
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform().")
        df = self._to_dataframe(data, row_indices=row_indices, col_indices=col_indices)
        return self._transform_2d(df)

    def save(self, filename="ft_transformer_preprocessor.joblib"):
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted yet.")

        state = {
            "feature_cols": self.feature_cols,
            "cat_cols": self.cat_cols,
            "numeric_impute_strategy": self.numeric_impute_strategy,
            "scaler_type": self.scaler_type,
            "clip_value": self.clip_value,
            "num_cols": self.num_cols,
            "valid_cat_cols_": self.valid_cat_cols_,
            "cat_maps": self.cat_maps,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "cat_idx": self.cat_idx,
            "cat_dims": self.cat_dims,
        }

        path = os.path.join(self.save_dir, filename)
        joblib.dump(state, path)
        print(f"FTTransformerPreprocessor saved to {path}")

    def load(self, filename="ft_transformer_preprocessor.joblib"):
        load_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Preprocessor state file not found: {load_path}")

        state = joblib.load(load_path)
        self.feature_cols = state["feature_cols"]
        self.cat_cols = state["cat_cols"]
        self.numeric_impute_strategy = state["numeric_impute_strategy"]
        self.scaler_type = state["scaler_type"]
        self.clip_value = state["clip_value"]
        self.num_cols = state["num_cols"]
        self.valid_cat_cols_ = state["valid_cat_cols_"]
        self.cat_maps = state["cat_maps"]
        self.imputer = state["imputer"]
        self.scaler = state["scaler"]
        self.cat_idx = state["cat_idx"]
        self.cat_dims = state["cat_dims"]
        self.is_fitted = True
        print(f"FTTransformerPreprocessor loaded from {load_path}")
