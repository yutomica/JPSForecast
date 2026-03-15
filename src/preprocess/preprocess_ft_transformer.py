import os
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from .base import BasePreprocessor

class FTTransformerPreprocessor(BasePreprocessor):
    """
    FT-Transformer用の前処理クラス
    - train.py から TabNetPreprocessor と同じ形で呼べるI/Fに合わせる
    - transform() は DataFrame を返す
    - cat_idx / cat_dims を保持する
    """

    def __init__(
        self,
        save_dir,
        feature_cols=None,
        cat_cols=None,
        numeric_impute_strategy="median",
        scaler_type="robust",   # "robust" | "standard" | "none"
    ):
        super().__init__(save_dir)
        self.feature_cols = feature_cols if feature_cols else []
        self.cat_cols = cat_cols if cat_cols else []

        self.numeric_impute_strategy = numeric_impute_strategy
        self.scaler_type = scaler_type

        # fit後に確定
        self.num_cols = []
        self.cat_idx = []   # train.py が hasattr(preprocessor, 'cat_idx') を見ているため、単数名で持つ
        self.cat_dims = []

        self.cat_maps = {}  # col -> dict(label -> int), 0 は unknown/reserved
        self.imputer = None
        self.scaler = None

    def _build_scaler(self):
        if self.scaler_type == "robust":
            return RobustScaler()
        if self.scaler_type == "standard":
            return StandardScaler()
        if self.scaler_type == "none":
            return None
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

        # 列定義
        valid_cat_cols = [c for c in self.cat_cols if c in df.columns]
        self.num_cols = [c for c in self.feature_cols if c not in valid_cat_cols]

        # カテゴリ列メタ情報
        self.cat_idx = []
        self.cat_dims = []
        self.cat_maps = {}

        for col in valid_cat_cols:
            self.cat_idx.append(df.columns.get_loc(col))

            ser = df[col].fillna("MISSING").astype(str)
            uniq = pd.Index(sorted(ser.unique()))

            # 0 を unknown 用に予約し、既知カテゴリは 1..K
            mapping = {v: i + 1 for i, v in enumerate(uniq)}
            self.cat_maps[col] = mapping
            self.cat_dims.append(len(mapping) + 1)

        # 数値列の imputer / scaler
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
            f"num_cols={len(self.num_cols)}, cat_cols={len(valid_cat_cols)}"
        )

    def transform(self, data, row_indices=None, col_indices=None):
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform().")

        df = self._to_dataframe(data, row_indices=row_indices, col_indices=col_indices)
        X = df[self.feature_cols].copy()

        # 欠落列があれば追加
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = np.nan

        X = X[self.feature_cols].copy()

        # 数値列処理
        if self.num_cols:
            X_num = X[self.num_cols].replace([np.inf, -np.inf], np.nan)
            X_num_imp = self.imputer.transform(X_num)

            if self.scaler is not None:
                X_num_imp = self.scaler.transform(X_num_imp)

            X.loc[:, self.num_cols] = X_num_imp.astype(np.float32)

        # カテゴリ列処理
        valid_cat_cols = [c for c in self.cat_cols if c in X.columns]
        for col in valid_cat_cols:
            mapping = self.cat_maps.get(col, {})
            ser = X[col].fillna("MISSING").astype(str)

            # 未知カテゴリは 0
            X.loc[:, col] = ser.map(mapping).fillna(0).astype(np.int64)

        return X

    def save(self, filename="ft_transformer_preprocessor.joblib"):
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted yet.")

        state = {
            "feature_cols": self.feature_cols,
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "cat_idx": self.cat_idx,
            "cat_dims": self.cat_dims,
            "cat_maps": self.cat_maps,
            "numeric_impute_strategy": self.numeric_impute_strategy,
            "scaler_type": self.scaler_type,
            "imputer": self.imputer,
            "scaler": self.scaler,
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
        self.num_cols = state["num_cols"]
        self.cat_idx = state["cat_idx"]
        self.cat_dims = state["cat_dims"]
        self.cat_maps = state["cat_maps"]
        self.numeric_impute_strategy = state["numeric_impute_strategy"]
        self.scaler_type = state["scaler_type"]
        self.imputer = state["imputer"]
        self.scaler = state["scaler"]

        self.is_fitted = True
        print(f"FTTransformerPreprocessor loaded from {load_path}")