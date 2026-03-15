import os
from typing import Dict, List
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from .base import BasePreprocessor


class GANDALFPreprocessor(BasePreprocessor):
    """
    GANDALF用の前処理クラス。

    設計方針:
    - GANDALF本体は「数値入力」を前提にするため、カテゴリ列はOne-Hot化して
      モデル側をシンプルな dense tabular network に保つ
    - 数値列は median 補完 + scaler (default: RobustScaler)
    - train.py の現行インターフェースに合わせ、memmap + row_indices/col_indices に対応
    - train.py の `hasattr(preprocessor, 'cat_idx')` に合わせるため、
      互換属性 `cat_idx`, `cat_dims` を持たせる（GANDALF本実装では未使用）
    """

    def __init__(
        self,
        save_dir,
        feature_cols=None,
        cat_cols=None,
        scaler_type: str = "robust",
        clip_value: float = None,
    ):
        super().__init__(save_dir)
        self.feature_cols = list(feature_cols) if feature_cols else []
        self.cat_cols = list(cat_cols) if cat_cols else []
        self.scaler_type = (scaler_type or "robust").lower()
        self.clip_value = clip_value

        self.encoders: Dict[str, LabelEncoder] = {}
        self.num_cols: List[str] = []
        self.valid_cat_cols: List[str] = []
        self.output_cols: List[str] = []
        self.onehot_cols: Dict[str, List[str]] = {}

        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "none":
            self.scaler = None
        else:
            self.scaler = RobustScaler()

        # train.py互換用（現行コードの `cat_idx` チェック対策）
        self.cat_idx = []
        self.cat_dims = []

    def _ensure_dataframe(self, data, row_indices=None, col_indices=None) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            missing = [c for c in self.feature_cols if c not in df.columns]
            for col in missing:
                df[col] = np.nan
            return df[self.feature_cols].copy()

        if row_indices is None:
            extracted = data[:, col_indices] if col_indices is not None else data
        else:
            if col_indices is not None:
                extracted = data[row_indices][:, col_indices]
            else:
                extracted = data[row_indices]
        return pd.DataFrame(extracted, columns=self.feature_cols)

    def fit(self, data):
        df = self._ensure_dataframe(data)
        self.valid_cat_cols = [c for c in self.cat_cols if c in df.columns]
        self.num_cols = [c for c in self.feature_cols if c not in self.valid_cat_cols]

        # カテゴリ列の学習
        self.encoders = {}
        self.onehot_cols = {}
        for col in self.valid_cat_cols:
            le = LabelEncoder()
            ser = df[col].fillna("MISSING").astype(str)
            le.fit(ser)
            self.encoders[col] = le
            self.onehot_cols[col] = [f"{col}__oh_{i}" for i in range(len(le.classes_))]

        # 数値列の補完 + スケーラ学習
        if self.num_cols:
            num_df = df[self.num_cols].replace([np.inf, -np.inf], np.nan)
            self.imputer.fit(num_df)
            num_array = self.imputer.transform(num_df)
            if self.scaler is not None:
                self.scaler.fit(num_array)

        self.output_cols = list(self.num_cols)
        for col in self.valid_cat_cols:
            self.output_cols.extend(self.onehot_cols[col])

        self.is_fitted = True
        print(
            f"GANDALF Preprocessor fitted. "
            f"num_cols={len(self.num_cols)}, cat_cols={len(self.valid_cat_cols)}, output_dim={len(self.output_cols)}"
        )

    def transform(self, data, row_indices=None, col_indices=None):
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform.")

        df = self._ensure_dataframe(data, row_indices=row_indices, col_indices=col_indices)
        parts = []
        part_cols = []

        # 数値列
        if self.num_cols:
            num_df = df[self.num_cols].replace([np.inf, -np.inf], np.nan)
            target_cols = self.imputer.feature_names_in_.tolist()
            num_array = self.imputer.transform(num_df[target_cols])
            if self.scaler is not None:
                num_array = self.scaler.transform(num_array)
            if self.clip_value is not None:
                num_array = np.clip(num_array, -float(self.clip_value), float(self.clip_value))
            num_array = num_array.astype(np.float32, copy=False)
            parts.append(num_array)
            part_cols.extend(self.num_cols)

        # カテゴリ列 -> One-Hot
        n_rows = len(df)
        for col in self.valid_cat_cols:
            le = self.encoders[col]
            ser = df[col].fillna("MISSING").astype(str)
            mapping = {label: idx for idx, label in enumerate(le.classes_)}
            default_idx = mapping.get("MISSING", 0)
            encoded = ser.map(mapping).fillna(default_idx).astype(int).to_numpy()
            oh = np.zeros((n_rows, len(le.classes_)), dtype=np.float32)
            oh[np.arange(n_rows), encoded] = 1.0
            parts.append(oh)
            part_cols.extend(self.onehot_cols[col])

        if parts:
            X = np.concatenate(parts, axis=1)
        else:
            X = np.empty((len(df), 0), dtype=np.float32)

        return pd.DataFrame(X, columns=part_cols, index=df.index)

    def save(self, filename="scaler.joblib"):
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted yet.")
        state = {
            "feature_cols": self.feature_cols,
            "cat_cols": self.cat_cols,
            "valid_cat_cols": self.valid_cat_cols,
            "num_cols": self.num_cols,
            "output_cols": self.output_cols,
            "onehot_cols": self.onehot_cols,
            "encoders": self.encoders,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "scaler_type": self.scaler_type,
            "clip_value": self.clip_value,
        }
        path = os.path.join(self.save_dir, filename)
        joblib.dump(state, path)
        print(f"GANDALF Preprocessor saved to {path}")

    def load(self, filename="scaler.joblib"):
        load_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Preprocessor state file not found: {load_path}")
        state = joblib.load(load_path)
        self.feature_cols = state["feature_cols"]
        self.cat_cols = state["cat_cols"]
        self.valid_cat_cols = state["valid_cat_cols"]
        self.num_cols = state["num_cols"]
        self.output_cols = state["output_cols"]
        self.onehot_cols = state["onehot_cols"]
        self.encoders = state["encoders"]
        self.imputer = state["imputer"]
        self.scaler = state["scaler"]
        self.scaler_type = state["scaler_type"]
        self.clip_value = state["clip_value"]

        # 互換属性
        self.cat_idx = []
        self.cat_dims = []

        self.is_fitted = True
        print(f"GANDALF Preprocessor loaded from {load_path}")
