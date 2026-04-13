import os
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from .base import BasePreprocessor


class ElasticNetPreprocessor(BasePreprocessor):
    """
    ElasticNet / 線形モデル向けの前処理クラス。

    設計方針:
    - StandardScaler による標準化は行わない
    - 数値列は Inf を NaN に変換した上で median 補完
    - カテゴリ列は LabelEncoder で整数化
    - 未知カテゴリは列ごとに専用 ID を確保して割り当てる
    - 学習済み encoder / mapping / imputer は save/load で永続化する

    注意:
    - train.py の呼び出し仕様に合わせ、memmap + row_indices / col_indices に対応
    - fit() にはカテゴリ列のエンコード辞書を学習する目的でサンプルデータを渡す想定
    - 欠損値は "MISSING" として既知カテゴリ扱いにする
    - 未知カテゴリは unknown_id_map_[col] に保存した専用 ID にマップする
    """

    def __init__(self, save_dir, feature_cols=None, cat_cols=None, **kwargs):
        super().__init__(save_dir)
        self.feature_cols = feature_cols if feature_cols else []
        self.cat_cols = cat_cols if cat_cols else []

        # 学習済みオブジェクト
        self.encoders = {}           # col -> LabelEncoder
        self.label_maps_ = {}        # col -> {label: int_id}
        self.unknown_id_map_ = {}    # col -> unknown 専用 int_id
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)

        # 学習済みメタ情報
        self.fitted_num_cols_ = []
        self.valid_cat_cols_ = []

    def _to_dataframe(self, data, row_indices=None, col_indices=None):
        if isinstance(data, pd.DataFrame):
            if self.feature_cols:
                return data[self.feature_cols].copy()
            return data.copy()

        if row_indices is None:
            extracted = data
        else:
            extracted = data[row_indices]

        if col_indices is not None:
            extracted = extracted[:, col_indices]

        return pd.DataFrame(extracted, columns=self.feature_cols)

    def fit(self, data):
        df = self._to_dataframe(data)

        if self.feature_cols:
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[self.feature_cols]
        else:
            self.feature_cols = df.columns.tolist()

        self.valid_cat_cols_ = [c for c in self.cat_cols if c in df.columns]
        self.encoders = {}
        self.label_maps_ = {}
        self.unknown_id_map_ = {}

        for col in self.valid_cat_cols_:
            le = LabelEncoder()
            ser = df[col].fillna("MISSING").astype(str)
            le.fit(ser)

            label_map = {label: i for i, label in enumerate(le.classes_)}
            unknown_id = len(le.classes_)

            self.encoders[col] = le
            self.label_maps_[col] = label_map
            self.unknown_id_map_[col] = unknown_id

        num_cols = [c for c in df.columns if c not in self.valid_cat_cols_]
        self.fitted_num_cols_ = num_cols

        if self.fitted_num_cols_:
            num_df = df[self.fitted_num_cols_].apply(pd.to_numeric, errors="coerce")
            num_df = num_df.replace([np.inf, -np.inf], np.nan)
            self.imputer.fit(num_df)

        self.is_fitted = True
        print(
            f"ElasticNet Preprocessor fitted. "
            f"n_features={len(self.feature_cols)}, "
            f"n_cat_cols={len(self.valid_cat_cols_)}, "
            f"n_num_cols={len(self.fitted_num_cols_)}"
        )
        if self.valid_cat_cols_:
            summary = ", ".join(
                [f"{col}:unknown_id={self.unknown_id_map_[col]}" for col in self.valid_cat_cols_]
            )
            print(f"Categorical unknown IDs -> {summary}")

    def transform(self, data, row_indices=None, col_indices=None):
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform().")

        df = self._to_dataframe(data, row_indices=row_indices, col_indices=col_indices)

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_cols].copy()

        # 数値列: Inf -> NaN, median 補完
        if self.fitted_num_cols_:
            num_df = df[self.fitted_num_cols_].apply(pd.to_numeric, errors="coerce")
            num_df = num_df.replace([np.inf, -np.inf], np.nan)
            df[self.fitted_num_cols_] = self.imputer.transform(num_df).astype(np.float32)

        # カテゴリ列: LabelEncoder 学習済みクラスを使って整数化
        # 未知カテゴリは列ごとの専用 ID にマップする
        for col in self.valid_cat_cols_:
            if col not in self.label_maps_:
                continue

            ser = df[col].fillna("MISSING").astype(str)
            label_map = self.label_maps_[col]
            unknown_id = self.unknown_id_map_[col]
            df[col] = ser.map(label_map).fillna(unknown_id).astype(np.int32)

        return df

    def save(self, filename="scaler.joblib"):
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted yet.")

        state = {
            "feature_cols": self.feature_cols,
            "cat_cols": self.cat_cols,
            "valid_cat_cols_": self.valid_cat_cols_,
            "fitted_num_cols_": self.fitted_num_cols_,
            "encoders": self.encoders,
            "label_maps_": self.label_maps_,
            "unknown_id_map_": self.unknown_id_map_,
            "imputer": self.imputer,
        }
        path = os.path.join(self.save_dir, filename)
        joblib.dump(state, path)
        print(f"ElasticNet Preprocessor saved to {path}")

    def load(self, filename="scaler.joblib"):
        load_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Preprocessor state file not found: {load_path}")

        state = joblib.load(load_path)
        self.feature_cols = state["feature_cols"]
        self.cat_cols = state.get("cat_cols", [])
        self.valid_cat_cols_ = state.get("valid_cat_cols_", [])
        self.fitted_num_cols_ = state.get("fitted_num_cols_", [])
        self.encoders = state.get("encoders", {})
        self.label_maps_ = state.get("label_maps_", {})
        self.unknown_id_map_ = state.get("unknown_id_map_", {})
        self.imputer = state["imputer"]
        self.is_fitted = True
        print(f"ElasticNet Preprocessor loaded from {load_path}")
