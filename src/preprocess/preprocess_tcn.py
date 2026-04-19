import os
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from .base import BasePreprocessor


class TCNPreprocessor(BasePreprocessor):
    """
    TCN用の時系列プリプロセッサ。

    train.py互換インターフェース:
      - fit(data)
      - transform(data, row_indices=None, col_indices=None)

    返却形式:
      - np.ndarray, shape = [N, window_size, F]

    重要な前提:
      - row_indices の過去方向に lookback を作るため、元データは「同一銘柄ごとに時系列順」で
        並んでいることが望ましい。
      - boundary_col が指定されている場合、異なる entity にまたがる過去行は自動的に無効化する。
    """

    COMMON_BOUNDARY_NAMES = ["scode", "code", "ticker", "symbol", "asset_id", "issue_code"]

    def __init__(
        self,
        save_dir,
        feature_cols=None,
        cat_cols=None,
        window_size=20,
        numeric_impute_strategy="median",
        scaler_type="robust",   # "robust" | "standard" | "none"
        pad_mode="zero",        # "zero" | "edge"
        boundary_col=None,
        clip_value=10.0,
    ):
        super().__init__(save_dir)
        self.feature_cols = feature_cols if feature_cols else []
        self.cat_cols = cat_cols if cat_cols else []
        self.window_size = int(window_size)
        self.numeric_impute_strategy = numeric_impute_strategy
        self.scaler_type = scaler_type
        self.pad_mode = pad_mode
        self.boundary_col = boundary_col
        self.clip_value = clip_value

        self.num_cols = []
        self.cat_maps = {}
        self.imputer = None
        self.scaler = None
        self.boundary_col_idx_ = None
        self.is_fitted = False

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

    def _resolve_boundary_col(self):
        if self.boundary_col is not None and self.boundary_col in self.feature_cols:
            self.boundary_col_idx_ = self.feature_cols.index(self.boundary_col)
            return

        for name in self.COMMON_BOUNDARY_NAMES:
            if name in self.feature_cols:
                self.boundary_col = name
                self.boundary_col_idx_ = self.feature_cols.index(name)
                return

        self.boundary_col = None
        self.boundary_col_idx_ = None

    def fit(self, data):
        df = self._to_dataframe(data)

        if not self.feature_cols:
            self.feature_cols = df.columns.tolist()

        df = df[self.feature_cols].copy()
        self._resolve_boundary_col()

        valid_cat_cols = [c for c in self.cat_cols if c in df.columns]
        self.num_cols = [c for c in self.feature_cols if c not in valid_cat_cols]
        self.cat_maps = {}

        for col in valid_cat_cols:
            ser = df[col].fillna("MISSING").astype(str)
            uniq = pd.Index(sorted(ser.unique()))
            self.cat_maps[col] = {v: i + 1 for i, v in enumerate(uniq)}  # 0 reserved for unknown

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
            f"TCNPreprocessor fitted. window_size={self.window_size}, "
            f"num_cols={len(self.num_cols)}, cat_cols={len(valid_cat_cols)}, boundary_col={self.boundary_col}"
        )

    def _transform_2d(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].copy()

        valid_cat_cols = [c for c in self.cat_cols if c in X.columns]
        for col in valid_cat_cols:
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

        return X.to_numpy(dtype=np.float32, copy=False)

    def transform(self, data, row_indices=None, col_indices=None):
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform().")

        # --- 推論パス (row_indices is None): 変更なし ---
        # メモリに収まる小さなDataFrameが渡されることを想定
        if row_indices is None:
            df = self._to_dataframe(data, row_indices=None, col_indices=col_indices)
            arr_2d = self._transform_2d(df)
            n, f = arr_2d.shape
            if n < self.window_size:
                pad = np.zeros((self.window_size - n, f), dtype=np.float32)
                arr_2d = np.vstack([pad, arr_2d])
                n = arr_2d.shape[0]
            idx = np.arange(self.window_size - 1, n)
            return self._build_sequence_from_2d(arr_2d, idx)

        # --- 学習パス (row_indices is not None): チャンク処理でメモリ効率化 ---
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.ndim != 1:
            raise ValueError("row_indices must be 1-dimensional.")

        chunk_size = 10000  # 一度に処理するサンプル数。メモリ使用量に応じて調整。
        results = []

        for i in range(0, len(row_indices), chunk_size):
            chunk_row_indices = row_indices[i:i + chunk_size]

            history_offsets = np.arange(self.window_size - 1, -1, -1, dtype=np.int64)
            idx_mat = chunk_row_indices[:, None] - history_offsets[None, :]
            valid_mask = idx_mat >= 0
            clipped_idx_mat = np.clip(idx_mat, 0, None)

            if col_indices is None:
                extracted = data[clipped_idx_mat]
            else:
                extracted = data[clipped_idx_mat][:, :, col_indices]

            flat_df = pd.DataFrame(
                extracted.reshape(-1, extracted.shape[-1]),
                columns=self.feature_cols,
            )
            flat_arr = self._transform_2d(flat_df)
            seq = flat_arr.reshape(len(chunk_row_indices), self.window_size, len(self.feature_cols)).astype(np.float32)

            # entity boundary protection
            if self.boundary_col_idx_ is not None:
                try:
                    local_idx = list(col_indices).index(self.boundary_col_idx_) if col_indices is not None else self.boundary_col_idx_
                    boundary_source = np.asarray(extracted[:, :, local_idx])
                    target_boundary = boundary_source[:, -1][:, None]
                    valid_mask &= (boundary_source == target_boundary)
                except (ValueError, IndexError):
                    # 境界列が選択されていない場合はスキップ
                    pass

            # Padding
            if self.pad_mode == "zero":
                seq[~valid_mask] = 0.0
            elif self.pad_mode == "edge":
                for j in range(seq.shape[0]):
                    if np.any(~valid_mask[j]):
                        edge_value = seq[j, -1, :]  # 最新のタイムステップの値
                        seq[j, ~valid_mask[j], :] = edge_value
            else:
                raise ValueError(f"Unknown pad_mode: {self.pad_mode}")

            results.append(seq)

        return np.concatenate(results, axis=0)

    def _build_sequence_from_2d(self, arr_2d: np.ndarray, row_indices: np.ndarray) -> np.ndarray:
        history_offsets = np.arange(self.window_size - 1, -1, -1, dtype=np.int64)
        idx_mat = row_indices[:, None] - history_offsets[None, :]
        valid_mask = idx_mat >= 0
        clipped = np.clip(idx_mat, 0, None)
        seq = arr_2d[clipped].astype(np.float32)
        seq[~valid_mask] = 0.0
        return seq

    def save(self, filename="tcn_preprocessor.joblib"):
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted yet.")

        state = {
            "feature_cols": self.feature_cols,
            "cat_cols": self.cat_cols,
            "window_size": self.window_size,
            "numeric_impute_strategy": self.numeric_impute_strategy,
            "scaler_type": self.scaler_type,
            "pad_mode": self.pad_mode,
            "boundary_col": self.boundary_col,
            "clip_value": self.clip_value,
            "num_cols": self.num_cols,
            "cat_maps": self.cat_maps,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "boundary_col_idx_": self.boundary_col_idx_,
        }

        path = os.path.join(self.save_dir, filename)
        joblib.dump(state, path)
        print(f"TCNPreprocessor saved to {path}")

    def load(self, filename="tcn_preprocessor.joblib"):
        load_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Preprocessor state file not found: {load_path}")

        state = joblib.load(load_path)
        self.feature_cols = state["feature_cols"]
        self.cat_cols = state["cat_cols"]
        self.window_size = state["window_size"]
        self.numeric_impute_strategy = state["numeric_impute_strategy"]
        self.scaler_type = state["scaler_type"]
        self.pad_mode = state["pad_mode"]
        self.boundary_col = state["boundary_col"]
        self.clip_value = state["clip_value"]
        self.num_cols = state["num_cols"]
        self.cat_maps = state["cat_maps"]
        self.imputer = state["imputer"]
        self.scaler = state["scaler"]
        self.boundary_col_idx_ = state["boundary_col_idx_"]
        self.is_fitted = True
        print(f"TCNPreprocessor loaded from {load_path}")
