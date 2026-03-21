import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .base import BasePreprocessor


class NBeatsPreprocessor(BasePreprocessor):
    """
    N-BEATS用の時系列前処理クラス。

    想定:
    - train.py から timeseries 用 preprocessor として呼ばれる
    - 入力は DataFrame または memmap/ndarray
    - 出力は (n_samples, window_size, n_features_per_timestep) の float32 配列

    特徴量名から time axis を推定する。代表的な suffix を優先的に認識し、
    認識できない場合は feature_cols の並び順を用いて window_size 単位で reshape する。
    """

    TIME_PATTERNS = [
        re.compile(r"^(?P<base>.+?)_lag(?P<step>\d+)$"),
        re.compile(r"^(?P<base>.+?)_t(?P<step>\d+)$"),
        re.compile(r"^(?P<base>.+?)\[(?P<step>\d+)\]$"),
        re.compile(r"^(?P<base>.+?)__(?P<step>\d+)$"),
        re.compile(r"^(?P<base>.+?)_(?P<step>\d+)$"),
    ]

    def __init__(self, save_dir, feature_cols=None, cat_cols=None, window_size=60):
        super().__init__(save_dir)
        self.feature_cols = feature_cols if feature_cols else []
        self.cat_cols = cat_cols if cat_cols else []
        self.window_size = int(window_size)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        self.num_features_per_timestep = None
        self.sequence_feature_names = None
        self.reshape_mode = None

    def fit(self, data):
        df = pd.DataFrame(data, columns=self.feature_cols).copy()
        df = self._sanitize_dataframe(df)
        self.imputer.fit(df)
        imputed = self.imputer.transform(df)
        self.scaler.fit(imputed)
        self.sequence_feature_names, self.reshape_mode = self._infer_sequence_layout(self.feature_cols)
        self.num_features_per_timestep = len(self.sequence_feature_names)
        self.is_fitted = True
        print(
            f"NBeats Preprocessor fitted. "
            f"window_size={self.window_size}, features_per_timestep={self.num_features_per_timestep}, mode={self.reshape_mode}"
        )

    def transform(self, data, row_indices=None, col_indices=None):
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted.")

        if isinstance(data, pd.DataFrame):
            df = data[self.feature_cols].copy()
        else:
            if row_indices is None:
                raise ValueError("row_indices must be provided when transforming ndarray/memmap input.")
            extracted = data[row_indices][:, col_indices] if col_indices is not None else data[row_indices]
            df = pd.DataFrame(extracted, columns=self.feature_cols)

        df = self._sanitize_dataframe(df)
        x = self.imputer.transform(df)
        x = self.scaler.transform(x)
        x = x.astype(np.float32)
        x = self._reshape_to_sequence(x)
        return x

    def save(self, filename="scaler.joblib"):
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted yet.")
        state = {
            "feature_cols": self.feature_cols,
            "cat_cols": self.cat_cols,
            "window_size": self.window_size,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "num_features_per_timestep": self.num_features_per_timestep,
            "sequence_feature_names": self.sequence_feature_names,
            "reshape_mode": self.reshape_mode,
        }
        path = os.path.join(self.save_dir, filename)
        joblib.dump(state, path)
        print(f"NBeats Preprocessor saved to {path}")

    def load(self, filename="scaler.joblib"):
        load_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Preprocessor state file not found: {load_path}")
        state = joblib.load(load_path)
        self.feature_cols = state["feature_cols"]
        self.cat_cols = state["cat_cols"]
        self.window_size = state["window_size"]
        self.imputer = state["imputer"]
        self.scaler = state["scaler"]
        self.num_features_per_timestep = state["num_features_per_timestep"]
        self.sequence_feature_names = state["sequence_feature_names"]
        self.reshape_mode = state["reshape_mode"]
        self.is_fitted = True
        print(f"NBeats Preprocessor loaded from {load_path}")

    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_cols]
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def _infer_sequence_layout(self, feature_cols: List[str]) -> Tuple[List[str], str]:
        parsed = []
        used = set()
        for col in feature_cols:
            hit = False
            for pattern in self.TIME_PATTERNS:
                m = pattern.match(col)
                if m:
                    base = m.group("base")
                    step = int(m.group("step"))
                    parsed.append((col, base, step))
                    used.add(col)
                    hit = True
                    break
            if not hit:
                continue

        if parsed:
            by_base = defaultdict(dict)
            for col, base, step in parsed:
                by_base[base][step] = col
            complete_bases = [
                base for base, step_map in by_base.items()
                if all(step in step_map for step in range(self.window_size))
            ]
            if complete_bases:
                ordered_cols = []
                for step in range(self.window_size):
                    for base in complete_bases:
                        ordered_cols.append(by_base[base][step])
                self.feature_cols = ordered_cols
                return complete_bases, "parsed_suffix"

        if len(feature_cols) % self.window_size != 0:
            raise ValueError(
                f"feature count ({len(feature_cols)}) is not divisible by window_size ({self.window_size}), "
                "and time suffix parsing did not find a complete sequence layout."
            )

        num_features_per_timestep = len(feature_cols) // self.window_size
        sequence_feature_names = [f"feature_{i}" for i in range(num_features_per_timestep)]
        return sequence_feature_names, "contiguous_chunk"

    def _reshape_to_sequence(self, x: np.ndarray) -> np.ndarray:
        if self.reshape_mode == "parsed_suffix":
            n = x.shape[0]
            f = self.num_features_per_timestep
            return x.reshape(n, self.window_size, f)

        n = x.shape[0]
        return x.reshape(n, self.window_size, self.num_features_per_timestep)
