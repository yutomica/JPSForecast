import os
import hashlib
import joblib
import json
import shutil
import time
import uuid
import tempfile
import numpy as np
import pandas as pd
import zarr
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
        sequence_cache_enabled=False,
        sequence_cache_dir=None,
        sequence_cache_wait_seconds=600,
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
        self.sequence_cache_enabled = self._as_bool(sequence_cache_enabled)
        self.sequence_cache_dir = sequence_cache_dir or os.path.join(
            tempfile.gettempdir(),
            "jps_tcn_sequence_cache",
        )
        self.sequence_cache_wait_seconds = int(sequence_cache_wait_seconds)

        self.num_cols = []
        self.cat_maps = {}
        self.imputer = None
        self.scaler = None
        self.boundary_col_idx_ = None
        self._index_metadata_cache_ = None
        self.is_fitted = False

    @staticmethod
    def _as_bool(value):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

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

    @staticmethod
    def _hash_array(value) -> str | None:
        if value is None:
            return None
        arr = np.asarray(value)
        h = hashlib.sha1()
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(np.ascontiguousarray(arr).view(np.uint8))
        return h.hexdigest()

    def _data_fingerprint(self, data_arr) -> dict:
        filename = getattr(data_arr, "filename", None)
        if filename:
            try:
                stat = os.stat(filename)
                return {
                    "kind": "memmap",
                    "filename": os.path.abspath(filename),
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                    "shape": tuple(data_arr.shape),
                    "dtype": str(data_arr.dtype),
                }
            except OSError:
                pass

        return {
            "kind": type(data_arr).__name__,
            "object_id": id(data_arr),
            "shape": tuple(getattr(data_arr, "shape", ())),
            "dtype": str(getattr(data_arr, "dtype", "")),
        }

    def _load_index_metadata(self, data_arr):
        filename = getattr(data_arr, "filename", None)
        if filename is None:
            raise ValueError(
                "TCN entity boundary protection requires a memmap filename when "
                "the boundary column is not present in feature_cols."
            )

        try:
            data_path = os.path.abspath(os.fspath(filename))
        except TypeError as exc:
            raise ValueError(f"Unable to resolve TCN memmap filename: {filename!r}") from exc
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"TCN memmap file not found: {data_path}")

        metadata_path = os.path.join(os.path.dirname(data_path), "index_meta.parquet")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                "TCN entity boundary metadata not found next to the memmap: "
                f"{metadata_path}"
            )

        try:
            stat = os.stat(metadata_path)
        except OSError as exc:
            raise OSError(f"Unable to stat TCN boundary metadata: {metadata_path}") from exc

        n_rows = int(data_arr.shape[0])
        cache_key = (metadata_path, stat.st_mtime_ns, stat.st_size, n_rows)
        cached = self._index_metadata_cache_
        if cached is not None and cached["key"] == cache_key:
            return cached["scode"], cached["date"], cached["fingerprint"]

        try:
            metadata = pd.read_parquet(metadata_path, columns=["scode", "date"])
        except Exception as exc:
            raise ValueError(
                "Unable to read required TCN boundary metadata columns "
                f"['scode', 'date'] from {metadata_path}"
            ) from exc

        if len(metadata) != n_rows:
            raise ValueError(
                "TCN boundary metadata row count does not match the feature memmap: "
                f"metadata={len(metadata)}, data={n_rows}, path={metadata_path}"
            )
        if metadata["scode"].isna().any():
            raise ValueError(f"TCN boundary metadata contains missing scode values: {metadata_path}")

        scode, _ = pd.factorize(metadata["scode"], sort=False)
        if np.any(scode < 0):
            raise ValueError(f"Unable to encode TCN boundary scode values: {metadata_path}")

        date = pd.to_datetime(metadata["date"], errors="coerce", utc=True)
        if date.isna().any():
            raise ValueError(f"TCN boundary metadata contains invalid date values: {metadata_path}")

        try:
            stat_after = os.stat(metadata_path)
        except OSError as exc:
            raise OSError(f"Unable to restat TCN boundary metadata: {metadata_path}") from exc
        if (stat_after.st_mtime_ns, stat_after.st_size) != (stat.st_mtime_ns, stat.st_size):
            raise RuntimeError(f"TCN boundary metadata changed while being read: {metadata_path}")

        scode_arr = np.asarray(scode, dtype=np.int64)
        date_arr = date.astype("int64").to_numpy(dtype=np.int64, copy=True)
        scode_arr.setflags(write=False)
        date_arr.setflags(write=False)
        fingerprint = {
            "path": metadata_path,
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }
        self._index_metadata_cache_ = {
            "key": cache_key,
            "scode": scode_arr,
            "date": date_arr,
            "fingerprint": fingerprint,
        }
        return scode_arr, date_arr, fingerprint

    def _preprocessor_fingerprint(self) -> dict:
        scaler_payload = {}
        if self.scaler is not None:
            for attr in ("center_", "scale_", "mean_", "var_"):
                if hasattr(self.scaler, attr):
                    scaler_payload[attr] = self._hash_array(getattr(self.scaler, attr))

        return {
            "version": "tcn_sequence_cache_v2",
            "feature_cols": self.feature_cols,
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "cat_maps": self.cat_maps,
            "window_size": self.window_size,
            "numeric_impute_strategy": self.numeric_impute_strategy,
            "scaler_type": self.scaler_type,
            "pad_mode": self.pad_mode,
            "boundary_col": self.boundary_col,
            "boundary_col_idx": self.boundary_col_idx_,
            "clip_value": self.clip_value,
            "imputer_statistics": self._hash_array(
                getattr(self.imputer, "statistics_", None)
            ),
            "scaler": scaler_payload,
        }

    def _sequence_cache_key(self, data_arr, row_indices, col_indices) -> str:
        h = hashlib.sha1()
        payload = {
            "data": self._data_fingerprint(data_arr),
            "preprocessor": self._preprocessor_fingerprint(),
            "index_metadata": (
                None
                if self.boundary_col_idx_ is not None
                else self._load_index_metadata(data_arr)[2]
            ),
            "col_indices": None if col_indices is None else [int(i) for i in col_indices],
            "n_rows": int(len(row_indices)),
        }
        h.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
        h.update(np.ascontiguousarray(row_indices).view(np.uint8))
        return h.hexdigest()

    def _is_cache_ready(self, zarr_dir, expected_shape) -> bool:
        complete_path = f"{zarr_dir}.complete"
        if not os.path.exists(zarr_dir) or not os.path.exists(complete_path):
            return False
        try:
            z = zarr.open(zarr_dir, mode="r")
            return tuple(z.shape) == tuple(expected_shape)
        except Exception:
            return False

    def _build_sequence_zarr(self, zarr_dir, data_arr, row_indices, col_indices):
        metadata_scode = None
        metadata_date = None
        if self.boundary_col_idx_ is None:
            metadata_scode, metadata_date, _ = self._load_index_metadata(data_arr)

        z = zarr.open(
            zarr_dir,
            mode='w',
            shape=(len(row_indices), self.window_size, len(self.feature_cols)),
            chunks=(2048, self.window_size, len(self.feature_cols)),
            dtype='float32'
        )

        chunk_size = 10000
        history_offsets = np.arange(self.window_size - 1, -1, -1, dtype=np.int64)
        for i in range(0, len(row_indices), chunk_size):
            chunk_row_indices = row_indices[i:i + chunk_size]

            idx_mat = chunk_row_indices[:, None] - history_offsets[None, :]
            valid_mask = idx_mat >= 0
            clipped_idx_mat = np.clip(idx_mat, 0, None)

            if col_indices is None:
                extracted = data_arr[clipped_idx_mat]
            else:
                extracted = data_arr[clipped_idx_mat][:, :, col_indices]

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
                except (ValueError, IndexError) as exc:
                    raise ValueError(
                        f"TCN boundary column {self.boundary_col!r} is not available "
                        "in the selected feature data."
                    ) from exc
            else:
                boundary_source = metadata_scode[clipped_idx_mat]
                date_source = metadata_date[clipped_idx_mat]
                target_boundary = metadata_scode[chunk_row_indices][:, None]
                target_date = metadata_date[chunk_row_indices][:, None]
                valid_mask &= boundary_source == target_boundary
                valid_mask &= date_source <= target_date

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

            z[i:i+chunk_size] = seq

    def _get_or_build_sequence_cache(self, data_arr, row_indices, col_indices):
        os.makedirs(self.sequence_cache_dir, exist_ok=True)
        key = self._sequence_cache_key(data_arr, row_indices, col_indices)
        zarr_dir = os.path.join(self.sequence_cache_dir, f"{key}.zarr")
        expected_shape = (len(row_indices), self.window_size, len(self.feature_cols))

        if self._is_cache_ready(zarr_dir, expected_shape):
            print(f"TCN sequence cache hit: {os.path.basename(zarr_dir)}")
            return zarr_dir

        lock_dir = f"{zarr_dir}.lock"
        start = time.time()
        while True:
            try:
                os.mkdir(lock_dir)
                break
            except FileExistsError:
                if self._is_cache_ready(zarr_dir, expected_shape):
                    print(f"TCN sequence cache hit: {os.path.basename(zarr_dir)}")
                    return zarr_dir
                if time.time() - start > self.sequence_cache_wait_seconds:
                    raise TimeoutError(f"Timed out waiting for TCN sequence cache lock: {lock_dir}")
                time.sleep(2)

        tmp_dir = f"{zarr_dir}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            if self._is_cache_ready(zarr_dir, expected_shape):
                return zarr_dir
            if os.path.exists(zarr_dir):
                shutil.rmtree(zarr_dir, ignore_errors=True)
            print(f"TCN sequence cache miss: building {os.path.basename(zarr_dir)}")
            self._build_sequence_zarr(tmp_dir, data_arr, row_indices, col_indices)
            os.replace(tmp_dir, zarr_dir)
            with open(f"{zarr_dir}.complete", "w", encoding="utf-8") as f:
                json.dump({"key": key, "shape": expected_shape}, f)
            return zarr_dir
        finally:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            try:
                os.rmdir(lock_dir)
            except OSError:
                pass

    def transform(self, data, row_indices=None, col_indices=None):
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform().")
            
        if isinstance(data, (str, os.PathLike)):
            import pyarrow.dataset as ds
            dataset = ds.dataset(data, format="parquet")
            table = dataset.to_table(columns=self.feature_cols)
            data_arr = table.to_pandas().to_numpy(dtype=np.float32)
            col_indices = None
        else:
            data_arr = data

        # --- 推論パス (row_indices is None): 変更なし ---
        # メモリに収まる小さなDataFrameが渡されることを想定
        if row_indices is None:
            df = self._to_dataframe(data_arr, row_indices=None, col_indices=col_indices)
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
        n_rows = int(data_arr.shape[0])
        if np.any(row_indices < 0):
            raise IndexError("row_indices contains negative values.")
        if np.any(row_indices >= n_rows):
            raise IndexError(
                f"row_indices contains values outside data range [0, {n_rows})."
            )

        if self.sequence_cache_enabled:
            return self._get_or_build_sequence_cache(data_arr, row_indices, col_indices)

        # Zarr を用いたオンディスクキャッシュ (3D化された時系列ウィンドウ)
        zarr_dir = os.path.join(tempfile.gettempdir(), f"tcn_cache_{uuid.uuid4().hex}.zarr")
        self._build_sequence_zarr(zarr_dir, data_arr, row_indices, col_indices)

        return zarr_dir

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
            "sequence_cache_enabled": self.sequence_cache_enabled,
            "sequence_cache_dir": self.sequence_cache_dir,
            "sequence_cache_wait_seconds": self.sequence_cache_wait_seconds,
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
        self.sequence_cache_enabled = state.get("sequence_cache_enabled", False)
        self.sequence_cache_dir = state.get(
            "sequence_cache_dir",
            os.path.join(tempfile.gettempdir(), "jps_tcn_sequence_cache"),
        )
        self.sequence_cache_wait_seconds = state.get("sequence_cache_wait_seconds", 600)
        self._index_metadata_cache_ = None
        self.is_fitted = True
        print(f"TCNPreprocessor loaded from {load_path}")
