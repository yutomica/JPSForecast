import numpy as np
import pandas as pd
import os
import joblib
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from .base import BasePreprocessor

class TCNPreprocessor(BasePreprocessor):
    def __init__(self, save_dir, feature_cols=None, cat_cols=None, window_size=20):
        super().__init__(save_dir)
        self.feature_cols = feature_cols if feature_cols else []
        self.cat_cols = []
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False

    def fit(self, data):
        """
        BasePreprocessor の抽象メソッドを満足させるために必須。
        一括データでの学習、または最初のチャンク学習として機能させます。
        """
        self.partial_fit(data)

    def partial_fit(self, data):
        print(f"  -- Fitting chunk (Rows: {len(data)})...")
        # 渡された DataFrame から数値列を抽出
        num_cols = [c for c in data.columns if c not in self.cat_cols]
        X_num = data[num_cols].replace([np.inf, -np.inf], np.nan)
        # 1. 欠損値補完の学習 (SimpleImputer もチャンク対応が必要な場合は別途検討)
        # 実際には統計量を維持するため、事前に計算するか、概算で対応します
        if not self.is_fitted:
            self.imputer.fit(X_num) # 最初のチャンクで型を固定
        # 1. partial_fit 内を修正
        X_imputed = self.imputer.transform(X_num)
        input_data = X_imputed.values if isinstance(X_imputed, pd.DataFrame) else X_imputed
        self.scaler.partial_fit(input_data)
        self.is_fitted = True

    def transform(self, data, row_indices=None, col_indices=None):
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted.")
        # 過去 window_size 分を確保するために必要な全期間を取得
        start_idx = max(0, min(row_indices) - self.window_size + 1)
        end_idx = max(row_indices) + 1
        total_rows = end_idx - start_idx
        print(f" - Transforming {len(row_indices)} samples (Scanning {total_rows} rows from disk)...")
        extracted = data[start_idx:end_idx, col_indices].copy()
        num_feature_names = [c for c in self.feature_cols if c not in self.cat_cols]
        num_idx = [i for i, c in enumerate(self.feature_cols) if c not in self.cat_cols]
        if num_idx:
            X_num_raw = extracted[:, num_idx]
            X_num_raw[np.isinf(X_num_raw)] = np.nan
            # DataFrame化して一括変換（ここは高速）
            X_num_df = pd.DataFrame(X_num_raw, columns=num_feature_names)
            X_imputed = self.imputer.transform(X_num_df)
            input_to_scaler = X_imputed.values if isinstance(X_imputed, pd.DataFrame) else X_imputed
            extracted[:, num_idx] = self.scaler.transform(input_to_scaler)
        # 従来の Python ループを廃止し、NumPy のストライド演算を使用
        # パディング: データの先頭付近でも window_size 分確保できるように 0 で埋める
        pad_width = ((self.window_size - 1, 0), (0, 0))
        padded = np.pad(extracted, pad_width, mode='constant', constant_values=0)
        # shape: (N_windows, 1, window_size, features)
        windows = sliding_window_view(padded, (self.window_size, extracted.shape[1]))
        windows = windows.squeeze(axis=1) # (N_windows, window_size, features)
        target_local_indices = row_indices - start_idx
        X_3d = windows[target_local_indices]
        print(f" - 3D Sequence construction complete. Shape: {X_3d.shape}")
        return X_3d

    def save(self, filename='preprocessor.joblib'):
        state = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_cols': self.feature_cols,
            'window_size': self.window_size,
            'is_fitted': self.is_fitted
        }
        path = os.path.join(self.save_dir, filename)
        os.makedirs(self.save_dir, exist_ok=True)
        joblib.dump(state, path)

    def load(self, filename='preprocessor.joblib'):
        path = os.path.join(self.save_dir, filename)
        state = joblib.load(path)
        self.scaler = state['scaler']
        self.imputer = state['imputer']
        self.feature_cols = state['feature_cols']
        self.window_size = state['window_size']
        self.is_fitted = state['is_fitted']