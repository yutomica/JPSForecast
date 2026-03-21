import numpy as np
import pandas as pd
import os
import joblib
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.special import ndtri
from .base import BasePreprocessor

class TCNPreprocessor(BasePreprocessor):
    def __init__(self, save_dir, feature_cols=None, cat_cols=None, window_size=20):
        super().__init__(save_dir)
        self.feature_cols = feature_cols if feature_cols else []
        self.cat_cols = cat_cols if cat_cols else []
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        self.label_encoders = {} # 文字列 -> ID マッピング
        self.embedding_info = []
        self.is_fitted = False
        self.epsilon = 1e-6 # Gauss Rank用

    def fit(self, data):
        """カテゴリ変数のラベルエンコーディングと数値変数の学習"""
        self.num_cols = [c for c in self.feature_cols if c not in self.cat_cols]
        self.num_indices = [self.feature_cols.index(c) for c in self.num_cols]
        self.cat_indices = {c: self.feature_cols.index(c) for c in self.cat_cols if c in self.feature_cols}

        embedding_count = 0 
        for col in self.cat_cols:
            if col in data.columns:
                unique_vals = data[col].dropna().unique()
                self.label_encoders[col] = {val: i + 1 for i, val in enumerate(unique_vals)}
                num_cat = len(unique_vals) + 1 # +1 は UNK(0) の分
                self.embedding_info.append({
                    'column_idx': self.cat_indices.get(col, data.columns.tolist().index(col)),
                    'num_categories': num_cat,
                    'embedding_dim': min(50, num_cat // 2 + 1)
                })
                embedding_count += 1
        print(str(embedding_count)+' features are categorical.')
                
        # 数値変数のスケーリング・補完の学習
        if self.num_cols:
            # pandasのreplaceは遅いため、NumPy配列化してから高速に処理
            num_data = data[self.num_cols].to_numpy(dtype=np.float32)
            num_data[np.isinf(num_data)] = np.nan
            self.imputer.fit(num_data)
            num_data_imp = self.imputer.transform(num_data)
            self.scaler.fit(num_data_imp)
            
        self.is_fitted = True

    def transform(self, data, row_indices=None, col_indices=None):
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted.")
        
        if row_indices is None:
            row_indices = np.arange(len(data))
        else:
            row_indices = np.asarray(row_indices)
            
        # 各行に対して過去 window_size 分のインデックスを計算
        # shape: (N, window_size)
        window_indices = row_indices[:, None] - np.arange(self.window_size - 1, -1, -1)
        
        # 範囲外(負)のインデックスは後で0埋めするためのマスクを作成し、0にクリップして抽出可能にする
        valid_mask = window_indices >= 0
        clipped_indices = np.clip(window_indices, 0, None)
        
        # 必要なユニークインデックスのみを抽出し、前処理の無駄を省く
        unique_indices, inverse_indices = np.unique(clipped_indices, return_inverse=True)
        
        # memmap等から、必要な行・列だけをピンポイントで抽出
        extracted = data[unique_indices]
        if col_indices is not None:
            extracted = extracted[:, col_indices]
            
        processed = np.empty((extracted.shape[0], extracted.shape[1]), dtype=np.float32)
            
        # カテゴリ変数のエンコーディング (未知の値は0)
        if hasattr(self, 'cat_indices') and self.cat_indices:
            for col, idx in self.cat_indices.items():
                if col in self.label_encoders:
                    mapping = self.label_encoders[col]
                    # np.vectorizeより高速なpandasのmapを利用
                    mapped_vals = pd.Series(extracted[:, idx]).map(mapping).fillna(0)
                    processed[:, idx] = mapped_vals.to_numpy(dtype=np.float32)
        
        # NaNとInfの処理、およびスケーリング (数値変数)
        if hasattr(self, 'num_indices') and self.num_indices:
            num_data = extracted[:, self.num_indices].astype(np.float32)
            num_data[np.isinf(num_data)] = np.nan
            num_data = self.imputer.transform(num_data)
            num_data = self.scaler.transform(num_data)
            processed[:, self.num_indices] = num_data

        # inverse_indices を用いて直接 3D配列 (N, window_size, features) を構築
        X_3d = processed[inverse_indices].reshape(len(row_indices), self.window_size, extracted.shape[1])
        
        # 範囲外(パディング対象)だった箇所を 0.0 で埋める
        X_3d[~valid_mask] = 0.0
        
        print(f" - 3D Sequence construction complete. Shape: {X_3d.shape}")
        return X_3d

    def save(self, filename='preprocessor.joblib'):
        state = {
            'label_encoders': self.label_encoders,
            'embedding_info': self.embedding_info,
            'feature_cols': self.feature_cols,
            'window_size': self.window_size,
            'is_fitted': self.is_fitted,
            'num_indices': getattr(self, 'num_indices', []),
            'cat_indices': getattr(self, 'cat_indices', {}),
            'imputer': self.imputer,
            'scaler': self.scaler
        }
        path = os.path.join(self.save_dir, filename)
        os.makedirs(self.save_dir, exist_ok=True)
        joblib.dump(state, path)

    def load(self, filename='preprocessor.joblib'):
        path = os.path.join(self.save_dir, filename)
        state = joblib.load(path)
        self.label_encoders = state['label_encoders']
        self.embedding_info = state['embedding_info']
        self.feature_cols = state['feature_cols']
        self.window_size = state['window_size']
        self.is_fitted = state['is_fitted']
        self.num_indices = state.get('num_indices', [])
        self.cat_indices = state.get('cat_indices', {})
        if 'imputer' in state: self.imputer = state['imputer']
        if 'scaler' in state: self.scaler = state['scaler']