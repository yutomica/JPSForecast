import numpy as np
import pandas as pd
import os
import joblib
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.ipc as ipc
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from .base import BasePreprocessor

class TabNetPreprocessor(BasePreprocessor):
    """
    TabNet用の前処理クラス
    """
    def __init__(self, save_dir, feature_cols=None, cat_cols=None):
        super().__init__(save_dir)
        self.feature_cols = feature_cols if feature_cols else []
        self.cat_cols = cat_cols if cat_cols else []
        # 状態保持用
        self.encoders = {} # col -> LabelEncoder
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        self.cat_idxs = []
        self.cat_dims = []

    def fit(self, data):
        """
        カテゴリ変数のLabelEncoding
        数値変数の欠損補完(median)
        cat_idxs, cat_dims の保持
        """
        df = pd.DataFrame(data,columns=self.feature_cols)
        # カテゴリ変数のLabelEncodingを実行
        valid_cat_cols = [c for c in self.cat_cols if c in df.columns]
        for col in valid_cat_cols:
            le = LabelEncoder()
            # 欠損値は 'Unknown' として扱うか、事前に埋める必要がある
            # ここでは簡易的に文字型にして欠損を埋めてからFit
            ser = df[col].fillna("MISSING").astype(str)
            le.fit(ser)
            self.encoders[col] = le
            # TabNet用にインデックスと次元数を記録
            self.cat_idxs.append(df.columns.get_loc(col))
            self.cat_dims.append(len(le.classes_))
        # 数値変数のImputer学習
        # カテゴリ変数はエンコード済みとして扱うため、ここでは数値列のみ対象にしたいが、
        # 簡易化のため全体に対してfitする（カテゴリ列は後で上書きされるので無視される前提）
        # ただし数値列のみ抽出してfitする方が安全
        num_cols = [c for c in df.columns if c not in valid_cat_cols]
        if num_cols:
            self.imputer.fit(df[num_cols])
        
        self.is_fitted = True
        print(f"TabNet Preprocessor fitted. Categorical cols: {self.cat_cols}")

    def _transform_2d(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].copy()
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = np.nan
        # 数値変数の欠損補完
        valid_cat_cols = [c for c in self.cat_cols if c in X.columns]
        num_cols = [c for c in X.columns if c not in valid_cat_cols]
        # コピーを作成
        if num_cols:
            # InfをNaNに
            X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
            # Impute
            target_cols = self.imputer.feature_names_in_.tolist()
            X[target_cols] = self.imputer.transform(X[target_cols])
        # カテゴリ変数のエンコーディング
        for col in valid_cat_cols:
            if col not in self.encoders.keys(): continue
            le = self.encoders[col]
            # 未知のラベル対応: 既知のものに置換、あるいは "MISSING" (Fit時にあれば)
            # ここではFit時と同じ変換を行う
            ser = X[col].fillna("MISSING").astype(str)
            # 未知ラベルは一旦 "MISSING" にするか、モード値にする等の対策が必要
            # 今回は簡易的に、le.classes_ にないものは 0 番目のクラスに置換する等の処理を入れる
            # (もっと厳密には Unknown 専用クラスを作るべき)
            # マッピング辞書作成
            param_map = {label: i for i, label in enumerate(le.classes_)}
            # mapで変換（見つからないものは0埋めなど）
            X[col] = ser.map(param_map).fillna(0).astype(int)
            
        return X.to_numpy(dtype=np.float32)

    def transform(self, data, row_indices=None, col_indices=None):
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted.")
            
        if row_indices is None:
            if isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                df = pd.DataFrame(data[:, col_indices] if col_indices is not None else data, columns=self.feature_cols)
            return self._transform_2d(df)

        import tempfile, uuid, zarr
        row_indices = np.asarray(row_indices, dtype=np.int64)
        zarr_dir = os.path.join(tempfile.gettempdir(), f"tabnet_cache_{uuid.uuid4().hex}.zarr")
        z = zarr.open(zarr_dir, mode='w', shape=(len(row_indices), len(self.feature_cols)), chunks=(10000, len(self.feature_cols)), dtype='float32')
        
        chunk_size = 50000
        for i in range(0, len(row_indices), chunk_size):
            extracted = data[row_indices[i:i + chunk_size]]
            if col_indices is not None: extracted = extracted[:, col_indices]
            z[i:i + chunk_size] = self._transform_2d(pd.DataFrame(extracted, columns=self.feature_cols))
            
        return zarr_dir

    def save(self, filename='scaler.joblib'):
        """
        TabNetPreprocessorの状態を保存する。
        Scalerオブジェクトではなく、カテゴリ列リスト等を辞書として保存。
        """
        if not self.is_fitted:
            # fitされていなければエラー、または何もしない
            raise ValueError("Preprocessor is not fitted yet.")
        state = {
            'encoders': self.encoders,
            'imputer': self.imputer,
            'cat_idxs': self.cat_idxs,
            'cat_dims': self.cat_dims,
            'feature_cols': self.feature_cols
        }
        path = os.path.join(self.save_dir, filename)
        joblib.dump(state, path)
        print(f"TabNet Preprocessor saved to {path}")

    def load(self, filename='scaler.joblib'):
        """
        保存された状態を復元する。
        """
        load_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Preprocessor state file not found: {load_path}")
        state = joblib.load(load_path)
        # 辞書から属性を復元
        self.encoders = state['encoders']
        self.imputer = state['imputer']
        self.cat_idxs = state['cat_idxs']
        self.cat_dims = state['cat_dims']
        self.feature_cols = state['feature_cols']
        self.is_fitted = True
        print(f"TabNet Preprocessor loaded from {load_path}")