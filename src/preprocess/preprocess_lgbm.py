import numpy as np
import pandas as pd
import os
import joblib
from .base import BasePreprocessor

class LGBMPreprocessor(BasePreprocessor):
    """
    LightGBM用の前処理クラス
    - 指定カラムの削除
    - 無限大(inf)の処理
    - カテゴリ変数の型変換 (object -> category)
    """
    def __init__(self, save_dir, feature_cols=None, cat_cols=None):
        super().__init__(save_dir)
        self.feature_cols = feature_cols if feature_cols else []
        self.cat_cols = cat_cols if cat_cols else []
        self.is_fitted = False

    def fit(self, X, y=None):
        # fitメソッドはデータ型判定を行うロジックのみ実装されるため、configの情報を使用すれば不要となる。
        self.is_fitted = True
        print(f"LGBM Preprocessor fitted via Schema-Driven approach.")
        return self

    def transform(self, data, row_indices=None, col_indices=None):
        """
        学習時: data=memmap, row_indices=行, col_indices=選定列
        推論時: data=DataFrame
        不要カラム削除、型変換を行う
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted.")
        if isinstance(data, pd.DataFrame):
            # 推論時：APIから取得した生のDataFrame
            df_processed = data[self.feature_cols].copy()
        else:
            # 学習時：memmapからスライシング
            # row_indices, col_indices を使って必要な次元だけをメモリに乗せる
            if col_indices is not None:
                extracted = data[row_indices][:, col_indices]
            else:
                extracted = data[row_indices]
            df_processed = pd.DataFrame(extracted, columns=self.feature_cols)
        # 無限大の処理
        df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        # カテゴリ変数の型変換
        for col in self.cat_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(-1).astype(int).astype('category')
        num_cols = [c for c in df_processed.columns if c not in self.cat_cols]
        df_processed[num_cols] = df_processed[num_cols].astype('float32')
        return df_processed

    def save(self, filename='scaler.joblib'):
        """
        LGBMPreprocessorの状態を保存する。
        Scalerオブジェクトではなく、カテゴリ列リスト等を辞書として保存。
        """
        if not self.is_fitted:
            # fitされていなければエラー、または何もしない
            raise ValueError("Preprocessor is not fitted yet.")
        state = {
            'cat_cols': self.cat_cols,
            'is_fitted': self.is_fitted,
            'feature_cols': self.feature_cols
        }
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, filename)
        joblib.dump(state, save_path)
        print(f"LGBM Preprocessor state saved to {save_path}")

    def load(self, filename='scaler.joblib'):
        """
        保存された状態を復元する。
        """
        load_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Preprocessor state file not found: {load_path}")
        state = joblib.load(load_path)
        # 辞書から属性を復元
        self.cat_cols = state.get('cat_cols', [])
        self.is_fitted = state.get('is_fitted', False)
        self.feature_cols = state.get('feature_cols', None)
        print(f"LGBM Preprocessor loaded from {load_path}")