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
    def __init__(self, save_dir, feature_cols=None):
        super().__init__(save_dir)
        self.feature_cols = feature_cols if feature_cols else []
        self.cat_cols = []

    def fit(self, df):
        """
        カテゴリ変数の特定などを行う
        """
        # object型 または category型 のカラムを特定
        self.cat_cols = df[self.feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        self.feature_names_ = df[self.feature_cols].columns.tolist()
        self.is_fitted = True
        print(f"LGBM Preprocessor fitted. Categorical cols: {self.cat_cols}")

    def transform(self, df):
        """
        不要カラム削除、型変換を行う
        """
        if not self.is_fitted:
            self.fit(df) # Fitされてなければその場でする
        df_processed = df.copy()
        # 無限大の処理 数値カラムに対してのみ実施
        num_cols = df_processed[self.feature_cols].select_dtypes(include=[np.number]).columns
        df_processed[num_cols] = df_processed[num_cols].replace([np.inf, -np.inf], np.nan)
        # カテゴリ変数の型変換
        for col in self.cat_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype('category')
        df_processed = df_processed[self.feature_names_]
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
            'feature_names_': self.feature_names_
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
        self.feature_names_ = state.get('feature_names_', None)
        print(f"LGBM Preprocessor loaded from {load_path}")