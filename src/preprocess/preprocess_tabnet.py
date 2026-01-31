import numpy as np
import pandas as pd
import os
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from .base import BasePreprocessor

class TabNetPreprocessor(BasePreprocessor):
    """
    TabNet用の前処理クラス
    """
    def __init__(self, save_dir, feature_cols=None):
        super().__init__(save_dir)
        self.feature_cols = feature_cols if feature_cols else []
        self.cat_cols = []                
        # 状態保持用
        self.encoders = {} # col -> LabelEncoder
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        self.feature_names = None
        self.cat_idxs = []
        self.cat_dims = []

    def fit(self, df):
        """
        カテゴリ変数のLabelEncoding
        数値変数の欠損補完(median)
        cat_idxs, cat_dims の保持
        """
        # object型 または category型 のカラムを特定
        self.cat_cols = df[self.feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        self.feature_names = df[self.feature_cols].columns.tolist()
        # カテゴリ変数のLabelEncodingを実行
        for col in self.cat_cols:
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
        num_cols = [c for c in df.columns if c not in self.cat_cols]
        if num_cols:
            self.imputer.fit(df[num_cols])
        
        self.is_fitted = True
        print(f"TabNet Preprocessor fitted. Categorical cols: {self.cat_cols}")

    def transform(self, df):
        """
        不要カラム削除、型変換を行う
        """
        if not self.is_fitted:
            self.fit(df) # Fitされてなければその場でする
        X = df[self.feature_names]
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = np.nan
        # 数値変数の欠損補完
        valid_cat_cols = [c for c in self.cat_cols if c in X.columns]
        num_cols = [c for c in X.columns if c not in valid_cat_cols]
        # コピーを作成
        X_processed = X.copy()
        if num_cols:
            # InfをNaNに
            X_processed[num_cols] = X_processed[num_cols].replace([np.inf, -np.inf], np.nan)
            # Impute
            target_cols = self.imputer.feature_names_in_.tolist()
            X_processed[target_cols] = self.imputer.transform(X_processed[target_cols])
        # カテゴリ変数のエンコーディング
        for col in valid_cat_cols:
            if col not in self.encoders.keys(): continue
            le = self.encoders[col]
            # 未知のラベル対応: 既知のものに置換、あるいは "MISSING" (Fit時にあれば)
            # ここではFit時と同じ変換を行う
            ser = X_processed[col].fillna("MISSING").astype(str)
            # 未知ラベルは一旦 "MISSING" にするか、モード値にする等の対策が必要
            # 今回は簡易的に、le.classes_ にないものは 0 番目のクラスに置換する等の処理を入れる
            # (もっと厳密には Unknown 専用クラスを作るべき)
            # マッピング辞書作成
            param_map = {label: i for i, label in enumerate(le.classes_)}
            # mapで変換（見つからないものは0埋めなど）
            X_processed[col] = ser.map(param_map).fillna(0).astype(int)
        return X_processed

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
            'feature_names': self.feature_names
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
        self.feature_names = state['feature_names']
        self.is_fitted = True
        print(f"TabNet Preprocessor loaded from {load_path}")