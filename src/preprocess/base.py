# src/preprocess/base.py
import os
import joblib
from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.scaler = None
        self.is_fitted = False
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, filename='scaler.joblib'):
        """Scalerなどの学習済みオブジェクトを保存"""
        if self.scaler is None:
            raise ValueError("Scaler is not defined or fitted.")
        path = os.path.join(self.save_dir, filename)
        joblib.dump(self.scaler, path)
        print(f"Preprocessor saved to {path}")

    def load(self, filename='scaler.joblib'):
        """保存されたオブジェクトをロード"""
        path = os.path.join(self.save_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found: {path}")
        self.scaler = joblib.load(path)
        self.is_fitted = True
        print(f"Preprocessor loaded from {path}")

    @abstractmethod
    def fit(self, X):
        """子クラスで必ず実装させる"""
        pass

    @abstractmethod
    def transform(self, X):
        """子クラスで必ず実装させる"""
        pass