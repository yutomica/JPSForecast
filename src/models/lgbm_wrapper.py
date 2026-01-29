import os
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
from .base import BaseModelWrapper

class LGBMWrapper(BaseModelWrapper):
    def __init__(self, task_type="regression", **params):
        self.task_type = task_type
        self.params = params
        self.model = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None):
        # LGBM専用のDataset構造に変換
        train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        valid_sets = [train_set]
        valid_names = ["train"]
        if X_valid is not None:
            valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
            valid_sets.append(valid_set)
            valid_names.append("valid")
        # 学習の実行
        self.model = lgb.train(
            params=self.params,
            train_set=train_set,
            valid_sets=valid_sets,
            valid_names=valid_names,
        )
        # 重要度の作成と保存
        self._create_feature_importance_df()
        # Feature Importanceの抽出とMLflow保存
        self._log_feature_importance()

    def _create_feature_importance_df(self):
        """重要度をデータフレーム形式で作成して属性に保持する"""
        if self.model is not None:
            self.feature_importances_ = pd.DataFrame({
                'feature': self.model.feature_name(),
                'importance_gain': self.model.feature_importance(importance_type='gain'),
                'importance_split': self.model.feature_importance(importance_type='split')
            }).sort_values(by='importance_gain', ascending=False)

    def _log_feature_importance(self):
        """
        特徴量重要度を計算・可視化し、MLflowのArtifactとして保存する
        """
        if self.model is None:
            return
        # 重要度の取得 (Gain: 目的関数の減少にどれだけ寄与したか)
        importance_df = pd.DataFrame({
            'feature': self.model.feature_name(),
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values(by='importance', ascending=False)
        # 上位30項目に絞ってプロット
        top_n = 30
        plot_df = importance_df.head(top_n)
        # プロットの作成
        plt.barh(plot_df['feature'], plot_df['importance'])
        plt.xlabel('Importance (Gain)')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()  # 上位が上に来るように
        plt.tight_layout()

        # 一時ファイルとして保存
        temp_path = "feature_importance.png"
        plt.savefig(temp_path)
        plt.close() # メモリ解放

        # MLflowに画像をアップロード
        if mlflow.active_run():
            mlflow.log_artifact(temp_path, artifact_path="plots")
            print(f"✅ Feature Importance plot saved to MLflow (Top {top_n})")
        
        # 不要な一時ファイルを削除
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def predict(self, X):
        return self.model.predict(X)