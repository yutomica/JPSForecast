import os
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
from .base import BaseModelWrapper

class LGBMWrapper(BaseModelWrapper):
    def __init__(self, task_type="regression", **params):
        self.task_type = task_type
        params.pop("use_time_decay", None)
        self.params = params
        self.model = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0):
        # LGBM専用のDataset構造に変換
        train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        valid_sets = [train_set]
        valid_names = ["train"]
        if X_valid is not None:
            valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
            valid_sets.append(valid_set)
            valid_names.append("valid")
        # 学習の実行
        evals_result = {}
        verbose_val = self.params.get("verbose", -1)
        callbacks = [lgb.record_evaluation(evals_result)]
        # - verboseが0以上の場合のみ、ログ出力コールバックを追加
        if verbose_val >= 0:
            # - 例えば 100 イテレーションごとにログを出す設定
            callbacks.append(lgb.log_evaluation(period=100))
        self.model = lgb.train(
            params=self.params,
            train_set=train_set,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks # 履歴を記録
        )
        # 重要度の作成と保存
        self._create_feature_importance_df()
        # Feature Importanceの抽出とMLflow保存
        self._log_feature_importance(model_idx)
        # ★ 学習曲線の保存処理を呼び出す
        self._log_learning_curve(evals_result, model_idx)

    def _log_learning_curve(self, evals_result, model_idx):
        """lgb.plot_metric を使用して学習曲線を保存し MLflow にアップロード"""
        # plot_metric を実行
        lgb.plot_metric(evals_result)
        plt.title("Learning Curve")
        plt.tight_layout()
        # 一時ファイルとして保存
        temp_path = f"learning_curve_m{model_idx}.png"
        plt.savefig(temp_path)
        plt.close()
        # MLflow に保存
        if mlflow.active_run():
            mlflow.log_artifact(temp_path, artifact_path="plots/learning_curves")
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def _create_feature_importance_df(self):
        """重要度をデータフレーム形式で作成して属性に保持する"""
        if self.model is not None:
            self.feature_importances_ = pd.DataFrame({
                'feature': self.model.feature_name(),
                'importance_gain': self.model.feature_importance(importance_type='gain'),
                'importance_split': self.model.feature_importance(importance_type='split')
            }).sort_values(by='importance_gain', ascending=False)

    def _log_feature_importance(self, model_idx):
        """特徴量重要度を計算・可視化し、MLflowのArtifactとして保存する"""
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
        temp_path = f"feature_importance_m{model_idx}.png"
        plt.savefig(temp_path)
        plt.close() # メモリ解放

        # MLflowに画像をアップロード
        if mlflow.active_run():
            mlflow.log_artifact(temp_path, artifact_path="plots/importance")
        
        # 不要な一時ファイルを削除
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        preds = self.model.predict(X)
        return preds