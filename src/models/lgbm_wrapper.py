import os
import tempfile
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import optuna
from hydra.utils import get_method
from .base import BaseModelWrapper
from .pruning import calculate_spearman_ic, log_epoch_metrics

class LGBMWrapper(BaseModelWrapper):
    def __init__(self, task_type="regression", **params):
        self.task_type = task_type
        params.pop("use_time_decay", None)
        params.pop("time_decay_rate", None)
        # カスタム目的関数および評価関数のパスを取得
        self.custom_objective_path = params.pop("custom_objective", None)
        self.custom_metric_path = params.pop("custom_metric", None)
        # デフォルトの目的関数と評価指標を設定
        if self.task_type == "classification":
            params["objective"] = params.get("objective", "binary")
            params["metric"] = params.get("metric", "binary_logloss")
        elif self.task_type == "multiclass":
            params["objective"] = params.get("objective", "multiclass")
            params["metric"] = params.get("metric", "multi_logloss")
        else:
            params["objective"] = params.get("objective", "regression")
            params["metric"] = params.get("metric", "rmse")
            
        # カスタム評価関数が指定された場合はLGBM組み込みの評価指標を無効化する
        # （first_metric_only=Trueの監視対象を確実にカスタム関数にするため）
        if self.custom_metric_path:
            params["metric"] = "None"
            
        # カスタム目的関数のパスが指定されていれば、params['objective'] に直接セットする (LightGBM 4.0.0以降の仕様)
        if self.custom_objective_path:
            params['objective'] = get_method(self.custom_objective_path)
            
        self.params = params
        self.model = None
        self.classes_ = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0, epoch_callback=None):
        # 多クラス分類用のラベル変換とクラス数設定
        if self.task_type == "multiclass":
            self.classes_ = np.unique(y_train)
            if "num_class" not in self.params:
                self.params["num_class"] = len(self.classes_)
            # -1, 0, 1 などのラベルを 0, 1, 2 の連番インデックスに変換
            y_train = np.searchsorted(self.classes_, y_train)
            if y_valid is not None:
                y_valid = np.searchsorted(self.classes_, y_valid)
        
        # paramsからEarly Stoppingとイテレーション数の設定を取り出す（内部コールバックとの競合を防ぐため）
        patience = self.params.pop("early_stopping_rounds", self.params.pop("early_stopping_round", self.params.pop("patience", 50)))
        num_boost_round = self.params.pop("n_estimators", self.params.pop("num_boost_round", 1000))
        # LGBM専用のDataset構造に変換
        # Early Stoppingの対象を正しく認識させるため、valid_setを先頭に配置する
        train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        valid_sets = []
        valid_names = []
        if X_valid is not None:
            valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
            valid_sets.append(valid_set)
            valid_names.append("valid")
        valid_sets.append(train_set)
        valid_names.append("train")
        # --- カスタム評価関数 (簡易IC) ---
        def custom_ic_eval(preds, train_data):
            labels = train_data.get_label()
            if self.task_type == "multiclass":
                # multiclassの場合、predsが平坦化されている場合があるため変形
                if preds.ndim == 1:
                    preds = preds.reshape(self.params["num_class"], -1).T
                # Score = P(target=+1) - P(target=-1)
                idx_plus = np.where(self.classes_ == 1)[0]
                idx_minus = np.where(self.classes_ == -1)[0]
                p_plus = preds[:, idx_plus[0]] if len(idx_plus) > 0 else np.zeros(preds.shape[0])
                p_minus = preds[:, idx_minus[0]] if len(idx_minus) > 0 else np.zeros(preds.shape[0])
                pred_scores = p_plus - p_minus
                orig_labels = self.classes_[labels.astype(int)]
                return 'ic', calculate_spearman_ic(pred_scores, orig_labels), True
            else:
                return 'ic', calculate_spearman_ic(preds, labels), True
        # --- Configで指定されたカスタム関数の動的読み込み ---
        fevals = []
        if self.custom_metric_path:
            fevals.append(get_method(self.custom_metric_path))
        fevals.append(custom_ic_eval)
        # 学習の実行
        evals_result = {}
        verbose_val = self.params.get("verbose", -1)
        callbacks = [lgb.record_evaluation(evals_result)]
        callbacks.append(lgb.early_stopping(stopping_rounds=patience, first_metric_only=True))
        callbacks.append(lgb.log_evaluation(period=10)) # 10イテレーションごとに進捗ログを出力
        # --- Epoch Callback (Pruning等) と MLflow Logging の実行 ---
        def lgbm_mlflow_callback(env):
            metrics = {}
            current_ic = 0.0
            for dataset_name, eval_name, eval_result, _ in env.evaluation_result_list:
                if dataset_name == 'valid' and eval_name == 'ic':
                    current_ic = eval_result
                elif eval_name != 'ic':
                    metrics[f"{dataset_name}_{eval_name}"] = eval_result
            if epoch_callback is not None and X_valid is not None:
                epoch_callback(epoch=env.iteration, current_score=current_ic)
            if metrics:
                log_epoch_metrics(model_idx, env.iteration, metrics)
        callbacks.append(lgbm_mlflow_callback)
        self.model = lgb.train(
            params=self.params,
            train_set=train_set,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=num_boost_round,
            feval=fevals,
            callbacks=callbacks # 履歴を記録
        )
        print(f"=== Best Iteration: {self.model.best_iteration} ===")
        
        # 重要度の作成と保存
        self._create_feature_importance_df()
        # Feature Importanceの抽出とMLflow保存
        self._log_feature_importance(model_idx)
        # ★ 学習曲線の保存処理を呼び出す
        self._log_learning_curve(evals_result, model_idx)

    def _log_learning_curve(self, evals_result, model_idx):
        """lgb.plot_metric を使用して学習曲線を保存し MLflow にアップロード"""
        if not mlflow.active_run():
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            # plot_metric を実行
            lgb.plot_metric(evals_result)
            plt.title("Learning Curve")
            plt.tight_layout()
            # 一時ファイルとして保存
            temp_path = os.path.join(tmpdir, f"learning_curve_m{model_idx}.png")
            plt.savefig(temp_path)
            plt.close()
            # MLflow に保存
            mlflow.log_artifact(temp_path, artifact_path="plots/learning_curves")

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
        if self.model is None or not mlflow.active_run():
            return

        with tempfile.TemporaryDirectory() as tmpdir:
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
            temp_path = os.path.join(tmpdir, f"feature_importance_m{model_idx}.png")
            plt.savefig(temp_path)
            plt.close() # メモリ解放

            # MLflowに画像をアップロード
            mlflow.log_artifact(temp_path, artifact_path="plots/importance")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        preds = self.model.predict(X)
        if self.task_type == "multiclass":
            # Score = P(target=+1) - P(target=-1)
            idx_plus = np.where(self.classes_ == 1)[0]
            idx_minus = np.where(self.classes_ == -1)[0]
            p_plus = preds[:, idx_plus[0]] if len(idx_plus) > 0 else np.zeros(preds.shape[0])
            p_minus = preds[:, idx_minus[0]] if len(idx_minus) > 0 else np.zeros(preds.shape[0])
            preds = p_plus - p_minus
        return preds