import os
import tempfile
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import optuna
import pyarrow as pa
import pyarrow.ipc as ipc
from hydra.utils import get_method
from .base import BaseModelWrapper
from .pruning import calculate_spearman_ic, log_epoch_metrics
import inspect

class LGBMWrapper(BaseModelWrapper):
    def __init__(self, task_type="regression", **params):
        self.task_type = task_type
        params.pop("use_time_decay", None)
        params.pop("time_decay_rate", None)
        # カスタム目的関数および評価関数のパスを取得
        self.custom_objective_path = params.pop("custom_objective", None)
        self.custom_metric_path = params.pop("custom_metric", None)
        self.builtin_metric_name = params.get("metric") # ログ出力用に元のmetricを保持
        self.early_stopping_metric_path = params.pop("early_stopping_metric", None)
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
        if self.custom_metric_path or self.early_stopping_metric_path:
            params["metric"] = "None"
            
        # カスタム目的関数のパスが指定されていれば、params['objective'] に直接セットする (LightGBM 4.0.0以降の仕様)
        if self.custom_objective_path:
            obj_func = get_method(self.custom_objective_path)
            # 引数に 'preds' を含まない場合はファクトリ関数とみなしてパラメータを渡す
            if 'preds' not in inspect.signature(obj_func).parameters:
                params['objective'] = obj_func(**params)
            else:
                params['objective'] = obj_func
            
        self.params = params
        self.model = None
        self.classes_ = None

    def _from_ipc_handle(self, X):
        """IPCバッファハンドルを受け取ってDataFrameに復元する"""
        if isinstance(X, pa.Buffer):
            with ipc.open_stream(X) as reader:
                table = reader.read_all()
            return table.to_pandas()
        return X

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0, epoch_callback=None, train_dates=None, valid_dates=None):
        # IPCハンドルのデコード
        X_train = self._from_ipc_handle(X_train)
        if X_valid is not None:
            X_valid = self._from_ipc_handle(X_valid)
            
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
        burn_in_rounds = self.params.pop("burn_in_rounds", 100)
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

        # --- 汎用評価指標をLGBM用に変換するヘルパー ---
        eval_cache = {} # 重複ラップを防ぐためのキャッシュ

        def _prepare_eval_func(func_or_path, name=None, direction_override=None):
            if func_or_path is None:
                return None
            
            cache_key = (func_or_path if isinstance(func_or_path, str) else id(func_or_path), direction_override)
            if cache_key in eval_cache:
                return eval_cache[cache_key]

            # パス（文字列）の場合は関数を取得
            if isinstance(func_or_path, str):
                if func_or_path.lower() == 'ic':
                    return custom_ic_eval
                func = get_method(func_or_path)
                inferred_name = func_or_path.split('.')[-1].replace('calc_', '').replace('_eval', '')
            else:
                func = func_or_path
                inferred_name = func.__name__.replace('calc_', '').replace('_eval', '')
                
            # ファクトリ関数の場合、パラメータを渡して実体化する
            sig = inspect.signature(func)
            if not any(p in sig.parameters for p in ["preds", "y_pred", "data", "y_true"]):
                func = func(**self.params)
                sig = inspect.signature(func)
            
            # NumPyベースの関数の場合、LGBM用にラップする
            if any(p in sig.parameters for p in ["dates", "y_true", "y_pred"]):
                metric_name = name or inferred_name or "custom"
                # _eval などの汎用名になってしまった場合のフォールバック
                if metric_name == "eval":
                    metric_name = "custom_metric"
                    
                is_higher_better = direction_override if direction_override is not None else (self.params.get("metric_direction", "maximize") == "maximize")
                from src.models.custom_metrics import create_lgbm_evaluator
                wrapped = create_lgbm_evaluator(
                    metric_name, func, train_dates, valid_dates, 
                    is_higher_better=is_higher_better
                )
                eval_cache[cache_key] = wrapped
                return wrapped
            
            eval_cache[cache_key] = func
            return func

        # --- Configで指定されたカスタム関数の動的読み込みと順序制御 ---
        all_eval_funcs = []
        if self.custom_metric_path:
            all_eval_funcs.append(_prepare_eval_func(self.custom_metric_path))
        all_eval_funcs.append(custom_ic_eval)

        fevals = []
        stopping_func = None
        if self.early_stopping_metric_path:
            stopping_func = _prepare_eval_func(
                self.early_stopping_metric_path, 
                direction_override=(self.params.get("metric_direction", "maximize") == "maximize")
            )
        
        if stopping_func:
            fevals.append(stopping_func)
            for func in all_eval_funcs:
                if func != stopping_func:
                    fevals.append(func)
        else:
            fevals = all_eval_funcs

        # 重複を削除しつつ順序を保持 (キャッシュのおかげで id 一致で判定可能)
        fevals = list(dict.fromkeys(fevals))

        # --- 目的関数に対応する評価指標をfevalsに手動で追加 ---
        if self.custom_objective_path:
            eval_path = self.custom_objective_path + "_eval"
            try:
                eval_func_inst = _prepare_eval_func(eval_path, direction_override=False)
                if eval_func_inst and eval_func_inst not in fevals:
                    fevals.append(eval_func_inst)
            except Exception:
                pass

        # 2. 組み込みの目的関数の場合 (objective パラメータに基づいて判断)
        orig_obj = self.params.get("objective", "regression")
        if isinstance(orig_obj, str):
            feval_names = [f.__name__ if hasattr(f, '__name__') else str(f) for f in fevals]
            
            if orig_obj == 'quantile' and 'quantile_eval' not in feval_names:
                q = self.params.get('alpha', 0.5)
                def quantile_eval(preds, data):
                    y = data.get_label()
                    res = y - preds
                    loss = np.mean(np.maximum(q * res, (q - 1) * res))
                    return 'quantile', loss, False
                fevals.append(quantile_eval)
            elif orig_obj == 'fair' and 'fair_eval' not in feval_names:
                c = self.params.get('fair_c', 1.0)
                def fair_eval(preds, data):
                    y = data.get_label()
                    x = np.abs(y - preds)
                    loss = np.mean(c * c * ((x / c) - np.log1p(x / c)))
                    return 'fair', loss, False
                fevals.append(fair_eval)
            elif orig_obj in ['regression', 'rmse', 'mse'] and 'rmse_eval' not in feval_names:
                def rmse_eval(preds, data):
                    y = data.get_label()
                    loss = np.sqrt(np.mean((y - preds)**2))
                    return 'rmse', loss, False
                fevals.append(rmse_eval)
            elif orig_obj == 'huber' and 'huber_eval' not in feval_names:
                delta = self.params.get('alpha', 1.0)
                def huber_eval(preds, data):
                    y = data.get_label()
                    residual = np.abs(y - preds)
                    loss = np.where(residual <= delta, 
                                    0.5 * residual**2, 
                                    delta * (residual - 0.5 * delta))
                    return 'huber', np.mean(loss), False
                fevals.append(huber_eval)

        # 学習の実行
        evals_result = {}
        callbacks = [lgb.record_evaluation(evals_result)]

        # --- カスタムEarly Stoppingのステート ---
        es_state = {
            "best_score": None,
            "best_iter": 0,
            "wait": 0
        }

        def unified_callback(env):
            metrics_to_log = {}
            current_ic = 0.0
            for dataset_name, eval_name, eval_result, _ in env.evaluation_result_list:
                metrics_to_log[f"{dataset_name}_{eval_name}"] = eval_result
                if dataset_name == 'valid' and eval_name == 'ic':
                    current_ic = eval_result
            if metrics_to_log:
                log_epoch_metrics(model_idx, env.iteration, metrics_to_log)
            
            if epoch_callback is not None and X_valid is not None:
                epoch_callback(epoch=env.iteration, current_score=current_ic)

            if env.evaluation_result_list:
                target_score = None
                target_higher_better = None
                for ds_name, ev_name, score, is_higher_better in env.evaluation_result_list:
                    if ds_name == 'valid':
                        target_score = score
                        target_higher_better = is_higher_better
                        break
                
                if target_score is not None:
                    if es_state["best_score"] is None or env.iteration <= burn_in_rounds:
                        es_state["best_score"] = target_score
                        es_state["best_iter"] = env.iteration
                        es_state["wait"] = 0
                    else:
                        improved = (target_score > es_state["best_score"]) if target_higher_better else (target_score < es_state["best_score"])
                        if improved:
                            es_state["best_score"] = target_score
                            es_state["best_iter"] = env.iteration
                            es_state["wait"] = 0
                        else:
                            es_state["wait"] += 1
                            if es_state["wait"] >= patience:
                                print(f"Early stopping, best iteration is:\n[{es_state['best_iter'] + 1}]")
                                import lightgbm.callback as lgb_cb
                                raise lgb_cb.EarlyStopException(es_state["best_iter"], env.evaluation_result_list)

        callbacks.append(unified_callback)
        callbacks.append(lgb.log_evaluation(period=10))
        self.model = lgb.train(
            params=self.params,
            train_set=train_set,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=num_boost_round,
            feval=fevals,
            callbacks=callbacks
        )
        
        if getattr(self.model, 'best_iteration', 0) == 0 and es_state["best_iter"] > 0:
            self.model.best_iteration = es_state["best_iter"] + 1
            
        print(f"=== Best Iteration: {self.model.best_iteration} ===")
        self._create_feature_importance_df()
        self._log_feature_importance(model_idx)
        self._log_learning_curve(evals_result, model_idx)

    def _log_learning_curve(self, evals_result, model_idx):
        if not mlflow.active_run() or not evals_result:
            return
        first_dataset = list(evals_result.keys())[0]
        first_metric = list(evals_result[first_dataset].keys())[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            lgb.plot_metric(evals_result, metric=first_metric)
            plt.title(f"Learning Curve ({first_metric})")
            plt.tight_layout()
            temp_path = os.path.join(tmpdir, f"learning_curve_m{model_idx}.png")
            plt.savefig(temp_path)
            plt.close()
            mlflow.log_artifact(temp_path, artifact_path="plots/learning_curves")

    def _create_feature_importance_df(self):
        if self.model is not None:
            self.feature_importances_ = pd.DataFrame({
                'feature': self.model.feature_name(),
                'importance_gain': self.model.feature_importance(importance_type='gain'),
                'importance_split': self.model.feature_importance(importance_type='split')
            }).sort_values(by='importance_gain', ascending=False)

    def _log_feature_importance(self, model_idx):
        if self.model is None or not mlflow.active_run():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            importance_df = pd.DataFrame({
                'feature': self.model.feature_name(),
                'importance': self.model.feature_importance(importance_type='gain')
            }).sort_values(by='importance', ascending=False)
            top_n = 30
            plot_df = importance_df.head(top_n)
            plt.barh(plot_df['feature'], plot_df['importance'])
            plt.xlabel('Importance (Gain)')
            plt.title(f'Top {top_n} Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            temp_path = os.path.join(tmpdir, f"feature_importance_m{model_idx}.png")
            plt.savefig(temp_path)
            plt.close()
            mlflow.log_artifact(temp_path, artifact_path="plots/importance")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        X = self._from_ipc_handle(X)
        preds = self.model.predict(X)
        if self.task_type == "multiclass":
            idx_plus = np.where(self.classes_ == 1)[0]
            idx_minus = np.where(self.classes_ == -1)[0]
            p_plus = preds[:, idx_plus[0]] if len(idx_plus) > 0 else np.zeros(preds.shape[0])
            p_minus = preds[:, idx_minus[0]] if len(idx_minus) > 0 else np.zeros(preds.shape[0])
            preds = p_plus - p_minus
        return preds
