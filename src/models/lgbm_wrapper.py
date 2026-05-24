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
from sklearn.metrics import average_precision_score, log_loss

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
        self.early_stopping_enabled = params.pop("early_stopping_enabled", True)
        self.early_stopping_ema_alpha = float(params.pop("early_stopping_ema_alpha", 1.0))
        self.early_stopping_smooth_window = int(params.pop("early_stopping_smooth_window", 1))

        # smooth_window が指定されている場合は alpha に変換 (EMAとして扱う)
        if self.early_stopping_smooth_window > 1 and self.early_stopping_ema_alpha == 1.0:
            self.early_stopping_ema_alpha = 2.0 / (self.early_stopping_smooth_window + 1.0)
        # デフォルトの目的関数と評価指標を設定
        if self.task_type in ["classification", "binary"]:
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
        burn_in_rounds = self.params.pop("burn_in_rounds", 200)
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

        # 監視対象の指標名を特定
        stopping_metric_name = "ic"
        if self.early_stopping_metric_path:
            if isinstance(self.early_stopping_metric_path, str):
                stopping_metric_name = self.early_stopping_metric_path.split('.')[-1].replace('calc_', '').replace('_eval', '')
            elif hasattr(self.early_stopping_metric_path, '__name__'):
                stopping_metric_name = self.early_stopping_metric_path.__name__.replace('calc_', '').replace('_eval', '')

        # --- カスタム評価関数 (簡易IC) ---
        def custom_ic_eval(preds, train_data):
            labels = train_data.get_label()
            if self.task_type == "multiclass":
                # multiclassの場合、predsが平坦化されている場合があるため変形
                if preds.ndim == 1:
                    preds = preds.reshape(self.params.get("num_class", 1), -1).T
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
            
            # デフォルトの方向性を取得
            is_higher_better_default = (self.params.get("metric_direction", "maximize") == "maximize")
            target_direction = direction_override if direction_override is not None else is_higher_better_default
            
            cache_key = (func_or_path if isinstance(func_or_path, str) else id(func_or_path), target_direction)
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
                    
                from src.models.custom_metrics import create_lgbm_evaluator
                wrapped = create_lgbm_evaluator(
                    metric_name, func, train_dates, valid_dates, 
                    is_higher_better=target_direction
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

        # 指標の方向性を確定
        metric_is_maximize = (self.params.get("metric_direction", "maximize") == "maximize")

        stopping_func = None
        if self.early_stopping_metric_path:
            stopping_func = _prepare_eval_func(
                self.early_stopping_metric_path, 
                direction_override=metric_is_maximize
            )
        
        # --- 目的関数に対応する評価指標の特定 ---
        obj_eval_func = None
        orig_obj = self.params.get("objective", "regression")
        
        if self.custom_objective_path:
            eval_path = self.custom_objective_path + "_eval"
            try:
                obj_eval_func = _prepare_eval_func(eval_path, direction_override=False)
            except Exception:
                pass
        elif isinstance(orig_obj, str):
            if orig_obj == 'binary':
                def binary_logloss_eval(preds, data):
                    y = data.get_label()
                    eps = 1e-15
                    p = np.clip(preds, eps, 1 - eps)
                    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                    return 'binary_logloss', loss, False
                obj_eval_func = binary_logloss_eval
            elif orig_obj == 'quantile':
                q = self.params.get('alpha', 0.5)
                def quantile_eval(preds, data):
                    y = data.get_label()
                    res = y - preds
                    loss = np.mean(np.maximum(q * res, (q - 1) * res))
                    return 'quantile', loss, False
                obj_eval_func = quantile_eval
            elif orig_obj == 'fair':
                c = self.params.get('fair_c', 1.0)
                def fair_eval(preds, data):
                    y = data.get_label()
                    x = np.abs(y - preds)
                    loss = np.mean(c * c * ((x / c) - np.log1p(x / c)))
                    return 'fair', loss, False
                obj_eval_func = fair_eval
            elif orig_obj in ['regression', 'rmse', 'mse']:
                def rmse_eval(preds, data):
                    y = data.get_label()
                    loss = np.sqrt(np.mean((y - preds)**2))
                    return 'rmse', loss, False
                obj_eval_func = rmse_eval
            elif orig_obj == 'huber':
                delta = self.params.get('alpha', 1.0)
                def huber_eval(preds, data):
                    y = data.get_label()
                    residual = np.abs(y - preds)
                    loss = np.where(residual <= delta, 
                                    0.5 * residual**2, 
                                    delta * (residual - 0.5 * delta))
                    return 'huber', np.mean(loss), False
                obj_eval_func = huber_eval

        # --- 指標の統合と順序制御 (コンソール表示順) ---
        fevals = []
        # 1. 目的関数の指標を最優先 (ES指標がない場合に先頭にするため)
        if obj_eval_func:
            fevals.append(obj_eval_func)
            
        # 2. ES指標を最優先に上書き (もしあれば、それが監視の主役になる)
        if stopping_func:
            # 同一関数が別目的（ログ用など）で既にあれば削除して先頭へ
            fevals = [f for f in fevals if f != stopping_func]
            fevals.insert(0, stopping_func)
            
        # 3. その他の指標 (ICなど)
        for func in all_eval_funcs:
            if func not in fevals:
                fevals.append(func)

        # 重複を削除しつつ順序を保持
        fevals = list(dict.fromkeys(fevals))

        # 学習の実行
        evals_result = {}
        callbacks = [lgb.record_evaluation(evals_result)]

        # --- カスタムEarly Stoppingのステート ---
        es_state = {
            "best_score": None,
            "best_iter": 0,
            "wait": 0,
            "monitored_metric": None,
            "ema_score": None
        }

        def unified_callback(env):
            metrics_to_log = {}
            current_ic = 0.0
            for dataset_name, eval_name, eval_result, _ in env.evaluation_result_list:
                metrics_to_log[f"{dataset_name}_{eval_name}"] = eval_result
                if dataset_name == 'valid' and eval_name == 'ic':
                    current_ic = eval_result

            if epoch_callback is not None and X_valid is not None:
                epoch_callback(epoch=env.iteration, current_score=current_ic)

            if env.evaluation_result_list:
                target_score = None
                target_name = None
                target_higher_better = None

                # 指定された監視指標を最優先で探し、なければ最初に見つかった valid 指標を使用
                for ds_name, ev_name, score, is_higher_better in env.evaluation_result_list:
                    if ds_name == 'valid':
                        if ev_name == stopping_metric_name:
                            target_score = score
                            target_name = ev_name
                            target_higher_better = is_higher_better
                            break
                        if target_score is None:
                            target_score = score
                            target_name = ev_name
                            target_higher_better = is_higher_better

                # target_score が NaN の場合は改善判定をスキップする
                if target_score is not None and not np.isnan(target_score):
                    # Smoothing (EMA)
                    if es_state["ema_score"] is None:
                        es_state["ema_score"] = float(target_score)
                    else:
                        es_state["ema_score"] = self.early_stopping_ema_alpha * target_score + (1.0 - self.early_stopping_ema_alpha) * es_state["ema_score"]
                    
                    smoothed_score = es_state["ema_score"]
                    if self.early_stopping_ema_alpha < 1.0:
                        metrics_to_log[f"valid_{target_name}_smoothed"] = smoothed_score

                    # 方向性の判定 (上位レイヤーでの不整合を防ぐため、明示的な metric_is_maximize も考慮)
                    effective_is_higher_better = target_higher_better
                    if target_name == stopping_metric_name:
                        effective_is_higher_better = metric_is_maximize

                    if not effective_is_higher_better:
                        improved = (es_state["best_score"] is None or np.isnan(es_state["best_score"]) or smoothed_score < es_state["best_score"])
                    else:
                        improved = (es_state["best_score"] is None or np.isnan(es_state["best_score"]) or smoothed_score > es_state["best_score"])

                    if improved:
                        es_state["best_score"] = float(smoothed_score)
                        es_state["best_iter"] = env.iteration
                        es_state["wait"] = 0
                        es_state["monitored_metric"] = target_name
                    else:
                        # No improvement. Only increment wait if we are past burn-in rounds.
                        if env.iteration >= burn_in_rounds:
                            es_state["wait"] += 1

                    # Burn-in期間終了時に、もしベストが初期すぎる（Burn-inの半分以下）場合は
                    # その時点のスコアでリセットして、真の学習フェーズでの探索を促す。
                    if env.iteration == burn_in_rounds:
                        if es_state["best_iter"] < (burn_in_rounds // 2):
                            old_best = es_state["best_iter"] + 1
                            es_state["best_score"] = float(smoothed_score)
                            es_state["best_iter"] = env.iteration
                            print(f"  ⚠️  Best iteration ([{old_best}]) was found too early during burn-in. Resetting best to iteration [{env.iteration + 1}] to avoid noise.")

                    if self.early_stopping_enabled and es_state["wait"] >= patience:
                        best_score_val = es_state["best_score"] if es_state["best_score"] is not None else 0.0
                        smooth_suffix = f" (Smoothed EMA alpha={self.early_stopping_ema_alpha:.3f})" if self.early_stopping_ema_alpha < 1.0 else ""
                        print(f"Early stopping, best iteration is:\n[{es_state['best_iter'] + 1}]")
                        print(f"Monitored metric: {es_state['monitored_metric']}{smooth_suffix} (Best Score: {best_score_val:.6f})")
                        import lightgbm.callback as lgb_cb
                        # LightGBM 4.0.0 以降のコールバック例外への対応
                        try:
                            raise lgb_cb.EarlyStopException(es_state["best_iter"], env.evaluation_result_list)
                        except AttributeError:
                            # 古いバージョンや環境向けのフォールバック
                            raise Exception("Early stopping triggered")
                elif target_score is not None and np.isnan(target_score):
                    # NaNの場合は改善なしとして扱うが、burn_in期間中などはwaitを増やさない
                    if env.iteration >= burn_in_rounds:
                        es_state["wait"] += 1

            if metrics_to_log:
                log_epoch_metrics(model_idx, env.iteration, metrics_to_log)

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
        # multiclass の場合は確率行列をそのまま返す (評価関数側で期待値変換を行うため)
        return preds

class LGBMOrdinalThresholdWrapper(BaseModelWrapper):
    def __init__(self, task_type="regression", **params):
        self.task_type = task_type
        # ポップするパラメータ
        self.ordinal_thresholds = params.pop("ordinal_thresholds", [1, 2, 3])
        self.ordinal_score_weights = params.pop("ordinal_score_weights", [0.2, 0.3, 0.5])
        self.output_mode = params.pop("output_mode", "weighted_score")
        self.enforce_monotonic_probs = params.pop("enforce_monotonic_probs", True)
        self.ordinal_pos_weight = params.pop("ordinal_pos_weight", {
            "enabled": True,
            "method": "sqrt_neg_pos_ratio",
            "max_scale_pos_weight": 5.0
        })
        self.early_stopping_enabled = params.pop("early_stopping_enabled", False)
        
        # 既存LGBMWrapperと同様のパラメータ処理
        params.pop("use_time_decay", None)
        params.pop("time_decay_rate", None)
        self.custom_objective_path = params.pop("custom_objective", None)
        self.custom_metric_path = params.pop("custom_metric", None)
        self.early_stopping_metric_path = params.pop("early_stopping_metric", None)
        
        self.params = params
        self.models_ = {}
        self.feature_importances_ = None

    def _from_ipc_handle(self, X):
        if isinstance(X, pa.Buffer):
            with ipc.open_stream(X) as reader:
                table = reader.read_all()
            return table.to_pandas()
        return X

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0, epoch_callback=None, train_dates=None, valid_dates=None):
        X_train = self._from_ipc_handle(X_train)
        if X_valid is not None:
            X_valid = self._from_ipc_handle(X_valid)
            
        base_sample_weight = sample_weight if sample_weight is not None else np.ones(len(y_train))
        
        # 学習設定の取得とparamsからの削除 (競合防止)
        num_boost_round = self.params.pop("n_estimators", self.params.pop("num_boost_round", 300))
        patience = self.params.pop("early_stopping_rounds", self.params.pop("early_stopping_round", self.params.pop("patience", 50)))

        for threshold in self.ordinal_thresholds:
            print(f"--- Training Threshold: {threshold} ---")
            y_train_bin = (y_train >= threshold).astype(int)
            
            # クラス不均衡補正
            sw = base_sample_weight.copy()
            pos_weight = 1.0
            if self.ordinal_pos_weight.get("enabled", False):
                n_pos = np.sum(y_train_bin == 1)
                n_neg = np.sum(y_train_bin == 0)
                if n_pos > 0:
                    ratio = n_neg / n_pos
                    pos_weight = np.sqrt(ratio)
                    pos_weight = min(pos_weight, self.ordinal_pos_weight.get("max_scale_pos_weight", 5.0))
                    sw[y_train_bin == 1] *= pos_weight
            
            train_set = lgb.Dataset(X_train, label=y_train_bin, weight=sw)
            valid_sets = [train_set]
            valid_names = ["train"]
            
            y_valid_bin = None
            if X_valid is not None:
                y_valid_bin = (y_valid >= threshold).astype(int)
                valid_set = lgb.Dataset(X_valid, label=y_valid_bin, reference=train_set)
                valid_sets.insert(0, valid_set)
                valid_names.insert(0, "valid")
            
            # パラメータ設定 (Binary固定)
            current_params = self.params.copy()
            current_params["objective"] = "binary"
            current_params["metric"] = "binary_logloss"
            # 競合する可能性のある不均衡補正パラメータを削除 (手動で重みを付けているため)
            current_params.pop("is_unbalance", None)
            current_params.pop("scale_pos_weight", None)
            # 多クラス用パラメータを削除
            current_params.pop("num_class", None)
            current_params.pop("metric_direction", None)
            
            callbacks = [lgb.log_evaluation(period=50)]
            if self.early_stopping_enabled and X_valid is not None:
                callbacks.append(lgb.early_stopping(stopping_rounds=patience))
                
            model = lgb.train(
                params=current_params,
                train_set=train_set,
                valid_sets=valid_sets,
                valid_names=valid_names,
                num_boost_round=num_boost_round,
                callbacks=callbacks
            )
            self.models_[threshold] = model
            
            # MLflow記録
            if mlflow.active_run():
                metrics = {
                    f"fold{model_idx}_thr{threshold}_positive_rate": float(np.mean(y_train_bin)),
                    f"fold{model_idx}_thr{threshold}_pos_weight": float(pos_weight),
                    f"fold{model_idx}_thr{threshold}_best_iteration": int(model.best_iteration)
                }
                if X_valid is not None:
                    valid_preds = model.predict(X_valid)
                    try:
                        metrics[f"fold{model_idx}_thr{threshold}_valid_ap"] = float(average_precision_score(y_valid_bin, valid_preds))
                        metrics[f"fold{model_idx}_thr{threshold}_valid_logloss"] = float(log_loss(y_valid_bin, valid_preds))
                    except Exception as e:
                        print(f"Failed to log metrics for threshold {threshold}: {e}")
                mlflow.log_metrics(metrics)

        self._create_feature_importance_df()
        self._log_feature_importance(model_idx)

    def predict(self, X):
        X = self._from_ipc_handle(X)
        
        # 各閾値モデルの予測 (P(y >= threshold))
        probs = {}
        for threshold, model in self.models_.items():
            probs[threshold] = model.predict(X)
            
        # 順序制約の補正
        if self.enforce_monotonic_probs:
            sorted_thresholds = sorted(self.models_.keys())
            for i in range(1, len(sorted_thresholds)):
                t_prev = sorted_thresholds[i-1]
                t_curr = sorted_thresholds[i]
                # P(y >= t_curr) <= P(y >= t_prev)
                probs[t_curr] = np.minimum(probs[t_curr], probs[t_prev])
                
        if self.output_mode == "weighted_score":
            score = np.zeros(len(X))
            for thr, weight in zip(self.ordinal_thresholds, self.ordinal_score_weights):
                score += weight * probs[thr]
            return score
        
        elif self.output_mode == "expected_class":
            # E[class] = P(y>=1) + P(y>=2) + P(y>=3) ... 
            return sum(probs.values())
        
        elif self.output_mode == "class_proba":
            # p0 = 1 - P(y>=1)
            # p1 = P(y>=1) - P(y>=2)
            # ...
            sorted_thrs = sorted(self.models_.keys())
            n_classes = len(sorted_thrs) + 1
            class_probs = np.zeros((len(X), n_classes))
            
            p_prev = np.ones(len(X))
            for i, thr in enumerate(sorted_thrs):
                p_ge = probs[thr]
                class_probs[:, i] = p_prev - p_ge
                p_prev = p_ge
            class_probs[:, -1] = p_prev
            return class_probs
        
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

    def _create_feature_importance_df(self):
        if not self.models_:
            return
            
        importance_list = []
        for threshold, model in self.models_.items():
            df = pd.DataFrame({
                'feature': model.feature_name(),
                f'gain_thr{threshold}': model.feature_importance(importance_type='gain'),
                f'split_thr{threshold}': model.feature_importance(importance_type='split')
            })
            importance_list.append(df.set_index('feature'))
            
        final_df = pd.concat(importance_list, axis=1)
        final_df['importance_gain'] = final_df[[c for c in final_df.columns if 'gain_thr' in c]].mean(axis=1)
        final_df['importance_split'] = final_df[[c for c in final_df.columns if 'split_thr' in c]].mean(axis=1)
        self.feature_importances_ = final_df.reset_index().sort_values(by='importance_gain', ascending=False)

    def _log_feature_importance(self, model_idx):
        if self.feature_importances_ is None or not mlflow.active_run():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            top_n = 30
            plot_df = self.feature_importances_.head(top_n)
            plt.figure(figsize=(10, 8))
            plt.barh(plot_df['feature'], plot_df['importance_gain'])
            plt.xlabel('Average Importance (Gain)')
            plt.title(f'Top {top_n} Feature Importance (Ordinal Avg)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            temp_path = os.path.join(tmpdir, f"feature_importance_m{model_idx}_ordinal.png")
            plt.savefig(temp_path)
            plt.close()
            mlflow.log_artifact(temp_path, artifact_path="plots/importance")
