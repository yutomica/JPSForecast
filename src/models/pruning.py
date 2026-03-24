import numpy as np
from scipy.stats import spearmanr
import mlflow
import optuna

def calculate_spearman_ic(preds, y_true):
    """予測値と正解値のスピアマン順位相関を計算する。"""
    ic = spearmanr(preds, np.asarray(y_true).flatten())[0]
    if np.isnan(ic):
        return 0.0
    return float(ic)

def execute_epoch_pruning(epoch_callback, epoch, valid_preds, y_valid):
    """エポックごとの枝刈りコールバックを実行する。"""
    if epoch_callback is not None:
        current_ic = calculate_spearman_ic(valid_preds, y_valid)
        epoch_callback(epoch=epoch, current_score=current_ic)

def log_epoch_metrics(model_idx: int, epoch: int, metrics: dict):
    """エポック単位のメトリクスをMLflowに記録する。"""
    if mlflow.active_run():
        for key, value in metrics.items():
            if value is not None and not np.isnan(value):
                mlflow.log_metric(f"fold{model_idx}_{key}", float(value), step=epoch)

def create_pruning_callback(client, experiment_id, parent_run_id=None, fold_idx=0, past_fold_scores=None, n_startup_trials=5, warmup_ratio=0.3, total_epochs=10, pruning_patience=3, pruning_margin=0.005):
    """
    MLflowの過去履歴を用いて、エポック単位でMedianPruner相当の判定を行うコールバックを生成します。
    過去のFoldのスコアが与えられた場合は、それらとの平均（蓄積スコア）を用いて判定します。
    """
    if past_fold_scores is None:
        past_fold_scores = []
    warmup_epochs = int(total_epochs * warmup_ratio)
    target_metric = f"fold{fold_idx}_accumulated_epoch_valid_ic"
    
    # 過去のRunを取得
    filter_string = f"tags.mlflow.parentRunId = '{parent_run_id}'" if parent_run_id else ""
    runs = client.search_runs(
        experiment_ids=[experiment_id], 
        filter_string=filter_string,
        max_results=1000
    )
    
    # エポックごとのスコア履歴を集計
    epoch_scores = {}
    for run in runs:
        try:
            history = client.get_metric_history(run.info.run_id, target_metric)
            for m in history:
                epoch = m.step
                if epoch not in epoch_scores:
                    epoch_scores[epoch] = []
                epoch_scores[epoch].append(m.value)
        except Exception:
            continue
            
    # n_startup_trials以上の履歴があるエポックのみ中央値を計算
    epoch_medians = {ep: np.median(scores) for ep, scores in epoch_scores.items() if len(scores) >= n_startup_trials}
    
    # 連続で下回った回数をカウントするクロージャ用変数
    state = {"underperform_count": 0}

    def pruning_callback(epoch, current_score):
        # 過去のFoldの確定スコアと現在のエポックのスコアの平均（蓄積スコア）を算出
        if past_fold_scores:
            accumulated_score = (sum(past_fold_scores) + current_score) / (len(past_fold_scores) + 1)
        else:
            accumulated_score = current_score

        # 現在のエポックの蓄積スコアをMLflowに記録（次回のTrialの中央値計算に必要）
        mlflow.log_metric(target_metric, accumulated_score, step=epoch)
        
        # 序盤のウォームアップ期間（例：3割）は枝刈りしない
        if epoch <= warmup_epochs:
            return
            
        if epoch in epoch_medians:
            median_score = epoch_medians[epoch]
            # マージンを引いた閾値を設定
            threshold = median_score - pruning_margin
            
            print(f"   [Pruning Check] Fold {fold_idx} Epoch {epoch} | Acc IC: {accumulated_score:.4f} | Median IC: {median_score:.4f} (Threshold: {threshold:.4f})")
            if accumulated_score < threshold:
                state["underperform_count"] += 1
                if state["underperform_count"] >= pruning_patience:
                    print(f"   ⚠️ [Pruning] Model underperformed threshold for {pruning_patience} consecutive checks. Pruning trial.")
                    raise optuna.exceptions.TrialPruned()
                else:
                    print(f"   ⚠️ [Pruning Warning] Score is below threshold. (Warning {state['underperform_count']}/{pruning_patience})")
            else:
                # 基準を上回ったらカウントをリセット
                state["underperform_count"] = 0
                
    return pruning_callback