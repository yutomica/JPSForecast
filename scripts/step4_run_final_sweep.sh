#!/bin/bash
# run_final_sweep.sh

set -e

# 並列処理時のスレッド競合（オーバーサブスクリプション）を防ぐための環境変数設定
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Joblib(loky)がワーカー起動時にスレッド数を全コア数に自動設定してしまうのを防ぐ
export LOKY_MAX_CPU_COUNT=1

# MLflowのバックエンドをtrain.pyと合わせる
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export exp_name="Final_Sweep"

# ドライランモードの判定
DRY_RUN=0
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

run_final_sweep() {
    local domain=$1
    local model=$2
    local target=$3
    local hparams=$4
    local n_jobs=$5
    local use_gpu=$6
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    shift 6
    local extra_args=("$@")

    # GPUを有効化する引数を追加 (LGBMとPyTorchモデルで分岐)
    local gpu_args=""
    if [ "$use_gpu" -eq 1 ]; then
        if [ "$model" = "lgbm" ]; then
            # lgbmについてはGPUを使用しない
            gpu_args=""
        else
            gpu_args="++hparams.device_name=auto"
        fi
    fi

    # ドライランモード時の設定上書き
    local data_arg="data=master"
    local dry_run_args=()
    if [ "$DRY_RUN" -eq 1 ]; then
        data_arg="data=sample"
        # Optunaの試行回数を最小限(2回)に制限
        dry_run_args+=("hydra.sweeper.n_trials=2")
        if [ "$model" = "lgbm" ]; then
            dry_run_args+=("++hparams.num_boost_round=2")
        else
            dry_run_args+=("++hparams.max_epochs=2")
        fi
        echo "⚠️ Running in DRY RUN mode: Using sample data and minimal epochs/trials."
    fi

    echo "============================================================"
    echo "Starting Final Sweep: $model ($domain)"
    echo "============================================================"
    echo "Creating Parent Run for $exp_name..."

    # 親ランを作成し、Run IDを取得
    local parent_run_id=$(uv run python -c "
import mlflow
from mlflow.tracking import MlflowClient
import datetime
mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}')
client = MlflowClient()
exp = client.get_experiment_by_name('${exp_name}')
if exp and exp.lifecycle_stage == 'deleted':
    client.restore_experiment(exp.experiment_id)
mlflow.set_experiment('${exp_name}')
run = mlflow.start_run(run_name=f'Sweep_{datetime.datetime.now().strftime(\"%Y%m%d_%H%M%S\")}')
print(run.info.run_id)
")

    echo "Parent Run ID: $parent_run_id"

    # Optunaの状態をDBに保存し、ワーカー間で共有できるようにする（ダッシュボード確認のためにも必須）
    local optuna_args=(
        "hydra.sweeper.storage=sqlite:///optuna.db"
        "hydra.sweeper.study_name=${exp_name}_${model}_${target}_${timestamp}"
    )

    # Hydraを実行。n_jobs > 1 の場合のみjoblibランチャーを使用
    if [ "${n_jobs}" -gt 1 ]; then
        echo "Running with joblib launcher (n_jobs=${n_jobs})"
        MLFLOW_PARENT_RUN_ID=$parent_run_id uv run python train.py -m \
            hydra/launcher=joblib \
            hydra.sweeper.n_jobs=${n_jobs} \
            hydra.launcher.n_jobs=${n_jobs} \
            ++hparams.num_threads=1 \
            domain=${domain} \
            target=${target} \
            ${data_arg} \
            features=features_${model}_${target}_fixed \
            model=${model} \
            hparams=${hparams} \
            period=${domain}_standard \
            cv=purged_kfold \
            mlflow.experiment_name="${exp_name}" \
            sweep=${model}_${target} \
            +mode=final_sweep \
            "${optuna_args[@]}" \
            $gpu_args \
            "${extra_args[@]}" \
            "${dry_run_args[@]}"
    else
        echo "Running sequentially (n_jobs=${n_jobs})"
        MLFLOW_PARENT_RUN_ID=$parent_run_id uv run python train.py -m \
            domain=${domain} \
            target=${target} \
            ${data_arg} \
            features=features_${model}_${target}_fixed \
            model=${model} \
            hparams=${hparams} \
            period=${domain}_standard \
            cv=purged_kfold \
            mlflow.experiment_name="${exp_name}" \
            sweep=${model}_${target} \
            +mode=final_sweep \
            "${optuna_args[@]}" \
            $gpu_args \
            "${extra_args[@]}" \
            "${dry_run_args[@]}"
    fi
        
    echo "Finished Feature Selection for $model ($domain)."
    echo ""
}

run_final_sweep "tac" "lgbm"    "tac_vol_scaled_asym_return"  "lgbm_tac_vol_scaled_asym_return_anc" "16" "0" \
    ++hparams.early_stopping_metric="src.models.custom_metrics.calc_ndcg_10" \
    ++hparams.metric_direction="maximize" \
    ++hparams.num_boost_round=1000 \
    ++hparams.custom_objective="src.models.custom_objectives.custom_asymmetric_mse" \
    ++hparams.custom_metric="src.models.custom_objectives.custom_asymmetric_mse_eval" \
    ++optimization_metric="worst_fold_ndcg_10"

run_final_sweep "tac" "lgbm"    "tac_max_neg_path"  "lgbm_tac_max_neg_path_anc" "16" "0" \
    ++hparams.early_stopping_metric="src.models.custom_metrics.calc_ap_severe" \
    ++hparams.metric_direction="maximize" \
    ++hparams.num_boost_round=1000 \
    ++hparams.objective="quantile" \
    ++hparams.metric="quantile" \
    ++hparams.alpha=0.1 \
    ++optimization_metric="worst_fold_ap_severe"

run_final_sweep "str" "lgbm"    "str_sharpe_adj"  "lgbm_str_sharpe_adj_anc" "16" "0" \
    ++hparams.early_stopping_metric="src.models.custom_metrics.calc_rank_ic_reb" \
    ++hparams.metric_direction="maximize" \
    ++preprocess.target_stratified_sampling.mode=mode_3 \
    '++preprocess.target_stratified_sampling.weight_dict={tail:1.5,center:0.5,other:1.0}' \
    ++preprocess.sampling.enabled=true \
    ++preprocess.sampling.interval=11 \
    ++hparams.num_boost_round=1000 \
    ++hparams.objective="fair" \
    ++hparams.metric="fair" \
    ++hparams.fair_c=10.0 \
    ++optimization_metric="daily_icir_reb"

run_final_sweep "str" "lgbm"    "str_mdd"  "lgbm_str_mdd_anc" "16" "0" \
    ++hparams.early_stopping_metric="src.models.custom_metrics.calc_pr_auc_30pt" \
    ++hparams.metric_direction="maximize" \
    ++preprocess.target_stratified_sampling.mode=mode_3 \
    '++preprocess.target_stratified_sampling.weight_dict={tail:3.0,center:0.5,other:1.0}' \
    ++preprocess.sampling.enabled=true \
    ++preprocess.sampling.interval=11 \
    ++hparams.num_boost_round=1000 \
    ++hparams.objective="tweedie" \
    ++hparams.metric="tweedie" \
    ++hparams.tweedie_variance_power=1.2 \
    ++optimization_metric="worst_fold_AP_severe_STR"