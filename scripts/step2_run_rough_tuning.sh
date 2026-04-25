#!/bin/bash
# run_rough_tuning.sh

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
DB_ABS_PATH="$(pwd)/mlflow.db"
export MLFLOW_TRACKING_URI="sqlite:///${DB_ABS_PATH}"
export exp_name="Rough_Tuning"

# Sweep実行用の共通関数
run_sweep() {
    local domain=$1
    local model=$2
    local target=$3
    local features=$4
    local n_jobs=$5
    local use_gpu=$6
    # 最初の6つの引数をシフトし、残りの引数を配列として保持する
    shift 6
    local extra_args=("$@")
    local timestamp=$(date +"%Y%m%d_%H%M%S")

    # GPUを有効化する引数を追加 (LGBMとPyTorchモデルで分岐)
    local gpu_args=""
    if [ "$use_gpu" -eq 1 ]; then
        if [ "$model" = "lgbm" ]; then
            gpu_args="++hparams.device_type=gpu"
        else
            gpu_args="++hparams.device_name=auto"
        fi
    fi

    echo "============================================================"
    echo "Starting Sweep: $model ($domain)"
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

    # Hydraを実行。n_jobs > 1 の場合のみjoblibランチャーを使用
    if [ "${n_jobs}" -gt 1 ]; then
        echo "Running with joblib launcher (n_jobs=${n_jobs})"
        MLFLOW_PARENT_RUN_ID=$parent_run_id uv run python train.py -m \
            hydra/launcher=joblib \
            hydra.sweeper.n_jobs=${n_jobs} \
            hydra.launcher.n_jobs=${n_jobs} \
            ++hparams.num_threads=1 \
            domain=${domain} \
            cv=cpcv \
            data=master_select \
            model=${model} \
            period=${domain}_standard \
            features=${features} \
            target=${target} \
            hparams=${model}_default \
            sweep=${model}_rough \
            mlflow.experiment_name="${exp_name}" \
            hydra.sweeper.study_name=rough_tuning_${model}_${target}_${timestamp} \
            $gpu_args \
            "${extra_args[@]}"
    else
        echo "Running sequentially (n_jobs=${n_jobs})"
        MLFLOW_PARENT_RUN_ID=$parent_run_id uv run python train.py -m \
            domain=${domain} \
            cv=cpcv \
            data=master_select \
            model=${model} \
            period=${domain}_standard \
            features=${features} \
            target=${target} \
            hparams=${model}_default \
            sweep=${model}_rough \
            mlflow.experiment_name="${exp_name}" \
            hydra.sweeper.study_name=rough_tuning_${model}_${target}_${timestamp} \
            $gpu_args \
            "${extra_args[@]}"
    fi
        
    echo "Finished $model ($domain)."
    echo ""
}


# run_sweep "tac" "lgbm" "tac_vol_scaled_asym_return" "features_lgbm_tac_vol_scaled_asym_return_rough" "8" "0" \
#     ++hparams.early_stopping_metric="src.models.custom_metrics.calc_ndcg_10" \
#     ++hparams.metric_direction="maximize" \
#     ++hparams.num_boost_round=1000 \
#     ++hparams.custom_objective="src.models.custom_objectives.custom_asymmetric_mse" \
#     ++hparams.custom_metric="src.models.custom_objectives.custom_asymmetric_mse_eval" \
#     hparams.min_child_samples="choice(50,100,200,500)"

# run_sweep "tac" "lgbm" "tac_max_neg_path" "features_lgbm_tac_max_neg_path_rough" "8" "0" \
#     ++hparams.early_stopping_metric="src.models.custom_metrics.calc_ap_severe" \
#     ++hparams.metric_direction="maximize" \
#     ++hparams.num_boost_round=1000 \
#     ++hparams.objective="quantile" \
#     ++hparams.metric="quantile" \
#     ++hparams.alpha=0.1 \
#     hparams.min_child_samples="choice(50,100,200,500)"

# run_sweep "str" "lgbm" "str_sharpe_adj" "features_lgbm_str_sharpe_adj_rough" "8" "0" \
#     ++hparams.early_stopping_metric="src.models.custom_metrics.calc_rank_ic_reb" \
#     ++hparams.metric_direction="maximize" \
#     ++preprocess.target_stratified_sampling.mode=mode_3 \
#     '++preprocess.target_stratified_sampling.weight_dict={tail:1.5,center:0.5,other:1.0}' \
#     ++preprocess.sampling.enabled=true \
#     ++preprocess.sampling.interval=11 \
#     ++hparams.num_boost_round=1000 \
#     ++hparams.objective="fair" \
#     ++hparams.metric="fair" \
#     ++hparams.fair_c=10.0 \
#     hparams.min_child_samples="choice(1000,2000,4000,8000)"

# run_sweep "str" "lgbm" "str_mdd" "features_lgbm_str_mdd_rough" "8" "0" \
#     ++hparams.early_stopping_metric="src.models.custom_metrics.calc_pr_auc_30pt" \
#     ++hparams.metric_direction="maximize" \
#     ++preprocess.target_stratified_sampling.mode=mode_3 \
#     '++preprocess.target_stratified_sampling.weight_dict={tail:3.0,center:0.5,other:1.0}' \
#     ++preprocess.sampling.enabled=true \
#     ++preprocess.sampling.interval=11 \
#     ++hparams.num_boost_round=1000 \
#     ++hparams.objective="tweedie" \
#     ++hparams.metric="tweedie" \
#     ++hparams.tweedie_variance_power=1.2 \
#     hparams.min_child_samples="choice(1000,2000,4000,8000)"

run_sweep "tac" "tcn" "tac_vol_scaled_asym_return" "features_tcn_tac_vol_scaled_asym_return_rough" "8" "1" \
    ++hparams.early_stopping_metric="src.models.custom_metrics.calc_ndcg_10" \
    ++hparams.metric_direction="maximize" \
    ++hparams.early_stopping_ema_alpha=0.33 \
    ++hparams.objective="asymmetric_mse" \
    ++preprocess.target_stratified_sampling.mode=mode_2 \
    ++preprocess.target_stratified_sampling.center_keep_ratio=0.2 \
    ++preprocess.target_stratified_sampling.other_keep_ratio=0.5 \
    hparams.window_size="choice(10,20,40)"

# run_sweep "tac" "ft_transformer" "tac_vol_scaled_asym_return" "features_ft_transformer_tac_vol_scaled_asym_return_rough" "8" "1" \
#     ++hparams.objective="asymmetric_mse" \
#     ++preprocess.target_stratified_sampling.mode=mode_2 \
#     ++preprocess.target_stratified_sampling.center_keep_ratio=0.2 \
#     ++preprocess.target_stratified_sampling.other_keep_ratio=0.5

# run_sweep "tac" "tcn" "tac_max_neg_path" "features_tcn_tac_max_neg_path_rough" "8" "1" \
#     ++hparams.objective="quantile" \
#     ++hparams.metric="quantile" \
#     ++hparams.alpha=0.1 \
#     ++preprocess.target_stratified_sampling.mode=mode_2 \
#     ++preprocess.target_stratified_sampling.center_keep_ratio=0.2 \
#     ++preprocess.target_stratified_sampling.other_keep_ratio=0.5 \
#     model.window_size.tac="choice(20,90)"

# run_sweep "tac" "ft_transformer" "tac_max_neg_path" "features_ft_transformer_tac_max_neg_path_rough" "8" "1" \
#     ++hparams.objective="quantile" \
#     ++hparams.metric="quantile" \
#     ++hparams.alpha=0.1 \
#     ++preprocess.target_stratified_sampling.mode=mode_2 \
#     ++preprocess.target_stratified_sampling.center_keep_ratio=0.2 \
#     ++preprocess.target_stratified_sampling.other_keep_ratio=0.5


# run_sweep "str" "tcn" "str_sharpe_adj" "features_tcn_str_sharpe_adj_rough" "8" "1" \
#     ++hparams.objective="fair" \
#     ++hparams.metric="fair" \
#     ++hparams.fair_c=10.0 \
#     ++preprocess.target_stratified_sampling.enabled=false \
#     ++preprocess.sampling.enabled=true \
#     ++preprocess.sampling.interval=11 \
#     model.window_size.str="choice(126,252)"

# run_sweep "str" "ft_transformer" "str_sharpe_adj" "features_ft_transformer_str_sharpe_adj_rough" "8" "1" \
#     ++hparams.objective="fair" \
#     ++hparams.metric="fair" \
#     ++hparams.fair_c=10.0 \
#     ++preprocess.target_stratified_sampling.enabled=false \
#     ++preprocess.sampling.enabled=true \
#     ++preprocess.sampling.interval=11

# run_sweep "str" "tcn" "str_mdd" "features_tcn_str_mdd_rough" "8" "1" \
#     ++hparams.objective="tweedie" \
#     ++hparams.metric="tweedie" \
#     ++hparams.tweedie_variance_power=1.2 \
#     ++preprocess.target_stratified_sampling.enabled=false \
#     ++preprocess.sampling.enabled=true \
#     ++preprocess.sampling.interval=11 \
#     model.window_size.str="choice(126,252)"

# run_sweep "str" "ft_transformer" "str_mdd" "features_ft_transformer_str_mdd_rough" "8" "1" \
#     ++hparams.objective="tweedie" \
#     ++hparams.metric="tweedie" \
#     ++hparams.tweedie_variance_power=1.2 \
#     ++preprocess.target_stratified_sampling.enabled=false \
#     ++preprocess.sampling.enabled=true \
#     ++preprocess.sampling.interval=11