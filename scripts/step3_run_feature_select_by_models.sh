#!/bin/bash
# run_feature_select_by_models.sh

set -e

# MLflowのバックエンドをtrain.pyと合わせる
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export exp_name="Feature_Selection"

# 特徴量選択（MDA）実行用の共通関数
run_feature_select() {
    local domain=$1
    local model=$2
    local target=$3
    local features=$4
    local hparams=$5

    # 最初の5つの引数をシフトし、残りの引数を配列として保持する
    shift 5
    local extra_args=("$@")

    # LGBM向けにGPUを有効化する引数を追加
    local gpu_args=""
    if [ "$USE_GPU" -eq 1 ] && [ "$model" = "lgbm" ]; then
        gpu_args="++hparams.device_type=gpu"
    fi

    echo "============================================================"
    echo "Starting Feature Selection: $model ($domain)"
    echo "============================================================"

    uv run python train.py \
        domain=${domain} \
        target=${target} \
        data=master_select \
        features=${features} \
        model=${model} \
        hparams=${hparams} \
        period=${domain}_standard \
        cv=purged_kfold \
        mlflow.experiment_name="${exp_name}" \
        +mode=feature_select \
        $gpu_args \
        "${extra_args[@]}"
        
    echo "Finished Feature Selection for $model ($domain)."
    echo ""
}

# run_feature_select "tac" "lgbm" "tac_vol_scaled_asym_return" "features_lgbm_tac_vol_scaled_asym_return_rough" "lgbm_tac_vol_scaled_asym_return_anc" \
#     ++hparams.early_stopping_metric="src.models.custom_metrics.calc_ndcg_10" \
#     ++hparams.metric_direction="maximize" \
#     ++hparams.num_boost_round=1000 \
#     ++hparams.custom_objective="src.models.custom_objectives.custom_asymmetric_mse" \
#     ++hparams.custom_metric="src.models.custom_objectives.custom_asymmetric_mse_eval" \
#     ++optimization_metric="ndcg_10"

run_feature_select "tac" "lgbm" "tac_max_neg_path" "features_lgbm_tac_max_neg_path_rough" "lgbm_tac_max_neg_path_anc" \
    ++hparams.early_stopping_metric="src.models.custom_metrics.calc_ap_severe" \
    ++hparams.metric_direction="maximize" \
    ++hparams.num_boost_round=1000 \
    ++hparams.objective="quantile" \
    ++hparams.metric="quantile" \
    ++hparams.alpha=0.1 \
    ++optimization_metric="AP_severe"

run_feature_select "str" "lgbm" "str_sharpe_adj" "features_lgbm_str_sharpe_adj_rough" "lgbm_str_sharpe_adj_anc" \
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
    ++optimization_metric="RankIC"

run_feature_select "str" "lgbm" "str_mdd" "features_lgbm_str_mdd_rough" "lgbm_str_mdd_anc" \
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
    ++optimization_metric="AP_severe_STR"


# run_feature_select "tac" "tcn" "tac_vol_scaled_asym_return" "features_tcn_tac_vol_scaled_asym_return_rough" "tcn_tac_vol_scaled_asym_return_anc" \
#     ++hparams.objective="asymmetric_mse" \
#     ++preprocess.target_stratified_sampling.mode=mode_2 \
#     ++preprocess.target_stratified_sampling.center_keep_ratio=0.2 \
#     ++preprocess.target_stratified_sampling.other_keep_ratio=0.5 \
#     ++optimization_metric="ndcg_10"
