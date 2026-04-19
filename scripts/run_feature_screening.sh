#!/bin/bash
# run_feature_screening.sh

# エラーが発生した時点でスクリプトを終了
set -e

# 1. 学習ターゲットの定義
# 実験計画書に基づき、短期(tac)と長期(str)の両方でスクリーニングを実施
TARGET_TYPES=("tac" "str")

echo "🚀 Starting Feature Screening Process..."

# python train.py \
#     +experiment=screening_lgbm \
#     domain=tac \
#     data=master \
#     target=tac_vol_scaled_asym_return \
#     mode=feature_screening \
#     period=tac_standard \
#     cv=purged_kfold \
#     mlflow.experiment_name="Feature_Screening" \
#     ++hparams.custom_objective="src.models.custom_objectives.custom_asymmetric_mse" \
#     ++hparams.custom_metric="src.models.custom_objectives.custom_asymmetric_mse_eval" &

# python train.py \
#     +experiment=screening_lgbm \
#     domain=tac \
#     data=master \
#     target=tac_max_neg_path \
#     mode=feature_screening \
#     period=tac_standard \
#     cv=purged_kfold \
#     mlflow.experiment_name="Feature_Screening" \
#     ++hparams.objective="quantile" \
#     ++hparams.metric="quantile" \
#     ++hparams.alpha=0.1 &

python train.py \
    +experiment=screening_lgbm \
    domain=str \
    data=master \
    target=str_sharpe_adj \
    mode=feature_screening \
    period=str_standard \
    cv=purged_kfold \
    mlflow.experiment_name="Feature_Screening" \
    ++preprocess.sampling.enabled=true \
    ++preprocess.sampling.interval=11 \
    ++preprocess.target_stratified_sampling.mode=mode_3 \
    '++preprocess.target_stratified_sampling.weight_dict={tail:1.5,center:0.5,other:1.0}' \
    ++hparams.objective="fair" \
    ++hparams.metric="fair" \
    ++hparams.fair_c=10.0 &

python train.py \
    +experiment=screening_lgbm \
    domain=str \
    data=master \
    target=str_mdd \
    mode=feature_screening \
    period=str_standard \
    cv=purged_kfold \
    mlflow.experiment_name="Feature_Screening" \
    ++preprocess.sampling.enabled=true \
    ++preprocess.sampling.interval=11 \
    ++preprocess.target_stratified_sampling.mode=mode_3 \
    '++preprocess.target_stratified_sampling.weight_dict={tail:3.0,center:0.5,other:1.0}' \
    ++hparams.objective="tweedie" \
    ++hparams.metric="tweedie" \
    ++hparams.tweedie_variance_power=1.2

# 全てのバックグラウンドプロセスの完了を待機
wait

echo "🎉 All screening tasks completed successfully."