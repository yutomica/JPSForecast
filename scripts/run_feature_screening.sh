#!/bin/bash
# run_feature_screening.sh

# エラーが発生した時点でスクリプトを終了
set -e

# 1. 学習ターゲットの定義
# 実験計画書に基づき、短期(tac)と長期(str)の両方でスクリーニングを実施
TARGET_TYPES=("tac" "str")

echo "🚀 Starting Feature Screening Process..."

python train.py \
    +experiment=screening_lgbm \
    domain=tac \
    target=tac_rank \
    mode=feature_screening \
    period=tac_standard \
    mlflow.experiment_name="Feature_Screening_TAC"

python train.py \
    +experiment=screening_lgbm \
    domain=tac \
    target=tac_gauss_rank \
    mode=feature_screening \
    period=tac_standard \
    mlflow.experiment_name="Feature_Screening_TAC"

python train.py \
    +experiment=screening_lgbm \
    domain=tac \
    target=tac_vol_scaled_residual \
    mode=feature_screening \
    period=tac_standard \
    mlflow.experiment_name="Feature_Screening_TAC"

python train.py \
    +experiment=screening_lgbm \
    domain=str \
    target=str_rank \
    mode=feature_screening \
    period=str_standard \
    mlflow.experiment_name="Feature_Screening_STR"

python train.py \
    +experiment=screening_lgbm \
    domain=str \
    target=str_risk_adj \
    mode=feature_screening \
    period=str_standard \
    mlflow.experiment_name="Feature_Screening_STR"

echo "🎉 All screening tasks completed successfully."