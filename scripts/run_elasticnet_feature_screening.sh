#!/bin/bash

# エラー発生時にスクリプトを停止
set -e

echo "======================================================"
echo "🚀 Starting ElasticNet Alpha Search via Hydra Sweeper"
echo "======================================================"

uv python train.py \
    experiment=search_elasticnet_alpha \
    domain=tac \
    data=master \
    target=tac_vol_scaled_asym_return \
    mode=feature_screening \
    period=tac_standard \
    cv=purged_kfold \
    mlflow.experiment_name="Feature_Screening" \
    ++hparams.loss="asymmetric_mse" &

python train.py \
    experiment=search_elasticnet_alpha \
    domain=tac \
    data=master \
    target=tac_max_neg_path \
    mode=feature_screening \
    period=tac_standard \
    cv=purged_kfold \
    mlflow.experiment_name="Feature_Screening" \
    ++hparams.loss="quantile" \
    ++hparams.metric="quantile" \
    ++hparams.alpha=0.1 \
    ++hparams.min_child_samples=10 &

python train.py \
    experiment=search_elasticnet_alpha \
    domain=str \
    data=master \
    target=str_sharpe_adj \
    mode=feature_screening \
    period=str_standard \
    cv=purged_kfold \
    mlflow.experiment_name="Feature_Screening" \
    ++preprocess.target_stratified_sampling.mode=mode_3 \
    '++preprocess.target_stratified_sampling.weight_dict={tail:3.0,center:0.5,other:1.0}' \
    ++hparams.loss="fair" \
    ++hparams.metric="fair" \
    ++hparams.fair_c=10.0 &

uv run python train.py -m \
    experiment=search_elasticnet_alpha \
    domain=str \
    data=master \
    target=str_mdd \
    mode=feature_screening \
    period=str_standard \
    cv=purged_kfold \
    mlflow.experiment_name="Feature_Screening" \
    ++preprocess.target_stratified_sampling.mode=mode_3 \
    '++preprocess.target_stratified_sampling.weight_dict={tail:3.0,center:0.5,other:1.0}' \
    ++hparams.loss="tweedie" \
    ++hparams.metric="tweedie" \
    ++hparams.tweedie_variance_power=1.2

echo "======================================================"
echo "✅ Alpha Search Completed. Check MLflow UI for the best alpha."
echo "======================================================"


python train.py \
    +experiment=screening_lgbm \
    domain=str \
    data=master \
    target=str_mdd \
    mode=feature_screening \
    period=str_standard \
    cv=purged_kfold \
    mlflow.experiment_name="Feature_Screening" \
    ++preprocess.target_stratified_sampling.mode=mode_3 \
    '++preprocess.target_stratified_sampling.weight_dict={tail:3.0,center:0.5,other:1.0}' \
    ++hparams.loss="tweedie" \
    ++hparams.metric="tweedie" \
    ++hparams.tweedie_variance_power=1.2 &