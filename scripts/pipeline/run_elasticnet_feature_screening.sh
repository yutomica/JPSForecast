#!/bin/bash
# run_elasticnet_feature_screening.sh

set -e

echo "======================================================"
echo "🚀 Starting ElasticNet Alpha Search via Hydra Sweeper"
echo "======================================================"

# TAC alpha
uv run python train.py \
    experiment=search_elasticnet_alpha \
    domain=tac \
    data=master \
    target=tac_alpha \
    model=elasticnet \
    hparams=elasticnet/base,elasticnet/tac_alpha \
    mode=feature_screening \
    period=tac_standard \
    cv=purged_kfold \
    mlflow.experiment_name="Feature_Screening"

# TAC risk
uv run python train.py \
    experiment=search_elasticnet_alpha \
    domain=tac \
    data=master \
    target=tac_risk \
    model=elasticnet \
    hparams=elasticnet/base,elasticnet/tac_risk \
    mode=feature_screening \
    period=tac_standard \
    cv=purged_kfold \
    mlflow.experiment_name="Feature_Screening"

# STR alpha
uv run python train.py \
    experiment=search_elasticnet_alpha \
    domain=str \
    data=master \
    target=str_alpha \
    model=elasticnet \
    hparams=elasticnet/base,elasticnet/str_alpha \
    mode=feature_screening \
    period=str_standard \
    cv=purged_kfold \
    mlflow.experiment_name="Feature_Screening"

# STR risk
uv run python train.py \
    experiment=search_elasticnet_alpha \
    domain=str \
    data=master \
    target=str_risk \
    model=elasticnet \
    hparams=elasticnet/base,elasticnet/str_risk \
    mode=feature_screening \
    period=str_standard \
    cv=purged_kfold \
    mlflow.experiment_name="Feature_Screening"

echo "======================================================"
echo "✅ Alpha Search Completed. Check MLflow UI."
echo "======================================================"
