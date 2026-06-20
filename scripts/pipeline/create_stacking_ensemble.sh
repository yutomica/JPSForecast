#!/bin/bash
# step5.5_create_stacking_ensemble.sh
# スタッキングモデル（メタモデル）を学習し、Productionステージとして登録します。

set -e

# 環境変数設定
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# MLflow設定
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export PYTHONPATH=$PYTHONPATH:.

domain="tac"
role="alpha_gr"
target="${domain}_${role}"
model="lgbm" # メタモデルとしてのLGBM

echo "============================================================"
echo "🚀 Creating STACKING ENSEMBLE Model for $target"
echo "============================================================"

# スタッキングアンサンブルの学習
# Base_OOFから特徴量を引き継ぎ、アンサンブルのパフォーマンスを検証・登録します
uv run python train.py \
    experiment=stacking_context_aware \
    domain=${domain} \
    target=${target} \
    period=${domain}_standard \
    +mode=stacking_ensemble \
    cv=anchored_walk_forward \
    ++mlflow.experiment_name="JPSForecast_Stacking_${target}" \
    ++mlflow.run_name="StackingEnsemble_${model}_${target}" 

echo "✅ Finished Stacking Ensemble model creation."
