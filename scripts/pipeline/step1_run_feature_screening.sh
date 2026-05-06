#!/bin/bash
# step1_run_feature_screening.sh

set -e

# 環境変数設定
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export exp_name="Feature_Screening"
export PYTHONPATH=$PYTHONPATH:.

run_screening() {
    local domain=$1
    local model=$2
    local role=$3 # 'alpha' or 'risk'
    local target="${domain}_${role}"
    shift 3
    local extra_args=("$@")
    
    echo "============================================================"
    echo "🚀 Starting Feature Screening: $model ($domain) - $role"
    echo "============================================================"

    uv run python train.py \
        experiment=screening_lgbm \
        domain=${domain} \
        data=master \
        target=${target} \
        period=${domain}_standard \
        cv=purged_kfold \
        ++mode=feature_screening \
        "${extra_args[@]}"
        
    echo "✅ Finished Screening for $model ($domain) - $role."
    echo ""
}

# # 1. TAC (Tactical) 攻め/守り
# run_screening "tac" "lgbm" "alpha" &
# run_screening "tac" "lgbm" "risk" &

# # 2. STR (Strategic) 攻め/守り
# run_screening "str" "lgbm" "alpha" &
# run_screening "str" "lgbm" "risk"

run_screening "tac" "lgbm" "alpha_gr" \
 ++preprocess.matrix_weight.enabled=true \
 ++preprocess.matrix_weight.cost_buffer=0.005

wait
echo "🎉 All screening tasks completed successfully."
