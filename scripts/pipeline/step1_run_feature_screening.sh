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
    local role=$2 # 'alpha' or 'risk'
    local target="${domain}_${role}"
    local exp_name="JPSForecast_${target}"
    shift 2
    local extra_args=("$@")
    
    echo "============================================================"
    echo "🚀 Starting Feature Screening: $model ($domain) - $role"
    echo "============================================================"

    uv run python train.py \
        experiment=lgbm_${domain}_${role} \
        domain=${domain} \
        data=master \
        target=${target} \
        period=${domain}_standard \
        cv=purged_kfold \
        ++mode=feature_screening \
        ++mlflow.experiment_name="${exp_name}" \
        ++mlflow.run_name="Step1_Feature_Screening" \
        ++hparams.max_depth=3 \
        ++hparams.num_leaves=7 \
        ++hparams.min_child_samples=2000 \
        ++hparams.feature_fraction=0.3 \
        ++hparams.extra_trees=true \
        ++hparams.bagging_fraction=0.7 \
        ++hparams.bagging_freq=1 \
        ++hparams.learning_rate=0.05 \
        ++hparams.num_boost_round=1000 \
        "${extra_args[@]}"
    
    # uv run python ./scripts/pipeline/feature_allocation.py target=${target} features=features_${domain}_${role}_init data=master
        
    echo "✅ Finished Screening for $model ($domain) - $role."
    echo ""
}

# run_screening "str" "alpha"
# run_screening "tac" "alpha_tb_5_3"
# run_screening "10d" "alpha_gr"
run_screening "20d" "alpha_gr" &
run_screening "40d" "alpha_gr"

# # 2. STR (Strategic) 攻め/守り
# run_screening "str" "lgbm" "alpha" &
# run_screening "str" "lgbm" "risk"

# run_screening "tac" "lgbm" "alpha_gr" \
#  ++preprocess.matrix_weight.enabled=true \
#  ++preprocess.matrix_weight.cost_buffer=0.005

wait
echo "🎉 All screening tasks completed successfully."
