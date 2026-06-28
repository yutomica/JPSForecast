#!/bin/bash
# step3_run_feature_select_by_models.sh

set -e

# 環境変数設定
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1

DB_ABS_PATH="$(pwd)/mlflow.db"
export MLFLOW_TRACKING_URI="sqlite:///${DB_ABS_PATH}"
export PYTHONPATH=$PYTHONPATH:.

run_feature_select() {
    local domain=$1
    local model=$2
    local role=$3 # 'alpha' or 'risk'
    local use_gpu=$4
    shift 4
    local extra_args=("$@")

    local target="${domain}_${role}"
    local features="features_${model}_${target}_rough"
    local exp_name="JPSForecast_${target}"

    local gpu_args=""
    if [ "$use_gpu" -eq 1 ]; then
        if [ "$model" = "lgbm" ]; then gpu_args="++hparams.device_type=gpu"; else gpu_args="++hparams.device_name=auto"; fi
    fi

    echo "============================================================"
    echo "Starting Feature Selection: $model ($domain) - $role"
    echo "============================================================"

    # hparams の指定を削除。Experiment 側の defaults で解決する。
    uv run python train.py \
        domain=${domain} \
        target=${target} \
        data=master_select \
        features=${features} \
        model=${model} \
        period=${domain}_standard \
        cv=purged_kfold \
        experiment=${model}_${target} \
        mlflow.experiment_name="${exp_name}" \
        ++mlflow.run_name="Step3_Feature_Selection_${model}_${target}" \
        ++mode=feature_select \
        $gpu_args \
        "${extra_args[@]}"
        
    echo "Finished Feature Selection for $model ($domain) - $role."
    echo ""
}

# --- Execution ---
run_feature_select "tac" "lgbm" "alpha_gr" "0"
# run_feature_select "str" "lgbm" "alpha"  "0" \
#     ++hparams.num_boost_round=1000 
# run_feature_select "tac" "gandalf" "alpha_gr"  "1"

# run_feature_select "tac" "tcn" "alpha_gr" "1"
# run_feature_select "10d" "lgbm" "alpha_gr" "0"
# run_feature_select "20d" "lgbm" "alpha_gr" "0"
# run_feature_select "40d" "lgbm" "alpha_gr" "0"