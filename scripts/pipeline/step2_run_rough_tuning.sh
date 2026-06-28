#!/bin/bash
# step2_run_rough_tuning.sh

set -e

# 並列処理時のスレッド競合を防ぐための環境変数設定
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1

# MLflow設定
DB_ABS_PATH="$(pwd)/mlflow.db"
export MLFLOW_TRACKING_URI="sqlite:///${DB_ABS_PATH}"
export PYTHONPATH=$PYTHONPATH:.

run_sweep() {
    local domain=$1
    local model=$2
    local role=$3 # 'alpha' or 'risk'
    local n_jobs=$4
    local use_gpu=$5
    shift 5
    local extra_args=("$@")
    
    local target="${domain}_${role}"
    local features="features_${model}_${target}_rough"
    local sweep="${model}_${target}_rough"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local exp_name="JPSForecast_${target}"

    local study_name=${OPTUNA_STUDY_NAME:-"rough_tuning_${model}_${target}_${timestamp}"}

    # 再開時のオフセット取得
    local trial_offset=$(uv run python -m src.utils.config_utils --action get_trial_count --storage "sqlite:///optuna.db" --study-name "${study_name}" --state COMPLETE)
    echo "Completed Trials (Offset): ${trial_offset}"

    # 設定ファイルから n_trials を取得
    local config_n_trials=$(grep "n_trials:" "config/sweep/${sweep}.yaml" | awk '{print $2}' | head -n 1)
    config_n_trials=$(echo "${config_n_trials}" | sed 's/[^0-9]//g')
    local total_n_trials=${OPTUNA_N_TRIALS:-${config_n_trials:-50}}
    local remaining_trials=$((total_n_trials - trial_offset))

    if [ "${remaining_trials}" -le 0 ]; then
        echo "All trials for ${study_name} are already completed. Skipping."
        return
    fi

    local gpu_args=""
    if [ "$use_gpu" -eq 1 ]; then
        if [ "$model" = "lgbm" ]; then
            gpu_args="++hparams.device_type=gpu"
        else
            gpu_args="++hparams.device_name=auto"
        fi
    fi

    echo "============================================================"
    echo "Starting Sweep: $model ($domain) - $role"
    echo "Study Name: $study_name"
    echo "============================================================"
    
    local parent_run_id=$(uv run python -m src.utils.mlflow_utils --action resolve_parent --tracking-uri "${MLFLOW_TRACKING_URI}" --experiment-name "${exp_name}" --study-name "${study_name}")

    # Hydra実行。hparamsを理論レイヤー（base継承済）に修正
    TRIAL_OFFSET=$trial_offset MLFLOW_PARENT_RUN_ID=$parent_run_id uv run python train.py -m \
        hydra/launcher=$( [ "${n_jobs}" -gt 1 ] && echo "joblib" || echo "basic" ) \
        $([ "${n_jobs}" -gt 1 ] && echo "hydra.sweeper.n_jobs=${n_jobs} hydra.launcher.n_jobs=${n_jobs}") \
        hydra.sweeper.n_trials=${remaining_trials} \
        ++hparams.num_threads=1 \
        domain=${domain} \
        target=${target} \
        data=master_select \
        model=${model} \
        period=${domain}_standard \
        features=${features} \
        experiment=${model}_${target} \
        ++mlflow.experiment_name="${exp_name}" \
        ++mlflow.run_name="Step2_Rough_Tuning_${model}_${target}" \
        sweep=${sweep} \
        cv=cpcv \
        hydra.sweeper.study_name="${study_name}" \
        $gpu_args \
        "${extra_args[@]}"

        
    echo "Finished $model ($domain) - $role."
    echo ""
}

# --- Execution ---
run_sweep "tac" "lgbm" "alpha_gr" "9" "0" 
# run_sweep "str" "lgbm" "alpha" "9" "0" 
# run_sweep "tac" "gandalf" "alpha_gr" "6" "1"

# run_sweep "tac" "tcn" "alpha_gr" "3" "1"

# run_sweep "10d" "lgbm" "alpha_gr" "9" "0" 
# run_sweep "20d" "lgbm" "alpha_gr" "9" "0" 
# run_sweep "40d" "lgbm" "alpha_gr" "9" "0" 