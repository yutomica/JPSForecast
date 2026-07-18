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
    local exp_name="JPSForecast_${target}"
    local parent_run_name="Step2_Rough_Tuning_${model}_${target}"
    if [ -n "${STEP2_PARENT_RUN_SUFFIX:-}" ]; then
        parent_run_name="${parent_run_name}_${STEP2_PARENT_RUN_SUFFIX}"
    fi
    if [ -n "${STEP2_PARENT_RUN_NAME:-}" ]; then
        parent_run_name="${STEP2_PARENT_RUN_NAME}"
    fi

    # 設定ファイルから n_trials を取得
    local config_n_trials=$(grep "n_trials:" "config/sweep/${sweep}.yaml" | awk '{print $2}' | head -n 1)
    config_n_trials=$(echo "${config_n_trials}" | sed 's/[^0-9]//g')
    local total_n_trials=${OPTUNA_N_TRIALS:-${config_n_trials:-50}}

    local gpu_args=()
    if [ "$use_gpu" -eq 1 ]; then
        if [ "$model" = "lgbm" ]; then
            gpu_args+=("++hparams.device_type=gpu")
        else
            gpu_args+=("++hparams.device_name=auto")
        fi
    fi

    local rough_artifact_args=(
        "++artifacts.log_model=false"
    )

    local tcn_cache_args=()
    if [ "$model" = "tcn" ]; then
        local tcn_sequence_cache_enabled="${TCN_SEQUENCE_CACHE_ENABLED:-false}"
        tcn_cache_args+=(
            "++hparams.sequence_cache_enabled=${tcn_sequence_cache_enabled}"
            "++hparams.log_learning_curve=false"
        )
        case "${tcn_sequence_cache_enabled}" in
            1|true|TRUE|True|yes|YES|Yes|on|ON|On)
                local tcn_sequence_cache_dir="${TCN_SEQUENCE_CACHE_DIR:-/private/tmp/jps_tcn_sequence_cache}"
                tcn_cache_args+=("++hparams.sequence_cache_dir=${tcn_sequence_cache_dir}")
                ;;
        esac
    fi

    local base_args=(
        "++hparams.num_threads=1"
        "domain=${domain}"
        "target=${target}"
        "data=master_select"
        "model=${model}"
        "period=${domain}_standard"
        "features=${features}"
        "experiment=${model}_${target}"
        "++mlflow.experiment_name=${exp_name}"
        "++mlflow.run_name=${parent_run_name}"
        "sweep=${sweep}"
        "cv=cpcv"
        "${gpu_args[@]}"
        "${rough_artifact_args[@]}"
        "${tcn_cache_args[@]}"
        "${extra_args[@]}"
    )

    local runner_args=(
        "--tracking-uri" "${MLFLOW_TRACKING_URI}"
        "--experiment-name" "${exp_name}"
        "--parent-run-name" "${parent_run_name}"
        "--sweep-config" "config/sweep/${sweep}.yaml"
        "--max-paths" "${total_n_trials}"
        "--n-jobs" "${n_jobs}"
        "--train-script" "train.py"
        "--child-run-name-style" "trial"
    )
    if [ -n "${STEP2_PARENT_RUN_ID:-}" ]; then
        runner_args+=("--parent-run-id" "${STEP2_PARENT_RUN_ID}")
    fi
    if [ "${STEP2_INCLUDE_DELETED_RUNS:-0}" = "1" ]; then
        runner_args+=("--include-deleted-runs")
    fi
    if [ "${STEP2_DRY_RUN:-0}" = "1" ]; then
        runner_args+=("--dry-run")
    fi
    for arg in "${base_args[@]}"; do
        runner_args+=("--base-arg" "${arg}")
    done

    echo "============================================================"
    echo "Starting Sweep: $model ($domain) - $role"
    echo "Parent Run Name: $parent_run_name"
    if [ -n "${STEP2_PARENT_RUN_ID:-}" ]; then
        echo "Parent Run ID: ${STEP2_PARENT_RUN_ID}"
    fi
    if [ "${STEP2_INCLUDE_DELETED_RUNS:-0}" = "1" ]; then
        echo "Include Deleted Runs: true"
    fi
    if [ "${STEP2_DRY_RUN:-0}" = "1" ]; then
        echo "Dry Run: true"
    fi
    echo "============================================================"

    uv run python -m src.utils.mlflow_grid_resume "${runner_args[@]}"

        
    echo "Finished $model ($domain) - $role."
    echo ""
}

# --- Execution ---
# run_sweep "tac" "lgbm" "alpha_gr" "9" "0" 
# run_sweep "str" "lgbm" "alpha" "9" "0" 
# run_sweep "tac" "gandalf" "alpha_gr" "3" "1"
# run_sweep "tac" "tcn" "alpha_gr" "2" "1"
run_sweep "tac" "lgbm" "tb_7_4" "9" "0" \
  ++hparams.is_unbalance=false \
  ++hparams.scale_pos_weight=3.0

# run_sweep "10d" "lgbm" "alpha_gr" "9" "0" 
# run_sweep "20d" "lgbm" "alpha_gr" "9" "0" 
# run_sweep "40d" "lgbm" "alpha_gr" "9" "0"
