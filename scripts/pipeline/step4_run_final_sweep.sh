#!/bin/bash
# step4_run_final_sweep.sh

set -e

# 環境変数設定
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1

export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export exp_name="Final_Sweep"
export PYTHONPATH=$PYTHONPATH:.

# ドライランモード
DRY_RUN=0
if [ "$1" = "--dry-run" ]; then DRY_RUN=1; shift; fi

run_final_sweep() {
    local domain=$1
    local model=$2
    local role=$3 # 'alpha' or 'risk'
    local n_jobs=$4
    local use_gpu=$5
    shift 5
    local extra_args=("$@")

    local target="${domain}_${role}"
    local features="features_${model}_${target}_fixed"
    local sweep="${model}_${target}_final"
    local timestamp=$(date +"%Y%m%d_%H%M%S")

    local study_name=${OPTUNA_STUDY_NAME:-"final_sweep_${model}_${target}_${timestamp}"}

    # 再開時のオフセット取得
    local trial_offset=$(uv run python -m src.utils.config_utils --action get_trial_count --storage "sqlite:///optuna.db" --study-name "${study_name}" --state COMPLETE)
    echo "Completed Trials (Offset): ${trial_offset}"

    # 設定ファイルから n_trials を取得
    local config_n_trials=$(grep "n_trials:" "config/sweep/${sweep}.yaml" | awk '{print $2}' | head -n 1)
    config_n_trials=$(echo "${config_n_trials}" | sed 's/[^0-9]//g')
    local total_n_trials=${OPTUNA_N_TRIALS:-${config_n_trials:-100}}
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

    local data_arg="data=master"
    local dry_run_args=()
    if [ "$DRY_RUN" -eq 1 ]; then
        data_arg="data=sample"
        remaining_trials=2
        dry_run_args+=("++hparams.max_epochs=2")
        echo "⚠️ Running in DRY RUN mode."
    fi

    echo "============================================================"
    echo "Starting Final Sweep: $model ($domain) - $role"
    echo "Study Name: $study_name"
    echo "============================================================"
    
    local parent_run_id=$(uv run python -m src.utils.mlflow_utils --action resolve_parent --tracking-uri "${MLFLOW_TRACKING_URI}" --experiment-name "${exp_name}" --study-name "${study_name}")

    local optuna_args=("hydra.sweeper.storage=sqlite:///optuna.db" "hydra.sweeper.study_name=${study_name}")

    # hparams の指定を削除。Experiment 側の defaults で解決する。
    TRIAL_OFFSET=$trial_offset MLFLOW_PARENT_RUN_ID=$parent_run_id uv run python train.py -m \
        hydra/launcher=$( [ "${n_jobs}" -gt 1 ] && echo "joblib" || echo "basic" ) \
        $([ "${n_jobs}" -gt 1 ] && echo "hydra.sweeper.n_jobs=${n_jobs} hydra.launcher.n_jobs=${n_jobs}") \
        hydra.sweeper.n_trials=${remaining_trials} \
        ++hparams.num_threads=1 \
        domain=${domain} \
        target=${target} \
        ${data_arg} \
        features=${features} \
        model=${model} \
        period=${domain}_standard \
        cv=purged_kfold \
        mlflow.experiment_name="${exp_name}" \
        sweep=${sweep} \
        +mode=final_sweep \
        +experiment=${model}_${target} \
        "${optuna_args[@]}" \
        $gpu_args \
        "${extra_args[@]}" \
        "${dry_run_args[@]}"

    echo "Finished Final Sweep for $model ($domain) - $role."
    echo ""
}

# --- Execution ---
# run_final_sweep "tac" "lgbm" "risk"  "8" "0"
# run_final_sweep "str" "lgbm" "risk"  "8" "0"
# run_final_sweep "tac" "elasticnet" "alpha" "10" "0"
# run_final_sweep "tac" "elasticnet" "risk" "10" "0"
# run_final_sweep "str" "elasticnet" "alpha" "10" "0"
# run_final_sweep "str" "elasticnet" "risk" "10" "0"


# run_final_sweep "tac" "tcn" "alpha" "8" "1"
run_final_sweep "tac" "tcn" "risk" "4" "1"

run_final_sweep "tac" "ft_transformer" "alpha" "2" "1"
run_final_sweep "tac" "ft_transformer" "risk" "2" "1"
# run_final_sweep "tac" "ft_transformer" "alpha" "2" "1"
