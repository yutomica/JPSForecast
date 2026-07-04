#!/bin/bash
# step4_run_final_sweep.sh
#
# Thin launcher for scripts/pipeline/auto_step4_hpo.py.
# Define run_final_sweep calls in the execution section at the bottom.

set -e

# Shell-level options.
DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

run_final_sweep() {
    local domain=$1
    local model=$2
    local role=$3
    local n_jobs=$4
    local use_gpu=$5
    shift 5

    local target="${domain}_${role}"
    local features="features_${model}_${target}_fixed"
    local sweep="${model}_${target}_final"
    local command_mode="${STEP4_MODE}"
    local data_config="${DATA_CONFIG}"
    local objectives=()
    local extra_args=()

    while [ "$#" -gt 0 ]; do
        if [ "$1" = "--" ]; then
            shift
            extra_args=("$@")
            break
        fi
        objectives+=("$1")
        shift
    done

    if [ "${#objectives[@]}" -eq 0 ]; then
        echo "ERROR: run_final_sweep requires at least one objective spec: metric:direction:label" >&2
        return 1
    fi

    if [ "${command_mode}" != "optimize" ] && [ "${command_mode}" != "refine" ]; then
        echo "ERROR: STEP4_MODE must be 'optimize' or 'refine': ${command_mode}" >&2
        return 1
    fi

    local cmd=(
        "${PYTHON_RUNNER[@]}" scripts/pipeline/auto_step4_hpo.py "${command_mode}"
        --domain "${domain}"
        --model "${model}"
        --role "${role}"
        --tracking-uri "${TRACKING_URI}"
        --storage "${OPTUNA_STORAGE}"
        --features "${features}"
        --data "${data_config}"
        --n-jobs "${n_jobs}"
    )

    if [ "${command_mode}" = "optimize" ]; then
        cmd+=(--sweep "${sweep}")
    fi

    for objective in "${objectives[@]}"; do
        cmd+=(--objective "${objective}")
    done

    if [ "${use_gpu}" -eq 1 ]; then
        cmd+=(--use-gpu)
    fi
    if [ "${DRY_RUN}" -eq 1 ]; then
        cmd+=(--dry-run)
    fi
    if [ -n "${TOTAL_TRIALS}" ]; then
        cmd+=(--total-trials "${TOTAL_TRIALS}")
    fi

    for arg in "${extra_args[@]}"; do
        cmd+=(--extra-arg "${arg}")
    done

    echo "============================================================"
    echo "Starting Final Sweep via auto_step4_hpo.py"
    echo "Target: ${target}"
    echo "Model: ${model}"
    echo "Sweep: ${sweep}"
    echo "Features: ${features}"
    echo "Mode: ${command_mode}"
    echo "Objectives: ${objectives[*]}"
    echo "n_jobs: ${n_jobs}"
    echo "============================================================"

    "${cmd[@]}"

    echo "Finished Final Sweep for ${model} (${domain}) - ${role}."
    echo ""
}

# --- Execution ---
# Runtime environment for stable parallel HPO execution.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1

# Step4 execution settings.
PYTHON_RUNNER=(uv run python)
TRACKING_URI="sqlite:///mlflow.db"
OPTUNA_STORAGE="sqlite:///optuna.db"
STEP4_MODE="optimize"  # "optimize" or "refine"
DATA_CONFIG="master"
TOTAL_TRIALS=36

# Objective spec format: metric:direction:label
run_final_sweep "tac" "lgbm" "alpha_gr" "9" "0" \
    "objective_tac_gr_guarded:maximize:guarded" \
    "objective_tac:maximize:tac" \
    "RankIC:maximize:rankic"

# Examples:
# run_final_sweep "str" "lgbm" "alpha" "8" "0" \
#     "rank_ic_reb_60d_multi_offset_icir_mean:maximize:reb_icir"
#
# run_final_sweep "tac" "gandalf" "alpha_gr" "6" "0" \
#     "objective_tac_gr_guarded:maximize:guarded"
#
# run_final_sweep "tac" "tcn" "alpha_gr" "3" "0" \
#     "RankIC:maximize:rankic"
#
# run_final_sweep "10d" "lgbm" "alpha_gr" "9" "0" \
#     "RankIC:maximize:rankic"
#
# run_final_sweep "20d" "lgbm" "alpha_gr" "9" "0" \
#     "RankIC:maximize:rankic"
