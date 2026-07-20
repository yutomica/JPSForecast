#!/bin/bash
# create_production_model.sh
#
# Step 6: Production model training.
# Reads the selected-candidate manifests written by Step5, retrains each
# selected candidate on Train+Valid with mode=production, then lets train.py
# register the resulting inference model to MLflow Production.

set -euo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1

export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export PYTHONPATH="${PYTHONPATH:-}:."

PYTHON_RUNNER=(uv run python)
DATA_CONFIG="master"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-5}"

run_selected_production() {
    local domain=$1
    local model=$2
    local role=$3
    shift 3

    local target="${domain}_${role}"
    local selected_manifest="config/promotion/selected_${model}_${target}.yaml"
    local cmd=(
        "${PYTHON_RUNNER[@]}" scripts/pipeline/train_selected_production.py
        --selected "${selected_manifest}"
        --tracking-uri "${MLFLOW_TRACKING_URI}"
        --data "${DATA_CONFIG}"
        --ensemble-size "${ENSEMBLE_SIZE}"
    )

    if [ "${DRY_RUN}" -eq 1 ]; then
        cmd+=(--dry-run)
    fi

    for arg in "$@"; do
        cmd+=(--extra-arg "${arg}")
    done

    echo "============================================================"
    echo "Creating PRODUCTION model from Step5 selection"
    echo "Target: ${target}"
    echo "Model: ${model}"
    echo "Selected manifest: ${selected_manifest}"
    echo "Ensemble size: ${ENSEMBLE_SIZE}"
    echo "============================================================"

    "${cmd[@]}"

    echo "Finished Production model creation for ${model} (${domain}) - ${role}."
    echo ""
}

# run_selected_production "tac" "lgbm" "alpha_gr"
# run_selected_production "tac" "gandalf" "alpha_gr"
# run_selected_production "tac" "tcn" "alpha_gr"
# run_selected_production "10d" "lgbm" "alpha_gr"
run_selected_production "tac" "lgbm" "tb_7_4"

# Add more targets here after Step5 writes the corresponding
# config/promotion/selected_${model}_${target}.yaml manifest.
