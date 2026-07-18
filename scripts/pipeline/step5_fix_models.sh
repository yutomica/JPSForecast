#!/bin/bash
# step5_fix_models.sh
#
# Step 5: Candidate Selection.
# This replaces the old fixed-hparams generation step. It evaluates Step4 HPO
# candidates on the fixed validation window defined by config/cv/fixed.yaml,
# then writes a leaderboard and selected-candidate manifest.

set -e

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

run_candidate_selection() {
    local domain=$1
    local model=$2
    local role=$3
    local top_n_per_study=$4
    local max_candidates=$5
    local n_jobs=$6
    local use_gpu=$7
    shift 7

    local target="${domain}_${role}"
    local features="features_${model}_${target}_fixed"
    local selected_output="config/promotion/selected_${model}_${target}.yaml"
    local selection_score="${SELECTION_SCORE:-auto}"
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
        echo "ERROR: run_candidate_selection requires at least one objective spec." >&2
        return 1
    fi

    local cmd=(
        "${PYTHON_RUNNER[@]}" scripts/pipeline/select_step4_candidate.py
        --domain "${domain}"
        --model "${model}"
        --role "${role}"
        --tracking-uri "${TRACKING_URI}"
        --storage "${OPTUNA_STORAGE}"
        --data "${DATA_CONFIG}"
        --features "${features}"
        --top-n-per-study "${top_n_per_study}"
        --max-candidates "${max_candidates}"
        --n-jobs "${n_jobs}"
        --output-dir "${OUTPUT_DIR}"
        --selected-output "${selected_output}"
        --selection-score "${selection_score}"
        # --promote-best-to-production
    )

    for objective in "${objectives[@]}"; do
        cmd+=(--objective "${objective}")
    done

    if [ "${use_gpu}" -eq 1 ]; then
        cmd+=(--use-gpu)
    fi
    if [ "${DRY_RUN}" -eq 1 ]; then
        cmd+=(--dry-run)
    fi

    for arg in "${extra_args[@]}"; do
        cmd+=(--extra-arg "${arg}")
    done

    echo "============================================================"
    echo "Starting Step5 Candidate Selection"
    echo "Target: ${target}"
    echo "Model: ${model}"
    echo "Features: ${features}"
    echo "Fixed CV: config/cv/fixed.yaml"
    echo "Top N per study: ${top_n_per_study}"
    echo "Max candidates: ${max_candidates}"
    echo "n_jobs: ${n_jobs}"
    echo "Selected output: ${selected_output}"
    echo "Selection score: ${selection_score}"
    echo "Promote best: Production"
    echo "Objectives: ${objectives[*]}"
    echo "============================================================"

    "${cmd[@]}"

    echo "Finished Candidate Selection for ${model} (${domain}) - ${role}."
    echo ""
}

# --- Execution ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1

PYTHON_RUNNER=(uv run python)
TRACKING_URI="sqlite:///mlflow.db"
OPTUNA_STORAGE="sqlite:///optuna.db"
DATA_CONFIG="master"
OUTPUT_DIR="reports/candidate_selection"

# run_candidate_selection domain model role top_n_per_study max_candidates n_jobs use_gpu objectives...
# top_n_per_study: each objective/study contributes up to this many top Optuna trials.
# max_candidates: total deduplicated candidates to evaluate after merging all objective/study candidates.
# n_jobs: number of candidate evaluation processes to run concurrently.
# run_candidate_selection "tac" "lgbm" "alpha_gr" "3" "12" "9" "0" \
#     "objective_tac_gr_guarded:maximize:guarded" \
#     "objective_tac:maximize:tac" \
#     "RankIC:maximize:rankic"

# run_candidate_selection "tac" "gandalf" "alpha_gr" "3" "12" "6" "1" \
#     "objective_tac_gr_guarded:maximize:guarded" \
#     "objective_tac:maximize:tac" \
#     "RankIC:maximize:rankic"

# run_candidate_selection "10d" "lgbm" "alpha_gr" "3" "12" "2" "0" \
#     "objective_10_gr_guarded:maximize:10gr" \
#     "objective_tac_gr_guarded:maximize:guarded" \
#     "RankIC:maximize:rankic"

# run_candidate_selection "tac" "tcn" "alpha_gr" "3" "12" "2" "1" \
#     "objective_tac_gr_guarded:maximize:guarded" \
#     "objective_tac:maximize:tac" \
#     "RankIC:maximize:rankic"

run_candidate_selection "tac" "lgbm" "tb_7_4" "3" "12" "2" "0" \
    "objective_tac_tb_hit_guarded:maximize:tb_hit"
