#!/bin/bash
# run_elasticnet_hpo.sh

set -e

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1

export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export exp_name="Final_Sweep"
export PYTHONPATH=$PYTHONPATH:.

run_elasticnet_sweep() {
    local domain=$1
    local role=$2 # 'alpha' or 'risk'
    local n_jobs=$3
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    shift 3
    local extra_args=("$@")

    local target="${domain}_${role}"
    local features="features_elasticnet_${target}"
    local sweep="elasticnet_${target}"
    # base -> target theory -> anchor の順で階層化
    local hparams="elasticnet/base,elasticnet/${target},anchor/elasticnet_${target}"

    local study_name="Final_Sweep_ElasticNet_${target}_${timestamp}"

    echo "============================================================"
    echo "🚀 Starting ElasticNet HPO: $model ($domain) - $role"
    echo "Study Name: $study_name"
    echo "============================================================"

    local parent_run_id=$(uv run python -m src.utils.mlflow_utils --action resolve_parent --tracking-uri "${MLFLOW_TRACKING_URI}" --experiment-name "${exp_name}" --study-name "${study_name}")

    MLFLOW_PARENT_RUN_ID=$parent_run_id uv run python train.py -m \
        hydra/launcher=joblib \
        hydra.sweeper.n_jobs=${n_jobs} \
        hydra.launcher.n_jobs=${n_jobs} \
        +experiment=search_elasticnet_alpha \
        domain=${domain} \
        data=master \
        target=${target} \
        features=${features} \
        model=elasticnet \
        hparams=${hparams} \
        sweep=${sweep} \
        period=${domain}_standard \
        cv=cpcv \
        mlflow.experiment_name="${exp_name}" \
        hydra.sweeper.storage="sqlite:///optuna.db" \
        hydra.sweeper.study_name="${study_name}" \
        "${extra_args[@]}"
        
    echo "✅ Finished HPO for $target ($domain)."
    echo ""
}

N_JOBS=8

# 1. TAC alpha
run_elasticnet_sweep "tac" "alpha" "${N_JOBS}" ++optimization_metric="ndcg_10"

# 2. TAC risk
run_elasticnet_sweep "tac" "risk"  "${N_JOBS}" ++optimization_metric="AP_severe"

# 3. STR alpha
run_elasticnet_sweep "str" "alpha" "${N_JOBS}" ++optimization_metric="RankIC"

# 4. STR risk
run_elasticnet_sweep "str" "risk"  "${N_JOBS}" ++optimization_metric="AP_severe_STR"

echo "🎉 All ElasticNet HPO tasks completed."
