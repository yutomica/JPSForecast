#!/bin/bash
# step0_target_probe.sh

set -e

# 環境変数設定
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export PYTHONPATH=$PYTHONPATH:.

# 動的メタデータの取得
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
DATA_SNAPSHOT="jq_2017_2025_03_v001"

run() {
    local domain=$1
    local model=$2
    local role=$3 # 'alpha' or 'risk'
    local target="${domain}_${role}"
    local exp_name="JPSForecast_${target}"
    shift 3
    local extra_args=("$@")
    
    # 設定ハッシュの簡易生成（引数から算出）
    local config_hash=$(echo "${extra_args[@]}" | md5 | cut -c1-8)

    echo "============================================================"
    echo "🚀 Starting Target Probe: $model ($domain) - $role"
    echo "============================================================"

    local domain_upper=$(echo "$domain" | tr '[:lower:]' '[:upper:]')
    local model_upper=$(echo "$model" | tr '[:lower:]' '[:upper:]')

    uv run python train.py \
        experiment=target_probe \
        domain=${domain} \
        model=${model} \
        data=master \
        features=features_target_probe_vol_excluded \
        target=${target} \
        period=${domain}_standard \
        cv=anchored_walk_forward \
        mlflow.experiment_name="${exp_name}" \
        ++mode="target_probe" \
        ++mlflow.run_name="TargetProbe_${domain}_${role}" \
        \
        ++mlflow.tags.project="JPSForecast" \
        ++mlflow.tags.domain="${domain_upper}" \
        ++mlflow.tags.task_family="${role}" \
        ++mlflow.tags.target_name="${target}" \
        ++mlflow.tags.target_version="v001" \
        ++mlflow.tags.model_name="${model_upper}" \
        ++mlflow.tags.model_family="tree" \
        ++mlflow.tags.stage="step0_target_probe" \
        ++mlflow.tags.cv_scheme="purged_anchored_walk_forward" \
        ++mlflow.tags.git_commit="${GIT_COMMIT}" \
        ++mlflow.tags.config_hash="${config_hash}" \
        \
        "${extra_args[@]}"
        
    echo "✅ Finished target probe for $model ($domain) - $role."
    echo ""
}

# run "tac" "lgbm_ordinal_threshold" "alpha_class"
# run "tac" "lgbm_ordinal_threshold" "alpha_upclass"
# run "tac" "lgbm_ordinal_threshold" "alpha_qlclass"
run "tac" "lgbm" "alpha_gr"
run "tac" "lgbm" "alpha"
# run "tac" "lgbm" "alpha_sector"
# run "tac" "lgbm" "alpha_linear"

# run "tac" "lgbm_ordinal_threshold" "risk"

run "str" "lgbm" "alpha" \
 ++preprocess.sampling.enabled=true \
 ++preprocess.sampling.interval=11
run "str" "lgbm" "alpha_gr" \
 ++preprocess.sampling.enabled=true \
 ++preprocess.sampling.interval=11
run "str" "lgbm_ordinal_threshold" "alpha_tb" \
 ++preprocess.sampling.enabled=true \
 ++preprocess.sampling.interval=11

run "str" "lgbm" "risk" \
 ++preprocess.sampling.enabled=true \
 ++preprocess.sampling.interval=11

# run "tac" "lgbm" "alpha_tb_5_2"
# run "tac" "lgbm" "alpha_tb_7_2"
# run "tac" "lgbm" "alpha_tb_10_2"
# run "tac" "lgbm" "alpha_tb_5_3"
# run "tac" "lgbm" "alpha_tb_7_3"
# run "tac" "lgbm" "alpha_tb_10_3"
# run "tac" "lgbm" "alpha_tb_7_4"
# run "tac" "lgbm" "alpha_tb_10_4"

wait
echo "🎉 All tasks completed successfully."
