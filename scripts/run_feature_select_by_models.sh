#!/bin/bash
# run_feature_select_by_models.sh

set -e

# MLflowのバックエンドをtrain.pyと合わせる
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export exp_name="Feature_Selection"

# 特徴量選択（MDA）実行用の共通関数
run_feature_select() {
    local domain=$1
    local model=$2
    local target=$3
    local features=$4
    local hparams=$5

    # LGBM向けにGPUを有効化する引数を追加
    local gpu_args=""
    if [ "$USE_GPU" -eq 1 ] && [ "$model" = "lgbm" ]; then
        gpu_args="++hparams.device_type=gpu"
    fi

    echo "============================================================"
    echo "Starting Feature Selection: $model ($domain)"
    echo "============================================================"

    uv run python train.py \
        domain=${domain} \
        target=${target} \
        data=master \
        features=${features} \
        model=${model} \
        hparams=${hparams} \
        period=${domain}_standard \
        cv=purged_kfold \
        mlflow.experiment_name="${exp_name}" \
        +mode=feature_select \
        $gpu_args
        
    echo "Finished Feature Selection for $model ($domain)."
    echo ""
}

run_feature_select "tac" "lgbm"    "tac_vol_scaled_asym_return"          "features_lgbm_tac_vol_scaled_asym_return_rough"        "lgbm_tac_vol_scaled_asym_return_anc" 