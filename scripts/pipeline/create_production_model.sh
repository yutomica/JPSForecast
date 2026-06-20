#!/bin/bash
# step6_create_production_model.sh
# 
# 承認済み・固定済みのハイパーパラメータと特徴量セットを用い、
# 全期間（Train+Valid）を使用してプロダクションモデルを本学習します。
# 学習後のモデルは MLflow の 'Production' ステージに登録されます。
#
# シード値を変更した複数モデル学習（アンサンブル）をサポートしており、
# model.ensemble_size 引数でアンサンブル数を指定可能です。

set -e

# 環境変数設定
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# MLflow設定
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export PYTHONPATH=$PYTHONPATH:.

run_production() {
    local domain=$1
    local model=$2
    local role=$3 
    local variant=${4:-"default"}
    shift 4
    local extra_args=("$@")

    local target="${domain}_${role}"
    # Step 5 で作成された固定ハイパーパラメータファイルを参照
    local hparams_fixed="${model}_${target}_${variant}"
    local features="features_${model}_${target}_fixed"
    local exp_name="JPSForecast_${target}"
    local run_name=${MLFLOW_RUN_NAME:-"Step6_Production_${model}_${target}"}
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local child_run_name="Prod_${model}_${timestamp}"

    echo "============================================================"
    echo "🚀 Creating PRODUCTION Model: $model ($domain) - $role"
    echo "   Target   : $target"
    echo "   HParams  : $hparams_fixed (from Step 5)"
    echo "   Variant  : $variant"
    echo "   Experiment: $exp_name"
    echo "   Run Name : $run_name"
    echo "   Child Run: $child_run_name"
    echo "============================================================"

    # 親ランのIDを解決または作成
    local parent_run_id=$(uv run python -m src.utils.mlflow_utils --action resolve_parent --tracking-uri "${MLFLOW_TRACKING_URI}" --experiment-name "${exp_name}" --parent-run-name "${run_name}")

    # mode=production を指定
    # train.py 内で Step 1 (最適Epoch探索) -> Step 2 (全データ本学習) が実行される
    MLFLOW_PARENT_RUN_ID=$parent_run_id uv run python train.py \
        domain=${domain} \
        target=${target} \
        data=master \
        features=${features} \
        model=${model} \
        hparams=${hparams_fixed} \
        period=${domain}_standard \
        cv=fixed \
        +mode=production \
        variant=${variant} \
        mlflow.experiment_name="${exp_name}" \
        ++mlflow.run_name="${run_name}" \
        ++mlflow.child_run_name="${child_run_name}" \
        "${extra_args[@]}"

    echo "✅ Finished Production model creation for $target."
    echo ""
}

run_production "tac" "lgbm" "alpha_gr" "v1_stable" \
    model.ensemble_size=5

run_production "tac" "gandalf" "alpha_gr" "v1_stable" \
    model.ensemble_size=5

# run_production "tac" "tcn" "alpha_gr" "v1_stable" \
#     model.ensemble_size=5

# run_production "str" "lgbm" "alpha" "v1_stable" \
#     model.ensemble_size=5

# run_production "10d" "lgbm" "alpha_gr" "v1_stable" \
#     model.ensemble_size=5

# run_production "20d" "lgbm" "alpha_gr" "v1_stable" \
#     model.ensemble_size=5