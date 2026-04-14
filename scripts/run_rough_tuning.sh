#!/bin/bash
# run_rough_tuning.sh

set -e

# MLflowのバックエンドをtrain.pyと合わせる
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export exp_name="Rough_Tuning"

# GPU使用フラグ (例: USE_GPU=1 ./scripts/run_rough_tuning.sh)
USE_GPU=${USE_GPU:-0}

# Sweep実行用の共通関数
run_sweep() {
    local domain=$1
    local model=$2
    local target=$3
    local features=$4
    # 最初の4つの引数をシフトし、残りの引数を配列として保持する
    shift 4
    local extra_args=("$@")
    local timestamp=$(date +"%Y%m%d_%H%M%S")

    # LGBM向けにGPUを有効化する引数を追加
    local gpu_args=""
    if [ "$USE_GPU" -eq 1 ] && [ "$model" = "lgbm" ]; then
        gpu_args="++hparams.device_type=gpu"
    fi

    echo "============================================================"
    echo "Starting Sweep: $model ($domain)"
    echo "============================================================"
    echo "Creating Parent Run for $exp_name..."

    # 親ランを作成し、Run IDを取得
    local parent_run_id=$(python -c "
import mlflow
from mlflow.tracking import MlflowClient
import datetime
mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}')
client = MlflowClient()
exp = client.get_experiment_by_name('${exp_name}')
if exp and exp.lifecycle_stage == 'deleted':
    client.restore_experiment(exp.experiment_id)
mlflow.set_experiment('${exp_name}')
run = mlflow.start_run(run_name=f'Sweep_{datetime.datetime.now().strftime(\"%Y%m%d_%H%M%S\")}')
print(run.info.run_id)
")

    echo "Parent Run ID: $parent_run_id"

    # Hydraを実行。環境変数経由で親IDをPython側に渡す
    MLFLOW_PARENT_RUN_ID=$parent_run_id python train.py -m \
        domain=${domain} \
        cv=purged_kfold \
        data=master \
        model=${model} \
        period=${domain}_standard \
        features=${features} \
        target=${target} \
        hparams=${model}_default \
        sweep=${model}_rough \
        mlflow.experiment_name="${exp_name}" \
        hydra.sweeper.study_name=rough_tuning_${model}_${target}_${timestamp} \
        $gpu_args \
        "${extra_args[@]}"
        
    echo "Finished $model ($domain)."
    echo ""
}

# -- LGBM
# run_sweep "tac" "lgbm" "tac_vol_scaled_asym_return" "features_lgbm_tac_vol_scaled_asym_return_rough" \
#     ++hparams.custom_objective="src.models.custom_objectives.custom_asymmetric_mse" \
#     ++hparams.custom_metric="src.models.custom_objectives.custom_asymmetric_mse_eval"

# run_sweep "tac" "lgbm" "tac_max_neg_path" "features_lgbm_tac_max_neg_path_rough" \
#     ++hparams.objective="quantile" \
#     ++hparams.metric="quantile" \
#     ++hparams.alpha=0.1 \
#     ++hparams.min_child_samples=10

run_sweep "str" "lgbm" "str_sharpe_adj" "features_lgbm_str_sharpe_adj_rough" \
    ++hparams.objective="fair" \
    ++hparams.metric="fair" \
    ++hparams.fair_c=10.0

run_sweep "str" "lgbm" "str_mdd" "features_lgbm_str_mdd_rough" \
    ++hparams.objective="tweedie" \
    ++hparams.metric="tweedie" \
    ++hparams.tweedie_variance_power=1.2

# -- TCN
# run_sweep "tac" "tcn" "tac_vol_scaled_asym_return" "features_tcn_tac_vol_scaled_asym_return_rough" \
#     ++hparams.objective="asymmetric_mse" \
#     +hydra.sweeper.params.model.window_size.tac="choice(20,90)"

# run_sweep "tac" "tcn" "tac_max_neg_path" "features_tcn_tac_max_neg_path_rough" \
#     ++hparams.objective="quantile" \
#     ++hparams.metric="quantile" \
#     ++hparams.alpha=0.1 \
#     ++hparams.min_child_samples=10 \
#     +hydra.sweeper.params.model.window_size.tac="choice(20,90)"

# run_sweep "str" "tcn" "str_sharpe_adj" "features_tcn_str_sharpe_adj_rough" \
#     ++hparams.objective="fair" \
#     ++hparams.metric="fair" \
#     ++hparams.fair_c=10.0 \
#     +hydra.sweeper.params.model.window_size.str="choice(126,252)"

# run_sweep "str" "tcn" "str_mdd" "features_tcn_str_mdd_rough" \
#     ++hparams.objective="tweedie" \
#     ++hparams.metric="tweedie" \
#     ++hparams.tweedie_variance_power=1.2 \
#     +hydra.sweeper.params.model.window_size.str="choice(126,252)"

# -- FT-Transformer
# run_sweep "tac" "ft_transfomer" "tac_vol_scaled_asym_return" "features_ft_transfomer_tac_vol_scaled_asym_return_rough" \
#     ++hparams.objective="asymmetric_mse"

# run_sweep "tac" "ft_transfomer" "tac_max_neg_path" "features_ft_transfomer_tac_max_neg_path_rough" \
#     ++hparams.objective="quantile" \
#     ++hparams.metric="quantile" \
#     ++hparams.alpha=0.1 \
#     ++hparams.min_child_samples=10

# run_sweep "str" "ft_transfomer" "str_sharpe_adj" "features_ft_transfomer_str_sharpe_adj_rough" \
#     ++hparams.objective="fair" \
#     ++hparams.metric="fair" \
#     ++hparams.fair_c=10.0

# run_sweep "str" "ft_transfomer" "str_mdd" "features_ft_transfomer_str_mdd_rough" \
#     ++hparams.objective="tweedie" \
#     ++hparams.metric="tweedie" \
#     ++hparams.tweedie_variance_power=1.2