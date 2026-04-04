#!/bin/bash
# run_final_sweep.sh

set -e

# MLflowのバックエンドをtrain.pyと合わせる
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export exp_name="Final_Sweep"

run_final_sweep() {
    local domain=$1
    local model=$2
    local target=$3
    local hparams=$4


    echo "============================================================"
    echo "Starting Final Sweep: $model ($domain)"
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
        target=${target} \
        data=master \
        features=features_${model}_${target} \
        model=${model} \
        hparams=${hparams} \
        period=${domain}_standard \
        cv=purged_kfold \
        mlflow.experiment_name="${exp_name}" \
        sweep=${model}_${target} \
        +mode=final_sweep
        
    echo "Finished Feature Selection for $model ($domain)."
    echo ""
}

# finished
# run_final_sweep "tac" "lgbm"    "tac_rank"  "lgbm_tac_rnk"
# run_final_sweep "tac" "lgbm"    "tac_gauss_rank"     "lgbm_tac_rnk"
# run_final_sweep "tac" "gandalf" "tac_gauss_rank"     "gandalf_tac_rnk"
# run_final_sweep "tac" "tcn"     "tac_vol_scaled_residual" "tcn_tac_scl"
# run_final_sweep "str" "lgbm"  "str_gauss_rank"     "lgbm_str_rnk"

# not yet
# run_final_sweep "str" "gandalf" "str_gauss_rank"           "gandalf_tac_rnk"
run_final_sweep "tac" "tabnet"  "tac_gauss_rank"     "tabnet_tac_rnk"
run_final_sweep "str" "tabnet"  "str_gauss_rank"           "tabnet_tac_rnk"
run_final_sweep "str" "lgbm"    "str_triple_barrier"      "lgbm_str_tpb"
run_final_sweep "tac" "lgbm"    "tac_tb_strategy_a"  "lgbm_tac_tpb"