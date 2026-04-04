#!/bin/bash
# run_rough_tuning.sh

set -e

# MLflowのバックエンドをtrain.pyと合わせる
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export exp_name="Rough_Tuning"


# Sweep実行用の共通関数
run_sweep() {
    local domain=$1
    local model=$2
    local target=$3
    local features=$4

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
        cv=rrv \
        data=master \
        model=${model} \
        period=${domain}_standard \
        features=${features} \
        target=${target} \
        hparams=${model}_default \
        sweep=${model}_rough \
        mlflow.experiment_name="${exp_name}"
        
    echo "Finished $model ($domain)."
    echo ""
}


# --- TAC (戦術モデル) ---
# run_sweep "tac" "lgbm"           "tac_rank"                "features_tac_LGBM_rough"        
# run_sweep "tac" "lgbm"           "tac_tb_strategy_a"       "features_tac_LGBM_rough"        
# run_sweep "tac" "tcn"            "tac_vol_scaled_residual" "features_tac_TimeSeries_rough"  
# run_sweep "tac" "tabnet"         "tac_rank"                "features_tac_TabNet_rough"      
# run_sweep "tac" "gandalf"        "tac_rank"                "features_tac_DeepTabular_rough" 
# run_sweep "tac" "ft_transformer" "tac_rank"                "features_tac_DeepTabular_rough" 
# run_sweep "str" "lgbm"           "str_rank"                "features_str_LGBM_rough"        
# run_sweep "str" "lgbm"           "str_consistency"         "features_str_LGBM_rough" 
# run_sweep "str" "tcn"            "str_risk_adj"            "features_str_TimeSeries_rough"  
run_sweep "str" "tabnet"         "str_gauss_rank"                "features_str_TabNet_rough"      
# run_sweep "str" "lgbm"           "str_triple_barrier"         "features_str_LGBM_rough" 
run_sweep "str" "gandalf"        "str_gauss_rank"                "features_str_DeepTabular_rough" 
# run_sweep "str" "ft_transformer" "str_rank"                "features_str_DeepTabular_rough" 