#!/bin/bash
# run_rough_tuning.sh

set -e
TARGET_TYPES=("tac" "str")
# TARGET_TYPES=("str")
# MODELS=("lgbm" "tabnet")
MODELS=("tabnet")

# MLflowのバックエンドをtrain.pyと合わせる
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

for TARGET in "${TARGET_TYPES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        TARGET_UPPER=$(echo "$TARGET" | tr '[:lower:]' '[:upper:]')
        MODEL_UPPER=$(echo "$MODEL" | tr '[:lower:]' '[:upper:]')
        
        # ドメイン(TARGET)ごとに学習ターゲットを設定
        # モデルとドメインの組み合わせでターゲットを切り替え
        case "${TARGET}_${MODEL}" in
            "tac_lgbm")
                TRAIN_TARGET="tac_rank"
                ;;
            "str_lgbm")
                TRAIN_TARGET="str_risk_adj"
                ;;
            "tac_tabnet")
                TRAIN_TARGET="tac_rank"
                ;;
            "str_tabnet")
                TRAIN_TARGET="str_rank"
                ;;
            *)
                echo "Skipping unknown combination: ${TARGET} ${MODEL}"
                continue
                ;;
        esac

        EXPERIMENT_NAME="Rough_Tuning_${TARGET_UPPER}_${MODEL_UPPER}"
        
        echo "Creating Parent Run for $EXPERIMENT_NAME..."
        
        # 1. 親ランを作成し、Run IDを取得
        # CLIの出力パースエラーを回避するため、Pythonワンライナーで確実にIDを取得する
        PARENT_RUN_ID=$(python -c "
import mlflow
from mlflow.tracking import MlflowClient
import datetime
mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}')
client = MlflowClient()
exp = client.get_experiment_by_name('${EXPERIMENT_NAME}')
if exp and exp.lifecycle_stage == 'deleted':
    client.restore_experiment(exp.experiment_id)
mlflow.set_experiment('${EXPERIMENT_NAME}')
run = mlflow.start_run(run_name=f'Sweep_{datetime.datetime.now().strftime(\"%Y%m%d_%H%M%S\")}')
print(run.info.run_id)
")
        
        echo "Parent Run ID: $PARENT_RUN_ID"

        # 2. Hydraを実行。環境変数経由で親IDをPython側に渡す
        # experimentはconfigに追加する項目なので '+' を付ける
        MLFLOW_PARENT_RUN_ID=$PARENT_RUN_ID python train.py -m \
            domain=${TARGET} \
            model=${MODEL} \
            period=${TARGET}_standard \
            features=${TARGET}_candidates \
            target=${TRAIN_TARGET} \
            hparams=${MODEL}_default \
            hydra/sweeper=optuna \
            +sweep=${MODEL}_rough \
            mlflow.experiment_name="$EXPERIMENT_NAME"
            
        echo "Finished $MODEL ($TARGET)."
    done
done