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

    python train.py \
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

# finished
# run_feature_select "tac" "lgbm"    "tac_rank"                "features_tac_LGBM_rough"        "lgbm_tac_rnk"    
# run_feature_select "tac" "tabnet"  "tac_rank"                "features_tac_TabNet_rough"      "tabnet_tac_rnk"  
# run_feature_select "tac" "gandalf" "tac_rank"                "features_tac_DeepTabular_rough" "gandalf_tac_rnk" 
run_feature_select "tac" "lgbm"    "tac_gauss_rank"          "features_tac_LGBM_rough"        "lgbm_tac_rnk" 
run_feature_select "tac" "tabnet"  "tac_gauss_rank"          "features_tac_TabNet_rough"      "tabnet_tac_rnk" 
run_feature_select "tac" "gandalf" "tac_gauss_rank"          "features_tac_DeepTabular_rough" "gandalf_tac_rnk" 
# run_feature_select "tac" "tcn"     "tac_vol_scaled_residual" "features_tac_TimeSeries_rough"  "tcn_tac_scl" 
run_feature_select "tac" "lgbm"    "tac_tb_strategy_a"       "features_tac_LGBM_rough"        "lgbm_tac_tpb" 
# run_feature_select "tac" "ft_transformer"     "tac_rank"       "features_tac_DeepTabular_rough"  "ft_transformer_tac_rnk" 
# run_feature_select "str" "lgbm"    "str_rank"                "features_str_LGBM_rough"        "lgbm_str_rnk"    
# run_feature_select "str" "lgbm"    "str_peer_alpha"          "features_str_LGBM_rough"        "lgbm_str_rnk"    
# run_feature_select "str" "lgbm"    "str_gauss_rank"          "features_str_LGBM_rough"        "lgbm_str_rnk"    
# run_feature_select "str" "gandalf"    "str_gauss_rank"          "features_str_DeepTabular_rough"        "gandalf_str_rnk"    
# run_feature_select "str" "tabnet"    "str_gauss_rank"          "features_str_TabNet_rough"        "tabnet_str_rnk"    
# run_feature_select "str" "lgbm"    "str_consistency"         "features_str_LGBM_rough"        "lgbm_str_scl"    
# run_feature_select "str" "lgbm"    "str_triple_barrier"      "features_str_LGBM_rough"        "lgbm_str_tpb"    
# not yet
# run_feature_select "tac" "ft_transformer"     "tac_gauss_rank" "features_tac_DeepTabular_rough"  "ft_transformer_tac_rnk" 
# run_feature_select "str" "tcn"     "str_risk_adj"            "features_str_TimeSeries_rough"  "tcn_str_scl"    
