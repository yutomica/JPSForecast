#!/bin/bash
# run_data_pipeline.sh

set -e
cd "$(dirname "$0")/../.."

echo "============================================================"
echo "Starting JPSForecast Data Preparation Pipeline"
echo "============================================================"

# 特徴量スクリーニング後、必要なカラムのみ保持した軽量データを作成する
# 実装名（tac_vol_scaled...）から役割名（tac_alpha）に変更
python scripts/data_prep/create_master_select_data.py \
 features_lgbm_tac_alpha_rough \
 features_lgbm_tac_risk_rough \
 features_lgbm_str_alpha_rough \
 features_lgbm_str_risk_rough \
 features_tcn_tac_alpha_rough \
 features_tcn_tac_risk_rough \
 features_tcn_str_alpha_rough \
 features_tcn_str_risk_rough \
 features_ft_transformer_tac_alpha_rough \
 features_ft_transformer_tac_risk_rough \
 features_ft_transformer_str_alpha_rough \
 features_ft_transformer_str_risk_rough

echo ""
echo "============================================================"
echo "✅ Data pipeline completed successfully!"
echo "============================================================"
