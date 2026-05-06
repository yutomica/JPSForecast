#!/bin/bash
# run_data_pipeline.sh

set -e
cd "$(dirname "$0")/../.."

echo "============================================================"
echo "Starting JPSForecast Data Preparation Pipeline"
echo "============================================================"

# # 1. 生データの標準化 (Standardize Raw Data)
# # 銘柄ごとのOHLCVデータを読み込み、テクニカル指標の計算やマーケットデータの結合を行います。
# echo ""
# echo "[Step 1/4] Running standardize_raw_data..."
# python -m scripts.data_prep.standardize_raw_data

# # 2. マスターデータ作成 (Create Master Data)
# # 標準化されたデータを結合し、横断面の正規化やターゲット（目的変数）の作成を行います。
# echo ""
# echo "[Step 2/4] Running create_master_data..."
# python -m scripts.data_prep.create_master_data
# python -m scripts.data_prep.create_master_data --sample

# # 3. データ検証 (Validate Master Data)
# # 作成されたマスターデータの欠損値や異常値を年別に検証し、レポートを出力します。
# echo ""
# echo "[Step 3/4] Running validate_master_data..."
# python -m scripts.data_prep.validate_master_data

# 4. 軽量データ作成 (Create Master Select Data)
# 特徴量スクリーニングの結果に基づき、各モデルの学習に必要なカラムのみを抽出した軽量版memmapを作成します。
echo ""
echo "[Step 4/4] Creating master select data..."
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
 features_ft_transformer_str_risk_rough \
 features_lgbm_tac_alpha_gr_rough

echo ""
echo "============================================================"
echo "✅ Data pipeline completed successfully!"
echo "============================================================"
