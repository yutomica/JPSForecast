#!/bin/bash
# run_update_pipeline.sh

set -e
cd "$(dirname "$0")/../.."

echo "============================================================"
echo "Starting JPSForecast Update Pipeline"
echo "============================================================"

# 1. 生データの標準化 (Standardize Raw Data)
# 銘柄ごとのOHLCVデータを読み込み、テクニカル指標の計算やマーケットデータの結合を行います。
echo ""
echo "[Step 1/5] Running standardize_raw_data..."
python -m scripts.data_prep.standardize_raw_data

# 2. マスターデータ作成 (Create Master Data)
# 標準化されたデータを結合し、横断面の正規化やターゲット（目的変数）の作成を行います。
echo ""
echo "[Step 2/5] Running create_master_data..."
python -m scripts.data_prep.create_master_data
python -m scripts.data_prep.create_master_data --sample

# 3. データ検証 (Validate Master Data)
# 作成されたマスターデータの欠損値や異常値を年別に検証し、レポートを出力します。
echo ""
echo "[Step 3/5] Running validate_master_data..."
python -m scripts.data_prep.validate_master_data

# 4. 軽量データ作成 (Create Master Select Data)
# 特徴量スクリーニングの結果に基づき、各モデルの学習に必要なカラムのみを抽出した軽量版memmapを作成します。
echo ""
echo "[Step 4/5] Creating master select data..."
python scripts/data_prep/create_master_select_data.py \
 features_lgbm_tac_alpha_gr_rough \
 features_lgbm_10d_alpha_gr_rough \
 features_lgbm_20d_alpha_gr_rough \
 features_lgbm_40d_alpha_gr_rough \
 features_lgbm_str_alpha_rough \
 features_tcn_tac_alpha_gr_rough \
 features_gandalf_tac_alpha_gr_rough

# 特徴量データの配置 (移動)
mkdir -p ./data/master_select
rm -rf ./data/master_select/features
mv ./data/master/features_select ./data/master_select/features
mv ./data/master/features_select_names.json ./data/master_select/feature_names.json

# 5. Productionモデル作成
echo ""
echo "[Step 5/5] Creating production models..."
./scripts/pipeline/create_production_model.sh

echo ""
echo "============================================================"
echo "✅ Data and model update pipeline completed successfully!"
echo "============================================================"
