#!/bin/bash
# run_data_pipeline.sh

set -e
cd "$(dirname "$0")/../.."

echo "============================================================"
echo "Starting JPSForecast Data Preparation Pipeline"
echo "============================================================"

# echo ""
# echo "[Init] Cleaning generated pipeline outputs..."
# rm -rf ./data/intermediate/date_chunks
# rm -rf ./data/master/features
# rm -rf ./data/master/features_select
# rm -rf ./data/master/temp_features_buffer
# rm -f ./data/master/index_meta.parquet
# rm -f ./data/master/feature_names.json
# rm -f ./data/master/features_select_names.json
# rm -rf ./data/master_select
# rm -rf ./data/sample/features
# rm -f ./data/sample/index_meta.parquet
# rm -f ./data/sample/feature_names.json

# # 1. 生データの標準化 (Standardize Raw Data)
# # 銘柄ごとのOHLCVデータを読み込み、テクニカル指標の計算やマーケットデータの結合を行います。
# echo ""
# echo "[Step 1/4] Running standardize_raw_data..."
# python -m scripts.data_prep.standardize_raw_data

# 2. マスターデータ作成 (Create Master Data)
# 標準化されたデータを結合し、横断面の正規化やターゲット（目的変数）の作成を行います。
echo ""
echo "[Step 2/4] Running create_master_data..."
python -m scripts.data_prep.create_master_data
python -m scripts.data_prep.create_master_data --sample

# 3. データ検証 (Validate Master Data)
# 作成されたマスターデータの欠損値や異常値を年別に検証し、レポートを出力します。
echo ""
echo "[Step 3/4] Running validate_master_data..."
python -m scripts.data_prep.validate_master_data

# 4. 軽量データ作成 (Create Master Select Data)
# 特徴量スクリーニングの結果に基づき、各モデルの学習に必要なカラムのみを抽出した軽量版memmapを作成します。
echo ""
echo "[Step 4/4] Creating master select data..."
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
cp ./data/master/index_meta.parquet ./data/master_select/index_meta.parquet

echo ""
echo "============================================================"
echo "✅ Data pipeline completed successfully!"
echo "============================================================"
