#!/bin/bash

# エラーが発生した場合にスクリプトを即座に停止する
set -e

# スクリプトが存在するディレクトリ（プロジェクトルート）に移動
cd "$(dirname "$0")"

echo "============================================================"
echo "Starting JPSForecast Data Preparation Pipeline"
echo "============================================================"

# 1. 生データの標準化 (Standardize Raw Data)
echo ""
echo "[Step 1/3] Running standardize_raw_data..."
python -m scripts.data_prep.standardize_raw_data

# 2. マスターデータ作成 (Create Master Data)
echo ""
echo "[Step 2/3] Running create_master_data..."
python -m scripts.data_prep.create_master_data

# 3. データ検証 (Validate Master Data)
echo ""
echo "[Step 3/3] Running validate_master_data..."
python -m scripts.data_prep.validate_master_data

echo ""
echo "============================================================"
echo "✅ Data pipeline completed successfully!"
echo "============================================================"