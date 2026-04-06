#!/bin/bash

# ==========================================
# Daily Prediction Batch Script
# ==========================================
# エラーが発生した時点でスクリプトを終了
set -e

# cronでの実行を想定し、スクリプトの配置場所からプロジェクトルートを自動で特定
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"

# ログディレクトリの作成
mkdir -p "${LOG_DIR}"

# 実行日時の取得 (例: 20260405_102429)
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/prediction_${CURRENT_TIME}.log"

# プロジェクトディレクトリに移動
cd "${PROJECT_DIR}" || exit 1

echo "==========================================" | tee -a "${LOG_FILE}"
echo "Starting Daily Prediction: $(date)" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

# predict.py の実行 (標準出力と標準エラー出力をターミナルとログファイル両方に出力)
python predict.py 2>&1 | tee -a "${LOG_FILE}"

echo "==========================================" | tee -a "${LOG_FILE}"
echo "✅ Prediction completed: $(date)" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
