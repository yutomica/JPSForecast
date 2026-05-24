#!/bin/bash

# ==========================================
# Daily Prediction Batch Script
# ==========================================
# エラーが発生した時点でスクリプトを終了
set -e

# 1. プロジェクトのルートディレクトリ
PROJECT_DIR="/Users/yuu/Projects/JPSForecast"

# 2. Pythonの実行パス
PYTHON_EXE="${PROJECT_DIR}/.venv/bin/python"

# 3. ログファイルの保存場所
LOG_DIR="${PROJECT_DIR}/logs"
# ログディレクトリの作成
mkdir -p "${LOG_DIR}"

# 実行日時の取得 (例: 20260405_102429)
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/prediction_${CURRENT_TIME}.log"

# プロジェクトディレクトリに移動
cd "${PROJECT_DIR}" || exit 1

echo "==========================================" | tee -a "${LOG_FILE}"
echo "Starting Daily Prediction: $(date "+%Y-%m-%d %H:%M:%S")" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

# predict.py の実行 (標準出力と標準エラー出力をターミナルとログファイル両方に出力)
# cron実行時は環境変数が引き継がれないため、必要に応じて仮想環境を有効化してください
# source "${PROJECT_DIR}/jps-env/bin/activate"
"$PYTHON_EXE" predict.py 2>&1 | tee -a "${LOG_FILE}"

echo "==========================================" | tee -a "${LOG_FILE}"
echo "✅ Prediction completed: $(date "+%Y-%m-%d %H:%M:%S")" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
