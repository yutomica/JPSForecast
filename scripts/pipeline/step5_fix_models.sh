#!/bin/bash
# step5_fix_models.sh
# 承認済みマニフェスト（YAML）に基づき、Optunaの最良試行から固定ハイパーパラメータを生成します。

set -e

# デフォルトのマニフェストファイル
MANIFEST=${1:-"config/promotion/default.yaml"}
EXTRA_ARGS=("${@:2}")

echo "============================================================"
echo "🎯 Starting Fixed Model Config Generation"
echo "   Manifest: ${MANIFEST}"
echo "============================================================"

# 並列処理時のスレッド競合を防ぐための環境変数設定
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# MLflowのバックエンドを一致させる
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export PYTHONPATH=$PYTHONPATH:.

# Pythonスクリプトの実行
uv run python scripts/pipeline/execute_promotion.py "${MANIFEST}" "${EXTRA_ARGS[@]}"

echo ""
echo "🎉 Fixed model config generation finished."
