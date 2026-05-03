#!/bin/bash
# step5_fix_model.sh
# 承認済みマニフェスト（YAML）に基づき、モデルの固定（Fix）とStaging昇格を一括実行します。

set -e

# デフォルトのマニフェストファイル
MANIFEST=${1:-"config/promotion/default.yaml"}

echo "============================================================"
echo "🎯 Starting Model Promotion Pipeline"
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

# Python司令塔スクリプトの実行
uv run python scripts/pipeline/execute_promotion.py "${MANIFEST}"

echo ""
echo "🎉 Promotion process finished."
echo "   Please run 'scripts/analysis/evaluate_holdout.py' next to verify scores."
