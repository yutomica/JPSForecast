#!/bin/bash
# setup_runpod.sh
set -e  # エラーが発生した時点で停止

echo "--- [1/5] Updating System Packages ---"
# プロクオンツ注：ビルドに必要な最小限のパッケージを導入
apt update
apt install -y pkg-config libmysqlclient-dev build-essential rsync curl

echo "--- [2/5] Ensuring 'uv' is installed ---"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv is already installed."
fi
export PATH="$HOME/.local/bin:$PATH"

# ハードリンク失敗によるフリーズを回避するための設定
export UV_LINK_MODE=copy

echo "--- [3/5] Cleaning up old environment ---"
# 既存の不完全・肥大化した環境をパージしてリセット
rm -rf .venv

echo "--- [4/5] Initializing venv with System Site Packages ---"
# 重要：RunPod標準のPythonを指定し、システム側の最適化済みPyTorchを継承する
# これにより CUDA initialization error を確実に回避します
uv venv --python /usr/bin/python3 --system-site-packages

echo "--- [5/5] Synchronizing Dependencies (Excluding Torch) ---"
# システム側のtorchを保護するため、インストール対象から除外
# uv.lockを尊重しつつ、GPUドライバとの整合性を維持します
source .venv/bin/activate
uv sync --no-install-package torch \
        --no-install-package nvidia-cuda-runtime-cu12 \
        --no-install-package nvidia-cuda-cupti-cu12 \
        --no-install-package nvidia-cudnn-cu12 \
        --no-install-package nvidia-cublas-cu12

echo "--- Environment Setup Complete ---"
echo "------------------------------------------------"
# 動作確認：GPUが正しく認識されるか自動チェック
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo "------------------------------------------------"
echo "To activate environment, run: source .venv/bin/activate"