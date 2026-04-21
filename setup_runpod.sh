#!/bin/bash
# setup_runpod.sh
set -e  # エラーが発生した時点で停止

echo "--- [1/4] Updating System Packages ---"
apt update
apt install -y pkg-config libmysqlclient-dev build-essential rsync curl

echo "--- [2/4] Ensuring 'uv' is installed ---"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv is already installed."
fi

# シェル再起動なしでパスを有効化
export PATH="$HOME/.local/bin:$PATH"

echo "--- [3/4] Python Toolchain Setup ---"
# プロジェクトが要求するPythonバージョンを導入
uv python install 3.11

echo "--- [4/4] Synchronizing Dependencies ---"
# 仮想環境がなければ作成し、pyproject.tomlに基づき同期
if [ ! -d ".venv" ]; then
    uv venv --python 3.11
fi

source .venv/bin/activate
uv sync

echo "--- Environment Setup Complete ---"
echo "To activate environment, run: source .venv/bin/activate"