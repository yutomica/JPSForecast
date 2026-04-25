#!/bin/bash

# 出力ファイル名
OUTPUT_FILE="project_context.txt"

# 初期化
echo "--- PROJECT ARCHITECTURE ---" > $OUTPUT_FILE

# 1. ディレクトリ構造の書き出し (treeコマンドを使用)
# data, logs, venv, .gitなどは除外
echo "Generating Directory Tree..."
echo "--- DIRECTORY TREE ---" >> $OUTPUT_FILE
tree -I "data|logs|outputs|venv|__pycache__|.git|.ipynb_checkpoints" >> $OUTPUT_FILE
echo -e "\n" >> $OUTPUT_FILE

# 2. ファイル内容の集約
echo "Collecting file contents..."

# 対象とする拡張子を定義
EXTENSIONS=("*.yaml" "*.py" "*.sh" "*.toml")

for ext in "${EXTENSIONS[@]}"; do
    find . -maxdepth 4 -name "$ext" \
    -not -path "*/data/*" \
    -not -path "*/logs/*" \
    -not -path "*/outputs/*" \
    -not -path "*/venv/*" \
    -not -path "*/.git/*" | while read -r file; do
        echo "--- FILE: $file ---" >> $OUTPUT_FILE
        cat "$file" >> $OUTPUT_FILE
        echo -e "\n--- END OF FILE ---\n" >> $OUTPUT_FILE
    done
done

echo "Done! $OUTPUT_FILE has been generated."