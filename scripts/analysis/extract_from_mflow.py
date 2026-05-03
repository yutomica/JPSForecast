import mlflow
import pandas as pd
import sys

# 1. 接続設定
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 2. 親ランの名前からIDを特定（画像から推測）
parent_name = sys.argv[1]
runs_df = mlflow.search_runs(
    filter_string=f"tags.mlflow.runName = '{parent_name}'",
    search_all_experiments=True
)

if runs_df.empty:
    raise ValueError(f"RunName '{parent_name}' を持つRunが見つかりません。名前や対象のDBが正しいか確認してください。")

parent_run = runs_df.iloc[0]
parent_id = parent_run['run_id']

# 3. その親IDを持つ子ラン（Nested Runs）をすべて取得
child_runs = mlflow.search_runs(
    filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
    search_all_experiments=True
)

# 4. CSV出力
child_runs.to_csv(parent_name+".csv", index=False)
print(f"完了：'{parent_name}' の子ラン {len(child_runs)} 件を出力しました。")