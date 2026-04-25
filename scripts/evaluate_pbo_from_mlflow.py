import argparse
import os
import sys
import mlflow
import pandas as pd

# プロジェクトルートにパスを通し、srcモジュールを読み込めるようにする
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.pbo_calculator import calculate_pbo_from_dataframe

def main():
    parser = argparse.ArgumentParser(description="MLflowから親Runの子Runを抽出し、データDLとPBO算出を一括で実施します。")
    parser.add_argument("--parent-run-name", type=str, required=True, help="対象となる親Runの名前 (例: LightGBM_Sweep)")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db", help="MLflowのTracking URI")
    parser.add_argument("--out-dir", type=str, default="./evaluation_results", help="結果（CSVやPlot）の出力ディレクトリ")
    parser.add_argument("--metric-prefix", type=str, default="path_", help="PBO計算対象のメトリクスプレフィックス")
    args = parser.parse_args()

    # 1. 接続設定とディレクトリ準備
    mlflow.set_tracking_uri(args.tracking_uri)
    os.makedirs(args.out_dir, exist_ok=True)

    # 2. 親ランの名前からIDを特定
    print(f"🔍 Searching for Parent Run: '{args.parent_run_name}'...")
    runs_df = mlflow.search_runs(
        filter_string=f"tags.mlflow.runName = '{args.parent_run_name}'",
        search_all_experiments=True
    )

    if runs_df.empty:
        raise ValueError(f"Parent Run Name '{args.parent_run_name}' を持つRunが見つかりません。名前や対象のDBが正しいか確認してください。")

    parent_run = runs_df.iloc[0]
    parent_id = parent_run['run_id']
    print(f"✅ Found Parent Run ID: {parent_id}")

    # 3. その親IDを持つ子ラン（Nested Runs）をすべて取得
    print("🔍 Extracting child runs...")
    child_runs = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
        search_all_experiments=True
    )

    if child_runs.empty:
        print("⚠️ 子Runが見つかりませんでした。データ抽出とPBO計算をスキップします。")
        return

    # 4. CSV出力
    csv_path = os.path.join(args.out_dir, f"{args.parent_run_name}_child_runs.csv")
    child_runs.to_csv(csv_path, index=False)
    print(f"✅ Extracted {len(child_runs)} child runs and saved to: {csv_path}")

    # 5. PBO算出とPlot出力
    print("📊 Calculating PBO...")
    plot_path = os.path.join(args.out_dir, f"{args.parent_run_name}_pbo_distribution.png")
    pbo_value = calculate_pbo_from_dataframe(runs_df=child_runs, metric_prefix=args.metric_prefix, plot_output_path=plot_path)
    
    print(f"\n🎯 PBO Calculation Complete! PBO Value: {pbo_value:.2%}")
    print(f"📉 PBO Plot saved to: {plot_path}\n")

if __name__ == "__main__":
    main()