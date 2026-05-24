import mlflow
import pandas as pd
import argparse
from pathlib import Path
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Extract step0_target_probe MLflow runs to CSV.")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db", help="MLflow Tracking URI")
    parser.add_argument("--output", type=str, default="target_probe_results.csv", help="Output CSV path")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    
    print(f"Searching for runs with tag 'stage' = 'step0_target_probe'...")
    
    # search_runs returns a pandas DataFrame directly
    try:
        df = mlflow.search_runs(
            filter_string="tags.stage = 'step0_target_probe'",
            search_all_experiments=True
        )
    except Exception as e:
        print(f"Error querying MLflow: {e}")
        sys.exit(1)
    
    if df.empty:
        print("No runs found with the specified tag.")
        return

    extract_cols = ['run_id', 'experiment_id', 'params.period', 'tags.target_name']
    extract_cols += [col for col in df.columns if col.find('top_bin_Future_') != -1]
    extract_cols += [col for col in df.columns if col.find('bot_bin_Future_') != -1]
    extract_cols += [col for col in df.columns if col.find('top10_Future_') != -1]
    extract_cols += [col for col in df.columns if col.find('bot10_Future_') != -1]
    df = df[extract_cols]

    # Sort by start_time if available for better readability
    if 'start_time' in df.columns:
        df = df.sort_values(by='start_time', ascending=False)
        
    print(f"Found {len(df)} runs. Exporting to {args.output}...")
    
    # Ensure output directory exists
    out_path = Path(args.output)
    if out_path.parent != Path(''):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
    df.to_csv(out_path, index=False)
    print(f"✅ Extracted data successfully saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
