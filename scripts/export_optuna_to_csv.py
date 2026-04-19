import argparse
import optuna
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Export Optuna study history for LLM evaluation.")
    parser.add_argument("--storage", type=str, default="sqlite:////Users/yuu/Projects/JPSForecast/optuna.db", help="Optuna DB path")
    parser.add_argument("--study-name", type=str, required=True, help="Optuna study name")
    parser.add_argument("--out", type=str, default="optuna_history_clean.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Loading study '{args.study_name}' from {args.storage}...")
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    
    # トライアル履歴をDataFrameとして取得
    df = study.trials_dataframe()
    
    # 完了したトライアル（COMPLETE）のみに絞り込み
    df = df[df["state"] == "COMPLETE"]
    
    # LLMの評価に必要な列（トライアル番号、スコア、パラメータ）のみを抽出
    cols_to_keep = ["number", "value"] + [c for c in df.columns if c.startswith("params_")]
    clean_df = df[cols_to_keep].sort_values("number")
    
    clean_df.to_csv(args.out, index=False)
    print(f"✅ Extracted {len(clean_df)} complete trials.")
    print(f"✅ Saved clean history to: {args.out}")

if __name__ == "__main__":
    main()
