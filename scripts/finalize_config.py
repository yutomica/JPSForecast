import argparse
import optuna
import yaml
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Finalize Hydra config with best Optuna parameters.")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna.db", help="Optuna DB path")
    parser.add_argument("--study-name", type=str, required=True, help="Optuna study name")
    parser.add_argument("--out", type=str, required=True, help="Output YAML path (e.g., config/hparams/lgbm_best.yaml)")
    parser.add_argument("--base-config", type=str, help="Base config YAML to merge with")
    args = parser.parse_args()

    print(f"🚀 Loading study '{args.study_name}' from {args.storage}...")
    try:
        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    except Exception as e:
        print(f"❌ Error loading study: {e}")
        return
    
    try:
        best_trial = study.best_trial
    except ValueError:
        print(f"❌ No complete trials found in study '{args.study_name}'.")
        return

    print(f"✅ Best Trial: #{best_trial.number}")
    print(f"✅ Best Value: {best_trial.value}")
    
    # Optunaのパラメータ（hparams.プレフィックスを削除）
    raw_params = best_trial.params
    best_params = {}
    for k, v in raw_params.items():
        new_k = k.replace("hparams.", "")
        best_params[new_k] = v
        
    # ベース設定がある場合は読み込んでマージ
    final_config = {}
    if args.base_config and os.path.exists(args.base_config):
        print(f"📦 Merging with base config: {args.base_config}")
        with open(args.base_config, "r") as f:
            final_config = yaml.safe_load(f) or {}
            
    # 最良パラメータで更新
    final_config.update(best_params)
    
    # 出力先ディレクトリの作成
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # YAMLとして保存
    with open(out_path, "w") as f:
        f.write(f"# Generated from Optuna study: {args.study_name}\n")
        f.write(f"# Best value (Objective): {best_trial.value}\n")
        f.write(f"# Trial number: {best_trial.number}\n")
        yaml.dump(final_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✨ Best parameters successfully saved to: {args.out}")

if __name__ == "__main__":
    main()
