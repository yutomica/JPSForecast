import os
import subprocess
import yaml
import argparse
import sys
from pathlib import Path
import time

def run_command(cmd, dry_run=False):
    # コマンド引数リストの正規化（型変換と空白削除）
    normalized_cmd = [str(c).strip() for c in cmd]
    
    print(f"\n🚀 Executing: {' '.join(normalized_cmd)}")
    sys.stdout.flush()

    if dry_run:
        print("⚠️ [Dry Run] Command skipped.")
        return True
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    try:
        process = subprocess.Popen(
            normalized_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1,
            env=env,
            shell=False
        )
        
        if process.stdout:
            for line in process.stdout:
                print(line, end="")
                sys.stdout.flush()
        
        process.wait()
        return process.returncode == 0
    except Exception as e:
        print(f"❌ Error executing command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="承認済みマニフェストに基づきモデルを固定・昇格させます。")
    parser.add_argument("manifest", type=str, help="プロモーション定義YAMLのパス (例: config/promotion/default.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="実行せずにコマンドのみ表示します")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"❌ Manifest not found: {args.manifest}")
        sys.exit(1)

    with open(args.manifest, "r") as f:
        config = yaml.safe_load(f)

    variant_global = config.get("variant_global", "default")
    promotions = config.get("promotions", [])

    print(f"🔮 Starting Promotion Process for Variant: {variant_global}")
    print(f"📄 Loaded {len(promotions)} definitions from {args.manifest}")

    for p in promotions:
        if not p.get("enabled", False):
            continue

        model = p["model"]
        domain = p["domain"]
        role = p["role"]
        study = p.get("study")
        
        if not study:
            print(f"⏭️ Skipping {model} ({domain}-{role}): No study name provided.")
            continue

        target = f"{domain}_{role}"
        variant = p.get("variant", variant_global)
        
        # 1. ハイパーパラメータの確定 (finalize_config.py)
        # ファイル名に variant を含めることで上書きを防止
        fix_hparams_filename = f"{model}_{target}_{variant}"
        fix_hparams_path = f"config/hparams/{fix_hparams_filename}.yaml"
        
        # マニフェストに base_hparams があればそれを使用、なければデフォルト推論
        base_hparams_raw = p.get("base_hparams")
        if not base_hparams_raw:
            if model == "lgbm":
                base_hparams_raw = f"lgbm/base,lgbm/{target},anchor/lgbm_{target}"
            elif model == "tcn":
                base_hparams_raw = f"tcn/base,tcn/{target},anchor/tcn_{target}"
            elif model == "ft_transformer":
                base_hparams_raw = f"ft_transformer/base,ft_transformer/{target},anchor/ft_transformer_{target}"
            else:
                base_hparams_raw = f"{model}/base"
        
        base_hparams_raw = base_hparams_raw.replace("{target}", target)
        base_config_args = ",".join([f"config/hparams/{h.strip()}.yaml" for h in base_hparams_raw.split(",")])

        print(f"\n" + "="*60)
        print(f"📦 Processing: {model} ({domain}) - {role}")
        print(f"   Variant: {variant}")
        print(f"   Study  : {study}")
        print("="*60)
        sys.stdout.flush()
        
        finalize_cmd = [
            "uv", "run", "python", "scripts/pipeline/finalize_config.py",
            "--storage", "sqlite:///optuna.db",
            "--study-name", study,
            "--base-config", base_config_args,
            "--out", fix_hparams_path
        ]
        
        if not run_command(finalize_cmd, args.dry_run):
            print(f"❌ Failed to finalize config for {model} {target}")
            continue

        # 2. 最終モデルの学習実行 (train.py)
        # experiment, mode は main.yaml の defaults に既に存在するため '+' は不要
        train_cmd = [
            "uv", "run", "python", "train.py",
            f"domain={domain}",
            f"target={target}",
            "data=master",
            f"features=features_{model}_{target}_fixed",
            f"experiment={model}_{target}",
            f"model={model}",
            f"hparams={fix_hparams_filename}",
            "+mode=fix",
            f"variant={variant}",
            f"++mlflow.tags.source_study={study}",
            f"period={domain}_standard",
            "cv=anchored_walk_forward",
            f"mlflow.experiment_name=JPSForecast_{target}",
            f"++mlflow.run_name=Step5_Promotion_{model}_{target}"
        ]
        
        if not run_command(train_cmd, args.dry_run):
            print(f"❌ Failed training for {model} {target}")
            continue

    print("\n✅ All enabled promotions completed.")

if __name__ == "__main__":
    main()
