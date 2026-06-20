import os
import yaml
import argparse
import sys

from src.utils.config_utils import finalize_hparams_from_optuna, find_latest_study_name


def print_dry_run_action(title: str, values: dict[str, str]) -> None:
    print(f"\n🚀 {title}:")
    for key, value in values.items():
        print(f"   {key}: {value}")
    print("⚠️ [Dry Run] Skipped.")


def main():
    parser = argparse.ArgumentParser(description="承認済みマニフェストに基づきモデルを固定・昇格させます。")
    parser.add_argument("manifest", type=str, help="プロモーション定義YAMLのパス (例: config/promotion/default.yaml)")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna.db", help="Optuna DB path")
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

        if args.dry_run:
            print_dry_run_action(
                "Finalize hparams",
                {
                    "study": study,
                    "storage": args.storage,
                    "base_config": base_config_args,
                    "out": fix_hparams_path,
                },
            )
        else:
            try:
                resolved_study = study
                if resolved_study.endswith("*"):
                    prefix = resolved_study[:-1]
                    resolved_study = find_latest_study_name(prefix, args.storage)
                    print(f"🔎 Found latest study: {resolved_study}")

                finalize_hparams_from_optuna(
                    study_name=resolved_study,
                    storage=args.storage,
                    out_path=fix_hparams_path,
                    base_config_path=base_config_args,
                )
            except Exception as e:
                print(f"❌ {e}")
                print(f"❌ Failed to finalize config for {model} {target}")
                continue

    print("\n✅ All enabled promotions completed.")

if __name__ == "__main__":
    main()
