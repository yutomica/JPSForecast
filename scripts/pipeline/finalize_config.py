import argparse
from pathlib import Path
from src.utils.config_utils import finalize_hparams_from_optuna, find_latest_study_name

def main():
    parser = argparse.ArgumentParser(description="Finalize Hydra config with best Optuna parameters.")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna.db", help="Optuna DB path")
    parser.add_argument("--study-name", type=str, required=True, help="Optuna study name (or prefix ending with *)")
    parser.add_argument("--out", type=str, required=True, help="Output YAML path (e.g., config/hparams/lgbm_best.yaml)")
    parser.add_argument("--base-config", type=str, help="Base config YAML to merge with (comma-separated for multiple files)")
    args = parser.parse_args()

    study_name = args.study_name
    if study_name.endswith("*"):
        prefix = study_name[:-1]
        try:
            study_name = find_latest_study_name(prefix, args.storage)
            print(f"🔎 Found latest study: {study_name}")
        except Exception as e:
            print(f"❌ {e}")
            return

    try:
        finalize_hparams_from_optuna(
            study_name=study_name,
            storage=args.storage,
            out_path=args.out,
            base_config_path=args.base_config
        )
    except Exception as e:
        print(f"❌ {e}")

if __name__ == "__main__":
    main()
