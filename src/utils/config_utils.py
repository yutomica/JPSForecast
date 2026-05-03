import optuna
import yaml
import os
from pathlib import Path
from typing import Optional, Any, Dict

def find_latest_study_name(prefix: str, storage: str) -> str:
    """
    指定されたプレフィックスで始まる最新のOptunaスタディ名を取得する。
    """
    summaries = optuna.get_all_study_summaries(storage=storage)
    # プレフィックスに一致し、かつ完了した試行があるものを抽出
    matching_studies = [s for s in summaries if s.study_name.startswith(prefix)]
    if not matching_studies:
        raise ValueError(f"No studies found with prefix: {prefix}")
    
    # スタディ名に含まれるタイムスタンプ（YYYYMMDD_HHMMSS）でソート
    # step4の命名規則: ${exp_name}_${model}_${target}_${timestamp}
    latest_study = max(matching_studies, key=lambda s: s.study_name)
    return latest_study.study_name

def get_trial_count(study_name: str, storage: str, state: Optional[str] = None) -> int:
    """指定されたスタディの現在の試行数を取得する。"""
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        if state == "COMPLETE":
            # 完了したユニークなパラメータセット数を数えるのが理想だが、
            # 簡易的に COMPLETE 状態の試行数を返す。
            return len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        return len(study.trials)
    except Exception:
        return 0

def finalize_hparams_from_optuna(
    study_name: str,
    storage: str,
    out_path: str | Path,
    base_config_path: Optional[str | Path | list] = None
) -> Dict[str, Any]:
    """
    Optunaのスタディから最良の試行パラメータを取得し、Hydra設定ファイルとして保存する。
    複数のベース設定を累積的にマージ可能。
    """
    print(f"🚀 Loading study '{study_name}' from {storage}...")
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        raise RuntimeError(f"Error loading study: {e}")
    
    try:
        best_trial = study.best_trial
    except ValueError:
        raise ValueError(f"No complete trials found in study '{study_name}'.")

    print(f"✅ Best Trial: #{best_trial.number}")
    print(f"✅ Best Value: {best_trial.value}")
    
    # Optunaのパラメータ（hparams.プレフィックスを削除）
    raw_params = best_trial.params
    best_params = {}
    for k, v in raw_params.items():
        new_k = k.replace("hparams.", "")
        best_params[new_k] = v
        
    # ベース設定の処理（単一文字列、リスト、またはカンマ区切り文字列に対応）
    final_config = {}
    if base_config_path:
        if isinstance(base_config_path, (str, Path)):
            paths = str(base_config_path).split(",")
        else:
            paths = base_config_path
            
        for path in paths:
            path = path.strip()
            if os.path.exists(path):
                print(f"📦 Merging with base config: {path}")
                with open(path, "r") as f:
                    config = yaml.safe_load(f) or {}
                    # defaults キーはHydraの動作と競合するため、最終ファイルからは除外してフラット化する
                    if "defaults" in config:
                        del config["defaults"]
                    final_config.update(config)
            else:
                print(f"⚠️ Base config not found, skipping: {path}")
            
    # 最良パラメータで最終更新
    final_config.update(best_params)
    
    # 出力先ディレクトリの作成
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # YAMLとして保存
    with open(out_path, "w") as f:
        f.write(f"# Generated from Optuna study: {study_name}\n")
        f.write(f"# Best value (Objective): {best_trial.value}\n")
        f.write(f"# Trial number: {best_trial.number}\n")
        yaml.dump(final_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✨ Best parameters successfully saved to: {out_path}")
    return final_config

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["get_trial_count"], required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--state", default=None)
    args = parser.parse_args()
    
    if args.action == "get_trial_count":
        print(get_trial_count(args.study_name, args.storage, args.state))
