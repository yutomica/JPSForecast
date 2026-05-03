import argparse
import os
import optuna
from optuna.visualization import plot_param_importances, plot_contour

def main():
    parser = argparse.ArgumentParser(description="Generate HTML plots from an Optuna study.")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna.db", help="Optuna DB path")
    parser.add_argument("--study-name", type=str, required=True, help="Optuna study name")
    parser.add_argument("--out-dir", type=str, default="./plots", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading study '{args.study_name}' from {args.storage}...")
    try:
        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    except Exception as e:
        print(f"❌ Failed to load study: {e}")
        return

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Number of finished trials: {len(completed_trials)}")
    if len(completed_trials) == 0:
        print("⚠️ No completed trials to plot.")
        return

    # 1. Parameter Importances (パラメータの重要度)
    print("Generating Parameter Importances plot...")
    fig_imp = plot_param_importances(study)
    imp_path = os.path.join(args.out_dir, f"{args.study_name}_importances.html")
    fig_imp.write_html(imp_path)
    
    # 2. Contour Plot (パラメータ間の等高線グラフ)
    print("Generating Contour plot...")
    fig_contour = plot_contour(study)
    contour_path = os.path.join(args.out_dir, f"{args.study_name}_contour.html")
    fig_contour.write_html(contour_path)

    print(f"✅ Plots saved successfully:\n  - {imp_path}\n  - {contour_path}")

if __name__ == "__main__":
    main()
