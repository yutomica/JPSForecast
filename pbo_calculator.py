import os
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple

def calculate_cpcv_pbo(
    experiment_id: str,
    metric_prefix: str = "path_",
    clip_eps: float = 1e-5,
    plot_output_path: Optional[str] = "pbo_logit_distribution.png",
    log_to_mlflow: bool = False
) -> float:
    """
    MLflowのRun履歴からCPCVの各パスのスコアを抽出し、PBO (Probability of Backtest Overfitting) を算出する。

    最適化対象スコア（RankICやAP_Severeなど）が大きいほど性能が良い（最大化問題）と想定しています。

    Args:
        experiment_id (str): MLflowのExperiment ID。
        metric_prefix (str): MLflowに記録されたメトリクスのプレフィックス（デフォルト: "path_"）。
        clip_eps (float): ロジット変換時のクリッピングの下限・上限のイプシロン（デフォルト: 1e-5）。
        plot_output_path (Optional[str]): PBO分布のヒストグラムの保存先パス。Noneの場合は保存しない。
        log_to_mlflow (bool): 生成したプロットを現在アクティブなMLflow RunにArtifactとして記録するかどうか。

    Returns:
        float: 算出されたPBOの値。0.0〜1.0の範囲で、0.5を超えると過学習の確率が非常に高いことを示す。

    Raises:
        ValueError: 必要なRunやメトリクスが見つからない場合、または有効なRun数が2未満の場合。
    """
    # 1. MLflowから該当Experimentの終了したRunを取得
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="status = 'FINISHED'",
        output_format="pandas"
    )
    if runs.empty:
        raise ValueError(f"Experiment ID '{experiment_id}' に完了したRunが見つかりません。")

    # 2. メトリクス列の特定
    train_cols = [c for c in runs.columns if c.startswith(f"metrics.{metric_prefix}") and c.endswith("_train_score")]
    valid_cols = [c for c in runs.columns if c.startswith(f"metrics.{metric_prefix}") and c.endswith("_valid_score")]

    if not train_cols or not valid_cols:
        raise ValueError(f"MLflowのログから '{metric_prefix}*_train_score' または '_valid_score' の列が見つかりません。")

    # 列名からパスインデックスを抽出してソート
    def extract_path_idx(col_name: str) -> int:
        parts = col_name.split('_')
        for p in parts:
            if p.isdigit():
                return int(p)
        return -1

    train_cols = sorted(train_cols, key=extract_path_idx)
    valid_cols = sorted(valid_cols, key=extract_path_idx)

    # 3. IS(Train)行列とOOS(Valid)行列の構築
    # 欠損値（NaN）を含むRunはエラーや未完とみなし除外する
    runs_clean = runs.dropna(subset=train_cols + valid_cols).copy()
    if runs_clean.empty:
        raise ValueError("すべてのRunにNaNのスコアが含まれており、有効なデータがありません。")

    # shape: (N_runs, P_paths) -> (P_paths, N_runs) へ転置して行列を構成
    is_matrix = runs_clean[train_cols].T
    oos_matrix = runs_clean[valid_cols].T

    # 行名をパスインデックス、列名を単純な連番（Runインデックス）にする
    is_matrix.index = [extract_path_idx(c) for c in train_cols]
    oos_matrix.index = [extract_path_idx(c) for c in valid_cols]
    is_matrix.columns = np.arange(is_matrix.shape[1])
    oos_matrix.columns = np.arange(oos_matrix.shape[1])

    is_mat = is_matrix.values
    oos_mat = oos_matrix.values
    P, N = is_mat.shape

    if N < 2:
        raise ValueError(f"PBOを計算するには少なくとも2つ以上の有効なRunが必要です。現在のRun数: {N}")

    # 4. PBO算出ロジックのベクトル化計算
    # 各パス (行) における IS (Train) スコアの最大値を持つ Run (n*) を特定
    n_star_indices = np.nanargmax(is_mat, axis=1)

    # OOS行列でのランク計算 (1-based, 昇順)
    oos_ranks = oos_matrix.rank(axis=1, method='average').values

    # 各パスにおいて、n* に該当するRunのOOSランクを抽出
    optimal_oos_ranks = oos_ranks[np.arange(P), n_star_indices]

    # 相対ランク ω_bar の計算
    omega_bar = (optimal_oos_ranks - 1.0) / (N - 1.0)

    # ロジット λ への変換とクリッピング
    omega_bar_clipped = np.clip(omega_bar, clip_eps, 1.0 - clip_eps)
    lambda_logits = np.log(omega_bar_clipped / (1.0 - omega_bar_clipped))

    # PBOの計算 (λ < 0 となる割合)
    pbo = float(np.mean(lambda_logits < 0))

    # 5. 可視化と保存
    if plot_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(plot_output_path)) or ".", exist_ok=True)
        plt.figure(figsize=(10, 6))
        sns.histplot(lambda_logits, bins=min(20, max(5, P // 5)), kde=True, color='lightsteelblue', edgecolor='black')
        plt.axvline(x=0, color='crimson', linestyle='--', linewidth=2, label=f'Logit = 0 (PBO = {pbo:.1%})')
        plt.title('Distribution of Logits (λ) for OOS Performance of IS-Optimal Strategies', fontsize=14)
        plt.xlabel('Logit (λ)', fontsize=12)
        plt.ylabel('Frequency (Number of Paths)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_output_path, dpi=300)
        plt.close()

        # アクティブなMLflow Runが存在する場合、Artifactとして記録
        if log_to_mlflow and mlflow.active_run():
            mlflow.log_artifact(plot_output_path, artifact_path="pbo_evaluation")

    return pbo
