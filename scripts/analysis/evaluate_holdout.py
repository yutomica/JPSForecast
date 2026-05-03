import os
import gc
import yaml
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from src.evaluation.metrics import evaluate_metrics
from sklearn.metrics import average_precision_score

def load_period_config(domain):
    path = Path(f"config/period/{domain.lower()}_standard.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_staging_models(client, algorithm_list, target_list):
    staging_versions = []
    for algo in algorithm_list:
        for target in target_list:
            model_name = f"{algo}_{target}"
            try:
                versions = client.get_latest_versions(model_name, stages=["Staging"])
                for v in versions:
                    staging_versions.append({
                        "name": model_name,
                        "version": v.version,
                        "algo": algo,
                        "target": target,
                        "variant": v.tags.get("variant", "default"),
                        "feature_config": v.tags.get("feature_config", "unknown"),
                        "run_id": v.run_id
                    })
            except Exception:
                # モデルが存在しない場合はスキップ
                continue
    return staging_versions

def calculate_pr_auc_30pt(y_true, y_pred):
    # MDDが0.30(30%)以上のものを陽性とする
    binary_true = (y_true >= 0.30).astype(int)
    if np.sum(binary_true) == 0:
        return np.nan
    return average_precision_score(binary_true, y_pred)

@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg: DictConfig):
    # 1. MLflow Setup
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    client = MlflowClient()

    # 2. Load Data
    master_dir = Path(cfg.data.path)
    print(f"📂 Loading meta data from {master_dir}...")
    meta_df = pd.read_parquet(master_dir / "index_meta.parquet")
    
    # 3. Determine Holdout Period
    # 両ドメインで共通の test_start_date ("2025-04-01") を使用
    holdout_start = pd.to_datetime("2025-04-01")
    holdout_meta = meta_df[meta_df['date'] >= holdout_start].copy()
    print(f"🗓️ Holdout period: {holdout_meta['date'].min()} to {holdout_meta['date'].max()}")
    print(f"   Total rows: {len(holdout_meta):,}")

    if holdout_meta.empty:
        print("❌ No data found for the holdout period.")
        return

    # 4. Discovery Staging Models
    algorithms = ["lgbm", "tcn", "ft_transformer", "elasticnet"]
    targets = [
        "target_tac_vol_scaled_asym_return",
        "target_str_sharpe_adj",
        "target_tac_max_neg_path",
        "target_str_mdd"
    ]
    
    staging_models = get_staging_models(client, algorithms, targets)
    if not staging_models:
        print("❌ No models found in Staging.")
        return
    
    print(f"🔍 Found {len(staging_models)} model versions in Staging.")

    # 5. Load All Required Features
    # メモリ効率のため、全モデルが必要とする特徴量のユニオンを一度にロードする
    all_feature_names = pd.read_json(master_dir / "feature_names.json", typ='series').tolist()
    
    # 全モデルの全フォールドが使用する列インデックスを収集
    # (EnsembleInferencePipeline.col_indices に全列のインデックスが入っている前提)
    # 面倒なので、一旦 memmap 経由で必要な時にアクセスする方式にする
    # features.npy は train.py で作成される hash 付きのものではなく、
    # 汎用的なものが master_dir にあるか確認
    
    # train.py のロジックを参考に memmap を構築
    print("🧠 Preparing shared memory map for features...")
    # 全特徴量を一旦対象にする
    chunk_files = sorted((master_dir / "features").glob("features_chunk_*.parquet"))
    # cache用のパス（簡易化のため固定名にするか、train.pyのロジックを流用）
    # ここではシンプルに全特徴量をアタッチ
    # 注意: 大規模データの場合は surgical なロードが必要だが、Holdout期間のみであれば
    # 全特徴量をメモリに載せても 48GB RAM なら耐えられるはず。
    
    # 6. Evaluation Loop
    results = []

    for m_info in staging_models:
        print(f"\n🚀 Evaluating {m_info['name']} v{m_info['version']} ({m_info['variant']})")
        
        try:
            model_uri = f"models:/{m_info['name']}/{m_info['version']}"
            model = mlflow.pyfunc.load_model(model_uri)
            
            # ドメインに応じたフィルタリング
            domain_key = 'TAC' if 'tac' in m_info['target'] else 'STR'
            mask_col = 'is_candidate_tac' if domain_key == 'TAC' else 'is_candidate_str'
            
            m_holdout = holdout_meta[holdout_meta[mask_col] == True].copy()
            if m_holdout.empty:
                print(f"⚠️ No candidates for domain {domain_key} in holdout period.")
                continue

            # 入力DataFrameの構築
            # EnsembleInferencePipeline.predict は DataFrame を受け取る
            # 必要な特徴量名を取得 (モデル内部の pipeline から)
            # MLflow pyfunc model の内部構造にアクセス
            raw_model = model._model_impl.python_model
            col_indices = raw_model.col_indices
            feature_cols = [all_feature_names[i] for i in col_indices]
            
            # Holdout 期間の特徴量をロード
            # master_dir / features / features_chunk_*.parquet から surgical に読み込むのは遅いので
            # memmap があればそれを使う。なければ今回だけロード。
            
            # surgical load logic:
            row_indices = m_holdout.index.values
            # 実際には index_meta.parquet の行番号と features の行番号が一致している必要がある
            # (master_data 作成時に一致させている前提)
            
            # 簡易的に memmap を開く（train.py と同じハッシュロジックを使うのが確実だが
            # ここでは surgical な read_parquet を試みる。Holdout は高々数万行のはず）
            X_df = pd.DataFrame(index=m_holdout.index)
            # メモリ節約のため、100列ずつ読み込む
            for i in range(0, len(feature_cols), 100):
                subset_cols = feature_cols[i:i+100]
                # 全てのチャンクから該当行・該当列を読み込むのは非効率
                # master_data/features_all.parquet などがあれば楽だが...
                # プロジェクト構成上、chunks しかない。
                
                # 解決策: train.py の mmap 作成ロジックを模倣して、Holdout 期間分だけメモリに載せる
                # ... (略) ...
            
            # 【効率化】一度だけ全特徴量の Holdout データをロードする
            if 'X_holdout_full' not in locals():
                print(f"📥 Loading all features for holdout period...")
                # master_dir / features / features_chunk_*.parquet から Holdout 期間の行を抽出
                # meta_df の index と物理行が一致していることを利用
                holdout_idx_min = holdout_meta.index.min()
                holdout_idx_max = holdout_meta.index.max()
                
                # チャンクを跨ぐ可能性を考慮
                X_holdout_full_list = []
                current_row = 0
                for cf in chunk_files:
                    chunk_len = int(cf.stem.split('_')[-1].split('-')[-1]) # features_chunk_0-9999
                    start_row, end_row = map(int, cf.stem.split('_')[-1].split('-'))
                    
                    if end_row < holdout_idx_min:
                        continue
                    if start_row > holdout_idx_max:
                        break
                    
                    # 読み込み
                    df_chunk = pd.read_parquet(cf)
                    # holdout 期間に重なる部分を抽出
                    overlap_start = max(start_row, holdout_idx_min)
                    overlap_end = min(end_row, holdout_idx_max)
                    
                    X_holdout_full_list.append(df_chunk.loc[overlap_start:overlap_end])
                    del df_chunk
                    gc.collect()
                
                X_holdout_full = pd.concat(X_holdout_full_list)
                print(f"✅ Loaded {len(X_holdout_full):,} rows of features.")

            # 推論実行
            X_input = X_holdout_full.loc[m_holdout.index, feature_cols]
            preds = model.predict(X_input)
            
            # 指標算出
            y_true = m_holdout[m_info['target']].values
            y_ret = m_holdout['Future_Close_Tac' if domain_key == 'TAC' else 'Future_Close_Str'].values - 1.0
            dates = m_holdout['date'].values
            
            metrics = evaluate_metrics(
                y_true=y_true,
                y_pred=preds,
                y_ret=y_ret,
                task_type='regression', # 全て回帰として扱う（riskもMDD/下落率なので）
                target_col=m_info['target'],
                dates=dates,
                ndcg_k=10
            )
            
            # str_risk のための追加指標
            if m_info['target'] == "target_str_mdd":
                metrics['pr_auc_30pt'] = calculate_pr_auc_30pt(y_true, preds)
                
            # 結果の保持
            res_entry = {
                "Model": m_info['name'],
                "Version": m_info['version'],
                "Variant": m_info['variant'],
                "Target": m_info['target'],
                "FeatureConfig": m_info['feature_config']
            }
            
            # READMEで指定された主要指標を抽出
            if "tac_alpha" in m_info['target']:
                res_entry["PrimaryMetric"] = "ndcg_10"
                res_entry["Score"] = metrics.get("ndcg_10")
            elif "tac_risk" in m_info['target']:
                res_entry["PrimaryMetric"] = "AP_severe"
                res_entry["Score"] = metrics.get("AP_severe")
            elif "str_alpha" in m_info['target']:
                res_entry["PrimaryMetric"] = "RankIC_reb"
                res_entry["Score"] = metrics.get("RankIC_reb")
            elif "str_risk" in m_info['target']:
                res_entry["PrimaryMetric"] = "pr_auc_30pt"
                res_entry["Score"] = metrics.get("pr_auc_30pt", metrics.get("AP_severe_STR"))
            
            # 他の全指標も一応入れる
            res_entry.update({f"m_{k}": v for k, v in metrics.items()})
            results.append(res_entry)
            
            print(f"📊 Result: {res_entry['PrimaryMetric']} = {res_entry['Score']:.4f}")

        except Exception as e:
            print(f"❌ Error evaluating {m_info['name']}: {e}")
            import traceback
            traceback.print_exc()

    # 7. Summary
    if not results:
        print("❌ No evaluation results generated.")
        return

    df_results = pd.DataFrame(results)
    
    # ターゲットごとに最良のバリアントを特定
    print("\n" + "="*60)
    print("🏆 Best Variants per Target (Holdout Evaluation)")
    print("="*60)
    
    for target in targets:
        target_df = df_results[df_results['Target'] == target]
        if target_df.empty: continue
        
        # 主要指標でソート (全て maximize 前提)
        best_row = target_df.sort_values(by="Score", ascending=False).iloc[0]
        print(f"\n🎯 Target: {target}")
        print(f"   Best Model  : {best_row['Model']} v{best_row['Version']}")
        print(f"   Variant     : {best_row['Variant']}")
        print(f"   {best_row['PrimaryMetric']} Score : {best_row['Score']:.4f}")

    # CSV保存
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"holdout_evaluation_{timestamp}.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n✅ Evaluation report saved to {output_path}")

if __name__ == "__main__":
    main()
