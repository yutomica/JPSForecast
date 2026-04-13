import numpy as np
import pandas as pd
import shap
from src.utils.evaluation import evaluate_metrics

def calculate_shap(model, X_valid):
    """
    Validデータを用いてSHAP値を計算し、特徴量ごとの平均絶対SHAP値を返す
    """
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_valid)
    # 回帰の場合、shap_valuesはndarray。クラス分類の場合はリストの可能性あり
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # バイナリ分類のPositiveクラスなどを想定
    abs_shap = np.abs(shap_values).mean(axis=0)
    return abs_shap

def calculate_mda(model, X_valid, y_valid, y_ret_valid, dates_for_shuffle, feature_cols, baseline_score, task_type, target_col, opt_metric="ic"):
    """
    各特徴量を順番にシャッフルして精度低下を測定する(MDA)
    """
    fold_mda = {}
    unique_dates = np.unique(dates_for_shuffle)
    
    for col_idx, col_name in enumerate(feature_cols):
        # メモリ節約のため、破壊的な変更を避けコピーを作成
        X_valid_permuted = X_valid.copy()
        # --- 日次クロスセクション内でのシャッフル ---
        for d in unique_dates:
            date_mask = (dates_for_shuffle == d)
            if isinstance(X_valid_permuted, pd.DataFrame):
                # DataFrameの場合
                date_pos = np.where(date_mask)[0]
                shuffled_values = np.random.permutation(X_valid_permuted.iloc[date_pos, col_idx].values)
                X_valid_permuted.iloc[date_pos, col_idx] = shuffled_values
            else:
                # ndarrayの場合 (TCNなど)
                date_indices = np.where(date_mask)[0]
                idx_perm = np.random.permutation(date_indices)
                if X_valid_permuted.ndim == 3:
                    X_valid_permuted[date_indices, :, col_idx] = X_valid_permuted[idx_perm, :, col_idx]
                else:
                    X_valid_permuted[date_indices, col_idx] = X_valid_permuted[idx_perm, col_idx]
                    
        # シャッフル後のデータで予測
        p_permuted = model.predict(X_valid_permuted)
        m_permuted = evaluate_metrics(y_valid, p_permuted, y_ret=y_ret_valid, task_type=task_type, target_col=target_col, dates=dates_for_shuffle)
        permuted_score = m_permuted.get(opt_metric, np.nan)
        # 精度低下幅を記録 (MDA)
        fold_mda[col_name] = baseline_score - permuted_score
        
    return fold_mda