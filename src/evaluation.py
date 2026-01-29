import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from scipy.stats import spearmanr

def calculate_bin_stats(df, score_col, target_col, model_type):
    df_calc = df.copy()
    if 'Entry_Price' in df_calc.columns and 'Future_Close' in df_calc.columns:
        df_calc['Entry_Price'] = df_calc['Entry_Price'].replace(0, np.nan)
        df_calc['Ret_High'] = (df_calc['Future_High'] / df_calc['Entry_Price']) - 1.0
        df_calc['Ret_Low'] = (df_calc['Future_Low'] / df_calc['Entry_Price']) - 1.0
        df_calc['Ret_Close'] = (df_calc['Future_Close'] / df_calc['Entry_Price']) - 1.0
    
    if model_type == 'classification':
        bins = np.arange(0, 1.05, 0.05)
        df_calc['bin'] = pd.cut(df_calc[score_col], bins=bins)
    else:
        try:
            df_calc['bin'] = pd.qcut(df_calc[score_col], q=20, duplicates='drop')
        except:
            df_calc['bin'] = pd.cut(df_calc[score_col], bins=10)

    agg_dict = {target_col: ['count', 'mean']}
    ret_cols = ['Ret_High', 'Ret_Low', 'Ret_Close']
    stats = ['mean', lambda x: x.quantile(0.05), 'median', lambda x: x.quantile(0.95)]
    
    for c in ret_cols:
        if c in df_calc.columns:
            agg_dict[c] = stats

    bin_stats = df_calc.groupby('bin', observed=False).agg(agg_dict)
    flat_cols = ["target_count", "target_mean"]
    stats_names = ['mean', 'q05', 'q50', 'q95']
    for rc in ret_cols:
        if rc in df_calc.columns:
            for sn in stats_names:
                flat_cols.append(f"{rc}_{sn}")
    
    if len(bin_stats.columns) == len(flat_cols):
        bin_stats.columns = flat_cols
    else:
        bin_stats.columns = [f"{c[0]}_{c[1]}" for c in bin_stats.columns]
    return bin_stats.reset_index()

def evaluate_metrics(y_true, y_pred, config, df_meta=None):
    metrics = {}
    if np.isnan(y_pred).any():
        if config['type'] == 'classification':
            metrics['AUC'] = 0.5
            metrics['LogLoss'] = 999.0
        else:
            metrics['RMSE'] = 999.0
            metrics['IC'] = 0.0
        return metrics

    if config['type'] == 'classification':
        try:
            auc = roc_auc_score(y_true, y_pred)
            metrics['AUC'] = auc
            loss = log_loss(y_true, y_pred)
            metrics['LogLoss'] = loss
        except:
            metrics['AUC'] = 0.5
    else:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['RMSE'] = rmse
        if df_meta is not None:
            df_rank = df_meta.copy()
            df_rank['score'] = y_pred
            df_rank['target'] = y_true
            ic_list = []
            for date, group in df_rank.groupby('date'):
                if len(group) < 2: continue
                corr, _ = spearmanr(group['target'], group['score'])
                if not np.isnan(corr):
                    ic_list.append(corr)
            metrics['IC'] = np.mean(ic_list) if ic_list else 0
    return metrics