import numpy as np
import pandas as pd
import shap
from tqdm.auto import tqdm
from src.evaluation.metrics import evaluate_metrics

def _ensure_float32_ndarray(X):
    """
    あらゆる入力形式（Zarrパス, DataFrame, PyArrow Buffer, ndarray等）を
    float32型のdenseなnumpy配列に変換するヘルパー関数。
    """
    X_out = None
    
    # 1. Zarrパス (str) の場合
    if isinstance(X, str) and X.endswith('.zarr'):
        import zarr
        X_out = zarr.open(X, mode='r')[:]
    # 2. DataFrameの場合
    elif isinstance(X, pd.DataFrame):
        X_out = X.values
    # 3. PyArrow Table等 (to_pandasを持つ) の場合
    elif hasattr(X, "to_pandas"):
        X_out = X.to_pandas().values
    # 4. PyArrow Buffer等、特殊なバッファ形式の場合
    elif type(X).__name__ == "Buffer":
        import pyarrow as pa
        import pyarrow.ipc
        try:
            with pa.ipc.open_stream(X) as reader:
                X_out = reader.read_all().to_pandas().values
        except Exception:
            try:
                with pa.ipc.open_file(X) as reader:
                    X_out = reader.read_all().to_pandas().values
            except Exception as e:
                raise TypeError(f"Failed to read pyarrow Buffer: {e}")
    # 5. すでにndarray等の場合
    else:
        X_out = X

    # 最終的に float32 型の連続したメモリ領域を確保する
    if hasattr(X_out, "astype"):
        return np.ascontiguousarray(X_out.astype(np.float32))
    
    return np.ascontiguousarray(np.array(X_out, dtype=np.float32))

def calculate_shap(model, X_valid):
    """
    Validデータを用いてSHAP値を計算し、特徴量ごとの平均絶対SHAP値を返す
    """
    # 堅牢な型変換を実行
    X_input = _ensure_float32_ndarray(X_valid)

    if "ElasticNet" in type(model).__name__:
        explainer = shap.LinearExplainer(model.model, X_input)
    else:
        explainer = shap.TreeExplainer(model.model)

    shap_values = explainer.shap_values(X_input)
    # 回帰の場合、shap_valuesはndarray。クラス分類の場合はリストの可能性あり
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # バイナリ分類のPositiveクラスなどを想定
    abs_shap = np.abs(shap_values).mean(axis=0)
    return abs_shap

def calculate_mda(model, X_valid, y_valid, y_ret_valid, dates_for_shuffle, feature_cols, baseline_score, task_type, target_col, opt_metric="ic", n_repeats=5, random_state=42):
    """
    各特徴量を順番にシャッフルして精度低下を測定する(MDA)
    """
    fold_mda = {}
    
    # 型変換とデータのコピーを作成
    X_mat = _ensure_float32_ndarray(X_valid).copy()
    
    # 実行環境（Apple Silicon等）での高速化のため、
    # 日付グループごとのシャッフルインデックスを事前にn_repeats回分計算してキャッシュする
    _, date_ids = np.unique(dates_for_shuffle, return_inverse=True)
    group_order = np.argsort(date_ids, kind='stable')
    
    rng = np.random.default_rng(random_state)
    shuffled_indices_list = []
    for _ in range(n_repeats):
        rand_vals = rng.random(len(date_ids))
        order = np.lexsort((rand_vals, date_ids))
        shuffled_indices = np.empty_like(date_ids)
        shuffled_indices[group_order] = order
        shuffled_indices_list.append(shuffled_indices)

    for col_idx, col_name in enumerate(tqdm(feature_cols, desc="Calculating MDA")):
        # --- 対象列の元の値を退避 ---
        if X_mat.ndim == 3:
            orig_col_data = X_mat[:, :, col_idx].copy()
        else:
            orig_col_data = X_mat[:, col_idx].copy()
                    
        scores = []
        for rep in range(n_repeats):
            shuffled_indices = shuffled_indices_list[rep]
            
            if X_mat.ndim == 3:
                X_mat[:, :, col_idx] = orig_col_data[shuffled_indices, :]
            else:
                X_mat[:, col_idx] = orig_col_data[shuffled_indices]
                        
            # シャッフル後のデータで予測
            p_permuted = model.predict(X_mat)
            m_permuted = evaluate_metrics(y_valid, p_permuted, y_ret=y_ret_valid, task_type=task_type, target_col=target_col, dates=dates_for_shuffle)
            permuted_score = m_permuted.get(opt_metric, np.nan)
            scores.append(permuted_score)
            
        # 平均精度低下幅を記録 (MDA)
        mda_val = baseline_score - np.nanmean(scores)
        fold_mda[col_name] = 0.0 if abs(mda_val) < 1e-15 else mda_val
        
        # --- 対象列を元の値に復元 (In-place) ---
        if X_mat.ndim == 3:
            X_mat[:, :, col_idx] = orig_col_data
        else:
            X_mat[:, col_idx] = orig_col_data
        
    return fold_mda

def calculate_cfi(model, X_valid, y_valid, y_ret_valid, dates_for_shuffle, feature_groups, feature_cols, baseline_score, task_type, target_col, opt_metric="ic"):
    """
    各特徴量グループに属する特徴量を同時にシャッフルして精度低下を測定する(CFI)
    """
    fold_cfi = {}
    unique_dates = np.unique(dates_for_shuffle)
    
    date_indices_list = [np.where(dates_for_shuffle == d)[0] for d in unique_dates]
    
    X_shufflable = None
    is_df = False

    if isinstance(X_valid, str) and X_valid.endswith('.zarr'):
        import zarr
        print("  [CFI] Loading data from Zarr cache for permutation...")
        X_shufflable = zarr.open(X_valid, mode='r')[:]
    else:
        X_shufflable = X_valid

    if isinstance(X_shufflable, pd.DataFrame):
        X_base = X_shufflable
        is_df = True
    elif isinstance(X_shufflable, np.ndarray):
        X_base = X_shufflable
    elif hasattr(X_shufflable, "to_pandas"): 
        X_base = X_shufflable.to_pandas()
        is_df = True
    elif type(X_shufflable).__name__ == "Buffer":
        import pyarrow as pa
        import pyarrow.ipc
        try:
            with pa.ipc.open_stream(X_shufflable) as reader:
                X_base = reader.read_all().to_pandas()
            is_df = True
        except Exception as e:
            try:
                with pa.ipc.open_file(X_shufflable) as reader:
                    X_base = reader.read_all().to_pandas()
                is_df = True
            except Exception as e2:
                raise TypeError(f"Failed to read pyarrow Buffer: stream err: {e}, file err: {e2}")
    else:
        raise TypeError(f"Unsupported data type for X_valid in calculate_cfi: {type(X_shufflable)}")
        
    col_to_idx = {col_name: idx for idx, col_name in enumerate(feature_cols)}
    
    if is_df:
        X_mat = X_base.values.copy()
        df_columns = X_base.columns
        df_index = X_base.index
    else:
        X_mat = X_base.copy()
        
    for group_name, cols_in_group in feature_groups.items():
        valid_cols = [col for col in cols_in_group if col in col_to_idx]
        if not valid_cols:
            continue
            
        group_col_indices = [col_to_idx[col] for col in valid_cols]
        
        if X_mat.ndim == 3:
            orig_group_data = X_mat[:, :, group_col_indices].copy()
            for d_pos in date_indices_list:
                if len(d_pos) > 1:
                    idx_perm = np.random.permutation(len(d_pos))
                    shuffled_pos = d_pos[idx_perm]
                    time_steps = np.arange(X_mat.shape[1])
                    X_mat[np.ix_(d_pos, time_steps, group_col_indices)] = X_mat[np.ix_(shuffled_pos, time_steps, group_col_indices)]
        else:
            orig_group_data = X_mat[:, group_col_indices].copy()
            for d_pos in date_indices_list:
                if len(d_pos) > 1:
                    idx_perm = np.random.permutation(len(d_pos))
                    shuffled_pos = d_pos[idx_perm]
                    X_mat[np.ix_(d_pos, group_col_indices)] = X_mat[np.ix_(shuffled_pos, group_col_indices)]
                    
        if is_df:
            X_pred_input = pd.DataFrame(X_mat, index=df_index, columns=df_columns)
            X_pred_input = X_pred_input.astype(X_base.dtypes)
        else:
            X_pred_input = X_mat

        if type(X_shufflable).__name__ == "Buffer":
            import pyarrow as pa
            import pyarrow.ipc
            sink = pa.BufferOutputStream()
            table_permuted = pa.Table.from_pandas(X_pred_input)
            with pa.ipc.new_stream(sink, table_permuted.schema) as writer:
                writer.write_table(table_permuted)
            X_pred_input = sink.getvalue()
            
        p_permuted = model.predict(X_pred_input)
        m_permuted = evaluate_metrics(y_valid, p_permuted, y_ret=y_ret_valid, task_type=task_type, target_col=target_col, dates=dates_for_shuffle)
        permuted_score = m_permuted.get(opt_metric, np.nan)
        cfi_val = baseline_score - permuted_score
        fold_cfi[group_name] = 0.0 if abs(cfi_val) < 1e-15 else cfi_val
        
        if X_mat.ndim == 3:
            X_mat[:, :, group_col_indices] = orig_group_data
        else:
            X_mat[:, group_col_indices] = orig_group_data
            
    return fold_cfi