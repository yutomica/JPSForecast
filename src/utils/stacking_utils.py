import os
import glob
import pandas as pd
import numpy as np
import mlflow
import pyarrow as pa
import pyarrow.ipc as ipc

def load_stacking_oof(cfg, client, meta_df, train_val_meta):
    """
    MLflowのレジストリから指定されたステージのモデルのOOF予測値をロードし、
    meta_dfおよびtrain_val_metaに結合する。
    """
    oof_cols = []
    if not cfg.get("stacking", {}).get("enabled", False):
        return meta_df, train_val_meta, oof_cols

    print("  🔹 [Stacking] Loading OOF predictions from MLflow via stacking_utils...")
    source_stage = cfg.stacking.get("source_stage", "Staging")
    target_models = cfg.stacking.get("target_models", [])
    stacking_df = None
    
    try:
        registered_models = client.search_registered_models()
        versions_to_fetch = []
        for rm in registered_models:
            if target_models:
                match = any([tm.lower() in rm.name.lower() for tm in target_models])
                if not match: continue
            for v in rm.latest_versions:
                if v.current_stage == source_stage:
                    versions_to_fetch.append((rm.name, v.run_id))
        
        for m_name, r_id in versions_to_fetch:
            print(f"    - Fetching OOF for {m_name} (Run: {r_id})...")
            local_dir = client.download_artifacts(r_id, "oof_data")
            csv_files = glob.glob(os.path.join(local_dir, "*.csv"))
            if csv_files:
                oof_sub = pd.read_csv(csv_files[0])
                if 'score' in oof_sub.columns:
                    col_name = f'oof_{m_name}'
                    oof_sub = oof_sub[['date', 'scode', 'score']].rename(columns={'score': col_name})
                    oof_sub['date'] = pd.to_datetime(oof_sub['date'])
                    oof_sub['scode'] = oof_sub['scode'].astype(str) # Force string
                    oof_cols.append(col_name)
                    if stacking_df is None:
                        stacking_df = oof_sub
                    else:
                        stacking_df = pd.merge(stacking_df, oof_sub, on=['date', 'scode'], how='outer')
        
        if stacking_df is not None:
            stacking_df = stacking_df.drop_duplicates(subset=['date', 'scode'])
            
            # --- Calculate Summary Statistics ---
            if len(oof_cols) > 0:
                print(f"    - Calculating OOF summary stats (mean, std) for {len(oof_cols)} models...")
                stacking_df['oof_mean'] = stacking_df[oof_cols].mean(axis=1)
                stacking_df['oof_std'] = stacking_df[oof_cols].std(axis=1).fillna(0.0)
                oof_cols.extend(['oof_mean', 'oof_std'])

            meta_df['date'] = pd.to_datetime(meta_df['date'])
            meta_df['scode'] = meta_df['scode'].astype(str) # Force string
            meta_df = meta_df.join(stacking_df.set_index(['date', 'scode']), on=['date', 'scode'], how='left')
            
            count_before = len(train_val_meta)
            train_val_meta['date'] = pd.to_datetime(train_val_meta['date'])
            train_val_meta['scode'] = train_val_meta['scode'].astype(str) # Force string
            train_val_meta = train_val_meta.join(stacking_df.set_index(['date', 'scode']), on=['date', 'scode'], how='left')
            train_val_meta = train_val_meta.dropna(subset=oof_cols)
            print(f"    - Merged OOF data. Dropped NaNs in train_val_meta: {count_before:,} -> {len(train_val_meta):,}")
        else:
            print("    ⚠️ No OOF data found for target models/stage.")
            
    except Exception as e:
        print(f"    ⚠️ Failed to load OOF data in stacking_utils: {e}")
        
    return meta_df, train_val_meta, oof_cols

def combine_features_with_oof(X, meta_df, row_indices, oof_cols):
    """
    既存の特徴量行列(X)にOOF予測値を結合する。
    X: numpy array or pd.DataFrame or pyarrow.Buffer
    """
    if not oof_cols:
        return X
    
    oof_values = meta_df.loc[row_indices, oof_cols].values
    
    # Handle pyarrow Buffer
    if isinstance(X, pa.Buffer):
        with ipc.open_stream(X) as reader:
            table = reader.read_all()
        X = table.to_pandas()
    
    if isinstance(X, pd.DataFrame):
        X_combined = X.copy()
        for i, col in enumerate(oof_cols):
            X_combined[col] = oof_values[:, i]
        return X_combined
    else:
        # If X is not a standard type (e.g. some other buffer), convert to numpy
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
            
        # Ensure 2D for hstack
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if oof_values.ndim == 1:
            oof_values = oof_values.reshape(-1, 1)
        return np.hstack([X, oof_values])
