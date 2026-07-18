import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class


def resolve_feature_columns(cfg, master_dir: Path):
    all_features = pd.read_json(master_dir / "feature_names.json", typ="series").tolist()

    features_choice = HydraConfig.get().runtime.choices.features
    print(f"🧬 Feature select: {features_choice}")

    feature_cols = cfg.features.get("feature_cols", all_features)
    feature_cols = list(dict.fromkeys(feature_cols))
    # Preserve the original early validation behavior, even though the selected
    # mmap is rebuilt in feature_cols order and no longer needs source indices.
    for col in feature_cols:
        all_features.index(col)

    cat_cols = cfg.features.get("cat_cols", [])
    print(f"  - Num of features: {len(feature_cols):,}")

    return feature_cols, cat_cols


def prepare_feature_memmap(master_dir: Path, n_rows: int, feature_cols: list[str]):
    features_dir = master_dir / "features"
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    print("  - Preparing shared memory map for raw features...")
    cols_hash = hashlib.md5(",".join(feature_cols).encode()).hexdigest()[:8]
    features_mmap_path = master_dir / f"features_array_{cols_hash}.npy"
    lock_path = master_dir / f"features_array_{cols_hash}.lock"

    if not features_mmap_path.exists() or lock_path.exists():
        try:
            if lock_path.exists() and (time.time() - lock_path.stat().st_mtime > 600):
                print("  - Found stale lock file. Removing it...")
                lock_path.unlink(missing_ok=True)

            lock_path.touch(exist_ok=False)
            print(f"  - Building mmap cache: {features_mmap_path.name}")
            chunk_files = sorted(features_dir.glob("features_chunk_*.parquet"))
            try:
                shape = (n_rows, len(feature_cols))
                mmap_arr = np.memmap(features_mmap_path, dtype="float32", mode="w+", shape=shape)

                current_row = 0
                for cf in chunk_files:
                    df_chunk = pd.read_parquet(cf, columns=feature_cols)
                    chunk_len = len(df_chunk)
                    mmap_arr[current_row : current_row + chunk_len] = df_chunk.values.astype("float32")
                    current_row += chunk_len

                mmap_arr.flush()
                del mmap_arr
                gc.collect()
            finally:
                lock_path.unlink(missing_ok=True)
        except FileExistsError:
            print("  - Waiting for other process to finish building mmap cache...")
            while lock_path.exists():
                time.sleep(2)

    print("  - Attaching to shared memory map...")
    features_array = np.memmap(
        features_mmap_path,
        dtype="float32",
        mode="r",
        shape=(n_rows, len(feature_cols)),
    )
    col_indices = list(range(len(feature_cols)))

    return features_array, features_mmap_path, col_indices


def fit_base_preprocessor(cfg, features_array, col_indices, feature_cols, cat_cols):
    print("🔹 Fitting preprocessor (Sampling 100k)...")
    prep_params = {
        "save_dir": ".",
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
    }
    if cfg.model.data_category == "timeseries":
        prep_params["window_size"] = cfg.hparams.get("window_size", 20)
        for key in (
            "sequence_cache_enabled",
            "sequence_cache_dir",
            "sequence_cache_wait_seconds",
        ):
            if key in cfg.hparams:
                prep_params[key] = cfg.hparams.get(key)

    preprocessor_class = get_class(cfg.model.preprocessor_target)
    base_preprocessor = preprocessor_class(**prep_params)
    sample_data = features_array[:100000, col_indices]
    base_preprocessor.fit(pd.DataFrame(sample_data, columns=feature_cols))

    model_meta_params = {}
    if hasattr(base_preprocessor, "cat_idx"):
        model_meta_params["cat_idx"] = base_preprocessor.cat_idx
    if hasattr(base_preprocessor, "cat_dims"):
        model_meta_params["cat_dims"] = base_preprocessor.cat_dims

    return base_preprocessor, model_meta_params
