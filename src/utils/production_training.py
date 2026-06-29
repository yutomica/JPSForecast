import copy

import numpy as np

from src.cv.cv_viz import log_split_info
from src.models.pipeline import FoldPipeline
from src.utils.stacking_utils import combine_features_with_oof
from src.utils.training_weights import calculate_fold_weights


def _build_production_params(cfg, full_params, best_iter):
    prod_params = copy.deepcopy(full_params)
    prod_params["ensemble_size"] = 1

    if best_iter is not None:
        if cfg.model.name.lower() in ["lgbm", "lightgbm"]:
            prod_params["num_boost_round"] = int(best_iter)
            prod_params["early_stopping_rounds"] = 0
        else:
            prod_params["max_epochs"] = int(best_iter)
            prod_params["patience"] = int(best_iter) + 1

    return prod_params


def train_production_fold(
    cfg,
    fold_idx,
    model_class,
    preprocessor,
    full_params,
    best_iter,
    train_idx,
    valid_idx,
    train_val_meta,
    meta_df,
    target_col,
    features_array,
    col_indices,
    oof_cols,
    unique_dates,
    pos_to_date,
    stratified_sampling_weights,
    fold_pipelines,
):
    ensemble_size = cfg.model.get("ensemble_size", 1)
    print(
        f"\n  🌟 [Production] Step 2: Training on Train+Valid data "
        f"(Ensemble Size: {ensemble_size}) using best_iter={best_iter}..."
    )

    full_train_idx = train_idx.append(valid_idx)
    full_train_dates = train_val_meta.loc[full_train_idx, "date"].unique()
    tr_pos_prod = np.where(np.isin(unique_dates, full_train_dates))[0]
    log_split_info(fold_idx, tr_pos_prod, np.array([]), pos_to_date, label="PROD")

    w_full = calculate_fold_weights(
        cfg,
        meta_df,
        full_train_idx,
        target_col,
        fold_idx,
        stratified_sampling_weights=stratified_sampling_weights,
        is_train=True,
    )

    X_full = preprocessor.transform(features_array, row_indices=full_train_idx, col_indices=col_indices)
    X_full = combine_features_with_oof(X_full, meta_df, full_train_idx, oof_cols)
    y_full = meta_df.loc[full_train_idx, target_col].values

    prod_params = _build_production_params(cfg, full_params, best_iter)
    base_seed = cfg.get("seed", 42)
    last_model = None

    for s in range(ensemble_size):
        if ensemble_size > 1:
            print(f"    - Training ensemble model {s+1}/{ensemble_size} with seed {base_seed + s}...")
        curr_params = copy.deepcopy(prod_params)
        curr_params["seed"] = base_seed + s
        curr_params["random_state"] = base_seed + s

        model_prod = model_class(task_type=cfg.target.task_type, **curr_params)
        model_prod.fit(
            X_full,
            y_full,
            X_valid=None,
            y_valid=None,
            sample_weight=w_full,
            model_idx=f"{fold_idx}_s{s}",
        )

        if s < ensemble_size - 1:
            fold_pipelines.append(FoldPipeline(preprocessor, model_prod))
        else:
            last_model = model_prod

    return last_model
