import mlflow
import numpy as np

from src.preprocess.weights import calculate_sample_weights, calculate_time_decay_weights
from src.utils.sampling import (
    apply_2d_matrix_weight,
    apply_hard_negative_weighting,
    apply_sampling,
    apply_target_stratified_sampling,
    make_sample_weight,
    make_train_fold_class_weight,
)


def apply_train_sampling_for_fold(cfg, meta_df, train_idx, target_col, interval, fold_idx):
    if cfg.get("preprocess", {}).get("sampling", {}).get("enabled", False):
        print("  🔹 Applying date-interval sampling...")
        count_before_sampling = len(train_idx)
        sampling_interval = cfg.preprocess.sampling.get("interval", interval)
        train_meta_subset = meta_df.loc[train_idx].copy()
        train_meta_processed = apply_sampling(train_meta_subset, sampling_interval)
        train_idx = train_meta_processed.index
        print(f"    - Samples reduced: {count_before_sampling:,} -> {len(train_idx):,}")

    stratified_sampling_weights = None
    if cfg.get("preprocess", {}).get("target_stratified_sampling", {}).get("enabled", False):
        sampling_cfg = cfg.preprocess.target_stratified_sampling
        mode = sampling_cfg.get("mode", "mode_1")
        print(f"  🔹 Applying target stratified sampling (mode: {mode})...")
        count_before_stratified = len(train_idx)
        train_meta_subset = meta_df.loc[train_idx].copy()
        train_meta_processed = apply_target_stratified_sampling(
            df=train_meta_subset,
            target_col=target_col,
            date_col="date",
            scode_col="scode",
            mode=mode,
            center_keep_ratio=sampling_cfg.get("center_keep_ratio", 0.25),
            other_keep_ratio=sampling_cfg.get("other_keep_ratio", 1.0),
            weight_dict=sampling_cfg.get("weight_dict", None),
            random_state=cfg.get("seed", 42) + fold_idx,
        )
        if mode in ["mode_1", "mode_2"]:
            train_idx = train_meta_processed.index
            print(f"    - Samples reduced: {count_before_stratified:,} -> {len(train_idx):,}")
        elif mode in ["mode_3", "mode_ap_severe"]:
            stratified_sampling_weights = train_meta_processed.loc[train_idx, "sample_weight"].values
            print(f"    - Weighting mode enabled. Sample count remains {len(train_idx):,}.")

    return train_idx, stratified_sampling_weights


def calculate_fold_weights(
    cfg,
    meta_df,
    idx,
    target_col,
    fold_idx,
    stratified_sampling_weights=None,
    is_train=True,
):
    w = np.ones(len(idx))
    w *= calculate_sample_weights(meta_df.loc[idx, "log_market_cap"].values, cfg.domain.name)

    if cfg.hparams.use_time_decay:
        decay_rate = cfg.hparams.get("time_decay_rate", 0.9999)
        w *= calculate_time_decay_weights(meta_df.loc[idx, "date"], decay_rate=decay_rate)

    if is_train and stratified_sampling_weights is not None:
        w *= stratified_sampling_weights

    if cfg.get("preprocess", {}).get("matrix_weight", {}).get("enabled", False):
        matrix_cfg = cfg.preprocess.matrix_weight
        cost_buffer = matrix_cfg.get("cost_buffer", 0.003)
        meta_subset = meta_df.loc[idx].copy()
        w *= apply_2d_matrix_weight(meta_subset, return_col="Future_Close", cost_buffer=cost_buffer)

    if cfg.get("preprocess", {}).get("hard_negative_weighting", {}).get("enabled", False):
        meta_subset = meta_df.loc[idx].copy()
        w *= apply_hard_negative_weighting(meta_subset)

    if cfg.get("preprocess", {}).get("class_weight", {}).get("enabled", False):
        cw_cfg = cfg.preprocess.class_weight
        num_classes = cw_cfg.get("num_classes", 4)
        clip_min = cw_cfg.get("clip_min", 1.0)
        clip_max = cw_cfg.get("clip_max", 10.0)
        y_series = meta_df.loc[idx, target_col]
        class_weight_dict, class_counts = make_train_fold_class_weight(
            y_series,
            num_classes=num_classes,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        w *= make_sample_weight(y_series, class_weight_dict)
        if is_train:
            total_n = class_counts.sum()
            for cls_idx in range(num_classes):
                mlflow.log_metric(f"fold{fold_idx}_class_count_{cls_idx}", float(class_counts[cls_idx]))
                mlflow.log_metric(f"fold{fold_idx}_class_weight_{cls_idx}", float(class_weight_dict[cls_idx]))
            if num_classes >= 4:
                pos_rate_5 = (class_counts[1] + class_counts[2] + class_counts[3]) / total_n
                pos_rate_7 = (class_counts[2] + class_counts[3]) / total_n
                pos_rate_10 = class_counts[3] / total_n
                mlflow.log_metric(f"fold{fold_idx}_positive_rate_5", float(pos_rate_5))
                mlflow.log_metric(f"fold{fold_idx}_positive_rate_7", float(pos_rate_7))
                mlflow.log_metric(f"fold{fold_idx}_positive_rate_10", float(pos_rate_10))
            print("    - Class weight mode enabled.")

    return w
