from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd



def _resolve_size(value: Optional[float | int], n_samples: int, name: str) -> Optional[int]:
    """
    Resolve a size parameter.

    Rules:
      - None -> None
      - int >= 1 -> 그대로使用
      - float in (0, 1) -> 全サンプル数に対する比率として ceil で整数化
    """
    if value is None:
        return None

    if isinstance(value, (np.integer, int)):
        out = int(value)
    elif isinstance(value, (np.floating, float)):
        v = float(value)
        if 0.0 < v < 1.0:
            out = int(np.ceil(n_samples * v))
        elif float(v).is_integer() and v >= 1.0:
            out = int(v)
        else:
            raise ValueError(
                f"{name} must be an int >= 1 or a float in (0, 1). Got: {value}"
            )
    else:
        raise TypeError(f"{name} must be int | float | None. Got: {type(value)}")

    if out < 1:
        raise ValueError(f"{name} must resolve to >= 1. Got: {out}")
    return out



def _purge_by_interval(
    train_idx: np.ndarray,
    valid_start: int,
    valid_end: int,
    purge_days: int = 0
) -> np.ndarray:
    """
    Remove train samples whose interval [i, i + purge_days] intersects
    validation interval [valid_start, valid_end].

    Intersects iff:
      (i <= valid_end) & (i + purge_days >= valid_start)
    """
    if train_idx.size == 0:
        return train_idx

    i = train_idx
    i_end = i + purge_days
    keep = ~((i <= valid_end) & (i_end >= valid_start))
    return i[keep]



def _apply_pre_valid_gap(train_idx: np.ndarray, valid_start: int, gap_size: int) -> np.ndarray:
    """
    Remove the last `gap_size` positions immediately before validation starts.

    This is not standard post-test embargo. In anchored walk-forward, train uses only
    historical samples, so classic post-validation embargo has no effect. Instead,
    this gap acts as an additional safety margin before validation.
    """
    if train_idx.size == 0 or gap_size <= 0:
        return train_idx

    cutoff = valid_start - gap_size
    return train_idx[train_idx < cutoff]


@dataclass
class AnchoredWalkForwardPurgedCV:
    """
    Anchored / expanding walk-forward validation on integer time axis.

    Compatible with the current train.py interface:
        cv = instantiate(cfg.cv, samples_info_sets=samples_info)
        for tr_pos, val_pos in cv.split(X=cv_input, groups=unique_dates):
            ...

    Parameters
    ----------
    n_splits : int
        Number of folds to generate.
    samples_info_sets : pd.Series
        Mapping start_pos -> end_pos (t1) on integer date positions.
    min_train_size : int | float
        Minimum anchored train size before first validation fold.
        - int  : number of date positions
        - float in (0,1): ratio of total samples
    val_size : int | float
        Validation window size per fold.
        - int  : number of date positions
        - float in (0,1): ratio of total samples
    step_size : int | float | None, default None
        Forward step per fold.
        If None, step_size = val_size.
    max_train_size : int | float | None, default None
        Optional cap on train size. If None, uses true anchored expanding window.
        If set, the latest max_train_size observations are used from the anchored
        history after purge/gap constraints.
    allow_incomplete_last_fold : bool, default False
        If True, the final fold may use a shorter validation window when the
        remaining tail is smaller than val_size.
    """

    n_splits: int
    samples_info_sets: pd.Series
    min_train_size: int | float
    val_size: int | float
    step_size: int | float | None = None
    max_train_size: int | float | None = None
    purge_days: int = 0
    embargo_days: int = 0
    allow_incomplete_last_fold: bool = False
    train_start_date: str | None = None
    test_start_date: str | None = None

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if not np.issubdtype(self.samples_info_sets.index.dtype, np.integer):
            raise TypeError("samples_info_sets.index must be integer positions")
        if not np.issubdtype(self.samples_info_sets.dtype, np.integer):
            raise TypeError("samples_info_sets values (t1) must be integer positions")

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        if n_samples != len(self.samples_info_sets):
            raise ValueError(
                f"X has {n_samples} samples but samples_info_sets has {len(self.samples_info_sets)}"
            )
        if groups is not None and len(groups) != n_samples:
            raise ValueError(
                f"groups has {len(groups)} elements but X has {n_samples} samples"
            )

        all_idx = np.arange(n_samples, dtype=np.int64)
        valid_indices = all_idx

        if self.train_start_date is not None and self.test_start_date is not None:
            if groups is None:
                raise ValueError("groups (dates) must be provided to filter by train_start_date and test_start_date.")
            dates_s = pd.to_datetime(groups)
            mask = (dates_s >= pd.to_datetime(self.train_start_date)) & (dates_s < pd.to_datetime(self.test_start_date))
            valid_indices = all_idx[mask]

        n_valid = len(valid_indices)
        if n_valid == 0:
            raise ValueError("No valid samples remain after applying date filters.")

        min_train_size = _resolve_size(self.min_train_size, n_valid, "min_train_size")
        val_size = _resolve_size(self.val_size, n_valid, "val_size")
        step_size = _resolve_size(self.step_size, n_valid, "step_size") if self.step_size is not None else val_size
        max_train_size = _resolve_size(self.max_train_size, n_valid, "max_train_size")

        if min_train_size is None or val_size is None or step_size is None:
            raise ValueError("min_train_size, val_size, and step_size must resolve to integers")

        if min_train_size >= n_valid:
            raise ValueError(
                f"min_train_size ({min_train_size}) must be smaller than valid samples ({n_valid})"
            )

        gap_size = self.purge_days + self.embargo_days

        yielded = 0
        for fold in range(self.n_splits):
            valid_start_pos = min_train_size + fold * step_size
            if valid_start_pos >= n_valid:
                break

            valid_stop_pos = valid_start_pos + val_size
            if valid_stop_pos > n_valid:
                if not self.allow_incomplete_last_fold:
                    break
                valid_stop_pos = n_valid

            if valid_stop_pos <= valid_start_pos:
                break

            val_idx = valid_indices[valid_start_pos:valid_stop_pos]

            # Anchored expanding history: [0 .. valid_start-1]
            train_idx = valid_indices[:valid_start_pos]
            valid_start_abs = val_idx[0]

            # Optional pre-validation gap (safety margin)
            train_idx = _apply_pre_valid_gap(train_idx, valid_start_abs, gap_size)

            # Purge overlap against the realized validation interval
            valid_end = int(val_idx[-1])
            train_idx = _purge_by_interval(train_idx, valid_start=valid_start_abs, valid_end=valid_end, purge_days=self.purge_days)

            # Optional cap on train length. Keep the most recent train observations.
            if max_train_size is not None and train_idx.size > max_train_size:
                train_idx = train_idx[-max_train_size:]

            if train_idx.size == 0:
                raise ValueError(
                    f"Fold {fold} produced empty train set after purge/gap. "
                    f"Try increasing min_train_size or reducing pct_embargo / val_size."
                )

            if np.intersect1d(train_idx, val_idx).size != 0:
                raise RuntimeError("Train and validation intersect after purge/gap (should not happen)")

            yielded += 1
            yield train_idx, val_idx

        if yielded == 0:
            raise ValueError(
                "No valid folds were generated. Check n_splits / min_train_size / val_size / step_size."
            )
