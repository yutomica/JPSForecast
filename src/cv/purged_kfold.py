# src/cv/purged_kfold.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


def _make_contiguous_blocks(indices: np.ndarray, n_splits: int) -> list[np.ndarray]:
    blocks = np.array_split(indices, n_splits)
    return [b.astype(np.int64, copy=False) for b in blocks if len(b) > 0]


def _block_interval(block: np.ndarray) -> tuple[int, int]:
    """Return (start,end) of a test block in integer-time axis; end=last element within block."""
    start = int(block[0])
    end = int(block[-1])
    return start, end


def _purge_by_interval(train_idx: np.ndarray, test_start: int, test_end: int, purge_days: int = 0) -> np.ndarray:
    """
    Remove train samples whose interval [i, i + purge_days] intersects test interval [test_start, test_end].
    Intersects iff (i <= test_end) & (i + purge_days >= test_start)
    """
    if len(train_idx) == 0:
        return train_idx
    i = train_idx
    i_end = i + purge_days
    keep = ~((i <= test_end) & (i_end >= test_start))
    return i[keep]


def _apply_embargo(train_idx: np.ndarray, test_end: int, embargo_size: int) -> np.ndarray:
    """Remove train samples whose start i falls in (test_end, test_end+embargo_size]."""
    if len(train_idx) == 0 or embargo_size <= 0:
        return train_idx
    i = train_idx
    keep = ~((i > test_end) & (i <= test_end + embargo_size))
    return i[keep]




@dataclass
class SimplePurgedKFold:
    """
    Self-contained PurgedKFold compatible with: for tr_pos, val_pos in cv.split(X):
      - returns (train_positions, test_positions)
    Assumes integer time axis:
      - samples_info_sets: pd.Series with index=start_pos (0..N-1), value=end_pos (int)
    """
    n_splits: int
    samples_info_sets: pd.Series
    purge_days: int = 0
    embargo_days: int = 0
    train_start_date: str | None = None
    test_start_date: str | None = None

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not np.issubdtype(self.samples_info_sets.index.dtype, np.integer):
            raise TypeError("samples_info_sets.index must be integer positions")
        if not np.issubdtype(self.samples_info_sets.dtype, np.integer):
            raise TypeError("samples_info_sets values (t1) must be integer positions")

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        if n_samples != len(self.samples_info_sets):
            raise ValueError(f"X has {n_samples} samples but samples_info_sets has {len(self.samples_info_sets)}")

        all_idx = np.arange(n_samples, dtype=np.int64)
        valid_indices = all_idx

        if self.train_start_date is not None and self.test_start_date is not None:
            if groups is None:
                raise ValueError("groups (dates) must be provided to filter by train_start_date and test_start_date.")
            dates_s = pd.to_datetime(groups)
            mask = (dates_s >= pd.to_datetime(self.train_start_date)) & (dates_s < pd.to_datetime(self.test_start_date))
            valid_indices = all_idx[mask]

        if len(valid_indices) == 0:
            raise ValueError("No valid samples remain after applying date filters.")

        blocks = _make_contiguous_blocks(valid_indices, self.n_splits)

        for k, test_block in enumerate(blocks):
            test_idx = test_block
            in_test = np.zeros(n_samples, dtype=bool)
            in_test[test_idx] = True
            train_idx = valid_indices[~in_test[valid_indices]]

            test_start, test_end = _block_interval(test_block)
            train_idx = _purge_by_interval(train_idx, test_start, test_end, self.purge_days)
            embargo_size = self.purge_days + self.embargo_days
            train_idx = _apply_embargo(train_idx, test_end, embargo_size)

            if np.intersect1d(train_idx, test_idx).size != 0:
                raise RuntimeError("Train and test intersect after purge/embargo (should not happen)")

            yield train_idx, test_idx
