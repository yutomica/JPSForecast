# src/cv/purged_kfold.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


def _make_contiguous_blocks(n_samples: int, n_splits: int) -> list[np.ndarray]:
    idx = np.arange(n_samples, dtype=np.int64)
    blocks = np.array_split(idx, n_splits)
    return [b.astype(np.int64, copy=False) for b in blocks if len(b) > 0]


def _block_interval(block: np.ndarray, t1: np.ndarray) -> tuple[int, int]:
    """Return (start,end) of a test block in integer-time axis; end=max(t1[i]) within block."""
    start = int(block[0])
    end = int(np.max(t1[block]))
    return start, end


def _purge_by_interval(train_idx: np.ndarray, t1: np.ndarray, test_start: int, test_end: int) -> np.ndarray:
    """
    Remove train samples whose interval [i, t1[i]] intersects test interval [test_start, test_end].
    Intersects iff (i <= test_end) & (t1[i] >= test_start)
    """
    if len(train_idx) == 0:
        return train_idx
    i = train_idx
    i_end = t1[i]
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
    pct_embargo: float = 0.0

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not (0.0 <= self.pct_embargo < 1.0):
            raise ValueError("pct_embargo must be in [0,1)")
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

        blocks = _make_contiguous_blocks(n_samples, self.n_splits)
        t1 = self.samples_info_sets.to_numpy(dtype=np.int64, copy=False)
        embargo_size = int(round(self.pct_embargo * n_samples))

        all_idx = np.arange(n_samples, dtype=np.int64)

        for k, test_block in enumerate(blocks):
            test_idx = test_block
            in_test = np.zeros(n_samples, dtype=bool)
            in_test[test_idx] = True
            train_idx = all_idx[~in_test]

            test_start, test_end = _block_interval(test_block, t1)
            train_idx = _purge_by_interval(train_idx, t1, test_start, test_end)
            train_idx = _apply_embargo(train_idx, test_end, embargo_size)

            if np.intersect1d(train_idx, test_idx).size != 0:
                raise RuntimeError("Train and test intersect after purge/embargo (should not happen)")

            yield train_idx, test_idx
