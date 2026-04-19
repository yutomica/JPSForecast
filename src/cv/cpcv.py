# src/cv/cpcv.py
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np
import pandas as pd


def _make_contiguous_blocks(indices: np.ndarray, n_splits: int) -> list[np.ndarray]:
    """Split indices into n_splits contiguous blocks (time-ordered)."""
    if n_splits <= 1:
        raise ValueError("n_splits must be >= 2")
    blocks = np.array_split(indices, n_splits)
    return [b.astype(np.int64, copy=False) for b in blocks if len(b) > 0]


def _blocks_to_intervals(blocks: list[np.ndarray]) -> list[tuple[int, int]]:
    """
    For each contiguous block, return (start, end) interval in integer-time axis.
    """
    intervals = []
    for b in blocks:
        start = int(b[0])
        end = int(b[-1])
        intervals.append((start, end))
    return intervals


def _purge_by_intervals(
    train_idx: np.ndarray,
    test_intervals: list[tuple[int, int]],
    purge_days: int = 0
) -> np.ndarray:
    """
    Remove train samples whose interval [i, i + purge_days] intersects any test interval [s,e].
    Intersection condition: (i <= e) & (i + purge_days >= s)
    """
    if len(train_idx) == 0:
        return train_idx

    i = train_idx
    i_end = i + purge_days

    keep = np.ones(len(i), dtype=bool)
    for s, e in test_intervals:
        keep &= ~((i <= e) & (i_end >= s))

    return i[keep]


def _apply_embargo(
    train_idx: np.ndarray,
    test_intervals: list[tuple[int, int]],
    embargo_size: int,
) -> np.ndarray:
    """
    Remove train samples whose start i falls in (end, end+embargo_size] for any test interval end.
    """
    if len(train_idx) == 0 or embargo_size <= 0:
        return train_idx

    i = train_idx
    keep = np.ones(len(i), dtype=bool)
    for _, e in test_intervals:
        lo = e
        hi = e + embargo_size
        keep &= ~((i > lo) & (i <= hi))
    return i[keep]


@dataclass
class SimpleCombinatorialPurgedKFold:
    """
    Self-contained CPCV compatible with: for tr_pos, val_pos in cv.split(X):
      - returns (train_positions, test_positions)
    Assumes integer time axis:
      - samples_info_sets: pd.Series with index=start_pos (0..N-1), value=end_pos (int)
    """
    n_splits: int
    n_test_splits: int
    samples_info_sets: pd.Series
    purge_days: int = 0
    embargo_days: int = 0
    train_start_date: str | None = None
    test_start_date: str | None = None

    def __post_init__(self) -> None:
        if not (0 < self.n_test_splits < self.n_splits):
            raise ValueError("Require 0 < n_test_splits < n_splits")
        # Expect integer axis
        if not np.issubdtype(self.samples_info_sets.index.dtype, np.integer):
            raise TypeError("samples_info_sets.index must be integer positions")
        if not np.issubdtype(self.samples_info_sets.dtype, np.integer):
            raise TypeError("samples_info_sets values (t1) must be integer positions")

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        from math import comb
        return comb(self.n_splits, self.n_test_splits)

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        # Align to samples_info_sets length (should match)
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

        # Enumerate combinations of test blocks
        for test_block_ids in combinations(range(len(blocks)), self.n_test_splits):
            test_blocks = [blocks[i] for i in test_block_ids]
            test_idx = np.concatenate(test_blocks).astype(np.int64, copy=False)

            # Train candidates are complement
            in_test = np.zeros(n_samples, dtype=bool)
            in_test[test_idx] = True
            train_idx = valid_indices[~in_test[valid_indices]]

            # Build test intervals per block and purge + embargo
            test_intervals = _blocks_to_intervals(test_blocks)
            train_idx = _purge_by_intervals(train_idx, test_intervals, self.purge_days)
            embargo_size = self.purge_days + self.embargo_days
            train_idx = _apply_embargo(train_idx, test_intervals, embargo_size)

            # Safety: ensure disjoint
            if np.intersect1d(train_idx, test_idx).size != 0:
                raise RuntimeError("Train and test intersect after purge/embargo (should not happen)")

            yield train_idx, test_idx
