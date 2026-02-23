# src/cv/cpcv.py
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import numpy as np
import pandas as pd
import exchange_calendars as ecals


def _make_contiguous_blocks(n_samples: int, n_splits: int) -> list[np.ndarray]:
    """Split [0..n_samples-1] into n_splits contiguous blocks (time-ordered)."""
    if n_splits <= 1:
        raise ValueError("n_splits must be >= 2")
    idx = np.arange(n_samples, dtype=np.int64)
    blocks = np.array_split(idx, n_splits)
    return [b.astype(np.int64, copy=False) for b in blocks if len(b) > 0]


def _blocks_to_intervals(blocks: list[np.ndarray], t1: np.ndarray) -> list[tuple[int, int]]:
    """
    For each contiguous block, return (start, end) interval in integer-time axis.
    end is max(t1[i]) within that block.
    """
    intervals = []
    for b in blocks:
        start = int(b[0])
        end = int(np.max(t1[b]))
        intervals.append((start, end))
    return intervals


def _purge_by_intervals(
    train_idx: np.ndarray,
    t1: np.ndarray,
    test_intervals: list[tuple[int, int]],
) -> np.ndarray:
    """
    Remove train samples whose interval [i, t1[i]] intersects any test interval [s,e].
    Intersection condition: (i <= e) & (t1[i] >= s)
    """
    if len(train_idx) == 0:
        return train_idx

    i = train_idx
    i_end = t1[i]

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


def add_t1_column(meta_df: pd.DataFrame, horizon: int, calendar_name: str = "XTKS") -> pd.DataFrame:
    """
    Add 't1' column (event end date) to metadata based on exchange calendar.
    Filters out samples where t1 is outside the calendar range.
    """
    cal = ecals.get_calendar(calendar_name)
    # Ensure date is datetime and normalized
    meta_df = meta_df.copy()
    meta_df['date'] = pd.to_datetime(meta_df['date']).dt.normalize()
    
    start_date = meta_df['date'].min()
    end_date = meta_df['date'].max()
    
    sessions = cal.sessions_in_range(start_date, end_date)
    if sessions.tz is not None:
        sessions = sessions.tz_convert(None)
    sessions = sessions.normalize()
    
    date_to_pos = {d: i for i, d in enumerate(sessions)}
    
    pos = meta_df['date'].map(date_to_pos)
    t1_pos = pos + horizon
    
    valid_mask = t1_pos < len(sessions)
    if not valid_mask.all():
        print(f"Dropping {(~valid_mask).sum()} samples due to horizon exceeding calendar range.")
            
    meta_df = meta_df.loc[valid_mask].copy()
    # Assign t1 date
    meta_df['t1'] = sessions[t1_pos[valid_mask].astype(int).values]
    
    return meta_df


def prepare_purged_cv_input(meta_df: pd.DataFrame) -> tuple[pd.Series, dict, np.ndarray]:
    """
    Prepare inputs for SimplePurgedKFold/CPCV from metadata with 't1' column.
    Returns:
        samples_info: pd.Series mapping start_pos -> end_pos (integer indices)
        date_to_indices: dict mapping date -> original dataframe indices
        dates: np.ndarray of unique dates corresponding to the integer positions
    """
    meta_df["date_floor"] = pd.to_datetime(meta_df["date"]).dt.normalize()
    meta_df["t1_floor"] = pd.to_datetime(meta_df["t1"]).dt.normalize()
    
    t1_per_date = meta_df.groupby("date_floor")["t1_floor"].max().sort_index()
    dates = t1_per_date.index.to_numpy()
    t1_vals = t1_per_date.to_numpy()
    
    start_pos = np.arange(len(dates), dtype=np.int64)
    end_pos = np.searchsorted(dates, t1_vals, side="right") - 1
    end_pos = np.clip(end_pos, 0, len(dates) - 1)
    end_pos = np.maximum(end_pos, start_pos)
    
    samples_info = pd.Series(end_pos, index=start_pos, name="t1")
    date_to_indices = meta_df.groupby("date_floor").groups
    
    return samples_info, date_to_indices, dates


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
    pct_embargo: float = 0.0

    def __post_init__(self) -> None:
        if not (0 < self.n_test_splits < self.n_splits):
            raise ValueError("Require 0 < n_test_splits < n_splits")
        if not (0.0 <= self.pct_embargo < 1.0):
            raise ValueError("pct_embargo must be in [0,1)")
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

        blocks = _make_contiguous_blocks(n_samples, self.n_splits)
        t1 = self.samples_info_sets.to_numpy(dtype=np.int64, copy=False)

        embargo_size = int(round(self.pct_embargo * n_samples))

        all_idx = np.arange(n_samples, dtype=np.int64)

        # Enumerate combinations of test blocks
        for test_block_ids in combinations(range(len(blocks)), self.n_test_splits):
            test_blocks = [blocks[i] for i in test_block_ids]
            test_idx = np.concatenate(test_blocks).astype(np.int64, copy=False)

            # Train candidates are complement
            in_test = np.zeros(n_samples, dtype=bool)
            in_test[test_idx] = True
            train_idx = all_idx[~in_test]

            # Build test intervals per block and purge + embargo
            test_intervals = _blocks_to_intervals(test_blocks, t1)
            train_idx = _purge_by_intervals(train_idx, t1, test_intervals)
            train_idx = _apply_embargo(train_idx, test_intervals, embargo_size)

            # Safety: ensure disjoint
            if np.intersect1d(train_idx, test_idx).size != 0:
                raise RuntimeError("Train and test intersect after purge/embargo (should not happen)")

            yield train_idx, test_idx
