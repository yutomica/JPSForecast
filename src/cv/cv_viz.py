# src/cv/cv_viz.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def segments_from_pos(pos_arr: Optional[np.ndarray]) -> List[Tuple[int, int]]:
    """Return list of contiguous [start,end] segments on integer position axis."""
    if pos_arr is None:
        return []
    pos_arr = np.asarray(pos_arr, dtype=np.int64)
    if pos_arr.size == 0:
        return []
    pos_arr = np.sort(pos_arr)
    breaks = np.where(np.diff(pos_arr) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, pos_arr.size - 1]
    return [(int(pos_arr[s]), int(pos_arr[e])) for s, e in zip(starts, ends)]


def fmt_segments(
    segs: List[Tuple[int, int]],
    label: str,
    pos_to_date: pd.Series,
    max_head: int = 2,
    max_tail: int = 1,
) -> str:
    """Format segments like TR[10..20](YYYY-MM-DD..YYYY-MM-DD) ..."""
    if not segs:
        return f"{label}(empty)"

    parts: List[str] = []
    show: List[Any] = []
    show.extend(segs[:max_head])
    if len(segs) > (max_head + max_tail):
        show.append(("...", "..."))
    if max_tail > 0 and len(segs) > max_head:
        show.extend(segs[-max_tail:])

    for a, b in show:
        if a == "...":
            parts.append("...")
            continue
        da = pd.to_datetime(pos_to_date.iloc[a]).date()
        db = pd.to_datetime(pos_to_date.iloc[b]).date()
        parts.append(f"{label}[{a}..{b}]({da}..{db})")

    return " ".join(parts)


def timeline(
    tr_pos: Optional[np.ndarray],
    va_pos: Optional[np.ndarray],
    n_days: int,
    width: int = 100,
) -> str:
    """
    ASCII timeline bar:
      V = valid/test, T = train, . = neither (purged/embargo)
    """
    tr = np.zeros(n_days, dtype=np.int8)
    va = np.zeros(n_days, dtype=np.int8)

    if tr_pos is not None and len(tr_pos) > 0:
        tr[np.asarray(tr_pos, dtype=np.int64)] = 1
    if va_pos is not None and len(va_pos) > 0:
        va[np.asarray(va_pos, dtype=np.int64)] = 1

    bins = np.linspace(0, n_days, num=width + 1, dtype=int)
    chars: List[str] = []
    for j in range(width):
        a, b = int(bins[j]), int(bins[j + 1])
        if b <= a:
            b = a + 1
        if va[a:b].any():
            chars.append("V")
        elif tr[a:b].any():
            chars.append("T")
        else:
            chars.append(".")
    return "".join(chars)


def summarize_split_for_logging(
    fold: int,
    tr_pos: np.ndarray,
    va_pos: np.ndarray,
    pos_to_date: pd.Series,
    timeline_width: int = 100,
) -> Dict[str, Any]:
    """Return a dict suitable for console + MLflow artifact (json/csv)."""
    tr_pos = np.asarray(tr_pos, dtype=np.int64)
    va_pos = np.asarray(va_pos, dtype=np.int64)

    tr_segs = segments_from_pos(tr_pos)
    va_segs = segments_from_pos(va_pos)

    tr_start = pd.to_datetime(pos_to_date.iloc[int(tr_pos.min())]).date() if tr_pos.size else None
    tr_end = pd.to_datetime(pos_to_date.iloc[int(tr_pos.max())]).date() if tr_pos.size else None
    va_start = pd.to_datetime(pos_to_date.iloc[int(va_pos.min())]).date() if va_pos.size else None
    va_end = pd.to_datetime(pos_to_date.iloc[int(va_pos.max())]).date() if va_pos.size else None

    bar = timeline(tr_pos, va_pos, n_days=len(pos_to_date), width=timeline_width)

    return {
        "fold": int(fold),
        "train_days": int(tr_pos.size),
        "valid_days": int(va_pos.size),
        "train_segs": int(len(tr_segs)),
        "valid_segs": int(len(va_segs)),
        "train_start": str(tr_start),
        "train_end": str(tr_end),
        "valid_start": str(va_start),
        "valid_end": str(va_end),
        "train_segments_str": fmt_segments(tr_segs, "TR", pos_to_date),
        "valid_segments_str": fmt_segments(va_segs, "VA", pos_to_date),
        "timeline": bar,
    }
