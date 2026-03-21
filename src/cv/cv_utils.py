
import numpy as np
import pandas as pd
import exchange_calendars as ecals


def add_t1_column(meta_df: pd.DataFrame, horizon: int, calendar_name: str = "XTKS") -> pd.DataFrame:
    """
    Add 't1' column (event end date) to metadata based on exchange calendar.
    Filters out samples where t1 is outside the calendar range.
    """
    cal = ecals.get_calendar(calendar_name)
    # Ensure date is datetime and normalized
    meta_df = meta_df.copy()
    meta_df['date'] = pd.to_datetime(meta_df['date']).dt.normalize()
    
    if meta_df.empty:
        return meta_df

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
        print(f" [add_t1_column] Dropping {(~valid_mask).sum()} samples due to horizon exceeding calendar range.")
            
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