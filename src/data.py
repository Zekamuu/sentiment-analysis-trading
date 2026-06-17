"""Data loading utilities.

Phase 0 deliverable: load the raw datasets into DataFrames indexed by a UTC
timestamp. No feature engineering, no alignment, no leakage handling here — that
is Phase 2 (see features.py). This module only gets bytes off disk into clean,
UTC-indexed frames so the Phase 0 checkpoint can pass.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_tweets(path: str | Path, encoding: str = "latin-1") -> pd.DataFrame:
    """Load the raw tweets CSV.

    Columns expected: tweet_id, text, date, retweets, replies, likes,
    location, followers, following. The file is not clean UTF-8, hence the
    latin-1 default. Returns a frame indexed by a UTC tz-aware 'date'.
    """
    df = pd.read_csv(path, encoding=encoding, on_bad_lines="skip")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
    return df


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load daily BTC OHLCV.

    Expected columns (case-insensitive, flexible): a date/timestamp column plus
    open, high, low, close, volume. Returns a frame indexed by a UTC tz-aware
    DatetimeIndex with lowercase OHLCV columns. See DATA_REQUIREMENTS.md for the
    file you must supply.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Find the timestamp column under any of the common names.
    date_col = next(
        (c for c in ("date", "timestamp", "time", "datetime", "day") if c in df.columns),
        None,
    )
    if date_col is None:
        raise ValueError(
            f"No date/timestamp column found in {path}. Columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df = (
        df.dropna(subset=[date_col])
        .sort_values(date_col)
        .set_index(date_col)
        .rename_axis("date")
    )

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"OHLCV file missing columns {sorted(missing)}. Found: {list(df.columns)}"
        )
    return df[["open", "high", "low", "close", "volume"]]
