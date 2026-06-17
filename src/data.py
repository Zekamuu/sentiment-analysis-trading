"""Data loading utilities.

Phase 0 deliverable: load the raw datasets into DataFrames indexed by a UTC
timestamp. No feature engineering, no alignment, no leakage handling here — that
is Phase 2 (see features.py). This module only gets bytes off disk into clean,
UTC-indexed frames so the Phase 0 checkpoint can pass.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


# Common column-name variants across tweet datasets, mapped to our logical names.
_TWEET_ALIASES = {
    "text": ["text", "tweet", "content", "body", "full_text"],
    "date": ["date", "timestamp", "created_at", "datetime", "time", "tweet_date"],
    "followers": ["followers", "follower_count", "followers_count", "user_followers",
                  "user_followers_count"],
}


def _pick(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def load_tweets(
    path: str | Path,
    encoding: str = "latin-1",
    columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load a tweets CSV into a UTC-indexed frame with standardized columns.

    Standardizes to: index = UTC tz-aware 'date', plus 'text' and (if present)
    'followers'. Column names are auto-detected across common dataset variants
    (see _TWEET_ALIASES); pass an explicit ``columns`` mapping
    ({"text": ..., "date": ..., "followers": ...}) to override detection.

    'followers' is optional — if absent, influence weighting falls back to a
    plain mean (see features.aggregate_sentiment_daily).
    """
    df = pd.read_csv(path, encoding=encoding, on_bad_lines="skip")
    columns = columns or {}

    text_col = columns.get("text") or _pick(list(df.columns), _TWEET_ALIASES["text"])
    date_col = columns.get("date") or _pick(list(df.columns), _TWEET_ALIASES["date"])
    foll_col = columns.get("followers") or _pick(list(df.columns), _TWEET_ALIASES["followers"])

    if text_col is None or date_col is None:
        raise ValueError(
            f"Could not find text/date columns in {path}. Found {list(df.columns)}. "
            f"Set data.tweets_columns in config.yaml to map them explicitly."
        )

    out = pd.DataFrame({"text": df[text_col]})
    if foll_col is not None:
        out["followers"] = pd.to_numeric(df[foll_col], errors="coerce")
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    out = out.dropna(subset=["date"]).sort_values("date").set_index("date")
    return out


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
