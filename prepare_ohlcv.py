"""One-time preprocessing: minute OHLCV (epoch seconds) -> daily OHLCV.

Input  : data/raw/btc_ohlcv_minute.csv  (Timestamp,Open,High,Low,Close,Volume;
         Timestamp is Unix epoch SECONDS at 1-minute resolution)
Output : data/raw/btc_ohlcv_daily.csv   (date,open,high,low,close,volume; one UTC day/row)

Resampling rule (standard OHLCV downsample):
    open = first, high = max, low = min, close = last, volume = sum

Run:  uv run python prepare_ohlcv.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ROOT, load_config

SRC = ROOT / "data" / "raw" / "btc_ohlcv_minute.csv"


def main() -> None:
    cfg = load_config()
    out = Path(cfg["data"]["ohlcv_path"])

    print(f"Reading {SRC} (this is a large minute file, give it a moment)...")
    df = pd.read_csv(
        SRC,
        dtype={
            "Timestamp": "int64",
            "Open": "float64", "High": "float64",
            "Low": "float64", "Close": "float64", "Volume": "float64",
        },
    )
    df.columns = [c.strip().lower() for c in df.columns]

    # Epoch seconds -> UTC tz-aware datetime index.
    df.index = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df.index.name = "date"
    print(f"  {len(df):,} minute bars, {df.index.min()} -> {df.index.max()}")

    # Downsample to daily. Drop any all-empty days the resample may introduce.
    daily = df.resample("1D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open", "high", "low", "close"])

    daily.to_csv(out)  # writes the 'date' index as the first column
    print(f"Wrote {len(daily):,} daily bars -> {out}")
    print(f"  {daily.index.min().date()} -> {daily.index.max().date()}")


if __name__ == "__main__":
    main()
