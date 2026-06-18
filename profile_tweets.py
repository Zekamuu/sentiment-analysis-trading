"""Profile the large tweets dataset: range, per-day coverage, and gaps.

Chunk-reads only the date column so a 2 GB file fits in memory. Reports the
overlap with our BTC OHLCV and any missing days in that window (the gaps we'd
need to forward-fill over).

Run:  uv run python profile_tweets.py
"""
from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.data import load_ohlcv

PATH = "data/raw/bitcoin_tweets_latest.csv"


def main() -> None:
    counts = None
    total = 0
    reader = pd.read_csv(PATH, usecols=["date"], chunksize=2_000_000,
                         on_bad_lines="skip", encoding="utf-8", engine="c")
    for i, chunk in enumerate(reader):
        d = pd.to_datetime(chunk["date"], errors="coerce", utc=True).dropna()
        total += len(d)
        day_counts = d.dt.floor("D").value_counts()
        counts = day_counts if counts is None else counts.add(day_counts, fill_value=0)
        print(f"  chunk {i}: {len(chunk):,} rows scanned, running total {total:,} dated")

    counts = counts.sort_index().astype(int)
    counts.rename("n_tweets").to_csv("results/tweet_day_counts.csv")
    tmin, tmax = counts.index.min(), counts.index.max()
    print(f"\nParsed {total:,} dated tweets across {len(counts):,} distinct days")
    print(f"Full tweet range: {tmin.date()} -> {tmax.date()} (saved per-day counts)")

    # The dense block = everything after the single largest empty gap, which
    # separates stray historical tweets from the modern high-volume crawl.
    full_all = pd.date_range(tmin, tmax, freq="D", tz="UTC")
    cov_all = counts.reindex(full_all, fill_value=0)
    empty_all = cov_all[cov_all == 0].index
    if len(empty_all):
        gap_lengths = empty_all.to_series().groupby(
            (empty_all.to_series().diff().dt.days != 1).cumsum()
        ).agg(["first", "last", "count"])
        biggest = gap_lengths.loc[gap_lengths["count"].idxmax()]
        dense_start = biggest["last"] + pd.Timedelta(days=1)
    else:
        dense_start = tmin

    # Overlap with OHLCV.
    cfg = load_config()
    ohlcv = load_ohlcv(cfg["data"]["ohlcv_path"])
    lo = max(dense_start, ohlcv.index.min())
    hi = min(tmax, ohlcv.index.max())
    print(f"OHLCV range: {ohlcv.index.min().date()} -> {ohlcv.index.max().date()}")
    print(f"Dense block starts {dense_start.date()} (after the largest gap)")
    print(f"Usable overlap window: {lo.date()} -> {hi.date()}")

    # Gaps: every calendar day in the overlap window with zero tweets.
    full = pd.date_range(lo, hi, freq="D", tz="UTC")
    covered = counts.reindex(full, fill_value=0)
    gaps = covered[covered == 0]
    print(f"\nDays in window: {len(full)} | with tweets: {(covered > 0).sum()} "
          f"({(covered > 0).mean():.1%}) | EMPTY (gaps): {len(gaps)}")
    print(f"Per-day tweet count in window: median {int(covered[covered>0].median())}, "
          f"min {int(covered[covered>0].min())}, max {int(covered.max())}")

    if len(gaps):
        # Collapse consecutive empty days into ranges for readability.
        g = gaps.index
        runs, start, prev = [], g[0], g[0]
        for day in g[1:]:
            if (day - prev).days == 1:
                prev = day
            else:
                runs.append((start, prev)); start = prev = day
        runs.append((start, prev))
        print(f"\nGap ranges ({len(runs)} runs):")
        for a, b in runs[:30]:
            n = (b - a).days + 1
            print(f"  {a.date()} -> {b.date()}  ({n} day{'s' if n > 1 else ''})")
        if len(runs) > 30:
            print(f"  ... and {len(runs) - 30} more runs")
    else:
        print("\nNo gaps — every day in the window has at least one tweet.")


if __name__ == "__main__":
    main()
