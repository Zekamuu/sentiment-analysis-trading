"""One-time preprocessing: large raw tweets dump -> small windowed+sampled file.

The raw file is ~2 GB / ~4.8M dated tweets and is NOT sorted by time. Scoring all
of it with VADER is needlessly slow, so we:
  1. chunk-read only the columns we need (date, text, followers),
  2. keep rows inside the modeling window,
  3. reservoir-sample up to K tweets PER DAY (Algorithm R, seeded) — a few thousand
     posts is plenty for a stable daily influence-weighted aggregate,
  4. sort by timestamp and write a compact Parquet that build_features.py consumes.

Run:  uv run python prepare_tweets.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow import csv as pacsv

from src.config import load_config
from src.data import _TWEET_ALIASES, _pick


def main() -> None:
    cfg = load_config()
    raw = cfg["data"]["tweets_raw_path"]
    out = Path(cfg["data"]["tweets_path"])
    enc = cfg["data"].get("tweets_encoding", "utf-8")
    K = int(cfg["data"].get("tweets_sample_per_day", 3000))
    lo = pd.Timestamp(cfg["modeling_window"]["start"], tz="UTC")
    hi = pd.Timestamp(cfg["modeling_window"]["end"], tz="UTC")
    rng = np.random.default_rng(cfg.get("random_seed", 42))

    # Detect the source column names once from the header.
    header = pd.read_csv(raw, nrows=0, encoding=enc)
    cols = {
        "text": _pick(list(header.columns), _TWEET_ALIASES["text"]),
        "date": _pick(list(header.columns), _TWEET_ALIASES["date"]),
        "followers": _pick(list(header.columns), _TWEET_ALIASES["followers"]),
    }
    usecols = [c for c in cols.values() if c]
    print(f"Source columns: {cols}; window {lo.date()} -> {hi.date()}, K={K}/day")

    # Per-day reservoirs: day -> dict(ts/text/foll lists), plus a seen counter.
    res: dict[pd.Timestamp, dict] = {}
    seen: dict[pd.Timestamp, int] = {}
    scanned = 0

    # pyarrow streaming CSV: tolerant of embedded newlines and skips malformed rows
    # (pandas' C tokenizer overflows on the occasional giant/garbled text field).
    reader = pacsv.open_csv(
        raw,
        read_options=pacsv.ReadOptions(block_size=1 << 26),
        parse_options=pacsv.ParseOptions(
            newlines_in_values=True, invalid_row_handler=lambda r: "skip"
        ),
        convert_options=pacsv.ConvertOptions(
            include_columns=usecols, column_types={c: "string" for c in usecols}
        ),
    )
    ci = 0
    while True:
        try:
            batch = reader.read_next_batch()
        except StopIteration:
            break
        ci += 1
        chunk = batch.to_pandas()
        scanned += len(chunk)
        ts = pd.to_datetime(chunk[cols["date"]], errors="coerce", utc=True)
        m = ts.notna() & (ts >= lo) & (ts <= hi)
        if not m.any():
            continue
        sub = chunk.loc[m]
        ts = ts.loc[m]
        days = ts.dt.floor("D")
        text = sub[cols["text"]].astype(str).to_numpy()
        foll = (pd.to_numeric(sub[cols["followers"]], errors="coerce").to_numpy()
                if cols["followers"] else np.full(len(sub), np.nan))
        ts_np = ts.to_numpy()

        for day, grp_idx in pd.Series(np.arange(len(sub)), index=days.to_numpy()).groupby(level=0):
            rows = grp_idx.to_numpy()
            r = res.get(day)
            if r is None:
                r = res[day] = {"ts": [], "text": [], "foll": []}
                seen[day] = 0
            for j in rows:
                seen[day] += 1
                if len(r["ts"]) < K:
                    r["ts"].append(ts_np[j]); r["text"].append(text[j]); r["foll"].append(foll[j])
                else:
                    # Algorithm R: replace a random slot with prob K/seen.
                    s = rng.integers(0, seen[day])
                    if s < K:
                        r["ts"][s] = ts_np[j]; r["text"][s] = text[j]; r["foll"][s] = foll[j]
        print(f"  chunk {ci}: scanned {scanned:,}, days held {len(res)}")

    frames = [pd.DataFrame({"date": r["ts"], "text": r["text"], "followers": r["foll"]})
              for r in res.values()]
    sampled = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    sampled.to_parquet(out)

    # Sidecar: TRUE in-window tweet count per day (sampling caps the sample, so the
    # real chatter-volume feature must come from these, not the sampled rows).
    true_counts = pd.Series(seen, name="true_count").sort_index()
    true_counts.index.name = "date"
    counts_path = out.with_name("tweet_true_counts.parquet")
    true_counts.to_frame().to_parquet(counts_path)
    print(f"Wrote true daily counts -> {counts_path}")

    print(f"\nKept {len(sampled):,} sampled tweets across {len(res)} days")
    print(f"  total in-window tweets seen: {sum(seen.values()):,}")
    print(f"  per-day sampled: min {min(len(r['ts']) for r in res.values())}, "
          f"max {max(len(r['ts']) for r in res.values())}")
    print(f"Wrote -> {out}")


if __name__ == "__main__":
    main()
