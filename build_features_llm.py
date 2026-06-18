"""Phase 7 — build a modeling table whose sentiment is the VADER+LLM fusion.

Reuses the price features / target / lagging / leakage safeguards of Phase 2, but
replaces the per-post sentiment with the confidence-weighted combination of VADER
and CryptoBERT. Output: data/processed/modeling_table_llm.parquet (same shape as
the Phase 5 table; only the sentiment columns differ).

Run (after prepare_llm.py):  uv run python build_features_llm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data import load_ohlcv
from src.features import (
    FEATURE_COLUMNS,
    TARGET,
    build_modeling_table,
    combine_vader_llm,
    score_tweets_vader,
)

OK, FAIL = "[ OK ]", "[FAIL]"


def main() -> int:
    cfg = load_config()
    tweets_path = Path(cfg["data"]["tweets_path"])
    llm_path = tweets_path.with_name("tweets_llm_scores.parquet")
    if not llm_path.exists():
        print(f"{FAIL} {llm_path} not found — run prepare_llm.py first.")
        return 1

    # Read in the SAME row order LLM scores were written in (RangeIndex == key).
    tw = pd.read_parquet(tweets_path)
    llm = pd.read_parquet(llm_path)
    assert len(tw) == len(llm), "tweet/LLM-score row counts differ"

    vader = score_tweets_vader(tw)
    combined = combine_vader_llm(vader, llm["llm_signed"], llm["llm_conf"])
    print(f"Fusion: VADER mean {vader.mean():+.3f}, LLM mean {llm['llm_signed'].mean():+.3f} "
          f"(conf {llm['llm_conf'].mean():.2f}) -> combined mean {combined.mean():+.3f}")

    tw = tw.assign(compound=combined.to_numpy())
    tw["date"] = pd.to_datetime(tw["date"], utc=True)
    tw = tw.set_index("date")

    ohlcv = load_ohlcv(cfg["data"]["ohlcv_path"])
    window = (cfg["modeling_window"]["start"], cfg["modeling_window"]["end"])
    counts_path = tweets_path.with_name("tweet_true_counts.parquet")
    true_counts = pd.read_parquet(counts_path)["true_count"] if counts_path.exists() else None

    table = build_modeling_table(tw, ohlcv, window, empty_bar=cfg.get("empty_bar", "zero"),
                                 true_counts=true_counts)
    out = tweets_path.with_name("modeling_table_llm.parquet")
    table.to_parquet(out)

    n_nan = int(table[FEATURE_COLUMNS + [TARGET]].isna().sum().sum())
    print(f"{OK if n_nan == 0 else FAIL} combined-sentiment table: {table.shape[0]} rows, "
          f"{n_nan} NaNs -> {out}")
    return 0 if n_nan == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
