"""Phase 2 — build the modeling table and verify its checkpoint.

Checkpoint (pathway Phase 2): one DataFrame where each row has lagged features
and a next-period-return target, NO NaNs in the modeling columns, and a spot-check
that row t's features predate the return they predict.

Run:  uv run python build_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.config import load_config
from src.data import load_ohlcv, load_tweets
from src.features import (
    FEATURE_COLUMNS,
    PRICE_FEATURES,
    SENTIMENT_FEATURES,
    TARGET,
    assert_no_lookahead,
    build_modeling_table,
)

OK, FAIL = "[ OK ]", "[FAIL]"
WINDOW = ("2022-06-08", "2023-05-28")  # tweet coverage


def main() -> int:
    cfg = load_config()
    failures = 0

    print("Loading tweets + OHLCV and scoring sentiment (VADER over ~107k posts)...")
    tweets = load_tweets(
        cfg["data"]["tweets_path"],
        cfg["data"].get("tweets_encoding", "latin-1"),
        cfg["data"].get("tweets_columns"),
    )
    ohlcv = load_ohlcv(cfg["data"]["ohlcv_path"])

    table = build_modeling_table(tweets, ohlcv, WINDOW)
    print(f"Modeling table: {table.shape[0]} rows x {table.shape[1]} cols, "
          f"{table.index.min().date()} -> {table.index.max().date()}")
    print(f"  price features    : {PRICE_FEATURES}")
    print(f"  sentiment features: {SENTIMENT_FEATURES}")
    print(f"  target            : {TARGET}")

    # 1. No NaNs in any modeling column.
    n_nan = int(table[FEATURE_COLUMNS + [TARGET]].isna().sum().sum())
    if n_nan == 0:
        print(f"{OK} no NaNs in modeling columns")
    else:
        print(f"{FAIL} {n_nan} NaNs in modeling columns"); failures += 1

    # 2. Target is a RETURN, not a price level (Design §1 rule 1).
    tgt = table[TARGET]
    looks_like_return = tgt.abs().max() < 1.0 and abs(tgt.mean()) < 0.05
    not_price = tgt.max() < 100  # BTC price is ~10k-60k; a return is ~O(0.01)
    if looks_like_return and not_price:
        print(f"{OK} target is a return: mean={tgt.mean():+.5f}, std={tgt.std():.5f}, "
              f"range=[{tgt.min():+.4f}, {tgt.max():+.4f}]")
    else:
        print(f"{FAIL} target does not look like a return: "
              f"mean={tgt.mean()}, max={tgt.max()}"); failures += 1

    # 3. No look-ahead: tweets feeding row t predate the t->t+1 return window.
    try:
        assert_no_lookahead(table, tweets, n_check=8)
        print(f"{OK} look-ahead spot-check passed (sampled rows: tweets predate target window)")
    except AssertionError as e:
        print(f"{FAIL} {e}"); failures += 1

    # 4. Sanity: feature/target correlation is small (no accidental echo of the target).
    leak_corr = table[FEATURE_COLUMNS].corrwith(table[TARGET]).abs().max()
    if leak_corr < 0.5:
        print(f"{OK} no feature trivially predicts the target (max |corr| = {leak_corr:.3f})")
    else:
        print(f"{FAIL} a feature has suspiciously high target corr ({leak_corr:.3f}) — "
              f"possible leak"); failures += 1

    # Save the processed table.
    out = Path(cfg["data"]["processed_path"])
    table.to_parquet(out)
    print(f"\nSaved modeling table -> {out}")
    print(table.head(3).to_string())

    print()
    if failures:
        print(f"{FAIL} Phase 2 NOT complete: {failures} check(s) failed.")
        return 1
    print(f"{OK} Phase 2 checkpoint passed — modeling table ready for the validation gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
