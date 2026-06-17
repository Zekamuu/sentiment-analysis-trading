"""Phase 0 checkpoint verifier.

Passes when: repo runs, dependencies import, config holds asset/frequency/N,
and the dataset loads into a DataFrame indexed by a UTC timestamp.

Run:  uv run python check_phase0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

OK, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def main() -> int:
    failures = 0

    # 1. Dependencies import.
    try:
        import numpy, pandas, statsmodels, pmdarima, matplotlib, yaml  # noqa: F401
        import vaderSentiment  # noqa: F401

        print(f"{OK} dependencies import (numpy {numpy.__version__}, "
              f"pandas {pandas.__version__}, statsmodels {statsmodels.__version__})")
    except Exception as e:  # pragma: no cover
        print(f"{FAIL} dependency import failed: {e}")
        return 1  # nothing else can run

    from src.config import load_config
    from src.data import load_ohlcv, load_tweets

    # 2. Config holds the three locked decisions.
    cfg = load_config()
    for key in ("asset", "bar_frequency", "holding_period_N"):
        if key in cfg and cfg[key] is not None:
            print(f"{OK} config.{key} = {cfg[key]!r}")
        else:
            print(f"{FAIL} config missing {key}")
            failures += 1

    # 3. Tweets load into a UTC-indexed DataFrame.
    tpath = cfg["data"]["tweets_path"]
    if Path(tpath).exists():
        tw = load_tweets(tpath, cfg["data"].get("tweets_encoding", "latin-1"),
                         cfg["data"].get("tweets_columns"))
        utc = getattr(tw.index, "tz", None) is not None and str(tw.index.tz) == "UTC"
        tag = OK if utc else FAIL
        failures += 0 if utc else 1
        print(f"{tag} tweets loaded: {len(tw):,} rows, "
              f"{tw.index.min().date()} -> {tw.index.max().date()}, UTC index = {utc}")
    else:
        print(f"{WARN} tweets file not found at {tpath}")

    # 4. OHLCV loads (you supply this — see DATA_REQUIREMENTS.md).
    opath = cfg["data"]["ohlcv_path"]
    if Path(opath).exists():
        oh = load_ohlcv(opath)
        utc = getattr(oh.index, "tz", None) is not None and str(oh.index.tz) == "UTC"
        tag = OK if utc else FAIL
        failures += 0 if utc else 1
        print(f"{tag} OHLCV loaded: {len(oh):,} bars, "
              f"{oh.index.min().date()} -> {oh.index.max().date()}, UTC index = {utc}")
    else:
        print(f"{WARN} OHLCV file not found at {opath} — supply it to complete Phase 0 data.")

    print()
    if failures:
        print(f"{FAIL} Phase 0 NOT complete: {failures} hard check(s) failed.")
        return 1
    print(f"{OK} Phase 0 checkpoint passed (warnings are pending data, not failures).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
