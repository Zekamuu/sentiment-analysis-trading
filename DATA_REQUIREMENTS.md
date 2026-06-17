# Data Requirements

What the project needs you to source from the internet. You handle acquisition;
the pipeline handles everything downstream of ingestion.

## Locked context (from `config.yaml`)

- **Asset:** BTC
- **Bar frequency:** 1 day
- **Holding period N:** 1 bar (forecast next-day return, hold 1 day)
- **Date window to cover:** **2022-06-08 → 2023-05-28** (the span of the tweets you already have, with a few days of buffer on each side ideally)

---

## 1. Tweets dataset — ⚠️ NEED A DENSER ONE

`Merged_Twitter_Data_IDsRemoved.csv` (present) works mechanically but is **too
sparse for the sentiment experiment**: it only has tweets on **36 of 355 days
(10%)** — about 3 irregular days per month, each with thousands of posts. With
that coverage, sentiment features are empty on 90% of bars, so Phase 5 could only
ever move ~36 of 355 predictions. We decided (Phase 2) to **source a denser set**.

### What to find

A crypto/Bitcoin tweets dataset with **continuous (near-daily) coverage** over a
**contiguous window of at least ~6–12 months**. Density matters more than total
volume: we want most days to have posts, not a few days with huge bursts.

**Required logical fields** (the loader auto-detects common column names — see
`_TWEET_ALIASES` in [src/data.py](src/data.py); override via `data.tweets_columns`
in [config.yaml](config.yaml) if needed):

| logical field | accepted column names | needed for |
|---|---|---|
| text | `text, tweet, content, body, full_text` | VADER sentiment (required) |
| date | `date, timestamp, created_at, datetime, time` | bar alignment (required) |
| followers | `followers, follower_count, user_followers...` | influence weighting (optional — falls back to plain mean if absent) |

Sentiment/engagement columns are not needed (we compute sentiment ourselves).

**Where to look** (Design §7.3):
- Kaggle **"Bitcoin Tweets"** sets (e.g. the ~16M sentiment-tagged set, or
  `bitcoin-tweets-2021`/`2022`) — these have near-daily timestamps.
- Kaggle **"Cryptocurrency Tweets"** (2022–2023) — general crypto chatter.
- HuggingFace crypto-tweet corpora with `created_at` + `user_followers`.

**Coverage check before committing to it:** the day-span should have tweets on
most days. We have BTC OHLCV for **2012 → 2026**, so any window your tweets cover
is fine on the price side.

### When you have it

1. Save the CSV somewhere in the project (e.g. `data/raw/tweets_dense.csv`).
2. Point `data.tweets_path` (and `tweets_encoding`/`tweets_columns` if needed) at it.
3. I'll re-run `uv run python build_features.py` to rebuild the modeling table and
   re-check coverage. If the file is very large (10M+ rows), VADER scoring can be
   slow — tell me and we'll sample or batch it.

> The current sparse file stays usable as a fallback; the pipeline is dataset-agnostic.

---

## 2. BTC daily OHLCV — ✅ DONE

Sourced as a **1-minute** file (epoch-seconds timestamps, 2012→2026) saved to
`data/raw/btc_ohlcv_minute.csv`, then resampled to **5,282 daily bars** by
[`prepare_ohlcv.py`](prepare_ohlcv.py) → `data/raw/btc_ohlcv_daily.csv`.
Re-run `uv run python prepare_ohlcv.py` if the raw file changes.

The original requirement below is kept for reference.

---

## 2. (reference) BTC daily OHLCV requirements

A single CSV of **daily** Bitcoin price bars covering at least **2022-06-08 → 2023-05-28**.

**Save it to:** `data/raw/btc_ohlcv_daily.csv`

**Required columns** (case-insensitive; loader lowercases and accepts a date column
named any of `date/timestamp/time/datetime/day`):

| column | meaning |
|---|---|
| date | one row per UTC day |
| open | daily open |
| high | daily high |
| low | daily low |
| close | daily close (used for log returns / the target) |
| volume | daily traded volume |

**Where to get it (free, no auth needed for most):**

- **CoinGecko** — BTC/USD daily market chart, easy CSV/JSON export.
- **Yahoo Finance** — ticker `BTC-USD`, "Historical Data", set the date range, Download.
- **Binance public klines** — `1d` interval, `BTCUSDT` (highest fidelity, UTC-aligned).
- **Kaggle / CryptoDataDownload** — "Bitcoin daily OHLCV" datasets.

**Acceptance checks the loader enforces:**
- Parses to a UTC tz-aware daily `DatetimeIndex`.
- Has all five OHLCV columns after lowercasing.
- Spans the tweet window (gaps inside the window are fine; we handle empty/missing bars in Phase 2).

> Pick **one** source and **record exactly which file and time range you used**
> (Design §8 reproducibility). Note it in the README once downloaded.

---

## Verify once OHLCV is in place

```bash
uv run python check_phase0.py
```

All hard checks should read `[ OK ]`; the OHLCV warning disappears once the file exists.
