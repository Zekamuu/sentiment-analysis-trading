# Data Requirements

What the project needs you to source from the internet. You handle acquisition;
the pipeline handles everything downstream of ingestion.

## Locked context (from `config.yaml`)

- **Asset:** BTC
- **Bar frequency:** 1 day
- **Holding period N:** 1 bar (forecast next-day return, hold 1 day)
- **Date window to cover:** **2022-06-08 → 2023-05-28** (the span of the tweets you already have, with a few days of buffer on each side ideally)

---

## 1. Tweets dataset — ✅ ALREADY PRESENT

`Merged_Twitter_Data_IDsRemoved.csv` (in the project root).

- ~107K usable tweets, daily timestamps, 2022-06-08 → 2023-05-28.
- Columns: `tweet_id, text, date, retweets, replies, likes, location, followers, following`.
- `followers` → influence weighting (Design §4.5). Engagement columns are bonus features.
- No sentiment column — correct; we compute it with VADER in Phase 2.
- Encoding is **latin-1** (not clean UTF-8); the loader handles this.

If you later want a richer/cleaner tweets set, the design (§7.3) suggests:
metadata-rich crypto-influencer sets with follower/influence fields. Any
replacement must keep **text + timestamp + follower count** at minimum.

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
