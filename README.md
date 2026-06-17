# Crypto Sentiment-Driven Trading Signal

End-to-end prototype: social sentiment + price features → ARIMAX return forecast →
buy/sell decision → leak-free fixed-holding-period backtest. See the design sheet
and build pathway for the *what/why* and *how/in-what-order*.

**Two non-negotiables:** the target is a **return** (never a price level), and the
evaluation is **leak-free and out-of-sample**.

## Locked decisions (Phase 0)

| Parameter | Value |
|---|---|
| Asset | BTC |
| Bar frequency | 1 day |
| Holding period N | 1 bar (next-day return, hold 1 day) |
| Transaction cost | 0.1% per trade |
| Train/test split | first 70% / last 30%, chronological |

All locked in [`config.yaml`](config.yaml).

## Setup

```bash
uv sync                       # create .venv and install pinned deps
uv run python check_phase0.py # verify the Phase 0 checkpoint
```

## Data

- Tweets: `Merged_Twitter_Data_IDsRemoved.csv` (already present, 2022-06-08 → 2023-05-28).
- BTC daily OHLCV: **you supply** `data/raw/btc_ohlcv_daily.csv` — see
  [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md).

> Data source actually used: BTC 1-minute OHLCV (epoch-seconds `Timestamp`, 2012-01-01 → 2026-06-17),
> resampled to 5,282 daily bars by [prepare_ohlcv.py](prepare_ohlcv.py). Raw minute file kept at
> `data/raw/btc_ohlcv_minute.csv`; modeling uses the 2022-06-08 → 2023-05-28 overlap with the tweets.

## Layout

```
config.yaml          asset, frequency, N, costs, threshold, data paths
check_phase0.py      Phase 0 checkpoint verifier
src/
  config.py          config loader
  data.py            raw tweets + OHLCV loaders (UTC-indexed)   [Phase 0]
  splits.py          chronological split + leak-free scaler     [Phase 1]
  backtest.py        fixed-hold backtest + baselines            [Phase 1]
  features.py        sentiment/price features, lag, target      [Phase 2]
  validate.py        ADF, Granger, leakage assertions           [Phase 3]
  model.py           ARIMAX fit/forecast/decision rule          [Phase 4-5]
data/                raw/ + processed/ (gitignored)
notebooks/  results/
```

## Progress

- [x] **Phase 0** — env, config, data loaders, repo structure
- [x] **Phase 1** — chronological split, leak-free scaler, backtest + baselines; random signal break-even after costs (`check_phase1.py`)
- [~] **Phase 2** — feature pipeline built & checkpoint passes (`build_features.py`); rebuilding on a **denser tweets dataset** (current set covers only 10% of days) before locking — see [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md)
- [ ] Phase 3 — pre-build validation gate (target, leak-free, ADF, Granger)
- [ ] Phase 4 — price-only ARIMAX (control)
- [ ] Phase 5 — add sentiment, compare
- [ ] Phase 6 — analysis & writeup
- [ ] Phase 7 — stretch goals
