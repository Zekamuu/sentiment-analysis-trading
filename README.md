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

- Tweets: `data/raw/bitcoin_tweets_latest.csv` (~2 GB, ~4.8M tweets, dense over
  2025-03 → 2026-03). Sampled to ≤3k/day and windowed by [prepare_tweets.py](prepare_tweets.py).
- BTC daily OHLCV: `data/raw/btc_ohlcv_daily.csv` — see [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md).

> Data sources actually used:
> - **OHLCV**: BTC 1-minute (epoch-seconds, 2012 → 2026-06-17), resampled to 5,282 daily
>   bars by [prepare_ohlcv.py](prepare_ohlcv.py). (`BTC1D.csv` ends 2021 and does NOT overlap
>   the tweet window, so it is unused.)
> - **Tweets**: a dense Bitcoin-tweets crawl; modeling window **2025-03-06 → 2026-03-02**
>   (362 daily bars, 55% have tweets, remainder forward-filled).
> Raw data lives under `data/` and is gitignored — only code + docs are committed.

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
- [x] **Phase 2** — modeling table on the dense 2025-03→2026-03 tweet block (362 bars, 55% daily coverage, sentiment forward-filled over gaps; `prepare_tweets.py` → `build_features.py`)
- [x] **Phase 3** — validation gate (`check_phase3.py`): target-is-return ✓, leak-free split ✓, ADF all stationary; Granger = **no significant sentiment lead** (honest null expectation)
- [x] **Phase 4** — price-only ARIMA(1,0,0) control (`check_phase4.py`): test slice is a BTC downtrend; model −25% cumulative (vs −27% B&H, −36% always-buy, −43% random), directional acc 0.46 — edge is exposure-reduction, not skill
- [ ] Phase 5 — add sentiment, compare
- [ ] Phase 6 — analysis & writeup
- [ ] Phase 7 — stretch goals
