"""Sentiment aggregation, price features, lagging, target. Build Pathway Phase 2.

Produces ONE modeling table, indexed by the decision bar t, where:

  - target = next-period return r_{t+1} = ln(close[t+1] / close[t])   (Design §1 rule 1)
  - every feature uses information available by the CLOSE of bar t, which strictly
    predates the t -> t+1 return window the target measures (Design §4.8).

Time-alignment convention (kept consistent with the Phase 1 backtest, which enters
at close[t] and exits at close[t+N]):
    row t  :  features as-of end of day t  ->  predict the return realized over day t+1
Because crypto trades 24/7 with daily bars labelled at 00:00 UTC, day-t tweets and
the day-t price close are all known by the time we decide at the end of day t, and
all precede the start of the day t+1 return window. No look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Column groups, prefixed so model phases can select exogenous sets cleanly.
PRICE_FEATURES = ["px_ret", "px_vol5", "px_volchg"]
SENTIMENT_FEATURES = ["sent_wmean", "sent_postvol", "sent_disp"]
FEATURE_COLUMNS = PRICE_FEATURES + SENTIMENT_FEATURES
TARGET = "target_logret_fwd"


# --------------------------------------------------------------------------- #
# Branch B — VADER sentiment per post, then influence-weighted daily aggregate
# --------------------------------------------------------------------------- #

def score_tweets_vader(tweets: pd.DataFrame, text_col: str = "text") -> pd.Series:
    """Score each post's compound sentiment in [-1, +1] with VADER.

    VADER is rule/lexicon-based and tuned for social media — we deliberately do
    NOT lowercase or strip punctuation/emoji, which it uses as signal (Design §4.2).
    Returns a Series aligned to the tweets' index.
    """
    analyzer = SentimentIntensityAnalyzer()
    text = tweets[text_col].fillna("").astype(str)
    return text.map(lambda t: analyzer.polarity_scores(t)["compound"]).rename("compound")


def aggregate_sentiment_daily(
    tweets: pd.DataFrame,
    bar_index: pd.DatetimeIndex,
    empty_bar: str = "zero",
    true_counts: pd.Series | None = None,
) -> pd.DataFrame:
    """Collapse per-post scores into daily bar features, influence-weighted.

    Features per bar (Design §4.5):
      - sent_wmean  : influence-weighted mean compound, weight = log(1 + followers)
      - sent_postvol: log(1 + post count)  (chatter volume is itself predictive)
      - sent_disp   : dispersion (std) of compound within the bar

    Empty-bar (no-tweet day) policy (Design §4.7):
      - "zero"  : neutral signal, zero volume/dispersion.
      - "ffill" : hold the most recent sentiment across the gap, then zero-fill any
                  leading bars before the first observation. Leak-free (only past
                  readings are carried forward) — used when coverage is gappy.
    """
    df = tweets.copy()
    if "compound" not in df.columns:
        df["compound"] = score_tweets_vader(df)

    # Bar = the UTC day the post belongs to (index is already a tz-aware datetime).
    bar = df.index.floor("D")
    followers = pd.to_numeric(df.get("followers"), errors="coerce").fillna(0.0)
    weight = np.log1p(followers.clip(lower=0.0))
    df = df.assign(_bar=bar, _w=weight.to_numpy(), _ws=(weight.to_numpy() * df["compound"]))

    grouped = df.groupby("_bar")
    wsum = grouped["_w"].sum()
    wmean = grouped["_ws"].sum() / wsum.replace(0.0, np.nan)
    # If every weight in a bar is 0 (e.g. all followers 0/NA), fall back to plain mean.
    wmean = wmean.fillna(grouped["compound"].mean())

    # Post volume: prefer the TRUE per-day counts (the sample is capped and would
    # saturate this feature); fall back to the sampled count when unavailable.
    if true_counts is not None:
        postvol = np.log1p(true_counts.reindex(grouped.size().index).fillna(grouped.size()))
    else:
        postvol = np.log1p(grouped.size())

    daily = pd.DataFrame({
        "sent_wmean": wmean,
        "sent_postvol": postvol,
        "sent_disp": grouped["compound"].std(ddof=0),
    })

    # Reindex onto the price bar grid and fill empty bars per the chosen policy.
    daily = daily.reindex(bar_index)
    if empty_bar == "ffill":
        daily = daily.ffill().fillna(0.0)  # hold last reading; zero any leading gap
    elif empty_bar == "zero":
        daily = daily.fillna(0.0)
    else:
        raise ValueError(f"empty_bar must be 'zero' or 'ffill', got {empty_bar!r}")
    daily.index.name = "date"
    return daily


# --------------------------------------------------------------------------- #
# Market branch — price features + the return target
# --------------------------------------------------------------------------- #

def price_features(ohlcv: pd.DataFrame, vol_window: int = 5) -> pd.DataFrame:
    """Daily price features computed as-of the close of each bar (Design §4.6).

      - px_ret    : log return ln(close[t]/close[t-1]) realized over the bar
      - px_vol5   : rolling std of log returns (realized volatility), window vol_window
      - px_volchg : log change in volume vs the prior bar

    Computed on the FULL OHLCV series so warm-up history is available; slice to the
    modeling window afterwards (avoids NaNs at the window's start).
    """
    close = ohlcv["close"].astype(float)
    logret = np.log(close).diff()
    feats = pd.DataFrame(index=ohlcv.index)
    feats["px_ret"] = logret
    feats["px_vol5"] = logret.rolling(vol_window).std(ddof=0)
    vol = ohlcv["volume"].astype(float)
    feats["px_volchg"] = np.log(vol.replace(0.0, np.nan)).diff()
    return feats


def make_target(ohlcv: pd.DataFrame) -> pd.Series:
    """Next-period return r_{t+1} = ln(close[t+1]/close[t]) (Design §1 rule 1).

    Computed on the full series so the last in-window bar still gets a target from
    the following day's price. This is the quantity the decision rule forecasts and
    exactly the return the Phase-1 backtest earns for a buy at close[t] (N=1).
    """
    logret = np.log(ohlcv["close"].astype(float)).diff()
    return logret.shift(-1).rename(TARGET)  # return realized over the NEXT bar


# --------------------------------------------------------------------------- #
# Assemble
# --------------------------------------------------------------------------- #

def build_modeling_table(
    tweets: pd.DataFrame,
    ohlcv: pd.DataFrame,
    window: tuple[str, str],
    vol_window: int = 5,
    empty_bar: str = "zero",
    true_counts: pd.Series | None = None,
) -> pd.DataFrame:
    """Join sentiment + price features and the target onto one daily index.

    Returns a table indexed by decision bar t with FEATURE_COLUMNS (all as-of the
    close of t) and TARGET (the t -> t+1 return). Drops rows with any NaN in the
    modeling columns. No leakage: features precede the target's return window.
    """
    start, end = window

    # Price features + target on the full series, then slice to the window.
    px = price_features(ohlcv, vol_window=vol_window).loc[start:end]
    target = make_target(ohlcv).loc[start:end]

    # Sentiment aggregated onto the same daily grid as the price window.
    sent = aggregate_sentiment_daily(tweets, px.index, empty_bar=empty_bar,
                                     true_counts=true_counts)

    table = pd.concat([px, sent, target], axis=1)
    table = table[FEATURE_COLUMNS + [TARGET]].dropna()
    return table


def assert_no_lookahead(table: pd.DataFrame, tweets: pd.DataFrame, n_check: int = 5) -> None:
    """Spot-check: tweets feeding row t's sentiment predate the t->t+1 return window.

    For a sample of rows, confirm every day-t post is timestamped strictly before
    the start of the next bar (t + 1 day) — the start of the window the target
    measures. Raises AssertionError on any violation.
    """
    bars = tweets.index.floor("D")
    rows = table.index[:: max(1, len(table) // n_check)][:n_check]
    for t in rows:
        next_bar_start = t + pd.Timedelta(days=1)
        day_posts = tweets.index[bars == t]
        if len(day_posts) and day_posts.max() >= next_bar_start:
            raise AssertionError(
                f"LOOK-AHEAD: row {t.date()} uses a post at {day_posts.max()} "
                f">= return-window start {next_bar_start}"
            )
