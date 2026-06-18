"""ADF, Granger, leakage assertions. Build Pathway Phase 3 (pre-build gate).

Two HARD gates (must pass before any model is fit):
  1. check_target_is_return  — the label is a return r_{t+1}, never a price level.
  2. check_leak_free_split    — chronological split + train-only scaling are wired,
     and the target is genuinely the NEXT bar's return (no contemporaneous leak).

Two INFORMATIONAL checks (recorded either way; they set honest expectations):
  3. adf_test         — stationarity of the target and sentiment features.
  4. granger_min_pvalue — does lagged sentiment help predict the return beyond the
     return's own past? Run on the TRAIN slice only, so the test set stays unseen.
"""
from __future__ import annotations

import os
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from src.features import FEATURE_COLUMNS, TARGET
from src.splits import TrainScaler, chronological_split


# --------------------------------------------------------------------------- #
# Hard gates
# --------------------------------------------------------------------------- #

def check_target_is_return(table: pd.DataFrame, target_col: str = TARGET) -> tuple[bool, str]:
    """Gate 1: the target must be a return (small, ~zero-mean), not a price level."""
    tgt = table[target_col]
    is_return = bool(tgt.abs().max() < 1.0 and abs(tgt.mean()) < 0.05)
    msg = (f"target '{target_col}': mean={tgt.mean():+.5f}, std={tgt.std():.5f}, "
           f"|max|={tgt.abs().max():.4f} "
           f"({'return-like' if is_return else 'NOT a return — looks like a level'})")
    return is_return, msg


def check_leak_free_split(table: pd.DataFrame, train_frac: float) -> tuple[bool, str]:
    """Gate 2: chronological split + train-only scaler wired, and target is the
    NEXT bar's return (target[t] == px_ret[t+1]) with no contemporaneous leak."""
    split = chronological_split(table.index, train_frac)

    # Scaler fits on train only and sees no test timestamp.
    scaler = TrainScaler().fit(table.loc[split.train_index, FEATURE_COLUMNS])
    try:
        scaler.assert_fit_was_leak_free(split.test_index)
        scaler_ok = True
    except AssertionError:
        scaler_ok = False

    # Target consistency: target[t] should equal the realized return on bar t+1
    # (px_ret[t+1]) — confirms it is the forward return, and differs from px_ret[t]
    # (so no feature equals the label on the same row).
    fwd_ok = bool(np.allclose(table[TARGET].to_numpy()[:-1],
                              table["px_ret"].to_numpy()[1:], atol=1e-9))
    same_row_corr = float(table["px_ret"].corr(table[TARGET]))

    ok = scaler_ok and fwd_ok and split.train_index.intersection(split.test_index).empty
    msg = (f"split {len(split.train_index)}/{len(split.test_index)} @ {split.cutoff.date()}; "
           f"scaler train-only={scaler_ok}; target==next-bar-return={fwd_ok}; "
           f"corr(px_ret_t, target_t)={same_row_corr:+.3f} (should be near 0)")
    return ok, msg


# --------------------------------------------------------------------------- #
# Informational checks
# --------------------------------------------------------------------------- #

def adf_test(series: pd.Series, alpha: float = 0.05) -> dict:
    """Augmented Dickey-Fuller. Null = unit root (non-stationary).

    Returns {stat, pvalue, stationary}. stationary=True means we reject the null
    (p < alpha) — the series is stationary, as ARIMA-family models require.
    """
    s = series.dropna()
    stat, pvalue, *_ = adfuller(s, autolag="AIC")
    return {"stat": float(stat), "pvalue": float(pvalue), "stationary": bool(pvalue < alpha)}


def granger_min_pvalue(
    target: pd.Series, feature: pd.Series, maxlag: int = 5, difference_feature: bool = False
) -> dict:
    """Does `feature` Granger-cause `target` (beyond target's own past lags)?

    grangercausalitytests controls for the target's autoregressive history, so a
    low p-value means the feature adds predictive information about the return.
    Returns the minimum ssr-F-test p-value across lags 1..maxlag and the best lag.
    """
    x = feature.diff() if difference_feature else feature
    data = pd.concat([target.rename("y"), x.rename("x")], axis=1).dropna()
    if len(data) <= maxlag + 2:
        return {"min_pvalue": float("nan"), "best_lag": None, "differenced": difference_feature}

    with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
        res = grangercausalitytests(data[["y", "x"]].to_numpy(), maxlag=maxlag)

    pvals = {lag: res[lag][0]["ssr_ftest"][1] for lag in res}
    best_lag = min(pvals, key=pvals.get)
    return {
        "min_pvalue": float(pvals[best_lag]),
        "best_lag": int(best_lag),
        "per_lag": {int(k): float(v) for k, v in pvals.items()},
        "differenced": difference_feature,
    }
