"""ARIMAX fit / forecast / decision rule. Build Pathway Phases 4-5.

Phase 4 = price-only ARIMA on the return series (the control). Phase 5 = the same
machinery with sentiment columns passed as the exogenous `exog` — so the ONLY
difference between the two phases is whether sentiment is supplied.

Forecasting is strictly chronological and leak-free:
  - ARIMA parameters are estimated on the TRAIN slice only.
  - One-step-ahead forecasts roll forward: the forecast of the return at bar t uses
    actual returns through bar t-1 (via Kalman filtering with append, refit=False),
    never future data and never re-estimating on test data.

Decision-bar convention (matches the modeling table / backtest): a forecast made at
decision bar t predicts r_{t+1}, the return the backtest earns entering at close[t].
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.backtest import BUY, NO_TRADE, SELL, fixed_hold_backtest

warnings.filterwarnings("ignore")  # SARIMAX convergence/spec chatter is noisy here


def select_order(
    returns: pd.Series,
    exog: pd.DataFrame | None = None,
    p_range=range(0, 4),
    q_range=range(0, 4),
    d: int = 0,
) -> tuple[int, int, int]:
    """Pick (p, d, q) by AIC over a small grid (ACF/PACF-scale search). d defaults
    to 0 because the target is already a stationary return (Phase 3 ADF confirmed)."""
    trend = "c" if d == 0 else "n"
    r = returns.reset_index(drop=True)
    ex = exog.reset_index(drop=True) if exog is not None else None
    best_aic, best = np.inf, (1, d, 0)
    for p in p_range:
        for q in q_range:
            if p == 0 and q == 0:
                continue
            try:
                res = SARIMAX(r, exog=ex, order=(p, d, q), trend=trend,
                              enforce_stationarity=False, enforce_invertibility=False
                              ).fit(disp=False)
                if np.isfinite(res.aic) and res.aic < best_aic:
                    best_aic, best = res.aic, (p, d, q)
            except Exception:
                continue
    return best


def one_step_forecasts(
    returns: pd.Series,
    n_train: int,
    order: tuple[int, int, int],
    exog: pd.DataFrame | None = None,
) -> pd.Series:
    """Roll one-step-ahead forecasts of r_{t+1} for every decision bar t.

    Params are estimated on returns[:n_train] (+ exog) ONLY. Returns a Series indexed
    like `returns` whose value at bar t is the forecast of the next bar's return
    (the quantity the decision rule acts on). The last bar's forecast is a genuine
    out-of-sample step beyond the data.
    """
    n = len(returns)
    d = order[1]
    trend = "c" if d == 0 else "n"
    orig_index = returns.index
    # Work on a positional index so SARIMAX.append doesn't choke on a tz-aware
    # DatetimeIndex that carries no frequency.
    r = returns.reset_index(drop=True)
    ex = exog.reset_index(drop=True) if exog is not None else None
    ex_train = ex.iloc[:n_train] if ex is not None else None

    res = SARIMAX(r.iloc[:n_train], exog=ex_train, order=order, trend=trend,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    if ex is not None:
        res = res.append(r.iloc[n_train:], exog=ex.iloc[n_train:], refit=False)
    else:
        res = res.append(r.iloc[n_train:], refit=False)

    # In-sample one-step-ahead predictions for positions 1..n-1 (each uses actuals
    # through the prior bar), then one out-of-sample step for the final bar.
    insample = res.get_prediction(start=1, dynamic=False).predicted_mean.to_numpy()
    if ex is not None:
        oos = float(res.forecast(steps=1, exog=ex.iloc[[-1]]).iloc[0])
    else:
        oos = float(res.forecast(steps=1).iloc[0])

    dec = np.empty(n)
    dec[: n - 1] = insample          # forecast of r_{t+1} for bars 0..n-2
    dec[n - 1] = oos                 # final bar: out-of-sample one-step
    return pd.Series(dec, index=orig_index, name="forecast")


def decision_rule(forecasts: pd.Series, threshold: float) -> pd.Series:
    """Threshold a forecast return into buy / sell / no-trade (Design §4.10)."""
    dec = pd.Series(NO_TRADE, index=forecasts.index, dtype=int)
    dec[forecasts > threshold] = BUY
    dec[forecasts < -threshold] = SELL
    return dec


def tune_threshold(
    forecasts: pd.Series,
    prices: pd.Series,
    holding_period_N: int,
    transaction_cost: float,
    n_grid: int = 12,
    min_trades: int = 5,
) -> float:
    """Choose the decision threshold on the TRAIN slice only (Design §4.10).

    Grids over quantiles of |forecast| and picks the threshold maximizing train
    cumulative net return, requiring at least `min_trades` trades so the choice
    isn't a one-lucky-trade artifact. Falls back to 0 if nothing qualifies.
    """
    cand = np.unique(np.concatenate([[0.0],
                     np.quantile(forecasts.abs(), np.linspace(0, 0.9, n_grid))]))
    best_thr, best_cum = 0.0, -np.inf
    for thr in cand:
        res = fixed_hold_backtest(decision_rule(forecasts, thr), prices,
                                  holding_period_N, transaction_cost)
        if res.n_trades >= min_trades and res.cumulative_net_return > best_cum:
            best_cum, best_thr = res.cumulative_net_return, thr
    return float(best_thr)
