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
    endog: pd.Series,
    n_train: int,
    order: tuple[int, int, int],
    exog: pd.DataFrame | None = None,
) -> pd.Series:
    """Roll one-step-ahead forecasts for every decision bar t.

    `endog` is the TARGET series (r_{t+1} indexed by decision bar t); `exog`, if
    given, is the contemporaneous feature row (sent_t), known at decision time t.
    The one-step-ahead prediction of endog[t] uses endog[:t] (actuals through t-1)
    plus exog[t] — all available at the close of bar t, so it is leak-free.

    Params are estimated on the first `n_train` observations ONLY; test actuals are
    filtered in via append(refit=False) without re-estimating. Returns a Series
    indexed like `endog` whose value at bar t is the forecast of r_{t+1}.
    """
    d = order[1]
    trend = "c" if d == 0 else "n"
    orig_index = endog.index
    # Work on a positional index so SARIMAX.append doesn't choke on a tz-aware
    # DatetimeIndex that carries no frequency.
    y = endog.reset_index(drop=True)
    ex = exog.reset_index(drop=True) if exog is not None else None
    ex_train = ex.iloc[:n_train] if ex is not None else None

    res = SARIMAX(y.iloc[:n_train], exog=ex_train, order=order, trend=trend,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    if ex is not None:
        res = res.append(y.iloc[n_train:], exog=ex.iloc[n_train:], refit=False)
    else:
        res = res.append(y.iloc[n_train:], refit=False)

    # One-step-ahead prediction for every position (each uses actuals through the
    # prior bar and the contemporaneous exog row). No out-of-sample tail step is
    # needed: endog[t] already IS r_{t+1}.
    pred = res.get_prediction(start=0, dynamic=False).predicted_mean
    return pd.Series(np.asarray(pred, dtype=float), index=orig_index, name="forecast")


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
