"""Fixed-holding-period backtest + baselines. Build Pathway Phase 1, steps 3-5.

Design §5.1. The backtest takes a per-bar DECISION series (buy / sell / no-trade)
and a PRICE series, and for every fired decision at bar t records the return from
the entry price at t to the price N bars later, minus a fixed transaction cost.
No position sizing, no stop-losses — one fixed-exit trade per fired signal.

Decision encoding (ints): +1 = buy (long), -1 = sell (short), 0 = no-trade.

CRITICAL honesty property (the whole point of building this first): a RANDOM
decision series must come out around break-even or negative after costs, and must
not beat buy-and-hold except by chance. If random looks profitable, there is a
look-ahead bug in how entry/exit prices are indexed — fix it before trusting any
later result.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BUY, SELL, NO_TRADE = 1, -1, 0


@dataclass(frozen=True)
class BacktestResult:
    """Aggregate outcome of running a decision series through the backtest."""

    n_trades: int
    avg_net_return: float        # mean net return per trade (after cost)
    hit_rate: float              # fraction of trades with net return > 0
    cumulative_net_return: float # compounded growth across trades, reinvested
    trade_returns: pd.Series     # per-trade net return, indexed by entry timestamp
    equity_curve: pd.Series      # (1+net).cumprod()-1 across trades, by entry time

    def as_row(self, name: str) -> dict:
        """Flat dict for building comparison tables (Phase 4-6)."""
        return {
            "strategy": name,
            "n_trades": self.n_trades,
            "avg_net_return": self.avg_net_return,
            "hit_rate": self.hit_rate,
            "cumulative_net_return": self.cumulative_net_return,
        }


def fixed_hold_backtest(
    decisions: pd.Series,
    prices: pd.Series,
    holding_period_N: int,
    transaction_cost: float,
) -> BacktestResult:
    """Run a decision series through the fixed-holding-period backtest.

    For each bar t where decision != 0 and a full N-bar exit exists:
        long  (+1): gross = price[t+N] / price[t] - 1
        short (-1): gross = -(price[t+N] / price[t] - 1)
        net = gross - transaction_cost      # cost = round-trip fee+spread per trade

    Bars in the last N positions cannot complete a trade (no exit bar yet) and are
    skipped — this is also where look-ahead would creep in if mis-indexed.
    """
    if holding_period_N < 1:
        raise ValueError("holding_period_N must be >= 1")

    # Align decisions onto the price index; treat anything missing as no-trade.
    decisions = decisions.reindex(prices.index).fillna(NO_TRADE).astype(int)
    px = prices.to_numpy(dtype=float)
    dec = decisions.to_numpy(dtype=int)
    idx = prices.index

    n = len(px)
    last_entry = n - holding_period_N  # entries at >= this position have no exit bar

    entry_times: list = []
    nets: list[float] = []
    for i in range(last_entry):
        side = dec[i]
        if side == NO_TRADE:
            continue
        gross = px[i + holding_period_N] / px[i] - 1.0
        if side == SELL:
            gross = -gross
        nets.append(gross - transaction_cost)
        entry_times.append(idx[i])

    trade_returns = pd.Series(nets, index=pd.Index(entry_times, name="entry_time"),
                              dtype=float)
    if len(trade_returns) == 0:
        empty = pd.Series(dtype=float)
        return BacktestResult(0, float("nan"), float("nan"), 0.0, empty, empty)

    equity = (1.0 + trade_returns).cumprod() - 1.0
    return BacktestResult(
        n_trades=int(len(trade_returns)),
        avg_net_return=float(trade_returns.mean()),
        hit_rate=float((trade_returns > 0).mean()),
        cumulative_net_return=float(equity.iloc[-1]),
        trade_returns=trade_returns,
        equity_curve=equity,
    )


# --------------------------------------------------------------------------- #
# Baselines (Design §5.2). always_buy and random are decision series the SAME
# backtest consumes. buy-and-hold is the hold-the-whole-period reference and is
# computed directly, because it is not a fixed-N strategy.
# --------------------------------------------------------------------------- #

def always_buy(prices: pd.Series) -> pd.Series:
    """Decision series that buys on every bar."""
    return pd.Series(BUY, index=prices.index, dtype=int)


def random_decisions(
    prices: pd.Series, seed: int, include_no_trade: bool = False
) -> pd.Series:
    """Coin-flip decision series — the harness's honesty test.

    Symmetric buy/sell (optionally with no-trade) so that, over a drifting market,
    long and short bets cancel and the expected gross return is ~0; after costs it
    should be slightly negative. A signal that is just market drift in disguise
    would (wrongly) look profitable here — this baseline exposes that.
    """
    rng = np.random.default_rng(seed)
    choices = [BUY, SELL, NO_TRADE] if include_no_trade else [BUY, SELL]
    draws = rng.choice(choices, size=len(prices))
    return pd.Series(draws, index=prices.index, dtype=int)


def buy_and_hold(prices: pd.Series, transaction_cost: float) -> BacktestResult:
    """Hold the asset across the whole period (Design §5.2 reference).

    One entry at the first bar, one exit at the last, a single round-trip cost.
    Reported in the same BacktestResult shape for side-by-side comparison.
    """
    gross = prices.to_numpy(dtype=float)
    net_curve = gross / gross[0] - 1.0 - transaction_cost
    equity = pd.Series(net_curve, index=prices.index)
    final = float(equity.iloc[-1])
    trade = pd.Series([final], index=pd.Index([prices.index[0]], name="entry_time"))
    return BacktestResult(
        n_trades=1,
        avg_net_return=final,
        hit_rate=float(final > 0),
        cumulative_net_return=final,
        trade_returns=trade,
        equity_curve=equity,
    )
