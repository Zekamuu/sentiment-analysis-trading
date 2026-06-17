"""Fixed-holding-period backtest + baselines. Build Pathway Phase 1, steps 3-5.

NOT YET IMPLEMENTED. Critical Phase 1 checkpoint: a RANDOM decision signal must
hover around break-even (or negative) after costs and must NOT beat buy-and-hold
except by chance. If random looks profitable, there is a look-ahead bug.
"""
from __future__ import annotations


def fixed_hold_backtest(*args, **kwargs):
    raise NotImplementedError("Phase 1: entry at price[t], exit at price[t+N], minus cost.")


def baseline_buy_and_hold(*args, **kwargs):
    raise NotImplementedError("Phase 1 baseline.")


def baseline_always_buy(*args, **kwargs):
    raise NotImplementedError("Phase 1 baseline.")


def baseline_random(*args, **kwargs):
    raise NotImplementedError("Phase 1 baseline — the harness's honesty test.")
