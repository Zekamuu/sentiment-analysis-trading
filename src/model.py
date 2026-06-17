"""ARIMAX fit / forecast / decision rule. Build Pathway Phases 4-5.

NOT YET IMPLEMENTED. Phase 4 = price-only ARIMAX (control). Phase 5 = add lagged
sentiment as exogenous X and compare. Target is the return; d is typically 0.
"""
from __future__ import annotations


def fit_arimax(*args, **kwargs):
    raise NotImplementedError("Phase 4/5: statsmodels SARIMAX / pmdarima on the return target.")


def forecast(*args, **kwargs):
    raise NotImplementedError("Phase 4/5: chronological forecast across the test slice.")


def decision_rule(*args, **kwargs):
    raise NotImplementedError("Phase 4/5: threshold forecast into buy/sell/no-trade.")
