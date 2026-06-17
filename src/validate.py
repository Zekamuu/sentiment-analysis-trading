"""ADF, Granger, leakage assertions. Build Pathway Phase 3 (pre-build gate).

NOT YET IMPLEMENTED. Hard gates: target is a return, split is leak-free.
Informational: ADF stationarity, Granger causality (sentiment -> return).
"""
from __future__ import annotations


def assert_target_is_return(*args, **kwargs):
    raise NotImplementedError("Phase 3: hard gate.")


def assert_leak_free_split(*args, **kwargs):
    raise NotImplementedError("Phase 3: hard gate.")


def adf_test(*args, **kwargs):
    raise NotImplementedError("Phase 3: stationarity check.")


def granger_causality(*args, **kwargs):
    raise NotImplementedError("Phase 3: informational lead-lag check.")
