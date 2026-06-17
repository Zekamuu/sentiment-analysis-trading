"""Sentiment aggregation, price features, lagging, target. Build Pathway Phase 2.

NOT YET IMPLEMENTED. Two non-negotiables enforced here later:
  - target = next-period RETURN r_{t+1}, never the price level (Design §1 rule 1).
  - features for bar t use information through t-1 only (Design §4.8).
"""
from __future__ import annotations


def vader_sentiment_per_bar(*args, **kwargs):
    raise NotImplementedError("Phase 2: VADER per post -> influence-weighted bar aggregate.")


def price_features(*args, **kwargs):
    raise NotImplementedError("Phase 2: log returns from close, optional vol/volume change.")


def make_target(*args, **kwargs):
    raise NotImplementedError("Phase 2: target = next-period return.")


def lag_features(*args, **kwargs):
    raise NotImplementedError("Phase 2: shift all features >=1 bar to prevent look-ahead.")
