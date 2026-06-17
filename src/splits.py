"""Chronological split + leak-free scaling. Build Pathway Phase 1, steps 1-2.

NOT YET IMPLEMENTED. The harness is built and validated against a RANDOM signal
in Phase 1, before any model exists.
"""
from __future__ import annotations


def chronological_split(*args, **kwargs):
    raise NotImplementedError("Phase 1: split by time (e.g. 70/30), never shuffle.")


def fit_train_scaler(*args, **kwargs):
    raise NotImplementedError("Phase 1: fit normalization on TRAIN slice only.")
