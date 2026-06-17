"""Chronological split + leak-free scaling. Build Pathway Phase 1, steps 1-2.

Two utilities, both built so time-order leakage is hard to introduce by accident:

  - chronological_split: split by TIME (first train_frac, then the rest). Never
    shuffles; returns the index labels of each slice, not reordered rows.
  - TrainScaler: standardization whose parameters are fit on the TRAIN slice
    ONLY and then applied to any slice. There is deliberately no fit_transform
    over a full series, so you cannot fit on test data by accident.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    """Result of a chronological split. Holds index labels, not copied rows."""

    train_index: pd.Index
    test_index: pd.Index
    cutoff: pd.Timestamp  # first timestamp belonging to the TEST slice

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Split(train={len(self.train_index)} bars "
            f"[{self.train_index.min()} -> {self.train_index.max()}], "
            f"test={len(self.test_index)} bars "
            f"[{self.test_index.min()} -> {self.test_index.max()}], "
            f"cutoff={self.cutoff})"
        )


def chronological_split(index: pd.Index, train_frac: float = 0.70) -> Split:
    """Split a time index into earlier-train / later-test by position.

    Parameters
    ----------
    index : a DatetimeIndex (or any ordered index) for the rows to split.
    train_frac : fraction of rows (by time order) assigned to training.

    Returns the index labels for each slice. NEVER shuffles.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if not index.is_monotonic_increasing:
        raise ValueError(
            "Index must be sorted ascending before splitting — refusing to "
            "split an unordered time index (that is how leakage sneaks in)."
        )

    n = len(index)
    k = int(n * train_frac)
    if k == 0 or k == n:
        raise ValueError(f"train_frac={train_frac} leaves an empty slice for n={n}")

    train_index, test_index = index[:k], index[k:]
    return Split(train_index=train_index, test_index=test_index, cutoff=test_index[0])


class TrainScaler:
    """Standardize features using statistics learned from the TRAIN slice only.

    Usage:
        scaler = TrainScaler().fit(df.loc[split.train_index])
        train_z = scaler.transform(df.loc[split.train_index])
        test_z  = scaler.transform(df.loc[split.test_index])

    There is no fit_transform on a full frame, by design: fitting is a separate,
    explicit step that you can only hand the training slice.
    """

    def __init__(self) -> None:
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None
        self.columns_: pd.Index | None = None
        self.fit_index_: pd.Index | None = None

    def fit(self, train_df: pd.DataFrame) -> "TrainScaler":
        self.columns_ = train_df.columns
        self.mean_ = train_df.mean()
        std = train_df.std(ddof=0)
        # Guard against zero-variance columns (would divide by zero).
        self.std_ = std.replace(0.0, 1.0)
        self.fit_index_ = train_df.index
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None:
            raise RuntimeError("TrainScaler.transform called before fit().")
        if not df.columns.equals(self.columns_):
            raise ValueError("Columns differ from those seen at fit time.")
        return (df - self.mean_) / self.std_

    def assert_fit_was_leak_free(self, test_index: pd.Index) -> None:
        """Sanity check: no test timestamp leaked into the fitted statistics."""
        if self.fit_index_ is None:
            raise RuntimeError("Scaler not fit yet.")
        overlap = self.fit_index_.intersection(test_index)
        if len(overlap) > 0:
            raise AssertionError(
                f"LEAKAGE: scaler was fit on {len(overlap)} timestamps that are "
                f"in the test slice. Fit on the training slice ONLY."
            )
