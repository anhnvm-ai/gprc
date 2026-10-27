"""Walk-forward purged-and-embargoed CV splits (López de Prado, 2018)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Tuple
import numpy as np
import pandas as pd

@dataclass
class WalkForwardSplit:
    train_end: str
    val_end: str
    test_start: str
    refit_freq: str = "M"
    purge_days: int = 22
    embargo_days: int = 5

    def split(self, index: pd.DatetimeIndex) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Yields (train_idx, val_idx, test_idx) numpy positional indices for each refit.
        First fold: train = [start, train_end], val = (train_end, val_end], test = first refit window.
        Subsequent folds: training expands; test rolls forward by `refit_freq`.
        Purge `purge_days` removed at the end of train; embargo `embargo_days` at start of test.
        """
        idx = pd.DatetimeIndex(index)
        train_end = pd.Timestamp(self.train_end)
        val_end   = pd.Timestamp(self.val_end)
        test_start = pd.Timestamp(self.test_start)
        # Refit anchors = month-ends within [test_start, idx[-1]]
        # pandas >= 2.2 deprecates "M"/"Q" in favor of "ME"/"QE"
        freq = self.refit_freq
        if freq in ("M", "Q", "Y"):
            freq = freq + "E"
        anchors = pd.date_range(test_start, idx[-1], freq=freq)
        prev_anchor = test_start - pd.Timedelta(days=1)
        for a in anchors:
            test_mask = (idx > prev_anchor) & (idx <= a)
            test_pos = np.where(test_mask)[0]
            if len(test_pos) == 0:
                prev_anchor = a; continue
            # Train = everything up to (test_start - embargo - purge); for fold k, expand to (prev_anchor - embargo)
            train_cut = idx[test_pos[0]] - pd.Timedelta(days=self.embargo_days + self.purge_days)
            train_mask = idx <= train_cut
            # Hold out a small val slice = last 6 months of train
            val_cut = idx[train_mask][-1] - pd.Timedelta(days=180)
            val_mask = (idx > val_cut) & (idx <= idx[train_mask][-1])
            train_only = train_mask & (~val_mask)
            yield (np.where(train_only)[0], np.where(val_mask)[0], test_pos)
            prev_anchor = a
