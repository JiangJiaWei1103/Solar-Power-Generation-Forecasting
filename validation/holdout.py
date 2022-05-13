"""
Holdout splitter definition.
Author: JiaWei Jiang

This file contains the definition of holdout data splitter splitting
a portion of data as unseen testing set for final evaluation.
"""
import random
from math import floor
from typing import List, Optional, Union

import numpy as np
import pandas as pd


class HoldoutSplitter(object):
    """Holdout splitter splitting a portion of data as unseen testing
    set for final evaluation.

    Parameters:
        n_splits: number of holdout sets
        chrono: whether to follow chronological order when splitting
        ts_col: name of column with time step identifier
        t_start: the first time step (inclusive) for each holdout set
        t_end: the last time step (inclusive) for each holdout set
        holdout_ratio: size ratio of each holdout set
            *Note: Even if samples have been sorted chronologically,
                using `holdout_ratio` might have data leakage issue
                because there's no explicit time step identifier
    """

    holdout: np.ndarray

    def __init__(
        self,
        n_splits: int = 1,
        chrono: bool = False,
        ts_col: Optional[str] = None,
        t_start: Optional[List[Union[int, str]]] = None,
        t_end: Optional[List[Union[int, str]]] = None,
        holdout_ratio: float = None,
    ):
        self.n_splits = n_splits
        self.chrono = chrono
        self.ts_col = ts_col
        self.t_start = t_start
        self.t_end = t_end
        self.holdout_ratio = holdout_ratio

        self._check_cfg()

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> None:
        """Assign fold numbers to corresponding indices.

        Parameters:
            X: X set with complete samples
        """
        n_samples = len(X)
        self.holdout = np.full((n_samples, 1), -1, dtype=np.int8)

        if self.chrono:
            if self.ts_col is not None:
                assert self.n_splits == len(self.t_start) == len(self.t_end), (
                    "Parameter `n_splits` must be equal to length of `t_start` "
                    "and length of `t_end`."
                )

                for fold, (ts, te) in enumerate(zip(self.t_start, self.t_end)):
                    mask = X[(X[self.ts_col] >= ts) & (X[self.ts_col]) <= te].index
                    self.holdout[mask, :] = fold
            else:
                holdout_size_fold = floor(len(X) * self.holdout_ratio)
                for fold in range(self.n_splits - 1, -1, -1):
                    ts = -(self.n_splits - fold) * holdout_size_fold
                    if fold == self.n_splits - 1:
                        te = None
                    else:
                        te = -(self.n_splits - fold - 1) * holdout_size_fold
                    self.holdout[ts:te] = fold
        else:
            holdout_size_fold = floor(len(X) * self.holdout_ratio)
            holdout_size = holdout_size_fold * self.n_splits
            if holdout_size > n_samples:
                raise ValueError(
                    "Total number of holdout samples is larger than total number of "
                    "the whole dataset, please decrease `n_splits` or `holdout_ratio`."
                )

            holdout_idx_all = random.sample([_ for _ in range(n_samples)], holdout_size)
            holdout_idx_all = np.reshape(holdout_idx_all, (self.n_splits, -1))
            for fold in range(self.n_splits):
                self.holdout[holdout_idx_all[fold], :] = fold

    def get_n_splits(self) -> int:
        """Return number of splitting iterations in holdout splitter.

        It can be seen as number of outer folds wrapping inner folds of
        cross validator.

        Return:
            n_splits: number of splitting iterations
        """
        return self.n_splits

    def get_holdout(self, fold: int) -> np.ndarray:
        """Return the indices of holdout samples for current fold.

        Parameters:
            fold: current fold number

        Return:
            holdout_idx: indices of holdout samples for current fold
        """
        if fold > np.max(self.holdout):
            fold = 0  # Reset fold number to extract the single holdout set
        holdout_idx = np.where(self.holdout == fold)[0]

        return holdout_idx

    def _check_cfg(self) -> None:
        """Check configuration of holdout splitter."""
        if self.ts_col is not None:
            assert self.chrono, "Parameter `chrono` must be True."
        elif self.holdout_ratio is None:
            raise ValueError("Parameter `holdout_ratio` must be set.")

        if self.n_splits == 1:
            print("Single holdout strategy is used...")
