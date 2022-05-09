"""
Time series validation schemes.
Author: JiaWei Jiang

This file contains customized time series validators, splitting dataset
following chronological ordering.
"""
import math
from typing import Iterator, Tuple

import numpy as np
import pandas as pd


class TSSplit:
    """Data splitter using the naive train/val split scheme.

    Parameters:
        train_ratio: ratio of training samples
        val_ratio: ratio of validation samples
    """

    def __init__(self, train_ratio: float, val_ratio: float):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Return indices of training and validation sets.

        Because this is the implementation of naive train/val split,
        returned Iterator is the pseudo one.

        Parameters:
            X: raw DataFrame

        Yield:
            tr_idx: training set indices for current split
            val_idx: validation set indices for current split
        """
        n_samples = len(X)
        train_end = math.floor(n_samples * self.train_ratio)
        val_end = train_end + math.floor(n_samples * self.val_ratio)

        tr_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)

        yield tr_idx, val_idx


class GroupTimeSeriesSplit:
    """Data splitter using the naive train/val split scheme. Also,
    grouped values appearing in val set won't exist in training set.


    For more detailed information, please see https://www.kaggle.com/
    competitions/ubiquant-market-prediction/discussion/304036.

    Parameters:
        n_folds: total number of folds
        oof_size: number of time identifiers in oof
        groups: column to group cv folds
    """

    def __init__(self, n_folds: int, oof_size: int, groups: pd.Series):
        self.n_folds = n_folds
        self.holdout_size = oof_size
        self.groups = groups

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Return indices of training and validation sets.

        Parameters:
            X: raw DataFrame

        Yield:
            tr_idx: training set indices for current split
            val_idx: validation set indices for current split
        """
        # Take the group column and get the unique values
        unique_time_ids = np.unique(self.groups.values)

        for fold in range(self.n_folds):
            val_range = (
                -self.holdout_size * (fold + 1),
                -self.holdout_size * fold,
            )
            if val_range[1] == 0:
                val_tids = unique_time_ids[val_range[0] :]
            else:
                val_tids = unique_time_ids[val_range[0] : val_range[1]]
            val_idx = X[X["time_id"].isin(val_tids)].index
            tr_idx = X[X["time_id"] < np.min(val_tids)].index

            yield tr_idx, val_idx
