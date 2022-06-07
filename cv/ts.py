"""
Time series validation schemes.
Author: JiaWei Jiang

This file contains customized time series validators, splitting dataset
following chronological ordering.
"""
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import indexable


class GPTSSplit(_BaseKFold):
    """Pseudo group time series cross validator.

    Note the issue that some of generators can only exist in validation
    set, because they were monitored near the end of the training set
    (e.g., generator with capacity 343.2 is recorded from 2021/10/09 to
    2021/10/28 in training set, accounting for only 20 samples.).

    Parameters:
        n_splits: number of splits (folds)
        oof_time_step: number of time steps (days in this case) in each
            validation fold
        max_train_time_step: max number of time steps in each training
            fold in rolling manner
            *Note: forward chaining is enabled if it's not set
    """

    def __init__(
        self,
        n_splits: int = 3,
        n_time_step_oof: int = 112,
        max_train_n_time_step: int = 0,
    ):
        if n_splits == 1:
            # Group time series split for single-fold scenario
            self.n_splits = n_splits
            self.shuffle = False
            self.random_state = None
        else:
            super().__init__(n_splits, shuffle=False, random_state=None)
        self.n_time_step_oof = n_time_step_oof
        self.max_train_n_time_step = max_train_n_time_step

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and validation
        sets.

        Parameters:
            X: training data
            y: always ignored, exists for compatibility
            groups: group labels (time identifiers in this case)

        Yileds:
            tr_idx: training set indices for current fold
            val_idx: validation set indices for current fold
        """
        X, y, groups = indexable(X, y, groups)
        t_unique = sorted(np.unique(groups))
        t_nunique = len(t_unique)
        tss_val = range(
            t_nunique - self.n_time_step_oof * self.n_splits,
            t_nunique,
            self.n_time_step_oof,
        )  # Start time index for validation set

        for ts_val in tss_val:
            te_val = ts_val + self.n_time_step_oof
            val_dates = t_unique[ts_val:te_val]
            tr_idx = groups[groups < t_unique[ts_val]].index
            val_idx = groups[groups.isin(val_dates)].index

            yield tr_idx, val_idx
