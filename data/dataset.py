"""
Dataset definitions.
Author: JiaWei Jiang

This file contains definitions of multiple datasets used in different
scenarios.
"""
# Import packages
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from metadata import TARGET, TID


class SinglePtDataset(Dataset):
    """Naive dataset for single point forecasting.

    Parameters:
        data: processed data, including features and predicting target
            *Note: Predicting target is optional, depending on the
                designated processes to run.
        t_window: lookback time window
        horizon: predicting horizon
    """

    _n_samples: int = None
    _X: pd.DataFrame = None
    _y: Optional[pd.Series] = None
    _have_y: bool = False

    def __init__(
        self,
        data: Tuple[pd.DataFrame, Optional[pd.Series]],
        t_window: int,
        horizon: int,
    ):
        self.data = data
        self.t_window = t_window
        self.horizon = horizon
        self.offset = t_window + horizon - 1
        if data[1] is not None:
            self._have_y = True
            self._proc_X_y()
        else:
            self._proc_X()

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        X, y = self._get_windowed_sample(idx)

        if self._have_y:
            return {
                "X": torch.tensor(X, dtype=torch.float32),
                "y": torch.tensor(y, dtype=torch.float32),
            }
        else:
            assert y is None
            return {
                "X": torch.tensor(X, dtype=torch.float32),
            }

    def _proc_X_y(self) -> None:
        """Process X and y data for generating data samples."""
        self._df = self.data[0].copy()
        self._df[TARGET] = self.data[1].values
        self._df = self._df.sort_values(["Capacity", "Date"]).reset_index(drop=True)

        # Assign sample identifiers
        sid, sid_ptr = [], 0
        for cp, gp in self._df.groupby("Capacity"):
            sid_gp = []
            for i in range(len(gp)):
                if i >= self.offset:
                    sid_gp.append(sid_ptr)
                    sid_ptr += 1
                else:
                    sid_gp.append(-1)
            sid += sid_gp
        self._df["SampleId"] = sid
        self._n_samples = (self._df["SampleId"] != -1).sum()

        # Split X and y data
        cols_to_drop = ["SampleId", TID, TARGET]
        self._X = self._df[[c for c in self._df.columns if c not in cols_to_drop]]
        self._y = self._df[TARGET]

    def _proc_X(self) -> None:
        """Process X data for generating testing data samples."""
        self._df = self.data[0].copy()
        self._df = self._df.sort_values(["Capacity", "Date"]).reset_index(drop=True)

        # Adjust sample identifiers
        self._df.rename({"ID": "SampleId"}, axis=1, inplace=True)
        self._df["SampleId"] = self._df["SampleId"] - 1
        self._n_samples = (self._df["SampleId"] != -2).sum()

        # Retrieve X data
        cols_to_drop = ["SampleId", TID]
        self._X = self._df[[c for c in self._df.columns if c not in cols_to_drop]]

    def _get_windowed_sample(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (X, y) sample based on idx passed into __getitem__.

        Parameters:
            idx: index of the sample to retrieve

        Return:
            X: X sample corresponding to the given index
            y: y sample corresponding to the given index
        """
        # Retrieve sample index based on sample identifier
        idx = self._df[self._df["SampleId"] == idx].index[0]
        X = self._X.iloc[idx - self.t_window + 1 : idx + 1].values
        if self._have_y:
            assert self._y is not None
            y = self._y.iloc[idx]

            return X, y
        else:
            return X, None
