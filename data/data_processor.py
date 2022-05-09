"""
Data processor definitions.
Author: JiaWei Jiang

This file contains the definition of data processor cleaning and
processing raw data before entering modeling phase. Because data
processing is case-specific, so I leave this part to users to customize
the procedure.
"""
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd

from metadata import COLS, TARGET
from validation.holdout import HoldoutSplitter


class DataProcessor:
    """Data processor processing raw data, and providing access to
    processed data ready to be fed into modeling phase.

    Parameters:
       file_path: path of the raw data
           *Note: File reading supports .parquet extension in default
               setting, which can be modified to customized one.
       dp_cfg: hyperparameters of data processor
    """

    holdout_splitter: Optional[HoldoutSplitter] = None
    _X: Union[pd.DataFrame, np.ndarray]
    _y: Union[pd.DataFrame, np.ndarray]

    def __init__(self, file_path: str, **dp_cfg: Any):
        #         self._df = pd.read_csv(file_path, parse_dates=["Date"])[COLS]
        self._df = pd.read_csv(file_path)[COLS]
        self._dp_cfg = dp_cfg
        self._setup()

    def run_before_cv(self) -> None:
        """Clean and process data before cross validation process.

        Holdout set splitting is also done in this process if holdout
        strategy is specified.

        Return:
            None
        """
        print("Run data cleaning and processing before data splitting...")
        self._df = self._df.sort_values("Date").reset_index(drop=True)
        self._split_X_y()
        self._holdout()

    def run_after_splitting(
        self,
        df_tr: Union[pd.DataFrame, np.ndarray],
        df_val: Union[pd.DataFrame, np.ndarray],
        fold: int,
    ) -> Tuple[
        Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], object
    ]:
        """Clean and process data after data splitting to avoid data
        leakage issue.

        Parameters:
            df_tr: training data
            df_val: validation data
            fold: current fold number

        Return:
            df_tr: processed training data
            df_val: processed validation data
            scaler: scaling object
        """
        print("Run data cleaning and processing after data splitting...")
        scaler = None

        return df_tr, df_val, scaler

    def get_df(self) -> Union[pd.DataFrame, np.ndarray]:
        """Return raw or processed DataFrame"""
        return self._df

    def get_X_y(
        self,
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]:
        """Return X set and y set."""
        return self._X, self._y

    def _setup(self) -> None:
        """Retrieve all parameters specified to process data."""
        # Before data splitting
        self.holdout_cfg = self._dp_cfg["holdout"]

        # After data splitting

    def _split_X_y(self) -> None:
        print("Start splitting X and y set...")
        self._X = self._df[
            [
                "Lat",
                "Lon",
                "Irradiance",
                "Temp",
                "Capacity",
                "Angle",
                "Irradiance_m",
                "Temp_m",
            ]
        ]
        self._y = self._df[TARGET]
        print("Done.")

    def _holdout(self) -> None:
        """Setup holdout splitter, and split the holdout sets."""
        holdout_n_splits = self.holdout_cfg["n_splits"]
        if holdout_n_splits == 0:
            print(
                "Holdout set splitting is disabled, so no local unseen "
                "testing data is used in evaluation process."
            )
        else:
            self.holdout_splitter = HoldoutSplitter(**self.holdout_cfg)

            print("Start splitting holdout sets for final evaluation...")
            self.holdout_splitter.split(self._X)
            print("Done.")
