"""
Data processor definitions.
Author: JiaWei Jiang

This file contains the definition of data processor cleaning and
processing raw data before entering modeling phase. Because data
processing is case-specific, so I leave this part to users to customize
the procedure.
"""
import os
import pickle
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

from metadata import PK, TARGET
from paths import DUMP_PATH
from validation.holdout import HoldoutSplitter

from .fe import FE


class DataProcessor:
    """Data processor processing raw data, and providing access to
    processed data ready to be fed into modeling phase.

    Parameters:
       file_path: path of the raw data
           *Note: File reading supports .parquet extension in default
               setting, which can be modified to customized one.
       dp_cfg: hyperparameters of data processor
    """

    fe: FE
    holdout_splitter: Optional[HoldoutSplitter] = None
    _X: Union[pd.DataFrame, np.ndarray]
    _y: Union[pd.DataFrame, np.ndarray]

    def __init__(self, file_path: str, **dp_cfg: Any):
        self._df = pd.read_csv(file_path)
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
        self._df = self._df.sort_values(PK).reset_index(drop=True)
        #         self._convert_irra_m()   # Deprecated (raw data has been converted)
        if self.drop_outliers is not None:
            self._drop_outliers()
        self._run_fe()

        # Split datasets and holdout
        self._split_X_y()
        self._holdout()

    def run_after_splitting(
        self,
        X_tr: Union[pd.DataFrame, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        fold: int,
    ) -> Tuple[
        Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], object
    ]:
        """Clean and process data after data splitting to avoid data
        leakage issue.

        Parameters:
            X_tr: X training set
            X_val: X validation set
            fold: current fold number

        Return:
            X_tr: processed X training set
            X_val: processed X validation set
            scl: fittet scaler
        """
        print("Run data cleaning and processing after data splitting...")
        scl = None
        if self.scale_cfg["type"] is not None:
            X_tr, X_val, scl = self._scale(X_tr, X_val)
            # Temporar workaround (dumping elsewhere)
            with open(os.path.join(DUMP_PATH, "trafos", f"fold{fold}.pkl"), "wb") as f:
                pickle.dump(scl, f)

        return X_tr, X_val, scl

    def get_df(self) -> Union[pd.DataFrame, np.ndarray]:
        """Return raw or processed DataFrame"""
        return self._df

    def get_X_y(
        self,
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]:
        """Return X set and y set."""
        return self._X, self._y

    def get_cat_feats(self) -> List[str]:
        """Return list of categorical features."""
        return self.fe.get_cat_feats()

    def _setup(self) -> None:
        """Retrieve all parameters specified to process data."""
        # Specify process mode
        if self._dp_cfg.get("infer") is not None:
            self._infer = True
        else:
            self._infer = False

        self.feats = self._dp_cfg["feats"]
        self.fe_cfg = self._dp_cfg["fe"]

        # Before data splitting
        self.drop_outliers = self._dp_cfg["drop_outliers"]
        self.holdout_cfg = self._dp_cfg["holdout"]

        # After data splitting
        self.scale_cfg = self._dp_cfg["scale"]

    def _convert_irra_m(self) -> None:
        """Convert the unit of `Irradiance_m` from Wh/m2 to MJ/m2."""
        print("Convert unit of `Irradiance_m`...")
        self._df["Irradiance_m"] = self._df["Irradiance_m"] / 1000 * 3.6

    def _drop_outliers(self) -> None:
        """Drop explicit outliers."""
        ols = []
        if self.drop_outliers == "top3":
            ol1 = self._df[self._df[TARGET] == 6752].index  # 314.88S 21/7/19
            ol2 = self._df[self._df[TARGET] == 3765].index  # 492.8S 21/1/27
            ol3 = self._df[self._df[TARGET] == 3187].index  # 438.3N 21/9/10
            for ol in [ol1, ol2, ol3]:
                if len(ol) == 0:
                    continue
                ols.append(ol[0])
        elif self.drop_outliers == "period":
            weird_period = (self._df["Date"] >= "2021-09-09") & (
                self._df["Date"] <= "2021-10-07"
            )
            ols = self._df[
                (self._df["Capacity"] == 438.3) & weird_period
            ].index.tolist()

        print(f"Start dropping {len(ols)} outliers...")
        self._df = self._df.drop(ols, axis=0).reset_index(drop=True)
        print("Done.")

    def _run_fe(self) -> None:
        """Setup feature engineer, and run feature engineering."""
        self.fe_cfg["infer"] = self._infer
        self.fe = FE(**self.fe_cfg)

        print("Start feature engineering...")
        self._df = self.fe.run(self._df)
        print("Done.")

    def _split_X_y(self) -> None:
        """Split data into X and y sets."""
        feats = self.feats
        # Add newly engineered features into feature set
        # (note that the ordering of df columns matters)
        for ft in self.fe.get_eng_feats():
            if ft not in feats:
                feats.append(ft)

        print("Start splitting X and y set...")
        print(f"Feature set:\n{feats}")
        #         self._X = self._df[self.feats]   # DL workaround (Date for Dataset)
        self._X = self._df[[f for f in feats if f != "Date"]]
        # Modify to Irradiance_m or Temp_m to train imputers
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

    # After data splitting
    def _scale(
        self,
        X_tr: Union[pd.DataFrame, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], Any]:
        """Scale numeric features.

        Support only pd.DataFrame now.

        Return:
            X_tr: scaled X training set
            X_val: scaled X validation set
            scl: fittet scaler
        """
        assert isinstance(X_tr, pd.DataFrame) and isinstance(X_val, pd.DataFrame)

        scl_type = self.scale_cfg["type"]
        cols_to_trafo = self.scale_cfg["cols"]

        if scl_type == "minmax":
            scl = MinMaxScaler()
        elif scl_type == "standard":
            scl = StandardScaler()
        elif scl_type == "quantile":
            n_quantiles = self.scale_cfg["n_quantiles"]
            scl = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="normal",
                random_state=168,
            )

        if cols_to_trafo == []:
            cols_to_trafo = _get_numeric_cols(X_tr)

        print(
            f"Start scaling features using {scl_type} trafo...\n"
            f"Feature list:\n{cols_to_trafo}"
        )
        X_tr[cols_to_trafo] = scl.fit_transform(X_tr[cols_to_trafo])
        X_val[cols_to_trafo] = scl.transform(X_val[cols_to_trafo])
        print("Done.")

        X_tr.fillna(0, inplace=True)
        X_val.fillna(0, inplace=True)

        return X_tr, X_val, scl


def _get_numeric_cols(df: pd.DataFrame) -> List[str]:
    """Return numeric column names.

    Parameters:
        df: data

    Return:
        numeric_cols: list of numeric column names.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    return numeric_cols
