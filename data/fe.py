"""
Feature engineer.
Author: JiaWei Jiang
"""
import pickle
from typing import List

import pandas as pd

from metadata import MODULE_META


class FE:
    """Feature engineer.

    Parameters:
        add_month: whether to add month indicator
        label_enc: list of features interpreted as categorical features
        mine_temp: whether to mine temperature-related features
    """

    EPS: float = 1e-7
    _df: pd.DataFrame = None
    _eng_feats: List[str] = []
    _cat_feats: List[str] = []

    def __init__(
        self,
        add_month: bool,
        add_module_meta: bool,
        label_enc: List[str],
        mine_temp: List[str],
        mine_irrad: List[str],
    ):
        self.add_month = add_month
        self.add_module_meta = add_module_meta
        self.label_enc = label_enc
        self.mine_temp = mine_temp
        self.mine_irrad = mine_irrad

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run feature engineering.

        Parameters:
            df: input DataFrame

        Return:
            self._df: DataFrame with engineered features
        """
        self._df = df.copy()

        if self.add_month:
            self._add_month()
        if self.add_module_meta:
            self._add_module_meta()
        if self.label_enc != []:
            self._encode_pseudo_cat()
        if self.mine_temp != []:
            self._mine_temp()
        if self.mine_irrad != []:
            self._mine_irrad()

        return self._df

    def get_eng_feats(self) -> List[str]:
        """Return list of all engineered features."""
        return self._eng_feats

    def get_cat_feats(self) -> List[str]:
        """Return list of categorical features."""
        return self._cat_feats

    def _add_month(self) -> None:
        """Add month indicator."""
        print("Adding month indicator...")
        self._df["Month"] = pd.to_datetime(self._df["Date"], format="%Y-%m-%d").dt.month
        print("Done.")

        self._eng_feats.append("Month")
        self._cat_feats.append("Month")

    def _add_module_meta(self) -> None:
        """Add metadata of generator module."""
        for feat, meta_map in MODULE_META.items():
            self._df[feat] = self._df["Module"].map(meta_map)
            self._eng_feats.append(feat)

    def _encode_pseudo_cat(self) -> None:
        """Apply label encoder on pseudo categorical features."""
        print(f"Encoding pseudo categorical features {self.label_enc}...")
        for feat in self.label_enc:
            with open(f"./data/trafos/label_enc/{feat}.pkl", "rb") as f:
                enc = pickle.load(f)
                self._df[feat] = enc.transform(self._df[feat])
        print("Done.")

        self._eng_feats += self.label_enc
        self._cat_feats += self.label_enc

    def _mine_temp(self) -> None:
        """Mine temperature-related features."""
        temp_feats = [
            "TempRange",
            "TempMax2Avg",
            "TempAvg2Min",
            "Temp_m2Temp",
            "TempRangeRatio",
            "TempMax2AvgRatio",
            "TempAvg2MinRatio",
            "Temp_m2TempRatio",
        ]

        print("Mining temperature-related features...")
        # Difference
        self._df["TempRange"] = self._df["TempMax"] - self._df["TempMin"]
        self._df["TempMax2Avg"] = self._df["TempMax"] - self._df["Temp"]
        self._df["TempAvg2Min"] = self._df["Temp"] - self._df["TempMin"]
        self._df["Temp_m2Temp"] = self._df["Temp_m"] - self._df["Temp"]
        # Ratio
        self._df["TempRangeRatio"] = self._df["TempRange"] / (
            self._df["TempMin"].abs() + self.EPS
        )
        self._df["TempMax2AvgRatio"] = self._df["TempMax2Avg"] / (
            self._df["Temp"].abs() + self.EPS
        )
        self._df["TempAvg2MinRatio"] = self._df["TempAvg2Min"] / (
            self._df["TempMin"].abs() + self.EPS
        )
        self._df["Temp_m2TempRatio"] = self._df["Temp_m2Temp"] / (
            self._df["Temp"].abs() + self.EPS
        )

        self._eng_feats += self.mine_temp
        temp_feats_to_drop = [f for f in temp_feats if f not in self.mine_temp]
        self._df.drop(temp_feats_to_drop, axis=1, inplace=True)
        print("Done.")

    def _mine_irrad(self) -> None:
        """Mine irradiance-related features."""
        irrad_feats = [
            "Irrad_m2Irrad",
            "Irrad_m2IrradRatio",
        ]
        print("Mining irradiance-related features...")
        # Difference
        self._df["Irrad_m2Irrad"] = self._df["Irradiance_m"] - self._df["Irradiance"]
        # Ratio
        self._df["Irrad_m2IrradRatio"] = self._df["Irrad_m2Irrad"] / (
            self._df["Irradiance"].abs() + self.EPS
        )

        self._eng_feats += self.mine_irrad
        irrad_feats_to_drop = [f for f in irrad_feats if f not in self.mine_irrad]
        self._df.drop(irrad_feats_to_drop, axis=1, inplace=True)
        print("Done.")
