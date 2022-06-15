"""
Feature engineer.
Author: JiaWei Jiang
"""
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from metadata import MODULE_META, PK
from paths import OOF_META_FEATS_PATH, TEST_META_FEATS_PATH


class FE:
    """Feature engineer.

    Parameters:
        add_month: whether to add month indicator
        add_module_meta: whether to add metadata of generator module
        label_enc: list of features interpreted as categorical features
        mine_temp: list of temperature-related features
        mine_irrad: list of irradiance-related features
        meta_feats: list of well-trained model versions
            *Note: Meta features are used for stacking or restacking.
                Model versions indicate the corresponding versions of
                predicting results.
        knn_meta_feats: list of well-trained model versions with k
            *Note: kNN meta features are used for pseudo stacking or
                restacking. Model versions indicate the corresponding
                versions of predicting results.
        infer: whether the process is in inference mode
    """

    MV2EID = {
        "l5": "lgbm-hjc3rp0j",
        "l6": "lgbm-54or6r30",
        "x10": "xgb-f9ahqzut",
    }  # Base model version to corresponding experiment identifier
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
        meta_feats: List[str],
        knn_meta_feats: Dict[str, Any],
        infer: bool = False,
    ):
        self.add_month = add_month
        self.add_module_meta = add_module_meta
        self.label_enc = label_enc
        self.mine_temp = mine_temp
        self.mine_irrad = mine_irrad
        self.meta_feats = meta_feats
        self.knn_meta_feats = knn_meta_feats

        self.infer = infer

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
        if self.meta_feats != []:
            self._add_meta_feats()
        if self.knn_meta_feats != {}:
            self._add_knn_meta_feats()

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
        print("Done.")

        self._eng_feats += self.mine_temp
        temp_feats_to_drop = [f for f in temp_feats if f not in self.mine_temp]
        self._df.drop(temp_feats_to_drop, axis=1, inplace=True)

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
        print("Done.")

        self._eng_feats += self.mine_irrad
        irrad_feats_to_drop = [f for f in irrad_feats if f not in self.mine_irrad]
        self._df.drop(irrad_feats_to_drop, axis=1, inplace=True)

    def _add_meta_feats(self) -> None:
        """Add meta features for stacking or restacking."""
        if self.infer:
            # Testing prediction is used
            meta_feats = pd.read_csv(TEST_META_FEATS_PATH)
        else:
            # Unseen prediction is used
            meta_feats = pd.read_csv(OOF_META_FEATS_PATH)

        print("Adding meta features...")
        meta_cols = []
        for model_v in self.meta_feats:
            meta_cols.append(self.MV2EID[model_v])
        meta_feats = meta_feats[PK + meta_cols]

        #         for meta_col in meta_cols:
        #             meta_feats[meta_col] = (meta_feats[meta_col]
        #                                     / meta_feats["Capacity"])

        self._df = self._df.merge(meta_feats, how="left", on=PK, validate="1:1")
        print("Done.")

        self._eng_feats += meta_cols

    def _add_knn_meta_feats(self) -> None:
        """Add meta features from kNN.

        Illustration of kNN meta column conversion:
            {
                "l5": 2,
                "l6": 3,
                ...
                <model version>: k
            }
            -> {
                "lgbm-hjc3rp0j": 2,
                "lgbm-54or6r30": 3,
                ...
                <experiment identifier>: k
            }
        """
        if self.infer:
            # Testing prediction is used
            meta_feats = pd.read_csv(TEST_META_FEATS_PATH)
        else:
            # Unseen prediction is used
            meta_feats = pd.read_csv(OOF_META_FEATS_PATH)

        # Load geographic kNN of each generator
        with open("./data/processed/gen_geo_knn.pkl", "rb") as f:
            geo_knn = pickle.load(f)

        print("Adding kNN meta features...")
        for model_v, k in self.knn_meta_feats.items():
            self.knn_meta_feats[self.MV2EID[model_v]] = self.knn_meta_feats.pop(model_v)

        knn_meta_cols = [
            f"{meta_col}_n{i}"
            for meta_col, k in self.knn_meta_feats.items()
            for i in range(k)
        ]
        knn_meta_dict: Dict[str, List[float]] = {col: [] for col in knn_meta_cols}
        for i, r in self._df.iterrows():
            meta_feats_date = meta_feats[meta_feats["Date"] == r["Date"]]
            cap = str(r["Capacity"])

            for meta_col, k in self.knn_meta_feats.items():
                knn = geo_knn[cap][:k]

                for i, cap_ in enumerate(knn):
                    cap_ = float(cap_)
                    df_knn = meta_feats_date[meta_feats_date["Capacity"] == cap_]

                    knn_meta_col = f"{meta_col}_n{i}"
                    if len(df_knn) != 0:
                        knn_meta_dict[knn_meta_col].append(
                            df_knn[meta_col].values[0]  # / cap_
                        )
                    else:
                        knn_meta_dict[knn_meta_col].append(np.nan)

        knn_meta_df = pd.DataFrame.from_dict(knn_meta_dict)
        self._df = pd.concat([self._df, knn_meta_df], axis=1)
        print("Done.")

        self._eng_feats += knn_meta_cols
