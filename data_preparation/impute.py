"""
Fix missing issue.
Author: JiaWei Jiang
"""
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import yaml

from data.fe import FE
from metadata import CLUST

# Variable definitions
TR_N_TO_IMPUTE = [
    "Temp",
    "Irradiance",
    "RH",
    "WSGust",
    "WDGust",
    "RHMin",
    "TempMin",
    "TempMax",
    "StnPresMin",
    "StnPresMax",
]
TR_S_TO_IMPUTE = [
    "Temp",
    "Irradiance",
    "StnPres",
    "StnPresMax",
    "StnPresMin",
    "TempMax",
    "TempMin",
    "RH",
    "RHMin",
    "WS",
    "WD",
    "WSGust",
    "WDGust",
    "Precp",
]
TEST_TO_IMPUTE = [
    "Temp",
    "WS",
    "WD",
    "WSGust",
    "WDGust",
    "RH",
    "RHMin",
    "TempMin",
    "TempMax",
    "StnPres",
    "StnPresMin",
    "StnPresMax",
    "Precp",
]


def impute_weather(df: pd.DataFrame, to_impute: List[str]) -> pd.DataFrame:
    """Impute weather related data.

    Parameters:
        df: input DataFrame
        to_impute: weather columns to impute

    Reture:
        df_imputed: imputed DataFrame
    """
    df_imputed = df.copy()

    for col in to_impute:
        col_na_idx = df_imputed[col].isna()
        df_imputed.loc[col_na_idx, col] = df_imputed.loc[col_na_idx, f"{col}_i"]

    return df_imputed


def impute_with_lgb(df: pd.DataFrame, to_impute: str) -> pd.DataFrame:
    """Impute missing `Temp_m` or `Irradiace_m` with pre-trained
    imputers.

    Parameters:
        df: input DataFrame
        to_impute: column to impute

    Reture:
        df_imputed: imputed DataFrame
    """
    df_imputed = df.copy()
    imputer_path = os.path.join("./data/imputers", to_impute)

    # Generate X set
    with open(os.path.join(imputer_path, "config/dp.yaml"), "r") as f:
        dp_cfg = yaml.full_load(f)
    feat_eng = FE(**dp_cfg["fe"])
    X = feat_eng.run(df_imputed)
    feats = [f for f in dp_cfg["feats"] if f != "Date"] + ["Month"]
    X = X[feats]

    # Load pre-trained imputers
    models = []
    model_path = os.path.join(imputer_path, "models", "whole")
    for model_file in sorted(os.listdir(model_path)):
        with open(os.path.join(model_path, model_file), "rb") as f:  # type: ignore
            models.append(pickle.load(f))  # type: ignore

    # Start inference
    pred = np.zeros(len(df_imputed))
    for model in models:
        pred += model.predict(X) / 5

    # Impute
    df_imputed[to_impute] = pred

    return df_imputed


def main() -> None:
    # Load training and tesing sets
    train = pd.read_csv("./data/processed/train.csv")
    test = pd.read_csv("./data/processed/test.csv")

    train["Location"] = train["Lat"].astype(str) + "-" + train["Lon"].astype(str)
    test["Location"] = test["Lat"].astype(str) + "-" + test["Lon"].astype(str)

    # Convert unit of `Irradiance`
    train["Irradiance_m"] = train["Irradiance_m"] / 1000 * 3.6
    test["Irradiance_m"] = test["Irradiance_m"] / 1000 * 3.6

    # Impute `Temp`
    train["Temp"] = train["Temp"].fillna(train["Temp_t"])
    test["Temp"] = test["Temp"].fillna(test["Temp_t"])

    # Split training set into different clusters
    train_n = train[train["Capacity"].isin(CLUST["N"])].reset_index(drop=True)
    train_s = train[train["Capacity"].isin(CLUST["S"])].reset_index(drop=True)

    # Impute weather data
    train_n = impute_weather(train_n, TR_N_TO_IMPUTE)
    train_s = impute_weather(train_s, TR_S_TO_IMPUTE)
    test = impute_weather(test, TEST_TO_IMPUTE)

    # Split training set into different clusters
    test_n = test[test["Capacity"].isin(CLUST["N"])].reset_index(drop=True)
    test_s = test[test["Capacity"].isin(CLUST["S"])].reset_index(drop=True)

    # Impute `Temp_m` and `Irradiance_m`
    train_n = impute_with_lgb(train_n, "Temp_m")
    train_n = impute_with_lgb(train_n, "Irradiance_m")
    test_n = impute_with_lgb(test_n, "Temp_m")
    test_n = impute_with_lgb(test_n, "Irradiance_m")

    # Concatenate data from different clusters
    train = pd.concat([train_n, train_s], ignore_index=True)
    test = pd.concat([test_n, test_s], ignore_index=True)

    # Correct `Temp`
    train["TempDiff"] = (train["Temp"] - train["Temp_t"]) / train["Temp_t"]
    train.loc[train["TempDiff"] > 0.01, "Temp"] = (
        train.loc[train["TempDiff"] > 0.01, ["Temp", "Temp_t"]].sum(axis=1) / 2
    )
    train.drop(["TempDiff"], axis=1, inplace=True)

    # Dump
    train.to_csv("./data/processed/train.csv", index=False)
    test.to_csv("./data/processed/test.csv", index=False)


if __name__ == "__main__":
    main()
