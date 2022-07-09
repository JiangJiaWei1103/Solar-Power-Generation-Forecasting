"""
Merge data.
Author: JiaWei Jiang
"""
import os

import pandas as pd

from metadata import AQ_STA, IRRA_STA, PK, TEMP_STA
from paths import RAW_DATA_PATH

# Variable definitions
CAP2STA_TEMP = {c: sta for sta, cap in TEMP_STA.items() for c in cap}
CAP2STA_IRRA = {c: sta for sta, cap in IRRA_STA.items() for c in cap}
CAP2STA_AQ = {c: sta for sta, cap in AQ_STA.items() for c in cap}


def merge_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Merge input DataFrame with weather data.

    Parameters:
        df: input DataFrame

    Return:
        df_merged: merged DataFrame
    """
    # Load weather data
    df_weather_codis = pd.read_csv("./data/processed/weather_codis.csv")
    df_weather_cwb = pd.read_csv("./data/processed/weather_cwb.csv")

    df_merged = df.copy()
    df_merged["TempSta"] = df_merged["Capacity"].map(CAP2STA_TEMP)
    df_merged["IrraSta"] = df_merged["Capacity"].map(CAP2STA_IRRA)
    df_merged = df_merged.merge(
        df_weather_codis,
        "left",
        left_on=["Date", "TempSta"],
        right_on=["Date", "StaName"],
        suffixes=("", "_t"),
    )
    df_merged = df_merged.merge(
        df_weather_cwb,
        "left",
        left_on=["Date", "IrraSta"],
        right_on=["Date", "StaName"],
        suffixes=("", "_i"),
    )

    return df_merged


def merge_aq(df: pd.DataFrame) -> pd.DataFrame:
    """Merge input DataFrame with air quality data.

    Parameters:
        df: input DataFrame

    Return:
        df_merged: merged DataFrame
    """
    # Load air quality data
    df_aq = pd.read_csv("./data/processed/aq.csv")

    df_merged = df.copy()
    df_merged["AQSta"] = df_merged["Capacity"].map(CAP2STA_AQ)
    df_merged = df_merged.merge(
        df_aq,
        "left",
        left_on=["Date", "AQSta"],
        right_on=["Date", "Station"],
    )

    return df_merged


def merge_pv(df: pd.DataFrame) -> pd.DataFrame:
    """Merge input DataFrame with pv data.

    Parameters:
        df: input DataFrame

    Return:
        df_merged: merged DataFrame
    """
    # Load pv data
    df_pv = pd.read_csv("./data/processed/pv.csv")

    df_merged = df.copy()
    df_merged = df_merged.merge(
        df_pv,
        "left",
        left_on=PK,
        right_on=PK,
    )

    return df_merged


def main() -> None:
    # Load training and tesing sets
    train = pd.read_csv(os.path.join(RAW_DATA_PATH, "train.csv"), parse_dates=["Date"])
    test = pd.read_csv(os.path.join(RAW_DATA_PATH, "test.csv"), parse_dates=["Date"])

    train["Date"] = train["Date"].astype(str)
    test["Date"] = test["Date"].astype(str)

    # Merge weather data
    train = merge_weather(train)
    test = merge_weather(test)

    # Merge air quality data
    train = merge_aq(train)
    test = merge_aq(test)

    # Merge pv data
    train = merge_pv(train)
    test = merge_pv(test)

    # Dump merged data
    train.to_csv("./data/processed/train.csv", index=False)
    test.to_csv("./data/processed/test.csv", index=False)


if __name__ == "__main__":
    main()
