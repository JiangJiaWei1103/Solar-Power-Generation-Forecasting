"""
Process weather data.
Author: JiaWei Jiang

Raw data is scraped from https://e-service.cwb.gov.tw/HistoryDataQuery/
and https://agr.cwb.gov.tw/NAGR/history/station_day.
"""
import os

import numpy as np
import pandas as pd

from paths import RAW_DATA_PATH

# Variable definitions
WEATHER_DATA_DIR = os.path.join(RAW_DATA_PATH, "weather")
COLS_TO_DROP_CODIS = [
    "ObsTime",
    "T Min Time",
    "T Max Time",
    "StnPresMinTime",
    "StnPresMaxTime",
    "PrecpMax10Time",
    "PrecpMax60Time",
    "RHMinTime",
    "WGustTime",
    "UVI Max Time",
    "month",
    "Td dew point",
    "SeaPres",
    "PrecpHour",
    "PrecpMax10",
    "PrecpMax60",
    "SunShine",
    "SunShineRate",
    "GloblRad",
    "VisbMean",
    "EvapA",
    "UVI Max",
    "Cloud Amount",
]
COLS_TO_DROP_CWB = [
    "ObsTime",
    "T Min Time",
    "T Max Time",
    "StnPresMinTime",
    "StnPresMaxTime",
    "PrecpMax10Time",
    "PrecpMax60Time",
    "RHMinTime",
    "WGustTime",
    "UVI Max Time",
    "month",
    "EvapA",
]
FINAL_COLS_CODIS = [
    "StnPres",
    "StnPresMax",
    "StnPresMin",
    "Temp",
    "TempMax",
    "TempMin",
    "RH",
    "RHMin",
    "WS",
    "WD",
    "WSGust",
    "WDGust",
    "Precp",
    "StaName",
    "Date",
]
FINAL_COLS_CWB = [
    "StnPres",
    "SeaPres",
    "StnPresMax",
    "StnPresMin",
    "Temp",
    "TempMax",
    "TempMin",
    "TdDewPoint",
    "RH",
    "RHMin",
    "WS",
    "WD",
    "WSGust",
    "WDGust",
    "Precp",
    "PrecpMax10",
    "PrecpMax60",
    "SunShine",
    "Irradiance",
    "StaName",
    "Date",
]
TO_REPLACE = {"...\xa0": np.nan, "X\xa0": np.nan, "T\xa0": np.nan}


def proc_weather_cwb_codis(df: pd.DataFrame) -> pd.DataFrame:
    """Process weather data scraped from CWB (CODIS).

    Parameters:
        df: raw weather data

    Return:
        df: processed weather data
    """
    df = df.copy()

    df["Date"] = df["month"] + "-" + df["ObsTime"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df.replace(TO_REPLACE, inplace=True)
    df.drop(COLS_TO_DROP_CODIS, axis=1, inplace=True)

    df.columns = FINAL_COLS_CODIS

    return df


def proc_weather_cwb(df: pd.DataFrame, sta_name: str) -> pd.DataFrame:
    """Process weather data downloaded from CWB.

    Parameters:
        df: raw weather data

    Return:
        df: processed weather data
    """
    df = df.copy()

    df.columns = df.iloc[0]
    cols_to_drop = [col for col in df.columns if col in COLS_TO_DROP_CWB] + ["StaCode"]
    df.drop(cols_to_drop, axis=1, inplace=True)
    df.drop(0, inplace=True)
    df.replace(TO_REPLACE, inplace=True)

    df["StaName"] = sta_name
    df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d")

    cols_ordered = [col for col in df.columns if col != "Date"] + ["Date"]
    df = df[cols_ordered]
    df.columns = FINAL_COLS_CWB

    return df


def main() -> None:
    df_weather_codis = []
    df_weather_cwb = []

    for platform, sta_names in {
        "codis": ["xx", "xs", "xw", "lz"],
        "cwb": ["aic", "tca", "tya"],
    }.items():
        for sta_name in sta_names:
            if platform == "codis":
                df = pd.read_csv(os.path.join(WEATHER_DATA_DIR, f"{sta_name}.csv"))
                df = proc_weather_cwb_codis(df)
                df_weather_codis.append(df)
            else:
                df = pd.read_csv(
                    os.path.join(WEATHER_DATA_DIR, f"{sta_name}.csv"), encoding="cp1252"
                )
                df = proc_weather_cwb(df, sta_name)
                df_weather_cwb.append(df)

    df_weather_codis = pd.concat(df_weather_codis, ignore_index=True)
    df_weather_cwb = pd.concat(df_weather_cwb, ignore_index=True)

    # Convert dtypes
    dtypes = {
        col: float for col in df_weather_codis.columns if col not in ["StaName", "Date"]
    }
    df_weather_codis = df_weather_codis.astype(dtypes)
    dtypes = {
        col: float for col in df_weather_cwb.columns if col not in ["StaName", "Date"]
    }
    df_weather_cwb = df_weather_cwb.astype(dtypes)

    # Dump processed data
    df_weather_codis.to_csv("./data/processed/weather_codis.csv", index=False)
    df_weather_cwb.to_csv("./data/processed/weather_cwb.csv", index=False)


if __name__ == "__main__":
    main()
