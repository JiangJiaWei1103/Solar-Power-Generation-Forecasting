"""
Process air quality data.
Author: JiaWei Jiang

Raw data is downloaded from https://airtw.epa.gov.tw/.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as kurt

from paths import RAW_DATA_PATH

# Variable definitions
AQ_DATA_DIR = os.path.join(RAW_DATA_PATH, "air_quality")
STATIONS = ["ty", "ki", "dy", "xx", "ch"]
COLS = [
    "SiteID",
    "SiteName",
    "County",
    "ItemID",
    "ItemNameCh",
    "ItemName",
    "ItemUnit",
    "Datetime",
    "Value",
]
COLS_TO_DROP = ["SiteName", "SiteID", "County", "ItemID", "ItemNameCh", "ItemUnit"]
AQ_ITEMS = ["CO", "NO2", "O3", "PM10", "PM2.5", "SO2"]
STATS = {
    "Mean": np.nanmean,
    "Min": np.min,
    "Max": np.max,
    "Median": np.median,
    "Std": np.std,
    "Skew": "skew",
    "Kurt": kurt,
}


def get_dy_2021_12() -> pd.DataFrame:
    """Process and return data from station `dy` in Dec, 2021.

    Data from station `dy` in Dec, 2021 has downloading issue, so the
    file is processed independently.

    Return:
        df: processed DataFrame from `dy` in Dec, 2021.
    """
    df = pd.read_csv(os.path.join(AQ_DATA_DIR, "dy", "2021-12.csv"))
    df.replace(
        {
            "#                              ": np.nan,
            "A                              ": np.nan,
            "x                              ": np.nan,
            "*                              ": np.nan,
        },
        inplace=True,
    )
    df[AQ_ITEMS] = df[AQ_ITEMS].astype("float32")
    df = df[df["Date"] >= "2021-11-31"]

    return df


def cal_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and return daily stats.

    Parameters:
        df: hourly air quality data

    Return:
        df_stats: daily stats of air quality data
    """
    stats = list(STATS.values())
    stats_name = list(STATS.keys())

    df_stats = df.groupby(["Date", "Station"]).agg(
        {
            "CO": stats,
            "NO2": stats,
            "O3": stats,
            "PM10": stats,
            "PM2.5": stats,
            "SO2": stats,
        }
    )
    df_stats.columns = [
        f"{item}{stn}" for item in df_stats.columns.levels[0] for stn in stats_name
    ]
    df_stats.reset_index(inplace=True)
    df_stats["Date"] = df_stats["Date"].astype(str)

    return df_stats


def main() -> None:
    df = []
    for sta in STATIONS:
        sta_path = os.path.join(AQ_DATA_DIR, sta)

        df_sta = []
        for i, file in enumerate(sorted(os.listdir(sta_path))):
            file_path = os.path.join(sta_path, file)
            if not file_path.endswith("csv") or file == "2021-12.csv":
                continue

            data = pd.read_csv(file_path)
            data.columns = COLS
            data.drop(COLS_TO_DROP, axis=1, inplace=True)
            data.drop_duplicates(keep="first", inplace=True, ignore_index=True)
            data = data.pivot(index="Datetime", columns="ItemName", values="Value")

            df_sta.append(data)

        df_sta = pd.concat(df_sta)
        df_sta = df_sta.replace({"x": np.nan}).astype("float32")
        df_sta.index = pd.to_datetime(df_sta.index, format="%Y-%m-%d %H:%M:%S")
        df_sta["Date"] = df_sta.index.date
        df_sta["Hour"] = df_sta.index.hour
        df_sta["Station"] = sta

        df.append(df_sta)

    df = pd.concat(df)
    df = pd.concat([df, get_dy_2021_12()], ignore_index=True)
    df_stats = cal_stats(df)

    # Dump processed data
    df_stats.to_csv("./data/processed/aq.csv", index=False)


if __name__ == "__main__":
    main()
