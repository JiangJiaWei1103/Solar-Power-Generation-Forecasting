# type: ignore
"""
PV LightHouse scraper.
Author: JiaWei Jiang
"""
import json
from time import sleep
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Variable definitions
URL = (
    "https://www2.pvlighthouse.com.au/calculators/solar%20path%20calculator/"
    "solar%20path%20calculator.aspx"
)
HEADER = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) "
        "Gecko/20100101 Firefox/88.0"
    ),
    "X-Requested-With": "XMLHttpRequest",
}
with open("./data/pv_lighthouse.json", "r") as f:
    PAYLOAD_ST = json.load(f)


def time2sec(t: List[str]) -> float:
    """Convert time second-unit representation.

    Parameters:
        t: raw time representation with format [hr, min, sec]

    Return:
        sec: total seconds
    """
    sec = 3600 * int(t[0]) + 60 * int(t[1]) + int(t[2])

    return sec


def main() -> None:
    df = pd.read_csv("./data/processed/0603/complete.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df_scraped = pd.DataFrame()

    sun_light_hr = []
    sun_rise = []
    sun_set = []
    zenith_angle = []
    air_mass_noon = []
    for i, r in tqdm(df.iterrows(), desc="Scraping"):
        # Setup payload
        lat, lon = r["Lat"], r["Lon"]
        year, month, day = r["Date"].year, r["Date"].month, r["Date"].day
        tilt_angle = lat + 15
        azimuth_angle = r["Angle"] + 180
        payload_dynamic = {
            "TabContainer1$TabPanel1$tbLatitude": f"{lat}",
            "TabContainer1$TabPanel1$tbLongitude": f"{lon}",
            "TabContainer1$TabPanel1$tbYear": f"{year}",
            "TabContainer1$TabPanel1$tbMonth": f"{month}",
            "TabContainer1$TabPanel1$tbDay": f"{day}",
            "TabContainer1$TabPanel1$tbModuleTiltAngle": f"{tilt_angle}",
            "TabContainer1$TabPanel1$tbModuleAzimuthAngle": f"{azimuth_angle}",
        }
        payload = {**PAYLOAD_ST, **payload_dynamic}

        # Post request and parse
        try:
            page = requests.post(URL, data=payload, headers=HEADER)
        except:
            print(f"Post request fails at iter {i}...")
            break
        soup = BeautifulSoup(page.text, "html.parser")

        # Arrange scraped features
        output = soup.find_all("span", class_="CalculatorTextboxOutput")
        sun_light_hr.append(time2sec(output[1].text.split(":")))
        sun_rise.append(time2sec(output[2].text.split(":")))
        sun_set.append(time2sec(output[3].text.split(":")))
        zenith_angle.append(float(output[4].text))
        air_mass_noon.append(float(output[5].text))

        sleep(0.1)

    df_scraped["SunLightHr"] = sun_light_hr
    df_scraped["SunRise"] = sun_rise
    df_scraped["SunSet"] = sun_set
    df_scraped["ZenithAngle"] = zenith_angle
    df_scraped["AirMassNoon"] = air_mass_noon

    assert len(df) == len(
        df_scraped
    ), "Length of scraped DataFrame must match that of raw one."
    df["Date"] = df["Date"].astype(str)
    df = pd.concat([df, df_scraped], axis=1)
    df.to_csv("./complete_scraped.csv", index=False)


if __name__ == "__main__":
    main()
