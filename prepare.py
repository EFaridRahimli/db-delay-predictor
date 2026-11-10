"""Build the training table from piebro/deutsche-bahn-data plus Open-Meteo weather.

One row per planned departure at our 16 stations, with the train's delay at its previous stops:
the information you have when the train leaves the stop before ours.
Writes external/hub/hub-YYYY-MM.parquet and external/weather.parquet.
"""
import json
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from collect import STATIONS

ROOT = Path(__file__).parent
EXT = ROOT / "external"
MONTHS = ["2025-11", "2025-12"] + [f"2026-{m:02d}" for m in range(1, 11)]  # piebro covers all stations since 2025-11
PIEBRO = "https://huggingface.co/datasets/piebro/deutsche-bahn-data/resolve/main/monthly_processed_data/data-{}.parquet"
HUB_EVA = {"0" + eva: name for name, (_, _, evas) in STATIONS.items() for eva in evas}  # piebro pads EVAs to 8 digits
COLS = ["eva", "id", "train_number", "line_number", "train_type", "final_destination_station", "delay_in_min",
        "arrival_is_canceled", "departure_is_canceled", "is_additional_stop", "is_replacement_train",
        "train_line_station_num", "arrival_planned_time", "departure_planned_time"]
WEATHER_VARS = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m"]


def download(month):
    path = EXT / "piebro" / f"data-{month}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(PIEBRO.format(month), part)
        part.rename(path)
    return path


def build_month(path):
    t = pq.read_table(path, columns=COLS).to_pandas()
    t["trip"] = t["id"].str.rsplit("-", n=1).str[0]  # ride id + start time; the ride id alone repeats daily
    t = t[t["trip"].isin(set(t.loc[t["eva"].isin(HUB_EVA), "trip"]))]
    # shortcut: trips that cross a month boundary lose their stops in the other month (a few per month).
    t = t.sort_values(["trip", "train_line_station_num"])
    g = t.groupby("trip", sort=False)
    t["prev_delay"] = g["delay_in_min"].shift(1)  # departure delay at the previous stop
    t["prev2_delay"] = g["delay_in_min"].shift(2)
    t["prev_canceled"] = g["departure_is_canceled"].shift(1)
    t["prev_departure_planned"] = g["departure_planned_time"].shift(1)
    t["stop_index"] = g.cumcount()

    h = t[t["eva"].isin(HUB_EVA) & t["departure_planned_time"].notna()].copy()
    h["station"] = h["eva"].map(HUB_EVA)
    h["run_minutes"] = (h["arrival_planned_time"] - h["prev_departure_planned"]).dt.total_seconds() / 60
    h["dwell_minutes"] = (h["departure_planned_time"] - h["arrival_planned_time"]).dt.total_seconds() / 60
    return h.drop(columns=["prev_departure_planned", "train_line_station_num"])


def fetch_weather():
    start, end = f"{MONTHS[0]}-01", (pd.Period(MONTHS[-1]).end_time.date()).isoformat()
    frames = []
    for name, (lat, lon, _) in STATIONS.items():
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               f"&start_date={start}&end_date={end}&hourly={','.join(WEATHER_VARS)}&timezone=Europe%2FBerlin")
        with urllib.request.urlopen(url, timeout=60) as r:
            hourly = json.load(r)["hourly"]
        df = pd.DataFrame(hourly).rename(columns={"time": "hour"})
        df["hour"], df["station"] = pd.to_datetime(df["hour"]), name
        frames.append(df)
    # Local times: the autumn DST hour appears twice; keep the first.
    return pd.concat(frames).drop_duplicates(["station", "hour"])


if __name__ == "__main__":
    (EXT / "hub").mkdir(parents=True, exist_ok=True)
    for month in MONTHS:
        out = EXT / "hub" / f"hub-{month}.parquet"
        if not out.exists():
            h = build_month(download(month))
            h.to_parquet(out, index=False)
            print(month, len(h), "departures")
    weather = fetch_weather()
    weather.to_parquet(EXT / "weather.parquet", index=False)
    print("weather rows", len(weather))
