"""Self-check for the previous-stop features in prepare.build_month. Run: python test_prepare.py"""
import tempfile
from pathlib import Path

import pandas as pd

from prepare import build_month

T = pd.Timestamp
rows = [  # trip A: Mainz -> Frankfurt Hbf (hub) -> Hanau; trip B never touches a hub
    ("08000240", "A-2611100800-1", 1, 3, None, T("2026-11-10 08:00")),
    ("08000105", "A-2611100800-2", 2, 7, T("2026-11-10 08:30"), T("2026-11-10 08:35")),
    ("08000150", "A-2611100800-3", 3, 9, T("2026-11-10 08:50"), None),
    ("08000240", "B-2611100900-1", 1, 0, None, T("2026-11-10 09:00")),
]
df = pd.DataFrame(rows, columns=["eva", "id", "train_line_station_num", "delay_in_min",
                                 "arrival_planned_time", "departure_planned_time"])
for col in ["train_number", "line_number", "train_type", "final_destination_station"]:
    df[col] = "x"
for col in ["arrival_is_canceled", "departure_is_canceled", "is_additional_stop", "is_replacement_train"]:
    df[col] = False

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "m.parquet"
    df.to_parquet(path)
    h = build_month(path)

assert len(h) == 1, h  # only the hub departure; Hanau has no departure, trip B has no hub stop
r = h.iloc[0]
assert r["station"] == "Frankfurt (Main) Hbf" and r["delay_in_min"] == 7
assert r["prev_delay"] == 3 and pd.isna(r["prev2_delay"]) and r["stop_index"] == 1
assert r["run_minutes"] == 30 and r["dwell_minutes"] == 5
print("ok")
