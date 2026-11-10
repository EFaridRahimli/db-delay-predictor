"""Collect Deutsche Bahn departures and arrivals for major hubs from the DB Timetables API.

Each run fetches the full change list (fchg) and the planned stops for the next two hours,
and stores only rows that are new or changed since the last stored state. Run it every 15 minutes.
Times stay in DB's raw format: YYMMddHHmm, Europe/Berlin local time.
"""
import csv
import gzip
import io
import os
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = "https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1"
BERLIN = ZoneInfo("Europe/Berlin")
ROOT = Path(__file__).parent
MIN_GAP = 1.05  # seconds between calls; the free plan allows 60 calls per minute

# name: (lat, lon, station IDs to poll). Big hubs split levels over several IDs
# (checked 2026-11-10); bus-only and empty IDs are left out.
STATIONS = {
    "Berlin Hbf": (52.525592, 13.369545, ["8098160", "8089021", "8089522"]),  # main ID 8011160 returns no trains
    "Berlin Ostbahnhof": (52.510972, 13.434567, ["8010255", "8089185"]),
    "Hamburg Hbf": (53.552736, 10.006909, ["8002549", "8098549"]),
    "München Hbf": (48.140232, 11.558335, ["8000261", "8098261", "8098262", "8098263"]),
    "Frankfurt (Main) Hbf": (50.107145, 8.663789, ["8000105", "8098105"]),
    "Köln Hbf": (50.94303, 6.958729, ["8000207"]),
    "Stuttgart Hbf": (48.784084, 9.181635, ["8000096", "8098096"]),
    "Düsseldorf Hbf": (51.219962, 6.794319, ["8000085"]),
    "Hannover Hbf": (52.376761, 9.741021, ["8000152"]),
    "Leipzig Hbf": (51.3454712, 12.3820639, ["8010205", "8098205"]),
    "Nürnberg Hbf": (49.445616, 11.082989, ["8000284"]),
    "Dortmund Hbf": (51.517896, 7.45929, ["8000080"]),
    "Essen Hbf": (51.451355, 7.014793, ["8000098"]),
    "Bremen Hbf": (53.083481, 8.813834, ["8000050"]),
    "Dresden Hbf": (51.040563, 13.732035, ["8010085"]),
    "Mannheim Hbf": (49.479354, 8.468921, ["8000244"]),
}

PLAN_COLS = ["stop_id", "eva", "category", "number", "line", "owner", "trip_type", "filter",
             "arr_pt", "arr_pp", "dep_pt", "dep_pp", "origin", "destination"]
CHANGE_COLS = ["stop_id", "eva", "arr_ct", "arr_cp", "arr_cs", "arr_clt",
               "dep_ct", "dep_cp", "dep_cs", "dep_clt", "messages"]
RUN_COLS = ["started_utc", "seconds", "calls", "errors", "plan_rows", "change_rows"]


def _attr(el, key):
    return el.get(key, "") if el is not None else ""


def parse_plan(xml, eva):
    rows = []
    for s in ET.fromstring(xml).iter("s"):
        tl, ar, dp = s.find("tl"), s.find("ar"), s.find("dp")
        path_in, path_out = _attr(ar, "ppth"), _attr(dp, "ppth")
        rows.append({
            "stop_id": s.get("id", ""), "eva": s.get("eva") or eva,
            "category": _attr(tl, "c"), "number": _attr(tl, "n"),
            "line": _attr(dp, "l") or _attr(ar, "l"),
            "owner": _attr(tl, "o"), "trip_type": _attr(tl, "t"), "filter": _attr(tl, "f"),
            "arr_pt": _attr(ar, "pt"), "arr_pp": _attr(ar, "pp"),
            "dep_pt": _attr(dp, "pt"), "dep_pp": _attr(dp, "pp"),
            "origin": path_in.split("|")[0] if path_in else "",
            "destination": path_out.split("|")[-1] if path_out else "",
        })
    return rows


def parse_changes(xml, eva):
    rows = []
    for s in ET.fromstring(xml).iter("s"):
        ar, dp = s.find("ar"), s.find("dp")
        codes = {f"{m.get('t', '')}:{m.get('c')}" for m in s.iter("m") if m.get("c")}
        rows.append({
            "stop_id": s.get("id", ""), "eva": s.get("eva") or eva,
            "arr_ct": _attr(ar, "ct"), "arr_cp": _attr(ar, "cp"),
            "arr_cs": _attr(ar, "cs"), "arr_clt": _attr(ar, "clt"),
            "dep_ct": _attr(dp, "ct"), "dep_cp": _attr(dp, "cp"),
            "dep_cs": _attr(dp, "cs"), "dep_clt": _attr(dp, "clt"),
            "messages": " ".join(sorted(codes)),
        })
    return rows


def latest_state(folder, days, cols):
    """Last stored values per (stop_id, eva), reading the given day folders in time order."""
    state = {}
    for day in days:
        for f in sorted((folder / day).glob("*.csv.gz")):
            with gzip.open(f, "rt", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    state[(row["stop_id"], row["eva"])] = tuple(row[c] for c in cols)
    return state


def new_rows(rows, state, cols):
    """Rows whose values differ from the stored state; also updates the state (handles duplicates within a run)."""
    out = []
    for row in rows:
        key, values = (row["stop_id"], row["eva"]), tuple(row[c] for c in cols)
        if state.get(key) != values:
            state[key] = values
            out.append(row)
    return out


def write_rows(path, rows, cols):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(buf.getvalue().encode("utf-8"))


def append_run_log(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RUN_COLS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(record)


_last_call = [0.0]


def get(path, headers):
    """GET with rate limiting and a few retries. Returns bytes, or None for 404 (hour outside the plan window)."""
    for attempt in range(3):
        wait = _last_call[0] + MIN_GAP - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(BASE + path, headers=headers), timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 and path.startswith("/plan/"):
                return None
            if e.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(10 * (attempt + 1))


def load_env(path):
    """Read KEY=value lines from a local .env file; real environment variables win."""
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip())


def main(data=ROOT / "data"):
    load_env(ROOT / ".env")
    headers = {"DB-Client-Id": os.environ["DB_CLIENT_ID"], "DB-Api-Key": os.environ["DB_API_KEY"],
               "Accept": "application/xml"}
    started = datetime.now(timezone.utc)
    # Past hours too, so a skipped or delayed scheduled run (up to about 4 h) leaves no plan gap.
    hours = [(started + timedelta(hours=k)).astimezone(BERLIN) for k in range(-1, 3)]
    days = [(started - timedelta(days=1)).strftime("%Y-%m-%d"), started.strftime("%Y-%m-%d")]
    stamp = started.strftime("%Y-%m-%dT%H:%M:%SZ")

    plan, changes, calls, errors = [], [], 0, 0
    t0 = time.monotonic()
    for name, (_, _, evas) in STATIONS.items():
        for eva in evas:
            jobs = [(f"/fchg/{eva}", parse_changes, changes)]
            jobs += [(f"/plan/{eva}/{h:%y%m%d}/{h:%H}", parse_plan, plan) for h in hours]
            for path, parse, sink in jobs:
                calls += 1
                if time.monotonic() - t0 > 360:  # save what we have before the 10-minute job timeout
                    errors += 1
                    continue
                try:
                    body = get(path, headers)
                    if body:
                        sink.extend(parse(body, eva))
                        if parse is parse_changes:  # extra/special trains appear only in fchg, with their own tl and pt
                            plan.extend(r for r in parse_plan(body, eva) if r["trip_type"] not in ("", "p"))
                except Exception as e:  # one bad station must not lose the whole run
                    errors += 1
                    print(f"error {name} {path}: {e}", file=sys.stderr)

    # Plans are static: keep the first row per stop (no value columns), so fchg and /plan copies don't alternate.
    plan_new = new_rows(plan, latest_state(data / "plan", days, []), [])
    change_new = new_rows(changes, latest_state(data / "changes", days, CHANGE_COLS), CHANGE_COLS)
    for row in change_new:
        row["observed_at"] = stamp
    file_name = f"{days[1]}/{started:%H%M%S}.csv.gz"
    if plan_new:
        write_rows(data / "plan" / file_name, plan_new, PLAN_COLS)
    if change_new:
        write_rows(data / "changes" / file_name, change_new, CHANGE_COLS + ["observed_at"])

    record = {"started_utc": stamp, "seconds": round((datetime.now(timezone.utc) - started).total_seconds()),
              "calls": calls, "errors": errors, "plan_rows": len(plan_new), "change_rows": len(change_new)}
    append_run_log(data / "runs" / f"{days[1]}.csv", record)
    print(record)
    if errors == calls:
        sys.exit("every API call failed")


if __name__ == "__main__":
    main()
