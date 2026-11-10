"""Train LightGBM delay models and compare them with two baselines on a time-based split.

Task: when a train leaves the stop before one of our 16 stations, predict its departure delay there
(regression, minutes) and whether it leaves more than 5 minutes late (classification).
Writes reports/metrics.json, reports/results.md and models/*.txt.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error

ROOT = Path(__file__).parent
EXT, REPORTS, MODELS = ROOT / "external", ROOT / "reports", ROOT / "models"
TRAIN_END, VALID_END = "2026-08-01", "2026-09-01"  # train 2025-11..2026-07, valid 2026-08, test 2026-09..2026-10
LATE = 5  # minutes; "late" means delay > 5
CATS = ["station", "eva", "train_type", "line_number"]
NUMS = ["prev_delay", "prev2_delay", "prev_canceled", "run_minutes", "dwell_minutes", "stop_index",
        "hour", "weekday", "is_additional_stop", "is_replacement_train",
        "temperature_2m", "precipitation", "snowfall", "wind_speed_10m"]
FEATURES = CATS + NUMS
LONG_DISTANCE = {"ICE", "IC", "EC", "ECE", "RJ", "RJX", "TGV", "NJ", "FLX", "EST", "EN", "WB"}


def load():
    df = pd.concat([pd.read_parquet(f) for f in sorted((EXT / "hub").glob("hub-*.parquet"))], ignore_index=True)
    df = df[~df["departure_is_canceled"].astype(bool) & df["delay_in_min"].notna()].copy()  # cancellations have no delay
    df["hour_ts"] = df["departure_planned_time"].dt.floor("h")
    weather = pd.read_parquet(EXT / "weather.parquet").rename(columns={"hour": "hour_ts"})
    df = df.merge(weather, on=["station", "hour_ts"], how="left")
    df["hour"] = df["departure_planned_time"].dt.hour
    df["weekday"] = df["departure_planned_time"].dt.weekday
    for col in ["prev_canceled", "is_additional_stop", "is_replacement_train"]:
        df[col] = df[col].astype(float)
    for col in CATS:  # categories set on the full table, so train and test codes match
        df[col] = df[col].fillna("none").astype("category")
    df["group"] = np.where(df["train_type"].astype(str).isin(LONG_DISTANCE), "long-distance",
                           np.where(df["train_type"].astype(str) == "S", "S-Bahn", "regional"))
    return df


def baseline_now(d):
    """Delay stays what it is now: the delay at the previous stop (0 for trains that start here)."""
    return d["prev_delay"].fillna(0).to_numpy()


def group_stat(train, values, d, stat):
    """Training statistic of `values` per station, train type, line and hour, falling back to coarser groups."""
    pred = pd.Series(np.nan, index=d.index)
    frame = train.assign(v=values)
    for keys in [["station", "train_type", "line_number", "hour"], ["station", "train_type", "hour"], ["station"]]:
        stats = frame.groupby(keys, observed=True)["v"].agg(stat).rename("m")
        pred = pred.fillna(d[keys].join(stats, on=keys)["m"])
    return pred.fillna(values.agg(stat))


def baseline_history(train, d):
    """Mean training delay per station, train type, line and hour (the 'historical mean' baseline)."""
    return group_stat(train, train["delay_in_min"], d, "mean").to_numpy()


def baseline_now_adjusted(train, d):
    """Strongest simple baseline: current delay plus the typical (median) change from the previous stop to ours;
    median history for trains that start here."""
    through = train[train["prev_delay"].notna()]
    delta = group_stat(through, through["delay_in_min"] - through["prev_delay"], d, "median")
    history = group_stat(train, train["delay_in_min"], d, "median")
    return (d["prev_delay"] + delta).fillna(history).to_numpy()


def scores(y, pred_minutes, late_pred):
    return {"mae": round(float(mean_absolute_error(y, pred_minutes)), 3),
            "f1_late": round(float(f1_score(y > LATE, late_pred)), 3)}


def breakdown(d, y, preds, by):
    rows = []
    for key, idx in d.groupby(by, observed=True).indices.items():
        row = {"group": str(key), "n": len(idx)}
        for name, (minutes, late) in preds.items():
            row[f"{name}_mae"] = round(float(mean_absolute_error(y[idx], minutes[idx])), 2)
            row[f"{name}_f1"] = round(float(f1_score(y[idx] > LATE, late[idx], zero_division=0)), 3)
        rows.append(row)
    return sorted(rows, key=lambda r: -r["n"])


def md_table(rows):
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main():
    df = load()
    t = df["departure_planned_time"]
    train, valid, test = df[t < TRAIN_END], df[(t >= TRAIN_END) & (t < VALID_END)], df[t >= VALID_END]
    y_train, y_valid, y_test = (s["delay_in_min"].to_numpy(dtype=float) for s in (train, valid, test))
    print(f"rows: train {len(train)}, valid {len(valid)}, test {len(test)}")

    params = dict(n_estimators=3000, learning_rate=0.05, num_leaves=127, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    stop = [lgb.early_stopping(100, verbose=False)]
    reg = lgb.LGBMRegressor(objective="l1", **params)  # L1 matches the MAE metric
    reg.fit(train[FEATURES], y_train, eval_set=[(valid[FEATURES], y_valid)], callbacks=stop)
    clf = lgb.LGBMClassifier(objective="binary", **params)
    clf.fit(train[FEATURES], y_train > LATE, eval_set=[(valid[FEATURES], y_valid > LATE)], callbacks=stop)

    # Decision thresholds for "late", chosen on validation only. The headline uses one threshold (it maximises
    # overall F1); trains that start at the station get their own threshold as a separate, reported use case.
    p_valid = clf.predict_proba(valid[FEATURES])[:, 1]
    origin_valid, origin_test = valid["prev_delay"].isna().to_numpy(), test["prev_delay"].isna().to_numpy()
    grid = np.arange(0.05, 0.9, 0.01)
    thresholds = {}
    for name, mask in [("all", np.ones_like(origin_valid)), ("starts_here", origin_valid)]:
        f1s = [f1_score(y_valid[mask] > LATE, p_valid[mask] >= g) for g in grid]
        thresholds[name] = round(float(grid[np.argmax(f1s)]), 2)

    reg_test = reg.predict(test[FEATURES])
    p_test = clf.predict_proba(test[FEATURES])[:, 1]
    clf_test = p_test >= thresholds["all"]
    starts_here_own_f1 = f1_score(y_test[origin_test] > LATE, p_test[origin_test] >= thresholds["starts_here"])
    now, adj, hist = baseline_now(test), baseline_now_adjusted(train, test), baseline_history(train, test)
    preds = {"model": (reg_test, clf_test), "now": (now, now > LATE),
             "now_adjusted": (adj, adj > LATE), "history": (hist, hist > LATE)}

    metrics = {
        "split": {"train": "2025-11..2026-07", "valid": "2026-08", "test": "2026-09..2026-10",
                  "rows": {"train": len(train), "valid": len(valid), "test": len(test)}},
        "late_threshold_minutes": LATE, "classifier_thresholds": thresholds,
        "trees": {"regressor": reg.best_iteration_, "classifier": clf.best_iteration_},
        "test": {name: scores(y_test, m, l) for name, (m, l) in preds.items()},
        "test_late_share": round(float((y_test > LATE).mean()), 3),
        "starts_here_f1_with_own_threshold": round(float(starts_here_own_f1), 3),
        "by_group": breakdown(test, y_test, preds, "group"),
        "by_origin": breakdown(test.assign(origin=np.where(test["prev_delay"].isna(), "starts here", "through train")),
                               y_test, preds, "origin"),
        # Planned minutes from the previous stop: roughly how far ahead the forecast is made.
        "by_lead_time": breakdown(test.assign(lead=pd.cut(test["run_minutes"], [-np.inf, 2, 9, 29, np.inf],
                                                          labels=["<=2 min", "3-9 min", "10-29 min", "30+ min"])
                                              .cat.add_categories("starts here").fillna("starts here")),
                                  y_test, preds, "lead"),
        "by_station": breakdown(test, y_test, preds, "station"),
        "by_hour": breakdown(test.assign(hours=pd.cut(test["hour"], [-1, 5, 9, 15, 19, 23],
                                                      labels=["0-5", "6-9", "10-15", "16-19", "20-23"])),
                             y_test, preds, "hours"),
        "feature_importance_gain": dict(sorted(zip(FEATURES, reg.booster_.feature_importance("gain").round().tolist()),
                                               key=lambda kv: -kv[1])),
    }

    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    reg.booster_.save_model(str(MODELS / "regressor.txt"))
    clf.booster_.save_model(str(MODELS / "classifier.txt"))
    overall = [{"method": n, **s} for n, s in metrics["test"].items()]
    sections = [("Test set, all departures", overall), ("By train group", metrics["by_group"]),
                ("Through trains vs trains starting at the station", metrics["by_origin"]),
                ("By planned minutes from the previous stop (how far ahead the forecast is)", metrics["by_lead_time"]),
                ("By station", metrics["by_station"]), ("By hour", metrics["by_hour"])]
    text = ("# Results\n\nGenerated by `train.py`. MAE in minutes; F1 for departing more than 5 minutes late.\n"
            "Baselines: `now` = delay at the previous stop; `now_adjusted` = that plus the typical change to our "
            "station (median history for trains starting here); `history` = mean delay per station, type, line, hour.\n\n")
    text += "\n\n".join(f"## {title}\n\n{md_table(rows)}" for title, rows in sections) + "\n"
    (REPORTS / "results.md").write_text(text, encoding="utf-8")
    print(json.dumps(metrics["test"], indent=1))


if __name__ == "__main__":
    main()
