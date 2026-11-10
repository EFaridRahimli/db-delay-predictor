# Deutsche Bahn delay predictor

Predicts how late a Deutsche Bahn train will leave one of 16 major German stations, and collects live data every 15 minutes to keep the model current.

## Results

**Task:** at the moment a train leaves its previous stop, predict its departure delay at our station (minutes), and whether it leaves more than 5 minutes late.

**Data and split:** past departures from [piebro/deutsche-bahn-data](https://huggingface.co/datasets/piebro/deutsche-bahn-data), split by time with no shuffling:

| Part | Months | Departures |
|---|---|---|
| Train | Nov 2025 – Jul 2026 | 3,741,644 |
| Validation | Aug 2026 | 391,353 |
| Test | Sep – Oct 2026 | 844,927 |

Test set results:

| Method | MAE (min) | F1, >5 min late |
|---|---|---|
| **LightGBM** | **1.76** | **0.784** |
| Current delay + typical change to this station (strongest baseline) | 1.99 | 0.756 |
| Current delay ("delay stays what it is now") | 2.03 | 0.751 |
| Historical mean per station, train type, line and hour | 4.56 | 0.485 |

The model has 11% lower MAE than the strongest baseline and +0.028 F1. It has the lowest MAE in every train group, hour band, lead time and station. On F1 it beats every baseline everywhere except two cases: it ties at Essen Hbf (0.663), and it loses for trains that start at the station (see limitations).

| Planned minutes from the previous stop | Departures | Model MAE | Strongest baseline MAE |
|---|---|---|---|
| 2 or less | 335,342 | 0.54 | 0.68 |
| 3-9 | 154,860 | 1.38 | 1.64 |
| 10-29 | 48,223 | 3.47 | 3.92 |
| 30 or more | 50,771 | 5.59 | 6.20 |

Most useful inputs: the train's delay at its previous stop, then its line, its delay two stops back, and the station level. Weather adds little. Full tables by station, train group and hour are in [reports/results.md](reports/results.md).

### Limitations

- **Short horizon.** The forecast is made when the train leaves the previous stop, which for 40% of test departures is 2 minutes or less before reaching our station. The gain holds at longer lead times, but errors grow (5.6 min MAE at 30+ min).
- **Trains that start at the station** (30% of the test set) have no current delay to use. With the shared threshold the model's F1 there is 0.123, below the historical mean baseline's 0.307. A threshold tuned on validation for these trains alone gives 0.333.
- **Labels come from piebro's data.** A stop with no live update is recorded as on time. The API does not say whether a changed time is a forecast or the actual time.
- **Weather** in the backtest is observed (Open-Meteo archive), not a forecast as it would be live.
- **Berlin Hbf upper level** (platforms 11-14) is missing from the API.
- Our own 15-minute collection has not been used for evaluation yet.

## Train it yourself

```
python prepare.py   # downloads 12 months (~4.3 GB) of piebro data and the weather; builds external/
python train.py     # trains, evaluates, writes reports/ and models/
```

## Data

`collect.py` runs on GitHub Actions at :07, :22, :37 and :52 every hour. It polls the DB Timetables API for these stations: Berlin Hbf, Berlin Ostbahnhof, Hamburg, München, Frankfurt (Main), Köln, Stuttgart, Düsseldorf, Hannover, Leipzig, Nürnberg, Dortmund, Essen, Bremen, Dresden and Mannheim (all Hbf unless named). It stores gzipped CSV in `data/`:

- `plan/YYYY-MM-DD/HHMMSS.csv.gz`: planned stops (train category, number, line, planned times and platforms, origin, destination).
- `changes/YYYY-MM-DD/HHMMSS.csv.gz`: changed times, platforms, cancellations and message codes. A row is stored only when a stop's values change; `observed_at` says when.
- `runs/YYYY-MM-DD.csv`: one line per run (calls, errors, rows), to spot gaps.

Folder dates and file names are UTC run times. Train times are DB's raw `YYMMddHHmm` in Europe/Berlin local time.

Known limits:

- The API returns no trains for the upper level of Berlin Hbf (platforms 11-14).
- `ct` can be a forecast or the actual time; the API does not say which.
- GitHub can delay or skip scheduled runs.

## Run locally

Put `DB_CLIENT_ID` and `DB_API_KEY` from the [DB API Marketplace](https://developers.deutschebahn.com/db-api-marketplace/apis/product/timetables) in a `.env` file, then:

```
pip install -r requirements.txt
python test_collect.py
python collect.py
```

## Credits

- Train data: Deutsche Bahn AG, DB Timetables API, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The data in `data/` is shared under the same license.
- Historical training data: [piebro/deutsche-bahn-data](https://github.com/piebro/deutsche-bahn-data) (CC BY 4.0).
- Weather: [Open-Meteo](https://open-meteo.com/) (CC BY 4.0).
