# Deutsche Bahn delay predictor

Collects live departures and arrivals for 16 major German stations every 15 minutes, to train a model that predicts train delays. Work in progress: the collector is running; the model comes next.

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

Train data: Deutsche Bahn AG, DB Timetables API, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The data in `data/` is shared under the same license.
