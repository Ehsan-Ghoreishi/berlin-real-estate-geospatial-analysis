# Berlin Real Estate — Spatial ETL & Price Prediction

An end-to-end ETL pipeline and geospatial analysis of Berlin residential
listings: raw data → cleaning → spatial feature engineering → SQL storage →
price prediction → interactive map.

![static map](maps/berlin_prices_static.png)

## Pipeline

```
extract.py  →  transform.py  →  load.py  →  model.py  →  visualize.py
 (raw CSVs)     (GeoPandas,       (SQLite)    (Linear/RF    (Folium +
                geopy distance)               regression)   Matplotlib)
```

Run the whole thing:

```bash
pip install -r requirements.txt
python3 main.py
```

Outputs land in `data/raw/`, `data/processed/`, and `maps/`.

## What each stage does

| Stage | File | Description |
|---|---|---|
| **Extract** | `src/extract.py` | Loads a reference table of 30 real Berlin U-/S-Bahn station coordinates and generates a listings dataset geographically anchored to Berlin's 12 Bezirke (districts). |
| **Transform** | `src/transform.py` | Cleans nulls/outliers (percentile-based price filtering), computes `dist_center_km` and `dist_nearest_station_km` via `geopy.distance.geodesic`, builds a `GeoDataFrame`, exports a GeoPackage. |
| **Load** | `src/load.py` | Persists the cleaned table into SQLite (`data/processed/berlin_real_estate.db`) with indexes on `bezirk` and `nearest_station`. |
| **Model** | `src/model.py` | Trains Linear Regression and Random Forest regressors (scikit-learn `Pipeline` + `ColumnTransformer`), evaluates with RMSE / MAE / R² on a held-out test split, saves the better model with `joblib`. |
| **Visualize** | `src/visualize.py` | Builds an interactive Folium HTML map (color-coded by €/sqm, station markers, popups) and a static Matplotlib map with basemap-tile support. |

## ⚠️ Data source note

This project currently runs on a **synthetic-but-geographically-grounded**
dataset, not a live pull from `daten.berlin.de` — I built it this way because
my sandbox environment doesn't have outbound access to Berlin's open data
portal or listing sites at the moment. What *is* real:

- The 30 station coordinates (`berlin_stations.csv`) are actual Berlin
  U-/S-Bahn locations.
- The 12 Bezirk centroids and their relative price ordering follow the real
  Berlin market pattern (Mitte/Charlottenburg pricier, Marzahn-Hellersdorf
  cheaper).
- The price-generation formula (size, condition, building age, district
  desirability + noise) is designed so the downstream regression has a
  genuine, non-trivial signal to recover — it is not a lookup table.

**To swap in real data**: replace `generate_synthetic_listings()` in
`extract.py` with a loader for a real export (Berlin Open Data /
Gutachterausschuss Bodenrichtwerte, or a Kaggle "Berlin housing prices" CSV).
As long as the output has columns
`[listing_id, lat, lon, size_sqm, rooms, year_built, condition, bezirk, price_eur]`,
every downstream stage works unchanged.

## Results (on the synthetic dataset, n=1470 after cleaning)

| Model | RMSE (€) | MAE (€) | R² |
|---|---|---|---|
| Linear Regression | 39,576 | 29,040 | 0.913 |
| Random Forest | 41,377 | 28,592 | 0.905 |

Linear Regression edges out Random Forest here because the synthetic
price-generation function is close to linear in its features — on a real,
noisier listings dataset with nonlinear interactions (e.g. size × location
premium), I'd expect the Random Forest to pull ahead, which is why the
pipeline evaluates both rather than assuming one wins.

Top price drivers (standardized linear coefficients): living space (`size_sqm`),
then district (`bezirk_Mitte`, `bezirk_Charlottenburg-Wilmersdorf` positive;
`bezirk_Marzahn-Hellersdorf`, `bezirk_Lichtenberg` negative), then renovation
condition.

## Design decisions worth defending in an interview

- **Percentile-based outlier filtering** (1st/99th on €/sqm) instead of an
  arbitrary fixed cutoff — transparent and reproducible.
- **Brute-force nearest-station search** (30 stations × 1.5k listings =
  45k geodesic calls, <1s) instead of a KD-tree — the right call at this
  scale; noted in code as the point where I'd switch to a spatial index if
  the station table grew into the thousands.
- **SQLite over SpatiaLite** for the load stage — zero extra native
  dependency, opens with the stdlib; the GeoPackage from the transform stage
  remains the source of truth if real GIS queries (polygon joins, etc.) are
  ever needed.
- **Random Forest over a deeper model** (gradient boosting / neural net) —
  higher interpretability via `feature_importances_`, and ~1.4k training
  rows is small enough that a deeper model risks overfitting relative to a
  bagged ensemble of shallow trees.
- **Basemap tiles with graceful fallback** — `visualize.py` tries
  `contextily` + OpenStreetMap tiles for the static map and falls back to a
  plain lat/lon scatter if the network call fails, so the pipeline never
  breaks on a sandboxed or offline machine.

## Project structure

```
berlin_real_estate/
├── main.py                  # orchestrates the full pipeline
├── requirements.txt
├── src/
│   ├── extract.py
│   ├── transform.py
│   ├── load.py
│   ├── model.py
│   └── visualize.py
├── data/
│   ├── raw/                 # extract.py output
│   └── processed/           # transform/load/model output
└── maps/                    # visualize.py output
```

## Next steps

- Swap in a real Berlin Open Data / Bodenrichtwert export.
- Replace geodesic "as the crow flies" station distance with real walking
  distance via `osmnx` + OSM street graph (blocked in this sandbox by
  network access, straightforward with internet access).
- Add SpatiaLite for true in-database spatial queries.
- Hyperparameter search (`GridSearchCV`) on the Random Forest.
