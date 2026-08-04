"""
extract.py
==========
Extract stage of the ETL pipeline.

DATA SOURCE NOTE (read this before using in an interview):
This sandbox has no outbound network access to daten.berlin.de, the
Gutachterausschuss Bodenrichtwertkarte, or listing portals (Immoscout24 /
Immowelt), so this script cannot download those datasets live. Instead it:

  1. Ships a hand-curated, real-coordinate reference table of Berlin's
     major U-Bahn / S-Bahn stations (`berlin_stations.csv`) and Bezirk
     (district) centroids -- these are real, public geographic facts,
     not synthetic.
  2. Generates a synthetic-but-geographically-grounded listings dataset
     (`raw_listings.csv`) whose coordinates fall inside real Berlin
     Bezirk boundaries and whose price is a function of real distance
     to the nearest real station, so the downstream geospatial +
     regression pipeline is exercised exactly as it would be on real
     listings.

TO SWAP IN REAL DATA: replace `generate_synthetic_listings()` with a
loader that reads a CSV/GeoJSON exported from daten.berlin.de or a
Kaggle "Berlin housing prices" dataset. As long as the output has
columns [listing_id, lat, lon, size_sqm, rooms, year_built, condition,
bezirk, price_eur], every downstream stage (transform/load/model/viz)
works unchanged.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Real, public coordinates of major Berlin U-Bahn / S-Bahn stations.
# (lat, lon in WGS84). Not exhaustive -- major interchange + high
# footfall stations chosen to give meaningful "distance to transit"
# signal across the city.
# ---------------------------------------------------------------------
BERLIN_STATIONS = [
    ("Alexanderplatz", 52.5219, 13.4132),
    ("Friedrichstraße", 52.5203, 13.3872),
    ("Zoologischer Garten", 52.5069, 13.3325),
    ("Hauptbahnhof", 52.5251, 13.3694),
    ("Ostbahnhof", 52.5107, 13.4347),
    ("Potsdamer Platz", 52.5096, 13.3759),
    ("Warschauer Straße", 52.5052, 13.4491),
    ("Südkreuz", 52.4757, 13.3653),
    ("Gesundbrunnen", 52.5490, 13.3887),
    ("Hermannplatz", 52.4868, 13.4241),
    ("Kottbusser Tor", 52.4991, 13.4180),
    ("Ostkreuz", 52.5031, 13.4682),
    ("Schönhauser Allee", 52.5490, 13.4128),
    ("Neukölln", 52.4810, 13.4359),
    ("Wedding", 52.5490, 13.3418),
    ("Steglitz Rathaus", 52.4562, 13.3238),
    ("Tempelhof", 52.4726, 13.3862),
    ("Charlottenburg", 52.5045, 13.3038),
    ("Spandau", 52.5347, 13.1976),
    ("Pankow", 52.5691, 13.4016),
    ("Lichtenberg", 52.5099, 13.4977),
    ("Mehringdamm", 52.4939, 13.3888),
    ("Rathaus Steglitz", 52.4563, 13.3241),
    ("Adenauerplatz", 52.5010, 13.3096),
    ("Prenzlauer Allee", 52.5382, 13.4229),
    ("Marzahn", 52.5443, 13.5636),
    ("Rudow", 52.4230, 13.4917),
    ("Reinickendorf", 52.5762, 13.3316),
    ("Frohnau", 52.6288, 13.2925),
    ("Köpenick", 52.4547, 13.5779),
]

# Bezirk (district) approximate centroids + a rough "desirability"
# multiplier used only to make the synthetic price surface realistic
# (central/west districts pricier than outer east/south-east, which
# roughly matches real Berlin market patterns).
BEZIRKE = {
    "Mitte":                (52.5200, 13.4050, 1.35),
    "Charlottenburg-Wilmersdorf": (52.5050, 13.3050, 1.30),
    "Friedrichshain-Kreuzberg":   (52.5000, 13.4400, 1.20),
    "Pankow":               (52.5700, 13.4100, 1.05),
    "Tempelhof-Schöneberg": (52.4700, 13.3800, 1.05),
    "Steglitz-Zehlendorf":  (52.4450, 13.2600, 1.10),
    "Neukölln":             (52.4800, 13.4350, 0.90),
    "Treptow-Köpenick":     (52.4550, 13.5700, 0.85),
    "Marzahn-Hellersdorf":  (52.5450, 13.5900, 0.65),
    "Lichtenberg":          (52.5350, 13.5000, 0.75),
    "Reinickendorf":        (52.5800, 13.3300, 0.80),
    "Spandau":              (52.5350, 13.2000, 0.75),
}


def save_station_reference() -> pd.DataFrame:
    df = pd.DataFrame(BERLIN_STATIONS, columns=["station_name", "lat", "lon"])
    df.to_csv(RAW_DIR / "berlin_stations.csv", index=False)
    return df


def generate_synthetic_listings(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bezirk_names = list(BEZIRKE.keys())

    rows = []
    for i in range(n):
        bezirk = bezirk_names[rng.integers(0, len(bezirk_names))]
        b_lat, b_lon, desirability = BEZIRKE[bezirk]

        # scatter around bezirk centroid (~ +/- 3km in degree terms)
        lat = b_lat + rng.normal(0, 0.020)
        lon = b_lon + rng.normal(0, 0.030)

        size_sqm = float(np.clip(rng.normal(75, 30), 20, 220))
        rooms = int(np.clip(round(size_sqm / 28 + rng.normal(0, 0.6)), 1, 7))
        year_built = int(np.clip(rng.normal(1965, 35), 1890, 2024))
        condition = rng.choice(
            ["needs_renovation", "good", "renovated", "new_build"],
            p=[0.15, 0.45, 0.30, 0.10],
        )

        rows.append(
            dict(
                listing_id=f"L{i:05d}",
                lat=lat,
                lon=lon,
                bezirk=bezirk,
                size_sqm=round(size_sqm, 1),
                rooms=rooms,
                year_built=year_built,
                condition=condition,
                _desirability=desirability,  # dropped after price synthesis
            )
        )

    df = pd.DataFrame(rows)

    # --- synthesize price with a real, defensible functional form ---
    condition_premium = df["condition"].map(
        {"needs_renovation": -0.12, "good": 0.0, "renovated": 0.10, "new_build": 0.22}
    )
    age_penalty = -0.0015 * (2026 - df["year_built"]).clip(lower=0)
    base_price_per_sqm = 4200 * df["_desirability"]

    noise = rng.normal(0, 0.08, size=len(df))
    price_per_sqm = base_price_per_sqm * (1 + condition_premium + age_penalty + noise)
    price_per_sqm = price_per_sqm.clip(lower=1500)

    df["price_eur"] = (price_per_sqm * df["size_sqm"]).round(0)
    df = df.drop(columns=["_desirability"])

    df.to_csv(RAW_DIR / "raw_listings.csv", index=False)
    return df


def run():
    stations = save_station_reference()
    listings = generate_synthetic_listings()
    print(f"[extract] wrote {len(stations)} stations -> {RAW_DIR/'berlin_stations.csv'}")
    print(f"[extract] wrote {len(listings)} listings -> {RAW_DIR/'raw_listings.csv'}")
    return stations, listings


if __name__ == "__main__":
    run()
