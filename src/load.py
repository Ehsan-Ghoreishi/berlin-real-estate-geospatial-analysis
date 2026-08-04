"""
load.py
=======
Load stage: persist the cleaned, feature-engineered listings into a
SQLite database (data/processed/berlin_real_estate.db, table `listings`).

SQLite (rather than SpatiaLite) is used deliberately for portability --
it requires no extra native extension and opens with the stdlib
sqlite3 module or any SQL browser. Geometry is kept as plain lat/lon
columns (already present) rather than a WKB blob, which is sufficient
for this project's read patterns (filter/aggregate by bezirk, join to
price). The GeoPackage produced by transform.py remains the
authoritative geospatial artifact if true GIS queries are ever needed.
"""

import sqlite3
from pathlib import Path

import pandas as pd

PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DB_PATH = PROC_DIR / "berlin_real_estate.db"


def run():
    csv_path = PROC_DIR / "listings_clean.csv"
    df = pd.read_csv(csv_path)

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("listings", conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_bezirk ON listings(bezirk)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_station ON listings(nearest_station)"
    )
    conn.commit()

    # sanity query
    check = pd.read_sql(
        "SELECT bezirk, COUNT(*) AS n, ROUND(AVG(price_per_sqm),0) AS avg_eur_sqm "
        "FROM listings GROUP BY bezirk ORDER BY avg_eur_sqm DESC",
        conn,
    )
    conn.close()

    print(f"[load] wrote {len(df)} rows -> {DB_PATH}")
    print(check.to_string(index=False))
    return DB_PATH


if __name__ == "__main__":
    run()
