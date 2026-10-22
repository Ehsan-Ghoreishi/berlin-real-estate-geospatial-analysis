"""
load.py
=======
Loads all cleaned tables into a single SQLite database
(data/processed/berlin_real_estate.db):
  - sales             (secondary market, 49k rows)
  - rentals           (29.4k rows)
  - new_construction  (9.8k rows)
  - kiez_monthly      (82 Ortsteile x 76 months time series)
  - stations          (135 real transit stations)
"""

import sqlite3

import pandas as pd

from src.config import DB_PATH, PROC_DIR

TABLES = {
    "sales": "sales_clean.csv",
    "rentals": "rentals_clean.csv",
    "new_construction": "new_construction_clean.csv",
    "kiez_monthly": "kiez_monthly_clean.csv",
    "stations": "stations_clean.csv",
}


def run():
    conn = sqlite3.connect(DB_PATH)

    for table, fname in TABLES.items():
        df = pd.read_csv(PROC_DIR / fname)
        df.to_sql(table, conn, if_exists="replace", index=False)
        print(f"[load] {table:18s} {len(df):>7,} rows -> {DB_PATH.name}")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_bezirk ON sales(bezirk)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rentals_bezirk ON rentals(bezirk)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_kiez_ortsteil ON kiez_monthly(ortsteil)")
    conn.commit()

    check = pd.read_sql(
        "SELECT bezirk, COUNT(*) AS n, ROUND(AVG(price_per_m2_eur),0) AS avg_eur_m2 "
        "FROM sales GROUP BY bezirk ORDER BY avg_eur_m2 DESC",
        conn,
    )
    conn.close()

    print(f"\n[load] wrote -> {DB_PATH}")
    print(check.to_string(index=False))
    return DB_PATH


if __name__ == "__main__":
    run()
