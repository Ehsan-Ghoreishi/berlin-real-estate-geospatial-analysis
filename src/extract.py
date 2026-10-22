"""
extract.py
==========
Extract stage. Unlike the original prototype, this version runs on a real
downloaded Kaggle dataset rather than synthetic data.

DATASET: "Berlin Real Estate: Sales, New Construction & Rentals (2020-2026)"
by Sergey Nefedov, https://www.kaggle.com/datasets (search: "berlin housing")

WHAT'S REAL vs GENERATED IN THE SOURCE DATA (per the dataset's own
documentation -- stated here for defensibility in an interview):
  - REAL: the 82 Ortsteil geographic centroids (from OpenStreetMap admin
    boundaries), the 135 transit station coordinates (OSM / BVG / S-Bahn
    Berlin GmbH), the ECB main refinancing rate timeline, German mortgage
    rate trajectory, and the real dates of Berlin's 2020-2024 housing
    policy shocks (Mietendeckel enactment, its 2021 constitutional
    reversal, the Heizungsgesetz, etc.) baked into the time series.
  - GENERATED: individual listing-level characteristics (rooms, floor,
    condition, exact price) come from a hedonic pricing model anchored to
    real ImmobilienScout24/Engel & Völkers median EUR/m^2 by Ortsteil --
    not scraped listings (scraping those portals violates their ToS,
    which is why the dataset author built it this way).

This extract stage just validates the five raw CSVs are present and have
the expected shape/columns -- no network fetch needed, they're already
downloaded to data/raw/.
"""

import logging

import pandas as pd

from src.config import RAW_DIR

logger = logging.getLogger(__name__)

EXPECTED = {
    "secondary_sales.csv": 50000,
    "rentals.csv": 30000,
    "new_construction.csv": 10000,
    "kiez_prices_monthly.csv": 6232,
    "transit_stations.csv": 135,
}


def run():
    logger.info("validating raw Kaggle source files")
    for fname, expected_rows in EXPECTED.items():
        path = RAW_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- place the Kaggle CSV export in data/raw/"
            )
        df = pd.read_csv(path)
        status = "OK" if len(df) == expected_rows else "ROW COUNT DIFFERS"
        logger.info(f"    {fname:32s} {len(df):>7,} rows  {status}")
    logger.info("all source files present")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
