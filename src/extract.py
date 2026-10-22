"""
extract.py
==========
Extract stage. Unlike the original prototype, this version runs on a real
downloaded Kaggle dataset rather than synthetic data.

DATASET: "Berlin Real Estate: Sales & Rentals 2020-2026" by Sergey Nefedov
(Apache 2.0), https://www.kaggle.com/datasets/sergionefedov/berlin-real-estate-sales-and-rentals-2020-2026

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

This extract stage validates that the five raw CSVs are present, carry
the columns later stages rely on, and have the expected row counts -- no
network fetch needed, they're already downloaded to data/raw/.
"""

import logging

import pandas as pd

from src.config import RAW_DIR

logger = logging.getLogger(__name__)

EXPECTED_ROWS = {
    "secondary_sales.csv": 50000,
    "rentals.csv": 30000,
    "new_construction.csv": 10000,
    "kiez_prices_monthly.csv": 6232,
    "transit_stations.csv": 135,
}

_LISTING_COLS = {"date_listed", "ortsteil", "bezirk", "lat", "lon", "area_m2", "rooms"}

# Columns the downstream stages actually depend on -- a renamed or dropped
# column should fail here, not as a KeyError three stages later.
REQUIRED_COLUMNS = {
    "secondary_sales.csv": _LISTING_COLS
    | {
        "price_eur",
        "price_per_m2_eur",
        "year_built",
        "building_era",
        "energy_class",
        "condition",
        "transit_distance_min",
        "to_brandenburg_gate_km",
        "mortgage_rate_at_listing",
    },
    "rentals.csv": _LISTING_COLS
    | {
        "warmmiete_eur_monthly",
        "rent_per_m2_kalt_eur",
    },
    "new_construction.csv": _LISTING_COLS | {"price_eur", "price_per_m2_eur"},
    "kiez_prices_monthly.csv": {
        "year_month",
        "ortsteil",
        "bezirk",
        "secondary_price_per_m2_eur",
        "kaltmiete_per_m2_monthly_eur",
        "n_listings_secondary",
        "n_listings_rental",
        "ecb_main_rate_pct",
        "avg_mortgage_rate_pct",
    },
    "transit_stations.csv": {"station_name", "line", "lat", "lon"},
}


class SchemaError(ValueError):
    """Raised when a raw source file is missing required columns."""


def validate_file(fname: str, df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS[fname] - set(df.columns)
    if missing:
        raise SchemaError(f"{fname} is missing required columns: {sorted(missing)}")
    expected_rows = EXPECTED_ROWS[fname]
    if len(df) != expected_rows:
        logger.warning(f"    {fname:32s} {len(df):>7,} rows  (expected {expected_rows:,})")
    else:
        logger.info(f"    {fname:32s} {len(df):>7,} rows  OK")


def run():
    logger.info("validating raw Kaggle source files")
    for fname in EXPECTED_ROWS:
        path = RAW_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- place the Kaggle CSV export in data/raw/")
        validate_file(fname, pd.read_csv(path))
    logger.info("all source files present with required columns")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
