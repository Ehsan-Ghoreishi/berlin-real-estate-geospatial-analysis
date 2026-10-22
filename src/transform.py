"""
transform.py
============
Cleans the three listing tables (secondary sales, rentals, new
construction) and adds one genuinely new geospatial feature: an
independently-computed nearest-transit-station distance via GeoPandas +
a scipy KD-tree in a projected metric CRS, built directly from the 135
real station coordinates --
NOT copied from the dataset's own `transit_distance_min` column.

This independent computation serves two purposes:
  1. It's real GeoPandas/KD-tree engineering work, not just column reuse.
  2. It doubles as a data-quality check: correlating our computed
     straight-line distance against the dataset's provided walking-time
     column is a legitimate validation step, and a defensible thing to
     show in an interview ("I didn't just trust the source column").
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.config import METRIC_CRS, PROC_DIR, RAW_DIR, WGS84_CRS, ensure_dirs

logger = logging.getLogger(__name__)


def load_raw():
    sales = pd.read_csv(RAW_DIR / "secondary_sales.csv", parse_dates=["date_listed"])
    rentals = pd.read_csv(RAW_DIR / "rentals.csv", parse_dates=["date_listed"])
    new_build = pd.read_csv(RAW_DIR / "new_construction.csv", parse_dates=["date_listed"])
    kiez_monthly = pd.read_csv(RAW_DIR / "kiez_prices_monthly.csv")
    stations = pd.read_csv(RAW_DIR / "transit_stations.csv")
    return sales, rentals, new_build, kiez_monthly, stations


def clean_percentile(df: pd.DataFrame, col: str, lo_q=0.01, hi_q=0.99) -> pd.DataFrame:
    lo, hi = df[col].quantile([lo_q, hi_q])
    before = len(df)
    df = df[df[col].between(lo, hi)]
    logger.info(f"{col}: dropped {before - len(df)} outliers outside [{lo:,.0f}, {hi:,.0f}]")
    return df


def nearest_station_kdtree(listings: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Vectorized nearest-station lookup using a KD-tree in projected
    (metric) space -- the scalable approach, appropriate here since we
    have 50k+ listings against 135 stations (unlike the small demo in
    the original prototype, brute force would be too slow at this scale)."""
    listings_gdf = gpd.GeoDataFrame(
        listings,
        geometry=gpd.points_from_xy(listings["lon"], listings["lat"]),
        crs=WGS84_CRS,
    ).to_crs(epsg=METRIC_CRS)  # ETRS89 / UTM zone 33N -- correct metric CRS for Berlin

    stations_gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
        crs=WGS84_CRS,
    ).to_crs(epsg=METRIC_CRS)

    station_coords = np.column_stack([stations_gdf.geometry.x, stations_gdf.geometry.y])
    listing_coords = np.column_stack([listings_gdf.geometry.x, listings_gdf.geometry.y])

    tree = cKDTree(station_coords)
    dist_m, idx = tree.query(listing_coords, k=1)

    listings = listings.copy()
    listings["computed_dist_nearest_station_km"] = (dist_m / 1000).round(3)
    listings["computed_nearest_station"] = stations.iloc[idx]["station_name"].values
    return listings


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["listing_year"] = df["date_listed"].dt.year
    df["listing_month"] = df["date_listed"].dt.month
    df["months_since_2020"] = (df["date_listed"].dt.year - 2020) * 12 + df["date_listed"].dt.month
    return df


def clean_sales(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["price_eur", "area_m2", "lat", "lon"]).copy()
    df = df[df["area_m2"].between(15, 400)]
    df = clean_percentile(df, "price_per_m2_eur")
    df = nearest_station_kdtree(df, stations)
    df = add_time_features(df)
    df["dist_center_km"] = df["to_brandenburg_gate_km"]
    return df.reset_index(drop=True)


def clean_rentals(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["warmmiete_eur_monthly", "area_m2", "lat", "lon"]).copy()
    df = df[df["area_m2"].between(15, 300)]
    df = clean_percentile(df, "rent_per_m2_kalt_eur")
    df = nearest_station_kdtree(df, stations)
    df = add_time_features(df)
    return df.reset_index(drop=True)


def clean_new_construction(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["price_eur", "area_m2", "lat", "lon"]).copy()
    df = clean_percentile(df, "price_per_m2_eur")
    df = nearest_station_kdtree(df, stations)
    df = add_time_features(df)
    return df.reset_index(drop=True)


def validate_distance_columns(sales: pd.DataFrame):
    """Sanity check: our independently-computed nearest-station distance
    should correlate with the dataset's own transit_distance_min (walking
    minutes). Prints Pearson correlation as a lightweight data-quality
    signal, not a formal test."""
    corr = sales["computed_dist_nearest_station_km"].corr(sales["transit_distance_min"])
    logger.info(
        f"validation: corr(computed_km, provided_transit_min) = {corr:.3f} "
        f"(expect positive -- further station distance -> longer walk time)"
    )


def run():
    ensure_dirs()
    sales, rentals, new_build, kiez_monthly, stations = load_raw()

    sales = clean_sales(sales, stations)
    rentals = clean_rentals(rentals, stations)
    new_build = clean_new_construction(new_build, stations)

    validate_distance_columns(sales)

    sales.to_csv(PROC_DIR / "sales_clean.csv", index=False)
    rentals.to_csv(PROC_DIR / "rentals_clean.csv", index=False)
    new_build.to_csv(PROC_DIR / "new_construction_clean.csv", index=False)
    kiez_monthly.to_csv(PROC_DIR / "kiez_monthly_clean.csv", index=False)
    stations.to_csv(PROC_DIR / "stations_clean.csv", index=False)

    logger.info(f"sales:          {len(sales):,} rows")
    logger.info(f"rentals:        {len(rentals):,} rows")
    logger.info(f"new_construction: {len(new_build):,} rows")

    # GeoPackage for the flagship sales table (GIS-ready artifact)
    gdf = gpd.GeoDataFrame(
        sales, geometry=gpd.points_from_xy(sales["lon"], sales["lat"]), crs=WGS84_CRS
    )
    gdf.to_file(PROC_DIR / "sales_clean.gpkg", driver="GPKG", layer="sales")
    logger.info(f"wrote GeoPackage -> {PROC_DIR / 'sales_clean.gpkg'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
