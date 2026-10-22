"""
visualize.py
============
Interactive Folium map (sampled secondary-sales listings, colour-coded by
EUR/m^2, plus all 135 real transit stations) and a static Matplotlib
fallback map, same basemap-fallback pattern as the original prototype.

Sampling: 49k markers would make an unusably heavy HTML file, so the
interactive map draws a stratified sample (proportional per Bezirk) of
2,000 listings -- enough to see the spatial price pattern clearly
without a multi-hundred-MB HTML file.
"""

import logging
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import MAP_DIR, PROC_DIR, RANDOM_STATE, WGS84_CRS, ensure_dirs

logger = logging.getLogger(__name__)

SAMPLE_SIZE = 2000


def price_color(p: float, lo: float, hi: float) -> str:
    t = 0.0 if hi == lo else max(0.0, min(1.0, (p - lo) / (hi - lo)))
    r = int(255 * t)
    g = int(255 * (1 - t))
    return f"#{r:02x}{g:02x}00"


def stratified_sample(df: pd.DataFrame, n: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    frac = n / len(df)
    parts = [g.sample(frac=frac, random_state=seed) for _, g in df.groupby("bezirk")]
    return pd.concat(parts, ignore_index=True)


def build_interactive_map(sample: pd.DataFrame, stations: pd.DataFrame) -> Path:
    m = folium.Map(location=[52.52, 13.405], zoom_start=11, tiles="cartodbpositron")

    lo, hi = sample["price_per_m2_eur"].quantile([0.05, 0.95])

    listings_fg = folium.FeatureGroup(name=f"Sales listings (n={len(sample):,} sample)")
    for _, r in sample.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=4,
            color=price_color(r["price_per_m2_eur"], lo, hi),
            fill=True,
            fill_opacity=0.75,
            weight=0.5,
            popup=(
                f"{r['ortsteil']}, {r['bezirk']}<br>"
                f"{r['area_m2']} m2, {r['rooms']} rooms, built {r['year_built']}<br>"
                f"energy class {r['energy_class']}, {r['condition']}<br>"
                f"€{r['price_eur']:,.0f} (€{r['price_per_m2_eur']:.0f}/m2)<br>"
                f"nearest: {r['computed_nearest_station']} "
                f"({r['computed_dist_nearest_station_km']:.2f} km)"
            ),
        ).add_to(listings_fg)
    listings_fg.add_to(m)

    stations_fg = folium.FeatureGroup(name="Transit stations (135 real)")
    for _, s in stations.iterrows():
        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=2,
            color="#0055ff",
            fill=True,
            fill_opacity=0.9,
            popup=f"{s['station_name']} ({s['line']})",
        ).add_to(stations_fg)
    stations_fg.add_to(m)

    folium.LayerControl().add_to(m)

    out_path = MAP_DIR / "berlin_prices_interactive.html"
    m.save(str(out_path))
    return out_path


def build_static_map(sample: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 9))
    basemap_ok = False
    try:
        import contextily as cx
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(
            sample,
            geometry=gpd.points_from_xy(sample["lon"], sample["lat"]),
            crs=WGS84_CRS,
        ).to_crs(epsg=3857)
        sc = ax.scatter(
            gdf.geometry.x,
            gdf.geometry.y,
            c=sample["price_per_m2_eur"],
            cmap="RdYlGn_r",
            s=8,
            alpha=0.7,
        )
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, timeout=3)
        ax.set_axis_off()
        basemap_ok = True
    except Exception as e:
        logger.info(f"basemap tiles unavailable ({type(e).__name__}), falling back")
        ax.clear()

    if not basemap_ok:
        sc = ax.scatter(
            sample["lon"],
            sample["lat"],
            c=sample["price_per_m2_eur"],
            cmap="RdYlGn_r",
            s=8,
            alpha=0.7,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("EUR / m²")
    ax.set_title(f"Berlin secondary-market sales, coloured by EUR/m²\n(n={len(sample):,} sample)")

    out_path = MAP_DIR / "berlin_prices_static.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run() -> None:
    ensure_dirs()
    sales = pd.read_csv(PROC_DIR / "sales_clean.csv")
    stations = pd.read_csv(PROC_DIR / "stations_clean.csv")

    sample = stratified_sample(sales, SAMPLE_SIZE)

    html_path = build_interactive_map(sample, stations)
    logger.info(f"wrote interactive map -> {html_path}")

    png_path = build_static_map(sample)
    logger.info(f"wrote static map -> {png_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
