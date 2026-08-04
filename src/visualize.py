"""
visualize.py
============
Two visual outputs:

1. maps/berlin_prices_interactive.html
   Folium map, one circle marker per listing colour-coded by
   price_per_sqm (red = expensive, green = cheap), plus station
   markers. Standalone HTML -- embeddable in a portfolio page.

2. maps/berlin_prices_static.png
   Matplotlib scatter of listings coloured by price_per_sqm, with an
   OpenStreetMap basemap via contextily when network access to tile
   servers is available, falling back to a plain axes plot (with
   district colouring) when it is not -- this sandbox has no route to
   tile servers, so the fallback path is what actually renders here.
"""

from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MAP_DIR = Path(__file__).resolve().parents[1] / "maps"
MAP_DIR.mkdir(parents=True, exist_ok=True)


def price_color(p, lo, hi):
    """green -> yellow -> red on price_per_sqm."""
    t = 0.0 if hi == lo else max(0.0, min(1.0, (p - lo) / (hi - lo)))
    r = int(255 * t)
    g = int(255 * (1 - t))
    return f"#{r:02x}{g:02x}00"


def build_interactive_map(df: pd.DataFrame, stations: pd.DataFrame) -> Path:
    m = folium.Map(location=[52.52, 13.405], zoom_start=11, tiles="cartodbpositron")

    lo, hi = df["price_per_sqm"].quantile([0.05, 0.95])

    listings_fg = folium.FeatureGroup(name="Listings")
    for _, r in df.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=4,
            color=price_color(r["price_per_sqm"], lo, hi),
            fill=True,
            fill_opacity=0.75,
            weight=0.5,
            popup=(
                f"{r['bezirk']}<br>"
                f"{r['size_sqm']} sqm, {r['rooms']} rooms<br>"
                f"€{r['price_eur']:,.0f} (€{r['price_per_sqm']:.0f}/sqm)<br>"
                f"nearest: {r['nearest_station']} "
                f"({r['dist_nearest_station_km']:.2f} km)"
            ),
        ).add_to(listings_fg)
    listings_fg.add_to(m)

    stations_fg = folium.FeatureGroup(name="Stations")
    for _, s in stations.iterrows():
        folium.Marker(
            location=[s["lat"], s["lon"]],
            icon=folium.Icon(color="blue", icon="train", prefix="fa"),
            popup=s["station_name"],
        ).add_to(stations_fg)
    stations_fg.add_to(m)

    folium.LayerControl().add_to(m)

    out_path = MAP_DIR / "berlin_prices_interactive.html"
    m.save(str(out_path))
    return out_path


def build_static_map(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 9))

    basemap_ok = False
    try:
        import contextily as cx
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            df, geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])],
            crs="EPSG:4326",
        ).to_crs(epsg=3857)
        sc = ax.scatter(
            gdf.geometry.x, gdf.geometry.y, c=df["price_per_sqm"],
            cmap="RdYlGn_r", s=10, alpha=0.8,
        )
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, timeout=3)
        ax.set_axis_off()
        basemap_ok = True
    except Exception as e:
        print(f"[visualize] basemap tiles unavailable ({type(e).__name__}), "
              f"falling back to plain scatter")
        ax.clear()  # discard any partial projected-CRS scatter drawn above

    if not basemap_ok:
        sc = ax.scatter(
            df["lon"], df["lat"], c=df["price_per_sqm"],
            cmap="RdYlGn_r", s=10, alpha=0.8,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("EUR / sqm")
    ax.set_title("Berlin listings coloured by price per sqm")

    out_path = MAP_DIR / "berlin_prices_static.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run():
    df = pd.read_csv(PROC_DIR / "listings_clean.csv")
    stations = pd.read_csv(RAW_DIR / "berlin_stations.csv")

    html_path = build_interactive_map(df, stations)
    print(f"[visualize] wrote interactive map -> {html_path}")

    png_path = build_static_map(df)
    print(f"[visualize] wrote static map -> {png_path}")


if __name__ == "__main__":
    run()
