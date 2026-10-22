import pandas as pd
import pytest

from src.transform import add_time_features, clean_percentile, nearest_station_kdtree


def test_clean_percentile_drops_extremes():
    df = pd.DataFrame({"v": list(range(1, 101))})
    out = clean_percentile(df, "v", lo_q=0.05, hi_q=0.95)
    assert out["v"].min() >= 5
    assert out["v"].max() <= 96
    assert len(out) < len(df)


def test_nearest_station_picks_closest(stations):
    listings = pd.DataFrame({
        "lat": [52.5215, 52.5070],  # next to Alexanderplatz, next to Zoo
        "lon": [13.4110, 13.3330],
    })
    out = nearest_station_kdtree(listings, stations)
    assert out["computed_nearest_station"].tolist() == ["Alexanderplatz", "Zoologischer Garten"]
    assert (out["computed_dist_nearest_station_km"] < 0.3).all()


def test_nearest_station_distance_is_metric(stations):
    # Alexanderplatz -> Zoologischer Garten is ~5.7 km as the crow flies
    listing = pd.DataFrame({"lat": [52.5219], "lon": [13.4132]})
    far = stations[stations["station_name"] == "Zoologischer Garten"].reset_index(drop=True)
    out = nearest_station_kdtree(listing, far)
    assert out["computed_dist_nearest_station_km"].iloc[0] == pytest.approx(5.7, abs=0.3)


def test_nearest_station_does_not_mutate_input(stations):
    listings = pd.DataFrame({"lat": [52.52], "lon": [13.41]})
    nearest_station_kdtree(listings, stations)
    assert list(listings.columns) == ["lat", "lon"]


def test_add_time_features():
    df = pd.DataFrame({"date_listed": pd.to_datetime(["2020-01-15", "2023-06-01"])})
    out = add_time_features(df)
    assert out["listing_year"].tolist() == [2020, 2023]
    assert out["listing_month"].tolist() == [1, 6]
    assert out["months_since_2020"].tolist() == [1, 42]
