import pandas as pd
import pytest


@pytest.fixture
def stations():
    # Three real Berlin stations, far enough apart to make nearest-neighbour unambiguous
    return pd.DataFrame(
        {
            "station_name": ["Alexanderplatz", "Zoologischer Garten", "Warschauer Straße"],
            "line": ["U2", "U2", "U1"],
            "lat": [52.5219, 52.5072, 52.5057],
            "lon": [13.4132, 13.3326, 13.4495],
        }
    )


@pytest.fixture
def kiez_monthly():
    """Two Ortsteile x two months, with listing counts that make the
    weighted and unweighted means differ."""
    return pd.DataFrame(
        {
            "year_month": ["2024-01", "2024-01", "2024-02", "2024-02"],
            "ortsteil": ["A", "B", "A", "B"],
            "bezirk": ["X", "Y", "X", "Y"],
            "secondary_price_per_m2_eur": [4000.0, 8000.0, 4400.0, 8800.0],
            "kaltmiete_per_m2_monthly_eur": [10.0, 20.0, 11.0, 22.0],
            "n_listings_secondary": [30, 10, 30, 10],
            "n_listings_rental": [10, 30, 10, 30],
            "ecb_main_rate_pct": [4.5, 4.5, 4.5, 4.5],
            "avg_mortgage_rate_pct": [3.9, 3.9, 3.8, 3.8],
        }
    )
