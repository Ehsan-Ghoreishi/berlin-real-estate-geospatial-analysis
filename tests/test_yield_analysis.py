import pandas as pd
import pytest

from src.yield_analysis import (
    citywide_yield_series,
    gross_yield_pct,
    kiez_cross_section_spread,
    ortsteil_yields,
)


def test_gross_yield_formula():
    # 10 EUR/m2/month * 12 / 4000 EUR/m2 = 3%
    assert gross_yield_pct(10, 4000) == pytest.approx(3.0)


def test_cross_section_spread_zero_for_constant_ratio(kiez_monthly):
    # fixture uses the same 1:400 rent/price ratio in every Ortsteil-month
    assert kiez_cross_section_spread(kiez_monthly) == pytest.approx(0, abs=1e-9)


def test_citywide_series_one_row_per_month(kiez_monthly):
    series = citywide_yield_series(kiez_monthly)
    assert series["year_month"].tolist() == ["2024-01", "2024-02"]


def _listings(ortsteil, n, months_ago, col, value):
    dates = pd.Timestamp("2026-06-01") - pd.DateOffset(months=months_ago)
    return pd.DataFrame(
        {"ortsteil": ortsteil, "bezirk": "X", "date_listed": [dates] * n, col: value}
    )


def test_ortsteil_yields_ranks_and_filters():
    sales = pd.concat(
        [
            _listings("Cheap", 40, 1, "price_per_m2_eur", 3000.0),
            _listings("Pricey", 40, 1, "price_per_m2_eur", 6000.0),
            _listings("Thin", 5, 1, "price_per_m2_eur", 3000.0),
        ]
    )
    rentals = pd.concat(
        [
            _listings("Cheap", 40, 1, "rent_per_m2_kalt_eur", 12.0),
            _listings("Pricey", 40, 1, "rent_per_m2_kalt_eur", 15.0),
            _listings("Thin", 40, 1, "rent_per_m2_kalt_eur", 12.0),
        ]
    )
    out = ortsteil_yields(sales, rentals, min_listings=30)
    assert out["ortsteil"].tolist() == ["Cheap", "Pricey"]  # Thin dropped, sorted by yield
    assert out.loc[0, "gross_yield_pct"] == pytest.approx(4.8)
    assert out.loc[1, "price_to_rent"] == pytest.approx(33.3)


def test_ortsteil_yields_ignores_old_listings():
    sales = pd.concat(
        [
            _listings("A", 40, 1, "price_per_m2_eur", 4000.0),
            _listings("A", 400, 24, "price_per_m2_eur", 1000.0),  # outside 12-month window
        ]
    )
    rentals = _listings("A", 40, 1, "rent_per_m2_kalt_eur", 10.0)
    out = ortsteil_yields(sales, rentals)
    assert out.loc[0, "sales_eur_m2"] == 4000
