import pytest

from src.policy_analysis import weighted_monthly_index


def test_index_is_listing_count_weighted(kiez_monthly):
    idx = weighted_monthly_index(kiez_monthly)
    jan = idx[idx["year_month"] == "2024-01"].iloc[0]
    # sales: (4000*30 + 8000*10) / 40 = 5000, not the naive mean of 6000
    assert jan["sales_eur_m2"] == pytest.approx(5000)
    # rent: (10*10 + 20*30) / 40 = 17.5
    assert jan["rent_eur_m2"] == pytest.approx(17.5)


def test_index_one_row_per_month_sorted(kiez_monthly):
    idx = weighted_monthly_index(kiez_monthly.sample(frac=1, random_state=0))
    assert idx["year_month"].tolist() == ["2024-01", "2024-02"]
    assert idx["date"].is_monotonic_increasing


def test_zero_listing_months_do_not_divide_by_zero(kiez_monthly):
    kiez_monthly["n_listings_secondary"] = 0
    idx = weighted_monthly_index(kiez_monthly)
    assert idx["sales_eur_m2"].notna().all()
