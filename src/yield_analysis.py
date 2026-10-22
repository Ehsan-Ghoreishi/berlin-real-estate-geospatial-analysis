"""
yield_analysis.py
=================
Buy-vs-rent comparison for Berlin:

    gross rental yield = (Kaltmiete EUR/m2/month * 12) / sales EUR/m2
    price-to-rent      = sales EUR/m2 / (Kaltmiete EUR/m2/month * 12)

Two views, each built from the source that can actually support it:

  1. Citywide yield over time, from the monthly Kiez series (listing-
     count weighted, like the index in policy_analysis.py).
  2. Per-Ortsteil yield, from listing-level medians (sales vs rentals)
     over the most recent 12 months.

Why not use the Kiez series for (2)? A data-quality check below shows its
rent/price ratio is identical across all 82 Ortsteile within any given
month (cross-sectional std ~0.001 pp) -- the dataset generator derives
Kiez rent from Kiez price with one citywide ratio per month. A ranking
built on it would be an artifact of the generator, not a finding, so
the per-Ortsteil view uses the independently generated listing tables.

Gross yield ignores Nebenkosten, vacancy, maintenance and transaction
costs -- it's a relative comparison, not an investment return.
"""

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import MAP_DIR, PROC_DIR, ensure_dirs

logger = logging.getLogger(__name__)

WINDOW_MONTHS = 12
MIN_LISTINGS = 30
TOP_N = 12


def gross_yield_pct(rent_m2_month, sales_m2):
    return rent_m2_month * 12 / sales_m2 * 100


def citywide_yield_series(kiez: pd.DataFrame) -> pd.DataFrame:
    def wavg(g, val_col, wt_col):
        return np.average(g[val_col], weights=g[wt_col].clip(lower=1))

    rows = []
    for ym, g in kiez.groupby("year_month"):
        sales = wavg(g, "secondary_price_per_m2_eur", "n_listings_secondary")
        rent = wavg(g, "kaltmiete_per_m2_monthly_eur", "n_listings_rental")
        rows.append(
            {
                "year_month": ym,
                "gross_yield_pct": gross_yield_pct(rent, sales),
                "ecb_rate_pct": g["ecb_main_rate_pct"].iloc[0],
            }
        )
    out = pd.DataFrame(rows).sort_values("year_month").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["year_month"] + "-01")
    return out


def kiez_cross_section_spread(kiez: pd.DataFrame) -> float:
    """Mean (over months) of the cross-Ortsteil std of the yield ratio."""
    y = gross_yield_pct(kiez["kaltmiete_per_m2_monthly_eur"], kiez["secondary_price_per_m2_eur"])
    return float(y.groupby(kiez["year_month"]).std().mean())


def ortsteil_yields(
    sales: pd.DataFrame,
    rentals: pd.DataFrame,
    window_months: int = WINDOW_MONTHS,
    min_listings: int = MIN_LISTINGS,
) -> pd.DataFrame:
    """Median EUR/m2 per Ortsteil from listing-level data over the last
    `window_months` months; Ortsteile with fewer than `min_listings` on
    either side are dropped as too thin to compare."""

    def recent(df):
        dates = pd.to_datetime(df["date_listed"])
        cutoff = dates.max() - pd.DateOffset(months=window_months)
        return df[dates > cutoff]

    s = recent(sales).groupby(["ortsteil", "bezirk"])["price_per_m2_eur"].agg(["median", "size"])
    r = (
        recent(rentals)
        .groupby(["ortsteil", "bezirk"])["rent_per_m2_kalt_eur"]
        .agg(["median", "size"])
    )
    out = s.join(r, lsuffix="_sales", rsuffix="_rent", how="inner").reset_index()
    out = out[(out["size_sales"] >= min_listings) & (out["size_rent"] >= min_listings)]

    out = out.rename(
        columns={
            "median_sales": "sales_eur_m2",
            "median_rent": "kaltmiete_eur_m2_month",
            "size_sales": "n_sales",
            "size_rent": "n_rentals",
        }
    )
    out["sales_eur_m2"] = out["sales_eur_m2"].round(0)
    out["kaltmiete_eur_m2_month"] = out["kaltmiete_eur_m2_month"].round(2)
    out["gross_yield_pct"] = gross_yield_pct(
        out["kaltmiete_eur_m2_month"], out["sales_eur_m2"]
    ).round(2)
    out["price_to_rent"] = (out["sales_eur_m2"] / (out["kaltmiete_eur_m2_month"] * 12)).round(1)
    return out.sort_values("gross_yield_pct", ascending=False).reset_index(drop=True)


def plot_yields(yields: pd.DataFrame, series: pd.DataFrame, top_n: int = TOP_N) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1.2]})

    ax1.plot(
        series["date"], series["gross_yield_pct"], color="#2ca02c", lw=2, label="Gross yield %"
    )
    ax1.set_ylabel("Citywide gross rental yield (%)", color="#2ca02c")
    ax1b = ax1.twinx()
    ax1b.plot(
        series["date"], series["ecb_rate_pct"], color="#d62728", lw=1.5, ls="--", label="ECB rate %"
    )
    ax1b.set_ylabel("ECB main refinancing rate (%)", color="#d62728")
    ax1.set_title("Citywide gross yield vs ECB rate (Kiez monthly series)")
    ax1.tick_params(axis="x", labelrotation=30)

    top_n = min(top_n, len(yields) // 2)
    subset = pd.concat([yields.head(top_n), yields.tail(top_n)])
    colors = ["#2ca02c"] * top_n + ["#d62728"] * top_n
    ax2.barh(subset["ortsteil"], subset["gross_yield_pct"], color=colors)
    ax2.invert_yaxis()
    # zoom the x-axis to the data range so the spread between Ortsteile is visible
    ax2.set_xlim(subset["gross_yield_pct"].min() - 0.3, subset["gross_yield_pct"].max() + 0.1)
    ax2.axvline(
        yields["gross_yield_pct"].median(),
        color="gray",
        ls="--",
        lw=1,
        label=f"median across {len(yields)} Ortsteile",
    )
    ax2.set_xlabel("Gross rental yield (%), last 12 months, listing medians")
    ax2.set_title(f"Highest vs lowest {top_n} Ortsteile")
    ax2.legend(loc="lower right")

    out_path = MAP_DIR / "rental_yield_by_ortsteil.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run():
    ensure_dirs()
    kiez = pd.read_csv(PROC_DIR / "kiez_monthly_clean.csv")
    sales = pd.read_csv(PROC_DIR / "sales_clean.csv")
    rentals = pd.read_csv(PROC_DIR / "rentals_clean.csv")

    spread = kiez_cross_section_spread(kiez)
    logger.info(
        f"data check: cross-Ortsteil yield std in Kiez series = {spread:.4f} pp "
        f"(near zero -> per-Ortsteil view uses listing-level data instead)"
    )

    series = citywide_yield_series(kiez)
    lo, hi = (
        series.loc[series["gross_yield_pct"].idxmin()],
        series.loc[series["gross_yield_pct"].idxmax()],
    )
    logger.info(
        f"citywide yield low {lo['gross_yield_pct']:.2f}% ({lo['year_month']}), "
        f"high {hi['gross_yield_pct']:.2f}% ({hi['year_month']})"
    )

    yields = ortsteil_yields(sales, rentals)
    yields.to_csv(PROC_DIR / "ortsteil_yields.csv", index=False)
    best, worst = yields.iloc[0], yields.iloc[-1]
    logger.info(
        f"{len(yields)} Ortsteile, median gross yield {yields['gross_yield_pct'].median():.2f}%"
    )
    logger.info(f"highest: {best['ortsteil']} ({best['bezirk']}) {best['gross_yield_pct']:.2f}%")
    logger.info(f"lowest:  {worst['ortsteil']} ({worst['bezirk']}) {worst['gross_yield_pct']:.2f}%")

    chart_path = plot_yields(yields, series)
    logger.info(f"wrote -> {chart_path}")
    return yields


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
