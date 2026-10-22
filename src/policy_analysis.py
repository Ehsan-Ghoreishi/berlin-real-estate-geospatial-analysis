"""
policy_analysis.py
===================
The flagship analysis: citywide price index (2020-2026) built from
kiez_prices_monthly.csv, weighted by listing counts per Ortsteil-month,
overlaid with the ECB main refinancing rate and annotated with real
Berlin housing-policy events. This is the piece that turns the project
from "another price predictor" into a market-structure story --
directly relevant to a fintech/risk-analyst audience.

Events plotted (all real, sourced from the dataset documentation):
  2020-02-23  Mietendeckel (rent freeze) takes effect
  2021-04-15  Constitutional Court strikes down Mietendeckel
  2022-02-24  Russia-Ukraine war begins (refugee housing demand surge)
  2022-07-01  ECB starts rate hikes
  2022-10-01  Energy crisis peak (Heizkosten shock)
  2024-01-01  Heizungsgesetz (heating law) takes effect
  2024-06-01  ECB starts cutting rates
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

EVENTS = [
    ("2020-02", "Mietendeckel enacted"),
    ("2021-04", "Mietendeckel struck down"),
    ("2022-02", "Ukraine war begins"),
    ("2022-07", "ECB hikes begin"),
    ("2022-10", "Energy crisis peak"),
    ("2024-01", "Heizungsgesetz"),
    ("2024-06", "ECB cuts begin"),
]


def weighted_monthly_index(df: pd.DataFrame) -> pd.DataFrame:
    """Citywide monthly average, weighted by each Ortsteil's listing
    count that month (so a big, liquid Kiez influences the index more
    than a thin one) -- the standard approach for a hedonic-style
    citywide index rather than a naive unweighted mean of Ortsteil
    averages."""
    def wavg(g, val_col, wt_col):
        w = g[wt_col].clip(lower=1)
        return np.average(g[val_col], weights=w)

    rows = []
    for ym, g in df.groupby("year_month"):
        rows.append({
            "year_month": ym,
            "sales_eur_m2": wavg(g, "secondary_price_per_m2_eur", "n_listings_secondary"),
            "rent_eur_m2": wavg(g, "kaltmiete_per_m2_monthly_eur", "n_listings_rental"),
            "ecb_rate_pct": g["ecb_main_rate_pct"].iloc[0],
            "mortgage_rate_pct": g["avg_mortgage_rate_pct"].iloc[0],
        })
    idx = pd.DataFrame(rows).sort_values("year_month").reset_index(drop=True)
    idx["date"] = pd.to_datetime(idx["year_month"] + "-01")
    return idx


def plot_policy_shock_chart(idx: pd.DataFrame) -> Path:
    fig, ax1 = plt.subplots(figsize=(13, 7))

    ax1.plot(idx["date"], idx["sales_eur_m2"], color="#1f77b4", lw=2, label="Secondary sales EUR/m²")
    ax1.set_ylabel("Sales price (EUR / m²)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax1b = ax1.twinx()
    ax1b.plot(idx["date"], idx["rent_eur_m2"], color="#2ca02c", lw=2, label="Kaltmiete EUR/m²/month")
    ax1b.set_ylabel("Rent (EUR / m² / month)", color="#2ca02c")
    ax1b.tick_params(axis="y", labelcolor="#2ca02c")

    ax2 = ax1.twinx()
    ax2.spines["right"].set_position(("outward", 60))
    ax2.plot(idx["date"], idx["ecb_rate_pct"], color="#d62728", lw=1.5, ls="--", label="ECB rate %")
    ax2.set_ylabel("ECB main refinancing rate (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ymin, ymax = ax1.get_ylim()
    for ym, label in EVENTS:
        d = pd.to_datetime(ym + "-01")
        if d < idx["date"].min() or d > idx["date"].max():
            continue
        ax1.axvline(d, color="gray", lw=0.8, ls=":", alpha=0.7)
        ax1.text(d, ymax * 0.985, label, rotation=90, fontsize=8,
                  ha="right", va="top", color="dimgray")

    ax1.set_title("Berlin housing market, 2020-2026: sales vs rent vs ECB rate,\n"
                   "with real policy/macro shocks annotated")
    ax1.set_xlabel("")
    fig.autofmt_xdate()

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    lines3, labels3 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
               loc="upper left", fontsize=9, framealpha=0.9)

    out_path = MAP_DIR / "policy_shock_timeline.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run():
    ensure_dirs()
    df = pd.read_csv(PROC_DIR / "kiez_monthly_clean.csv")
    idx = weighted_monthly_index(df)
    idx.to_csv(PROC_DIR / "citywide_monthly_index.csv", index=False)

    # quick printed callouts for the README / write-up
    pre = idx[idx["date"] < "2021-04-15"]["rent_eur_m2"].mean()
    post = idx[(idx["date"] >= "2021-04-15") & (idx["date"] < "2021-10-01")]["rent_eur_m2"].mean()
    logger.info(f"avg kaltmiete EUR/m2 before Mietendeckel reversal: {pre:.2f}")
    logger.info(f"avg kaltmiete EUR/m2 in 6mo after reversal:        {post:.2f} "
          f"({(post/pre-1)*100:+.1f}%)")

    peak_rate_row = idx.loc[idx["ecb_rate_pct"].idxmax()]
    # local trough during the ECB-hike cycle specifically (2023), not the
    # global series minimum which is trivially the first month before
    # any multi-year growth had accumulated
    cycle_window = idx[(idx["date"] >= "2023-01-01") & (idx["date"] <= "2023-12-31")]
    trough_sales_row = cycle_window.loc[cycle_window["sales_eur_m2"].idxmin()]
    logger.info(f"ECB rate peak: {peak_rate_row['ecb_rate_pct']:.2f}% in {peak_rate_row['year_month']}")
    logger.info(f"sales price local trough during 2023 rate-hike cycle: "
          f"{trough_sales_row['sales_eur_m2']:.0f} EUR/m2 in {trough_sales_row['year_month']}")

    chart_path = plot_policy_shock_chart(idx)
    logger.info(f"wrote -> {chart_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
