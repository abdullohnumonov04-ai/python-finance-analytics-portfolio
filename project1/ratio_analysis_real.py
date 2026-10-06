"""
PROJECT 1: COMPANY RATIO ANALYSIS
=================================
Compares 3 companies from the same industry over 5 years using six core ratios
plus a DuPont breakdown of ROE.

DATA POLICY
This portfolio version is REAL-DATA ONLY. It never creates synthetic company data.
Place the sourced dataset at data/financials.csv; the script fails loudly if it is absent.

Units: all money columns in the same currency unit (e.g. USD millions).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 0. SETTINGS
# ----------------------------------------------------------------------------
DATA_FILE = Path("data/financials.csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
DATA_FILE.parent.mkdir(exist_ok=True)

REQUIRED_COLS = [
    "company", "year", "revenue", "net_income", "total_assets",
    "total_equity", "total_debt", "current_assets", "current_liabilities",
]

# Consistent colours so each company looks the same on every chart
COLOURS = {}  # filled after data is loaded


# ----------------------------------------------------------------------------
# 1. REAL DATASET
# ----------------------------------------------------------------------------
def require_real_data(path: Path) -> None:
    """Fail loudly instead of silently creating or using fictional data."""
    if not path.exists():
        raise FileNotFoundError(
            f"Real dataset not found at {path}. "
            "Download/copy the sourced annual-report dataset before running."
        )


# ----------------------------------------------------------------------------
# 2. LOAD + CLEAN DATA
# ----------------------------------------------------------------------------
def load_and_clean(path: Path) -> pd.DataFrame:
    """Load the CSV, standardise it and run data-quality checks."""
    df = pd.read_csv(path)

    # Standardise column names: "Net Income " -> "net_income"
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))

    missing_cols = set(REQUIRED_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {sorted(missing_cols)}")

    df["company"] = df["company"].str.strip()

    # Real-world files often contain "1,234", "(250)" for negatives, "-", "n/a"
    num_cols = [c for c in REQUIRED_COLS if c not in ("company",)]
    for col in num_cols:
        s = df[col].astype(str).str.strip()
        s = s.str.replace(",", "", regex=False)
        s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)   # (250) -> -250
        df[col] = pd.to_numeric(s.replace({"-": np.nan, "n/a": np.nan, "": np.nan}),
                                errors="coerce")

    # Remove duplicates and sort
    before = len(df)
    df = df.drop_duplicates(subset=["company", "year"]).sort_values(["company", "year"])
    if len(df) < before:
        print(f"[clean] Removed {before - len(df)} duplicate company-year rows")

    # Missing values: report only. Never fabricate missing annual-report figures.
    n_missing = int(df[num_cols].isna().sum().sum())
    if n_missing:
        print(f"[clean] {n_missing} missing numeric values retained as NaN (no imputation)")

    # Sanity checks an analyst would always run
    assert (df["revenue"] > 0).all(), "Revenue must be positive"
    assert (df["total_equity"] > 0).all(), "Negative equity breaks ROE/D-E - investigate"
    assert (df["total_equity"] < df["total_assets"]).all(), "Equity should be below assets"
    years_per_co = df.groupby("company")["year"].nunique()
    if years_per_co.nunique() != 1:
        print(f"[warn] Companies have different numbers of years:\n{years_per_co}")

    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. CALCULATE RATIOS
# ----------------------------------------------------------------------------
def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add the six core ratios plus DuPont components (year-end balances)."""
    r = df.copy()
    r["roe_pct"] = r["net_income"] / r["total_equity"] * 100
    r["roa_pct"] = r["net_income"] / r["total_assets"] * 100
    r["net_margin_pct"] = r["net_income"] / r["revenue"] * 100
    r["current_ratio"] = r["current_assets"] / r["current_liabilities"]
    r["debt_to_equity"] = r["total_debt"] / r["total_equity"]
    r["asset_turnover"] = r["revenue"] / r["total_assets"]

    # DuPont: ROE = Net margin x Asset turnover x Equity multiplier
    r["equity_multiplier"] = r["total_assets"] / r["total_equity"]
    r["roe_dupont_pct"] = (r["net_income"] / r["revenue"] * r["asset_turnover"]
                           * r["equity_multiplier"] * 100)
    assert np.allclose(r["roe_pct"], r["roe_dupont_pct"]), "DuPont check failed"
    return r


# ----------------------------------------------------------------------------
# 4. SCORECARD: WHO IS BEST, WHO IMPROVED?
# ----------------------------------------------------------------------------
def build_scorecard(r: pd.DataFrame) -> pd.DataFrame:
    """Rank companies on 5-year average level and on change over time."""
    first_year, last_year = r["year"].min(), r["year"].max()
    metrics = ["roe_pct", "roa_pct", "net_margin_pct", "current_ratio",
               "debt_to_equity", "asset_turnover"]

    avg = r.groupby("company")[metrics].mean()
    # Rank each metric on observed values only; no missing value is converted to a fake number.
    change = (r[r["year"] == last_year].set_index("company")[metrics]
              - r[r["year"] == first_year].set_index("company")[metrics])

    # Ranking: 1 = best. Lower debt-to-equity is better; higher is better for the rest
    ranks = avg.rank(ascending=False, na_option="keep")
    ranks["debt_to_equity"] = avg["debt_to_equity"].rank(ascending=True)

    score = pd.DataFrame({
        "avg_rank_observed_ratios": ranks.mean(axis=1),
        "avg_ROE_%": avg["roe_pct"],
        f"ROE_change_{first_year}_{last_year}_pp": change["roe_pct"],
        f"margin_change_{first_year}_{last_year}_pp": change["net_margin_pct"],
        f"D/E_change_{first_year}_{last_year}": change["debt_to_equity"],
    }).sort_values("avg_rank_observed_ratios")
    return score.round(2)


# ----------------------------------------------------------------------------
# 5. CHARTS
# ----------------------------------------------------------------------------
def style_axes(ax, title, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_ratio_panels(r: pd.DataFrame) -> None:
    """Six-panel trend chart: one panel per ratio, one line per company."""
    panels = [
        ("roe_pct", "Return on Equity (ROE)", "%"),
        ("roa_pct", "Return on Assets (ROA)", "%"),
        ("net_margin_pct", "Net Profit Margin", "%"),
        ("current_ratio", "Current Ratio", "x"),
        ("debt_to_equity", "Debt-to-Equity", "x"),
        ("asset_turnover", "Asset Turnover", "x"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, title, unit) in zip(axes.ravel(), panels):
        for co, g in r.groupby("company"):
            ax.plot(g["year"], g[col], marker="o", lw=2, label=co, color=COLOURS[co])
        style_axes(ax, title, unit)
        ax.set_xticks(sorted(r["year"].unique()))
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Retail Peer Group: Key Financial Ratios", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.01, "Source: company annual reports / SEC filings; all populated values are real sourced data.",
             fontsize=8, color="grey")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUT_DIR / "ratio_trends.png", dpi=200)
    plt.close(fig)


def plot_dupont(r: pd.DataFrame) -> None:
    """Show WHY ROE differs: margin vs. efficiency vs. leverage (latest year)."""
    latest = r[r["year"] == r["year"].max()].set_index("company")
    comps = [("net_margin_pct", "Net margin (%)"),
             ("asset_turnover", "Asset turnover (x)"),
             ("equity_multiplier", "Equity multiplier (x)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (col, title) in zip(axes, comps):
        ax.bar(latest.index, latest[col], color=[COLOURS[c] for c in latest.index])
        for i, v in enumerate(latest[col]):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        style_axes(ax, title, "")
    fig.suptitle(f"DuPont Breakdown of ROE, {r['year'].max()}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "dupont_breakdown.png", dpi=200)
    plt.close(fig)


def plot_improvement(r: pd.DataFrame) -> None:
    """Bar chart of ROE change first -> last year (who improved?)."""
    y0, y1 = r["year"].min(), r["year"].max()
    roe = r.pivot(index="company", columns="year", values="roe_pct")
    change = (roe[y1] - roe[y0]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(change.index, change.values,
            color=["#c0392b" if v < 0 else "#27ae60" for v in change.values])
    for i, v in enumerate(change.values):
        ax.text(v, i, f" {v:+.1f} pp", va="center", ha="left" if v >= 0 else "right")
    ax.axvline(0, color="black", lw=0.8)
    style_axes(ax, f"Change in ROE, {y0} to {y1} (percentage points)", "")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roe_improvement.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------------
def main():
    require_real_data(DATA_FILE)
    df = load_and_clean(DATA_FILE)
    ratios = calculate_ratios(df)

    palette = ["#1f4e79", "#e67e22", "#7f8c8d", "#8e44ad"]
    for co, colour in zip(sorted(ratios["company"].unique()), palette):
        COLOURS[co] = colour

    ratios.round(3).to_csv(OUT_DIR / "ratio_table.csv", index=False)
    scorecard = build_scorecard(ratios)
    scorecard.to_csv(OUT_DIR / "scorecard.csv")

    plot_ratio_panels(ratios)
    plot_dupont(ratios)
    plot_improvement(ratios)

    # Plain-English summary for the console / README
    last = ratios["year"].max()
    print("\n=== LATEST-YEAR RATIOS ===")
    cols = ["company", "roe_pct", "roa_pct", "net_margin_pct",
            "current_ratio", "debt_to_equity", "asset_turnover"]
    print(ratios[ratios["year"] == last][cols].round(2).to_string(index=False))
    print("\n=== SCORECARD (1 = best average rank) ===")
    print(scorecard.to_string())
    best = scorecard.index[0]
    roe_change_col = [c for c in scorecard.columns if c.startswith("ROE_change")][0]
    improver = scorecard[roe_change_col].idxmax()
    print(f"\nBest overall performer (average rank): {best}")
    print(f"Biggest ROE improvement: {improver}")
    print(f"Charts and tables saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
