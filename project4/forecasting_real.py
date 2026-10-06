"""
PROJECT 4: FINANCIAL FORECASTING (MONTHLY REVENUE / SALES)
==========================================================
Workflow: prepare data -> time-based train/test split -> baseline models ->
forecast models -> evaluate with RMSE and MAPE -> forecast the next 12 months.

DATA POLICY
This portfolio version is REAL-DATA ONLY. The dataset is the U.S. Census Bureau's
monthly, seasonally adjusted retail sales series (FRED: MRTSSM44X72USS).
The script fails loudly if the sourced CSV is absent.

MODELS
  1. Naive                  - next month = last month (the "do nothing" baseline)
  2. Seasonal naive         - next month = same month last year (strong baseline)
  3. Trend + seasonality regression (numpy least squares)
  4. Holt-Winters           - exponential smoothing (needs: pip install statsmodels;
                              skipped automatically if not installed)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path("data/monthly_revenue.csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
DATA_FILE.parent.mkdir(exist_ok=True)

TEST_MONTHS = 12        # hold out the most recent 12 months for testing
SEASON = 12             # monthly data -> yearly seasonality
FORECAST_HORIZON = 12   # months to forecast beyond the data


# ----------------------------------------------------------------------------
# 1. REAL DATA
# ----------------------------------------------------------------------------
def load_data(path: Path) -> pd.Series:
    """Load, clean and return the real monthly series."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df = df.drop_duplicates("date").set_index("date")
    if "revenue" not in df.columns:
        raise ValueError("Expected columns: date, revenue")
    s = pd.to_numeric(df["revenue"], errors="coerce").asfreq("MS")
    n_missing = int(s.isna().sum())
    if n_missing:
        raise ValueError(f"Real dataset contains {n_missing} missing months; do not interpolate sourced observations.")
    return s

def require_real_data(path: Path) -> None:
    """Fail loudly instead of silently creating synthetic observations."""
    if not path.exists():
        raise FileNotFoundError(
            f"Real dataset not found at {path}. Place the sourced FRED CSV there."
        )


# ----------------------------------------------------------------------------
# 2. MODELS - each takes (train series, number of months to forecast)
# ----------------------------------------------------------------------------
def naive_forecast(train: pd.Series, h: int) -> np.ndarray:
    return np.repeat(train.iloc[-1], h)


def seasonal_naive_forecast(train: pd.Series, h: int) -> np.ndarray:
    last_season = train.iloc[-SEASON:].to_numpy()
    return np.array([last_season[i % SEASON] for i in range(h)])


def _design_matrix(index: pd.DatetimeIndex, t0: pd.Timestamp) -> np.ndarray:
    """Columns: intercept, linear time trend, 11 month dummies."""
    t = ((index.year - t0.year) * 12 + (index.month - t0.month)).to_numpy()
    months = pd.get_dummies(index.month).reindex(columns=range(2, 13), fill_value=0)
    return np.column_stack([np.ones(len(index)), t, months.to_numpy(dtype=float)])


def regression_forecast(train: pd.Series, h: int) -> np.ndarray:
    X = _design_matrix(train.index, train.index[0])
    coef, *_ = np.linalg.lstsq(X, train.to_numpy(), rcond=None)
    future_idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS")
    return _design_matrix(future_idx, train.index[0]) @ coef


def holt_winters_forecast(train: pd.Series, h: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(train, trend="add", seasonal="add",
                                 seasonal_periods=SEASON).fit()
    return model.forecast(h).to_numpy()


MODELS = {
    "Naive": naive_forecast,
    "Seasonal naive": seasonal_naive_forecast,
    "Trend + season regression": regression_forecast,
    "Holt-Winters": holt_winters_forecast,
}


# ----------------------------------------------------------------------------
# 3. EVALUATION
# ----------------------------------------------------------------------------
def rmse(actual, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2)))


def mape(actual, pred) -> float:
    actual, pred = np.asarray(actual), np.asarray(pred)
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


# ----------------------------------------------------------------------------
# 4. CHARTS
# ----------------------------------------------------------------------------
def plot_backtest(train, test, preds, results) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(train.index[-36:], train.iloc[-36:], color="grey", label="Training data")
    ax.plot(test.index, test, color="black", lw=2.5, label="Actual (test)")
    for name, p in preds.items():
        ax.plot(test.index, p, lw=1.8, ls="--", label=name)
    ax.set_title("Backtest: last 12 months held out", loc="left", fontweight="bold")
    ax.set_ylabel("Revenue (USD m)"); ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    order = results.sort_values("MAPE %")
    bars = ax.barh(order.index, order["MAPE %"], color="#1f4e79")
    for b, (r, m) in zip(bars, zip(order["RMSE"], order["MAPE %"])):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {m:.1f}%  (RMSE {r:.1f})",
                va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, order["MAPE %"].max() * 1.5)
    ax.set_title("Test error (lower is better)", loc="left", fontweight="bold")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "backtest_comparison.png", dpi=200)
    plt.close(fig)


def plot_future(series, future_idx, forecast, band) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(series.index, series, color="#1f4e79", label="History")
    ax.plot(future_idx, forecast, color="#e67e22", lw=2.5, marker="o", ms=4, label="Forecast")
    ax.fill_between(future_idx, forecast - band, forecast + band, color="#e67e22", alpha=0.2,
                    label="Approx. range (+/- 1.96 x test RMSE)")
    ax.set_title("12-month revenue forecast", loc="left", fontweight="bold", fontsize=13)
    ax.set_ylabel("Revenue (USD m)")
    ax.legend(frameon=False); ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "future_forecast.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------------
def main():
    require_real_data(DATA_FILE)
    series = load_data(DATA_FILE)
    print(f"Data: {series.index[0]:%b %Y} to {series.index[-1]:%b %Y} ({len(series)} months)")

    # Time-based split: NEVER shuffle time-series data (that leaks the future)
    train, test = series.iloc[:-TEST_MONTHS], series.iloc[-TEST_MONTHS:]

    preds, rows = {}, {}
    for name, fn in MODELS.items():
        try:
            p = fn(train, len(test))
        except ImportError as exc:
            raise RuntimeError("statsmodels is required for the Holt-Winters comparison") from exc
        preds[name] = p
        rows[name] = {"RMSE": rmse(test, p), "MAPE %": mape(test, p)}
    results = pd.DataFrame(rows).T.round(2)
    results.to_csv(OUT_DIR / "model_comparison.csv")
    print("\n=== TEST-SET ERRORS (last 12 months) ===")
    print(results.sort_values("MAPE %").to_string())

    best = results["MAPE %"].idxmin()
    base = results.loc["Seasonal naive", "MAPE %"]
    print(f"\nBest model: {best} (MAPE {results.loc[best, 'MAPE %']:.1f}% vs "
          f"seasonal-naive {base:.1f}%)")

    plot_backtest(train, test, preds, results)

    # Refit best model on ALL data and forecast the future
    forecast = MODELS[best](series, FORECAST_HORIZON)
    future_idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(),
                               periods=FORECAST_HORIZON, freq="MS")
    band = 1.96 * results.loc[best, "RMSE"]
    plot_future(series, future_idx, forecast, band)
    pd.DataFrame({"forecast": forecast.round(1)}, index=future_idx).to_csv(
        OUT_DIR / "forecast_next_12_months.csv")

    # Residual diagnostics: are errors biased in one direction?
    bias = float(np.mean(test.to_numpy() - preds[best]))
    print(f"Average forecast error (bias) of {best}: {bias:+.2f} (positive = under-forecasting)")
    print(f"Next-12-month forecast total: {forecast.sum():,.0f} vs last 12 months actual: "
          f"{series.iloc[-12:].sum():,.0f} ({forecast.sum() / series.iloc[-12:].sum() - 1:+.1%})")
    print(f"Saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
