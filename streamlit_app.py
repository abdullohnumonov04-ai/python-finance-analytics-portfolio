from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Finance Analytics Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def money_m(value: float) -> str:
    return f"${value:,.1f}m"


def pct(value: float) -> str:
    return f"{value:.2f}%"


def show_source(text: str):
    st.caption(text)


# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
p1 = load_module("portfolio_project1", ROOT / "project1" / "ratio_analysis_real.py")
p3 = load_module("portfolio_project3", ROOT / "project3" / "dcf_valuation_real.py")
p4 = load_module("portfolio_project4", ROOT / "project4" / "forecasting_real.py")


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("Select Project")
project = st.sidebar.selectbox(
    "",
    [
        "Home",
        "Project 1 - Company Ratio Analysis",
        "Project 2 - Risk & Return Analysis",
        "Project 3 - DCF Valuation",
        "Project 4 - Financial Forecasting",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("Python • pandas • NumPy • Matplotlib • Statsmodels • yfinance • Streamlit")


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("📈 Finance Analytics Portfolio")
st.write(
    "Python finance and analytics portfolio covering financial statement analysis, "
    "market risk, DCF valuation and time-series forecasting using real sourced data."
)


# -----------------------------------------------------------------------------
# HOME
# -----------------------------------------------------------------------------
if project == "Home":
    st.subheader("Portfolio overview")
    cols = st.columns(4)
    cards = [
        ("Project 1", "Company Ratio Analysis", "ROE, ROA, margin, liquidity, leverage, DuPont"),
        ("Project 2", "Stock Risk & Return", "Returns, volatility, beta, Sharpe, VaR, drawdown"),
        ("Project 3", "DCF Valuation", "FCFF, CAPM/WACC, terminal value, scenarios"),
        ("Project 4", "Financial Forecasting", "Baselines, regression, Holt-Winters, RMSE, MAPE"),
    ]
    for col, (num, title, desc) in zip(cols, cards):
        with col:
            st.metric(num, title)
            st.caption(desc)

    st.markdown("### How to use this dashboard")
    st.write(
        "Choose a project from the left-hand menu. Each page loads its real dataset or "
        "documented modelling inputs, recalculates the analysis, displays the underlying "
        "table, and renders the key charts directly in Streamlit."
    )
    st.info(
        "Project 2 is intentionally real-data-only. If Yahoo Finance cannot be reached and "
        "no real cached CSV is present, the page will show an explicit data-availability error "
        "instead of generating simulated market prices."
    )


# -----------------------------------------------------------------------------
# PROJECT 1
# -----------------------------------------------------------------------------
elif project == "Project 1 - Company Ratio Analysis":
    st.header("Project 1 - Company Ratio Analysis")
    show_source("Data: Walmart, Costco and Target annual-report / SEC filing figures supplied in project1/data/financials.csv.")

    data_path = ROOT / "project1" / "data" / "financials.csv"
    df = p1.load_and_clean(data_path)
    ratios = p1.calculate_ratios(df)

    palette = ["#1f4e79", "#e67e22", "#7f8c8d", "#8e44ad"]
    p1.COLOURS.clear()
    for company, colour in zip(sorted(ratios["company"].unique()), palette):
        p1.COLOURS[company] = colour

    latest_year = int(ratios["year"].max())
    latest = ratios[ratios["year"] == latest_year]
    scorecard = p1.build_scorecard(ratios)

    c1, c2, c3 = st.columns(3)
    best = scorecard.index[0]
    roe_change_col = [c for c in scorecard.columns if c.startswith("ROE_change")][0]
    improver = scorecard[roe_change_col].idxmax()
    c1.metric("Latest year", latest_year)
    c2.metric("Best overall performer", best)
    c3.metric("Biggest ROE improver", improver)

    st.subheader("Latest-year ratios")
    display_cols = [
        "company", "year", "roe_pct", "roa_pct", "net_margin_pct",
        "current_ratio", "debt_to_equity", "asset_turnover",
    ]
    st.dataframe(latest[display_cols].round(2), use_container_width=True, hide_index=True)

    st.subheader("Five-year scorecard")
    st.dataframe(scorecard, use_container_width=True)

    st.subheader("Source dataset")
    with st.expander("Show raw sourced financial statement data"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Ratio trends")
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
        for company, g in ratios.groupby("company"):
            ax.plot(g["year"], g[col], marker="o", lw=2, label=company, color=p1.COLOURS[company])
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel(unit)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(sorted(ratios["year"].unique()))
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader(f"DuPont breakdown of ROE — {latest_year}")
    latest_indexed = latest.set_index("company")
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (col, title) in zip(
        axes2,
        [
            ("net_margin_pct", "Net margin (%)"),
            ("asset_turnover", "Asset turnover (x)"),
            ("equity_multiplier", "Equity multiplier (x)"),
        ],
    ):
        values = latest_indexed[col]
        ax.bar(values.index, values.values, color=[p1.COLOURS[c] for c in values.index])
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    st.pyplot(fig2, clear_figure=True)
    plt.close(fig2)


# -----------------------------------------------------------------------------
# PROJECT 2
# -----------------------------------------------------------------------------
elif project == "Project 2 - Risk & Return Analysis":
    st.header("Project 2 - Risk & Return Analysis")
    st.write("AAPL, MSFT and JPM compared with the S&P 500 using daily adjusted-close prices.")
    show_source("Source: Yahoo Finance via yfinance, or a committed real cached CSV at project2/data/daily_prices.csv.")

    p2 = load_module("portfolio_project2", ROOT / "project2" / "stock_risk_return_real.py")

    try:
        prices, source = p2.get_prices()
        prices = prices.loc[(prices.index >= p2.START) & (prices.index <= p2.END)].dropna()
        if len(prices) < 2:
            raise ValueError("Insufficient real observations after date filtering.")
    except Exception as exc:
        st.error("Real stock data is not available for this run.")
        st.code(str(exc))
        st.markdown(
            "Place a real file at `project2/data/daily_prices.csv` with columns "
            "`Date,AAPL,MSFT,JPM,^GSPC`, then refresh this page. Do not replace it with simulated data."
        )
        st.stop()

    returns = prices.pct_change().dropna()
    summary = p2.build_summary(prices, returns)

    st.success(f"Data source: {source} | {len(prices):,} trading observations | {prices.index[0].date()} to {prices.index[-1].date()}")
    st.subheader("Risk & return summary")
    st.dataframe(summary.drop(columns=["Drawdown peak", "Drawdown trough"]).round(2), use_container_width=True)

    with st.expander("Show real daily price dataset"):
        st.dataframe(prices.tail(250), use_container_width=True)

    cumulative = (1 + returns).cumprod() - 1
    drawdown = prices / prices.cummax() - 1
    rolling_vol = returns.rolling(60).std() * np.sqrt(p2.TRADING_DAYS) * 100
    corr = returns.corr()
    focus = p2.STOCKS[0]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ticker in cumulative.columns:
        lw, ls = (1.5, "--") if ticker == p2.BENCHMARK else (2, "-")
        axes[0, 0].plot(cumulative.index, cumulative[ticker] * 100, lw=lw, ls=ls, label=ticker)
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title("Cumulative return (%)", loc="left", fontweight="bold")

    for ticker in drawdown.columns:
        axes[0, 1].plot(drawdown.index, drawdown[ticker] * 100, lw=1.5, label=ticker)
    axes[0, 1].set_title("Drawdown from previous peak (%)", loc="left", fontweight="bold")

    for ticker in rolling_vol.columns:
        axes[0, 2].plot(rolling_vol.index, rolling_vol[ticker], lw=1.5, label=ticker)
    axes[0, 2].set_title("60-day rolling volatility (annualised, %)", loc="left", fontweight="bold")

    im = axes[1, 0].imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1, 0].set_xticks(range(len(corr))); axes[1, 0].set_xticklabels(corr.columns)
    axes[1, 0].set_yticks(range(len(corr))); axes[1, 0].set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            axes[1, 0].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    axes[1, 0].set_title("Correlation of daily returns", loc="left", fontweight="bold")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].hist(returns[focus] * 100, bins=60, alpha=0.85)
    axes[1, 1].axvline(returns[focus].quantile(0.05) * 100, ls="--", label="5th percentile")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].set_title(f"{focus}: daily return distribution", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Daily return (%)")

    for ticker in summary.index:
        x = summary.loc[ticker, "Annualised volatility %"]
        y = summary.loc[ticker, "Annualised return % (CAGR)"]
        axes[1, 2].scatter(x, y, s=120)
        axes[1, 2].annotate(ticker, (x, y), xytext=(6, 6), textcoords="offset points")
    axes[1, 2].set_title("Risk vs return", loc="left", fontweight="bold")
    axes[1, 2].set_xlabel("Annualised volatility (%)")
    axes[1, 2].set_ylabel("Annualised return (%)")

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# -----------------------------------------------------------------------------
# PROJECT 3
# -----------------------------------------------------------------------------
elif project == "Project 3 - DCF Valuation":
    st.header("Project 3 - DCF Valuation")
    show_source("Company: Target Corporation (TGT). Historical base-year inputs are sourced; forecast rates and CAPM assumptions are explicit modelling assumptions.")

    w = p3.calc_wacc(p3.BASE)
    res = p3.run_dcf(p3.BASE)
    sens = p3.sensitivity_table(p3.BASE, w["wacc"])
    scen = p3.run_scenarios(p3.BASE)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("WACC", f"{w['wacc']:.2%}")
    c2.metric("DCF value / share", f"${res['value_per_share']:.2f}")
    c3.metric("Market price", f"${p3.BASE['share_price']:.2f}")
    c4.metric("Upside / (downside)", f"{(res['value_per_share']/p3.BASE['share_price']-1):+.1%}")

    st.subheader("Forecast free cash flow")
    st.dataframe(res["forecast"].round(1), use_container_width=True)

    st.subheader("Valuation summary")
    valuation = pd.DataFrame({
        "Metric": [
            "PV of 5-year FCF", "PV of terminal value", "Terminal value % of EV",
            "Enterprise value", "Net debt", "Equity value", "Implied terminal EV/EBITDA"
        ],
        "Value": [
            res["pv_fcf_sum"], res["pv_tv"], res["tv_share_of_ev"] * 100,
            res["enterprise_value"], res["net_debt"], res["equity_value"], res["implied_exit_ev_ebitda"]
        ],
    })
    st.dataframe(valuation, use_container_width=True, hide_index=True)

    st.subheader("Sensitivity: DCF value per share")
    st.dataframe(sens.round(2), use_container_width=True)

    st.subheader("Bear / Base / Bull")
    st.dataframe(scen.round(2), use_container_width=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fc = res["forecast"]
    axes[0, 0].bar(fc.index, fc["fcf"], label="Free cash flow")
    axes[0, 0].plot(fc.index, fc["pv_fcf"], marker="o", label="Present value")
    axes[0, 0].set_title("Forecast FCF (USD m)", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].pie([res["pv_fcf_sum"], res["pv_tv"]], labels=["PV of 5-year FCF", "PV of terminal value"], autopct="%1.0f%%", startangle=90)
    axes[0, 1].set_title("Enterprise value composition", loc="left", fontweight="bold")

    vals = sens.to_numpy(dtype=float)
    im = axes[1, 0].imshow(vals, cmap="RdYlGn", aspect="auto")
    axes[1, 0].set_xticks(range(vals.shape[1])); axes[1, 0].set_xticklabels(sens.columns)
    axes[1, 0].set_yticks(range(vals.shape[0])); axes[1, 0].set_yticklabels(sens.index)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            axes[1, 0].text(j, i, f"{vals[i,j]:.1f}", ha="center", va="center", fontsize=8)
    axes[1, 0].set_xlabel("Terminal growth"); axes[1, 0].set_ylabel("WACC")
    axes[1, 0].set_title("Value/share sensitivity", loc="left", fontweight="bold")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].bar(scen.index, scen["Value per share"])
    axes[1, 1].axhline(p3.BASE["share_price"], ls="--", label=f"Market ${p3.BASE['share_price']:.2f}")
    for i, value in enumerate(scen["Value per share"]):
        axes[1, 1].text(i, value, f"${value:.1f}", ha="center", va="bottom")
    axes[1, 1].set_title("DCF value by scenario", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False)

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    with st.expander("Show modelling assumptions"):
        assumptions = pd.Series(p3.BASE, name="Value")
        st.dataframe(assumptions.to_frame(), use_container_width=True)


# -----------------------------------------------------------------------------
# PROJECT 4
# -----------------------------------------------------------------------------
elif project == "Project 4 - Financial Forecasting":
    st.header("Project 4 - Financial Forecasting")
    show_source("Data: U.S. Census Bureau monthly retail sales series distributed through FRED (MRTSSM44X72USS), stored in project4/data/monthly_revenue.csv.")

    data_path = ROOT / "project4" / "data" / "monthly_revenue.csv"
    p4.require_real_data(data_path)
    series = p4.load_data(data_path)

    train, test = series.iloc[:-p4.TEST_MONTHS], series.iloc[-p4.TEST_MONTHS:]
    preds, rows = {}, {}
    for name, fn in p4.MODELS.items():
        preds[name] = fn(train, len(test))
        rows[name] = {"RMSE": p4.rmse(test, preds[name]), "MAPE %": p4.mape(test, preds[name])}
    results = pd.DataFrame(rows).T.sort_values("MAPE %")
    best = results.index[0]
    forecast = p4.MODELS[best](series, p4.FORECAST_HORIZON)
    future_idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=p4.FORECAST_HORIZON, freq="MS")
    band = 1.96 * results.loc[best, "RMSE"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Observations", f"{len(series):,} months")
    c2.metric("Best model", best)
    c3.metric("Best test MAPE", f"{results.loc[best, 'MAPE %']:.2f}%")

    st.subheader("Model comparison")
    st.dataframe(results.round(2), use_container_width=True)

    st.subheader("Source dataset")
    st.dataframe(series.to_frame("revenue").tail(120), use_container_width=True)

    st.subheader("Next 12 months")
    future_df = pd.DataFrame({"forecast": forecast, "lower_bound": forecast - band, "upper_bound": forecast + band}, index=future_idx)
    st.dataframe(future_df.round(1), use_container_width=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].plot(train.index[-36:], train.iloc[-36:], label="Training data")
    axes[0].plot(test.index, test, lw=2.5, label="Actual test")
    for name, values in preds.items():
        axes[0].plot(test.index, values, lw=1.5, ls="--", label=name)
    axes[0].set_title("Backtest: last 12 months held out", loc="left", fontweight="bold")
    axes[0].set_ylabel("Revenue (USD m)")
    axes[0].legend(frameon=False, fontsize=8)

    order = results.sort_values("MAPE %")
    axes[1].barh(order.index, order["MAPE %"])
    axes[1].invert_yaxis()
    axes[1].set_title("Test MAPE (lower is better)", loc="left", fontweight="bold")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(12, 5.5))
    ax2.plot(series.index, series, label="History")
    ax2.plot(future_idx, forecast, lw=2.5, marker="o", ms=4, label="Forecast")
    ax2.fill_between(future_idx, forecast - band, forecast + band, alpha=0.2, label="Approx. range")
    ax2.set_title("12-month revenue forecast", loc="left", fontweight="bold")
    ax2.set_ylabel("Revenue (USD m)")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25)
    ax2.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2, clear_figure=True)
    plt.close(fig2)

    total_forecast = float(forecast.sum())
    previous_12m = float(series.iloc[-12:].sum())
    st.info(
        f"Next-12-month forecast total: {total_forecast:,.0f} USD m vs "
        f"{previous_12m:,.0f} USD m for the preceding 12 months "
        f"({total_forecast / previous_12m - 1:+.1%})."
    )
