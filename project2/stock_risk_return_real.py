"""
PROJECT 2: STOCK MARKET RISK & RETURN ANALYSIS — REAL-DATA ONLY
================================================================
Analyses AAPL, MSFT and JPM against the S&P 500 using daily adjusted-close
prices. Metrics: cumulative/annualised return, volatility, beta, correlation,
Sharpe ratio, maximum drawdown and 1-day 95% historical VaR.

There is deliberately NO simulated fallback. The script either loads real
Yahoo Finance data (directly or from a real cached CSV) or stops with an error.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

STOCKS = ["AAPL", "MSFT", "JPM"]
BENCHMARK = "^GSPC"
START, END = "2020-01-01", "2024-12-31"
RISK_FREE_RATE = 0.03
TRADING_DAYS = 252
DATA_FILE = Path("data/daily_prices.csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def download_prices(tickers, start, end) -> pd.DataFrame:
    import yfinance as yf
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError("Yahoo Finance returned no data")
    # yfinance may return MultiIndex columns or a single-level frame.
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.reindex(columns=tickers)
    if prices.isna().all().any():
        missing = list(prices.columns[prices.isna().all()])
        raise ValueError(f"No downloaded data for: {missing}")
    return prices.dropna(how="all")


def load_cached_prices(path: Path, tickers) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Real cached data not found at {path}")
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        raise ValueError(f"Cached data is missing tickers: {missing}")
    prices = df[tickers].apply(pd.to_numeric, errors="coerce")
    return prices.dropna(how="all")


def get_prices() -> tuple[pd.DataFrame, str]:
    tickers = STOCKS + [BENCHMARK]
    # Prefer a committed real cache for reproducible portfolio runs.
    if DATA_FILE.exists():
        p = load_cached_prices(DATA_FILE, tickers)
        return p, "local real cache"
    # Otherwise, try a live Yahoo Finance download; never simulate.
    try:
        return download_prices(tickers, START, END), "live Yahoo Finance"
    except Exception as exc:
        raise RuntimeError(
            "Real stock data unavailable. Provide data/daily_prices.csv or run with internet access. "
            "No simulated fallback is permitted."
        ) from exc


def max_drawdown(price: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    running_peak = price.cummax()
    dd = price / running_peak - 1
    trough = dd.idxmin()
    peak = price.loc[:trough].idxmax()
    return dd.min(), peak, trough


def beta_vs_market(stock_ret: pd.Series, market_ret: pd.Series) -> float:
    cov = np.cov(stock_ret, market_ret)
    return cov[0, 1] / cov[1, 1]


def build_summary(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    rows = {}
    for t in prices.columns:
        total_ret = prices[t].iloc[-1] / prices[t].iloc[0] - 1
        cagr = (1 + total_ret) ** (1 / years) - 1
        vol = returns[t].std() * np.sqrt(TRADING_DAYS)
        mdd, peak, trough = max_drawdown(prices[t])
        rows[t] = {
            "Total return %": total_ret * 100,
            "Annualised return % (CAGR)": cagr * 100,
            "Annualised volatility %": vol * 100,
            "Sharpe ratio": (cagr - RISK_FREE_RATE) / vol,
            "Beta vs S&P 500": 1.0 if t == BENCHMARK else beta_vs_market(returns[t], returns[BENCHMARK]),
            "Max drawdown %": mdd * 100,
            "Drawdown peak": peak.date(),
            "Drawdown trough": trough.date(),
            "1-day 95% VaR %": returns[t].quantile(0.05) * 100,
        }
    return pd.DataFrame(rows).T


def style(ax, title, ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def build_dashboard(prices, returns, summary, data_source: str) -> None:
    cumulative = (1 + returns).cumprod() - 1
    drawdown = prices / prices.cummax() - 1
    rolling_vol = returns.rolling(60).std() * np.sqrt(TRADING_DAYS) * 100
    corr = returns.corr()
    focus = STOCKS[0]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax = axes[0, 0]
    for t in cumulative.columns:
        lw, ls = (1.5, "--") if t == BENCHMARK else (2, "-")
        ax.plot(cumulative.index, cumulative[t] * 100, lw=lw, ls=ls, label=t)
    ax.legend(frameon=False); style(ax, "Cumulative return (%)", "%")

    ax = axes[0, 1]
    for t in drawdown.columns:
        ax.plot(drawdown.index, drawdown[t] * 100, lw=1.5, label=t)
    style(ax, "Drawdown from previous peak (%)", "%")

    ax = axes[0, 2]
    for t in rolling_vol.columns:
        ax.plot(rolling_vol.index, rolling_vol[t], lw=1.5, label=t)
    style(ax, "60-day rolling volatility (annualised, %)", "%")

    ax = axes[1, 0]
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Correlation of daily returns", fontsize=11, fontweight="bold", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    ax.hist(returns[focus] * 100, bins=60, color="#1f4e79", alpha=0.85)
    ax.axvline(returns[focus].quantile(0.05) * 100, color="red", ls="--", label="5th percentile (95% VaR)")
    ax.legend(frameon=False); style(ax, f"{focus}: distribution of daily returns", "days")
    ax.set_xlabel("Daily return (%)")

    ax = axes[1, 2]
    for t in summary.index:
        x = summary.loc[t, "Annualised volatility %"]
        y = summary.loc[t, "Annualised return % (CAGR)"]
        ax.scatter(x, y, s=120)
        ax.annotate(t, (x, y), xytext=(6, 6), textcoords="offset points")
    style(ax, "Risk vs return", "Annualised return (%)")
    ax.set_xlabel("Annualised volatility (%)")

    fig.suptitle(f"Stock Risk & Return Dashboard — {data_source}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "stock_dashboard.png", dpi=200)
    plt.close(fig)


def main():
    prices, source = get_prices()
    # Constrain the real cache to the documented project window.
    prices = prices.loc[(prices.index >= START) & (prices.index <= END)].dropna()
    if len(prices) < 2:
        raise ValueError("Insufficient real observations after date filtering")
    returns = prices.pct_change().dropna()
    summary = build_summary(prices, returns)
    summary.round(2).to_csv(OUT_DIR / "risk_return_summary.csv")
    returns.corr().round(3).to_csv(OUT_DIR / "correlation_matrix.csv")
    build_dashboard(prices, returns, summary, source)
    pd.set_option("display.width", 220)
    print(f"Data source: {source}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()} ({len(prices)} trading days)")
    print("\n=== RISK & RETURN SUMMARY ===")
    print(summary.drop(columns=["Drawdown peak", "Drawdown trough"]).astype(float).round(2).to_string())
    print("\n=== WORST DRAWDOWN WINDOWS ===")
    print(summary[["Drawdown peak", "Drawdown trough"]].to_string())
    print("\n=== CORRELATION OF DAILY RETURNS ===")
    print(returns.corr().round(2).to_string())
    print(f"\nSaved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
