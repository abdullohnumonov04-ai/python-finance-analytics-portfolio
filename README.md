# Python Finance & Analytics Portfolio

Four reproducible Python projects designed for finance, accounting, investment,
FP&A, risk, and business-analytics applications. The projects use real sourced
data and explicitly avoid presenting simulated output as empirical findings.

## Projects

### 1. Company Ratio Analysis
Five-year peer benchmarking using ROE, ROA, net margin, current ratio,
debt-to-equity, asset turnover, and DuPont analysis.

Data: Walmart, Costco, and Target annual-report financial statement figures.

Entry point: `project1/ratio_analysis_real.py`

### 2. Stock Market Risk & Return Analysis
Daily market-risk analysis for AAPL, MSFT, and JPM versus the S&P 500,
covering return, volatility, beta, correlation, Sharpe ratio, maximum
drawdown, and historical VaR.

Data source: Yahoo Finance via `yfinance`.

Entry point: `project2/stock_risk_return_real.py`

Important: this project is real-data-only. It must not fall back to synthetic
prices. If real Yahoo data cannot be downloaded, the script stops before
publishing numerical findings.

### 3. Simple DCF Valuation
Five-year unlevered FCF valuation with CAPM/WACC, Gordon-growth terminal value,
sensitivity analysis, and Bear/Base/Bull scenarios.

Company: Target Corporation.

Entry point: `project3/dcf_valuation_real.py`

### 4. Financial Forecasting
Monthly U.S. retail-sales forecasting using a time-based holdout, naive and
seasonal-naive baselines, regression, and Holt-Winters exponential smoothing.

Data source: U.S. Census Bureau series distributed through FRED.

Entry point: `project4/forecasting_real.py`

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run everything

```bash
python run_all.py
```

## Run a project individually

```bash
python project1/ratio_analysis_real.py
python project2/stock_risk_return_real.py
python project3/dcf_valuation_real.py
python project4/forecasting_real.py
```

## Outputs

Each project writes tables and charts to its own `outputs/` folder when the
script is run successfully.

## Reproducibility and sourcing

Each README/report should record the exact data source, fiscal period or date
range, access date, and any modelling assumptions. Numerical claims in the
portfolio should only be copied from generated output files produced by the
current run.

See `PORTFOLIO_RESULTS.md` for the numerical findings currently produced in
this workspace and the corresponding CV-ready wording.
