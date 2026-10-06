# Portfolio Real-Data Conversion — Results & CV Copy

Research / run date: 1 October 2026. All numerical claims below are taken from the scripts actually executed in this workspace. Project 2 is explicitly marked incomplete because a real daily price dataset was not available for local execution.

## Project 1 — Company Ratio Analysis

**Key Findings**

Costco finished FY2024 with ROE 31.19%, ROA 10.55%, net margin 2.90%, current ratio 0.97x, D/E 0.25x and asset turnover 3.64x. Its ROE rose 9.49pp from 2020 to 2024. The scorecard's average observed-ratio rank was 1.33.

**CV bullets**

- Analysed five years of financial statements for 3 large U.S. retailers in Python (pandas, matplotlib), calculating six profitability, liquidity, leverage and efficiency ratios to benchmark performance.
- Applied DuPont analysis to show that Costco's 31.19% 2024 ROE reflected a 2.90% net margin, 3.64x asset turnover and 2.96x equity multiplier; flagged Target's 1.09x D/E and Walmart's 0.83x current ratio as lender watchpoints.
- Built a reusable data-cleaning and validation workflow (type conversion, duplicate checks, balance-sheet sanity tests) and a six-panel visual report that communicates findings to non-technical readers.

**Note:** Target FY2020 current assets/current liabilities remain missing in the sourced dataset and are not imputed.

## Project 2 — Stock Market Risk & Return Analysis

**Status:** real-only script completed and executed, but the workspace could not retrieve the required Yahoo daily price rows. No numerical findings or numeric CV bullets are published. The former synthetic fallback was removed.

**CV bullet status:** do not add a numeric Project 2 bullet until the local real-data run produces the actual summary table.

## Project 3 — Simple DCF Valuation

**Key Findings**

- WACC: 8.11%; CAPM cost of equity: 9.48%; after-tax cost of debt: 2.00%; capital weights: 82% equity / 18% debt.
- Base DCF value: $155.18 per share versus a $156.69 market-price snapshot, a -1.0% difference.
- Terminal value contributed 76.6% of enterprise value; implied terminal EV/EBITDA was 9.4x.
- Sensitivity grid: $104.11–$288.10 per share; Bear/Base/Bull: $114.44 / $155.18 / $207.51.

**CV bullets**

- Developed a DCF valuation model in Python for Target Corporation, forecasting five-year unlevered free cash flow and estimating an 8.11% WACC using CAPM and market-value capital weights.
- Valued Target at $155.18 per share versus a $156.69 market price (-1.0%), and used sensitivity and scenario analysis to show a value range of $104.11 to $288.10 per share across the WACC/terminal-growth grid.
- Sense-checked the output with terminal value at 76.6% of enterprise value and an implied exit EV/EBITDA multiple of 9.4x, documenting assumptions and sources.

## Project 4 — Financial Forecasting

**Key Findings**

- Holt-Winters achieved 0.79% test MAPE and $7,248.97m RMSE on the 12-month holdout, compared with 4.26% MAPE and $33,433.37m RMSE for seasonal naive.
- The next-12-month Holt-Winters forecast totals approximately $9.42tn, 5.2% above the preceding 12-month actual total of $8.95tn.

**CV bullets**

- Built a time-series forecasting workflow in Python (pandas, numpy, statsmodels) to predict monthly U.S. retail sales, using a time-based train/test split to avoid data leakage.
- Benchmarked Holt-Winters against naive and seasonal-naive baselines, reducing test MAPE from 4.26% to 0.79% (RMSE $7,248.97m), and selected the final model on out-of-sample accuracy rather than in-sample fit.
- Delivered a 12-month forecast of approximately $9.42tn in aggregate sales, with an uncertainty band based on 1.96 × test RMSE and a limitations summary covering structural breaks and sample size.

## Recommended GitHub order

1. Project 1 — Company Ratio Analysis
2. Project 3 — Simple DCF Valuation
3. Project 2 — Stock Risk & Return (after the real run)
4. Project 4 — Forecasting

