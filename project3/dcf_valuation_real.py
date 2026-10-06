"""
PROJECT 3: SIMPLE DCF VALUATION — REAL COMPANY VERSION
========================================================
Five-year unlevered FCF DCF for Target Corporation (TGT), using FY2024
reported financials as the base year and a 30-Sep-2026 market-price snapshot.

Historical inputs are sourced from Target annual reports / SEC filings.
Forecast rates, terminal growth, beta and other modelling assumptions are
explicitly identified below; they are NOT presented as historical facts.
"""

from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

BASE = {
    "company": "Target Corporation (TGT)",
    # FY2024 actuals — USD millions unless stated otherwise.
    "revenue_0": 106566.0,             # FY2024 net sales
    "ebit_margin": 5566.0 / 106566.0,  # FY2024 operating income / net sales
    "tax_rate": 1170.0 / 5261.0,       # FY2024 tax provision / earnings before income taxes
    "da_pct_rev": 2981.0 / 106566.0,   # FY2024 D&A / net sales
    "capex_pct_rev": 2891.0 / 106566.0,# FY2024 additions to property/plant/equipment / sales
    # Working-capital investment is set to zero because Target has negative
    # operating working capital in the sourced FY2024/FY2023 balance sheets.
    "nwc_pct_change_rev": 0.0,

    # Forecast assumptions: explicit modelling choices, not historical facts.
    "revenue_growth": [0.033, 0.030, 0.027, 0.025, 0.023],

    # CAPM inputs / capital structure snapshot.
    "risk_free": 0.0515,                # 10Y U.S. Treasury snapshot used for the model
    "equity_risk_premium": 0.0433,      # Damodaran implied ERP reference assumption
    "beta": 1.00,                        # transparent modelling assumption; test in sensitivity table
    "pre_tax_cost_of_debt": 411.0 / ((15940.0 + 16038.0) / 2),
    "total_debt": 15940.0,
    "cash": 4959.0,
    "shares_out": 455.6,
    "share_price": 156.69,              # 30-Sep-2026 market-price snapshot
    "terminal_growth": 0.025,
}


def calc_wacc(a: dict) -> dict:
    market_cap = a["share_price"] * a["shares_out"]
    total_capital = market_cap + a["total_debt"]
    ke = a["risk_free"] + a["beta"] * a["equity_risk_premium"]
    kd_after_tax = a["pre_tax_cost_of_debt"] * (1 - a["tax_rate"])
    we, wd = market_cap / total_capital, a["total_debt"] / total_capital
    return {
        "cost_of_equity": ke,
        "cost_of_debt_after_tax": kd_after_tax,
        "weight_equity": we,
        "weight_debt": wd,
        "wacc": we * ke + wd * kd_after_tax,
    }


def forecast_fcf(a: dict) -> pd.DataFrame:
    rows = []
    rev_prev = a["revenue_0"]
    for yr, g in enumerate(a["revenue_growth"], start=1):
        revenue = rev_prev * (1 + g)
        ebit = revenue * a["ebit_margin"]
        nopat = ebit * (1 - a["tax_rate"])
        da = revenue * a["da_pct_rev"]
        capex = revenue * a["capex_pct_rev"]
        d_nwc = (revenue - rev_prev) * a["nwc_pct_change_rev"]
        fcf = nopat + da - capex - d_nwc
        rows.append({"year": yr, "revenue": revenue, "ebit": ebit, "nopat": nopat,
                     "da": da, "capex": capex, "increase_nwc": d_nwc, "fcf": fcf})
        rev_prev = revenue
    return pd.DataFrame(rows).set_index("year")


def run_dcf(a: dict, wacc: float | None = None, g: float | None = None) -> dict:
    wacc = calc_wacc(a)["wacc"] if wacc is None else wacc
    g = a["terminal_growth"] if g is None else g
    if wacc <= g:
        raise ValueError("WACC must be greater than terminal growth")
    fc = forecast_fcf(a)
    n = len(fc)
    discount = (1 + wacc) ** fc.index.to_numpy()
    pv_fcf = fc["fcf"].to_numpy() / discount
    terminal_value = fc["fcf"].iloc[-1] * (1 + g) / (wacc - g)
    pv_tv = terminal_value / (1 + wacc) ** n
    enterprise_value = pv_fcf.sum() + pv_tv
    net_debt = a["total_debt"] - a["cash"]
    equity_value = enterprise_value - net_debt
    value_per_share = equity_value / a["shares_out"]
    return {
        "forecast": fc.assign(pv_fcf=pv_fcf), "wacc": wacc, "g": g,
        "pv_fcf_sum": pv_fcf.sum(), "terminal_value": terminal_value, "pv_tv": pv_tv,
        "enterprise_value": enterprise_value, "net_debt": net_debt,
        "equity_value": equity_value, "value_per_share": value_per_share,
        "tv_share_of_ev": pv_tv / enterprise_value,
        "implied_exit_ev_ebitda": terminal_value / (fc["ebit"].iloc[-1] + fc["da"].iloc[-1]),
    }


def sensitivity_table(a: dict, wacc_base: float) -> pd.DataFrame:
    waccs = np.round(wacc_base + np.arange(-0.015, 0.0151, 0.005), 4)
    gs = np.arange(0.015, 0.0351, 0.005)
    grid = pd.DataFrame(index=[f"{w:.1%}" for w in waccs],
                        columns=[f"{x:.1%}" for x in gs], dtype=float)
    for w in waccs:
        for x in gs:
            grid.loc[f"{w:.1%}", f"{x:.1%}"] = run_dcf(a, wacc=w, g=x)["value_per_share"]
    grid.index.name = "WACC \\ terminal growth"
    return grid


def run_scenarios(a: dict) -> pd.DataFrame:
    # Scenarios are intentionally moderate: +/-1pp growth and margins of
    # 4.5% / 5.22% / 6.0%, rather than the unrealistic 15% / 21% margins in the demo.
    shifts = {
        "Bear": dict(growth_shift=-0.01, margin=0.045, g=0.020),
        "Base": dict(growth_shift=0.0, margin=a["ebit_margin"], g=a["terminal_growth"]),
        "Bull": dict(growth_shift=+0.01, margin=0.060, g=0.030),
    }
    rows = {}
    for name, s in shifts.items():
        sc = deepcopy(a)
        sc["revenue_growth"] = [x + s["growth_shift"] for x in a["revenue_growth"]]
        sc["ebit_margin"] = s["margin"]
        sc["terminal_growth"] = s["g"]
        res = run_dcf(sc)
        rows[name] = {
            "Value per share": res["value_per_share"],
            "Upside vs market %": (res["value_per_share"] / a["share_price"] - 1) * 100,
            "Enterprise value": res["enterprise_value"],
            "TV % of EV": res["tv_share_of_ev"] * 100,
        }
    return pd.DataFrame(rows).T


def make_charts(a, res, sens, scen) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    fc = res["forecast"]
    ax.bar(fc.index, fc["fcf"], color="#1f4e79", label="Free cash flow")
    ax.plot(fc.index, fc["pv_fcf"], color="#e67e22", marker="o", label="Present value")
    ax.set_title("Forecast free cash flow (USD m)", loc="left", fontweight="bold")
    ax.set_xlabel("Forecast year"); ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.pie([res["pv_fcf_sum"], res["pv_tv"]],
           labels=["PV of 5-year FCF", "PV of terminal value"], autopct="%1.0f%%",
           colors=["#1f4e79", "#e67e22"], startangle=90)
    ax.set_title("Where the enterprise value comes from", loc="left", fontweight="bold")

    ax = axes[1, 0]
    vals = sens.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap="RdYlGn", aspect="auto",
                   vmin=min(vals.min(), a["share_price"] * 0.6),
                   vmax=max(vals.max(), a["share_price"] * 1.4))
    ax.set_xticks(range(vals.shape[1])); ax.set_xticklabels(sens.columns)
    ax.set_yticks(range(vals.shape[0])); ax.set_yticklabels(sens.index)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.1f}", ha="center", va="center", fontsize=9)
    ax.set_xlabel("Terminal growth"); ax.set_ylabel("WACC")
    ax.set_title(f"Value per share sensitivity (market price {a['share_price']:.2f})",
                 loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    colours = {"Bear": "#c0392b", "Base": "#7f8c8d", "Bull": "#27ae60"}
    ax.bar(scen.index, scen["Value per share"], color=[colours[s] for s in scen.index])
    ax.axhline(a["share_price"], color="black", ls="--", label=f"Market price {a['share_price']:.2f}")
    for i, v in enumerate(scen["Value per share"]):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    ax.set_title("DCF value per share by scenario", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    for ax in (axes[0, 0], axes[1, 1]):
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"DCF Valuation: {a['company']}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "dcf_dashboard.png", dpi=200)
    plt.close(fig)


def main():
    w = calc_wacc(BASE)
    res = run_dcf(BASE)
    sens = sensitivity_table(BASE, w["wacc"])
    scen = run_scenarios(BASE)
    print("=== WACC ===")
    print(f"Cost of equity (CAPM): {w['cost_of_equity']:.2%} | After-tax cost of debt: {w['cost_of_debt_after_tax']:.2%} | "
          f"Weights E/D: {w['weight_equity']:.0%}/{w['weight_debt']:.0%} | WACC: {w['wacc']:.2%}")
    print("\n=== FREE CASH FLOW FORECAST (USD m) ===")
    print(res["forecast"].round(1).to_string())
    print("\n=== VALUATION ===")
    print(f"PV of FCF: {res['pv_fcf_sum']:,.0f} | PV of terminal value: {res['pv_tv']:,.0f} ({res['tv_share_of_ev']:.0%} of EV)")
    print(f"Enterprise value: {res['enterprise_value']:,.0f} | Net debt: {res['net_debt']:,.0f} | Equity value: {res['equity_value']:,.0f}")
    print(f"Implied terminal EV/EBITDA: {res['implied_exit_ev_ebitda']:.1f}x")
    upside = res["value_per_share"] / BASE["share_price"] - 1
    print(f"DCF value per share: {res['value_per_share']:.2f} vs market {BASE['share_price']:.2f} -> {upside:+.1%}")
    print("\n=== SENSITIVITY (value per share) ===")
    print(sens.round(2).to_string())
    print("\n=== SCENARIOS ===")
    print(scen.round(2).to_string())
    sens.round(2).to_csv(OUT_DIR / "sensitivity_table.csv")
    scen.round(2).to_csv(OUT_DIR / "scenarios.csv")
    res["forecast"].round(1).to_csv(OUT_DIR / "fcf_forecast.csv")
    pd.DataFrame([{
        "company": BASE["company"], "base_revenue_usd_m": BASE["revenue_0"],
        "wacc": w["wacc"], "market_price": BASE["share_price"],
        "dcf_value_per_share": res["value_per_share"],
        "terminal_value_share_of_ev": res["tv_share_of_ev"],
        "implied_exit_ev_ebitda": res["implied_exit_ev_ebitda"],
    }]).to_csv(OUT_DIR / "valuation_summary.csv", index=False)
    make_charts(BASE, res, sens, scen)
    print(f"\nSaved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
