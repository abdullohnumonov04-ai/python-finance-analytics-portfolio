from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECTS = [
    ("Project 1 - Company Ratio Analysis", ROOT / "project1", "ratio_analysis_real.py"),
    ("Project 2 - Stock Market Risk & Return", ROOT / "project2", "stock_risk_return_real.py"),
    ("Project 3 - Simple DCF Valuation", ROOT / "project3", "dcf_valuation_real.py"),
    ("Project 4 - Financial Forecasting", ROOT / "project4", "forecasting_real.py"),
]

for name, cwd, script in PROJECTS:
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    result = subprocess.run([sys.executable, script], cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(f"\n{name} failed with exit code {result.returncode}.")

print("\nAll four projects completed successfully.")
