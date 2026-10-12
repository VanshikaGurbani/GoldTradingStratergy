"""Correlation between gold ETF (GLD) prices and Google Trends search interest.

Runs on the datasets saved in this folder so the numbers are reproducible
(Google Trends returns different values on every live pull).

Usage (from the repo root):
    python correlation/correlation_stats.py
"""
from pathlib import Path

import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATASETS = ["gold_etf_trends_data.csv", "gold_etf_trends_detailed_data.csv"]


def summarize(name: str) -> str:
    df = pd.read_csv(HERE / name, parse_dates=["Date"])
    price = "ETF_Price" if "ETF_Price" in df else "Close_Price"
    interest = df["Search_Interest"]
    returns = df[price].pct_change()

    r, p = stats.pearsonr(df[price], interest)
    rho, p_rho = stats.spearmanr(df[price], interest)

    changes = pd.concat([returns, interest.diff()], axis=1).dropna()
    r_chg, p_chg = stats.pearsonr(changes.iloc[:, 0], changes.iloc[:, 1])

    ahead = pd.concat([returns.shift(-1), interest], axis=1).dropna()
    r_next, p_next = stats.pearsonr(ahead.iloc[:, 0], ahead.iloc[:, 1])

    return "\n".join([
        f"=== {name}: n={len(df)}, {df['Date'].min().date()} to {df['Date'].max().date()}",
        f"Search interest: mean {interest.mean():.1f}, min {interest.min()}, max {interest.max()}",
        f"Pearson  (price vs interest):                  r={r:.4f}   p={p:.4g}",
        f"Spearman (price vs interest):                  rho={rho:.4f} p={p_rho:.4g}",
        f"Pearson  (price return vs change in interest): r={r_chg:.4f}   p={p_chg:.4g}  n={len(changes)}",
        f"Pearson  (interest vs next-period return):     r={r_next:.4f}   p={p_next:.4g}  n={len(ahead)}",
    ])


if __name__ == "__main__":
    report = "\n\n".join(summarize(name) for name in DATASETS)
    print(report)
    (HERE / "correlation_results.txt").write_text(report + "\n", encoding="utf-8")
