#!/usr/bin/env python3
"""Mann–Kendall trend test + Sen's slope for wildfire burned-area data.

Source: Wildfire_data.xlsx, Sheet1, rows 3–25 (2001–2023).
Series: low/mid/high latitudes, Other regions, PF->SFG, PF.

IMPORTANT: The input indices are normalized to 0–1. Multiplying by 100
expresses the change in points of the 0–100 normalized index per year;
it does NOT estimate the percentage change in the physical burned area.

Requirements: artifact_tool, numpy, scipy, matplotlib (stdlib csv).
Run: python wildfire_all_latitudes_mk_sen.py --input Wildfire_data.xlsx
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, theilslopes
from artifact_tool import Blob, SpreadsheetFile


# 0-based indices within the input Excel row (A=0, ..., J=9).
SERIES = [
    ("Low (0-30°)", "Other regions", 1),   # B
    ("Low (0-30°)", "PF->SFG", 2),         # C (no values)
    ("Low (0-30°)", "PF", 3),              # D (no values)
    ("Mid (30-60°)", "Other regions", 4),  # E
    ("Mid (30-60°)", "PF->SFG", 5),        # F
    ("Mid (30-60°)", "PF", 6),             # G
    ("High (60-90°)", "Other regions", 7), # H
    ("High (60-90°)", "PF->SFG", 8),       # I
    ("High (60-90°)", "PF", 9),            # J
]


def mk_sen(years: np.ndarray, y: np.ndarray) -> dict:
    """Two-sided asymptotic MK test (tie and continuity correction) + Sen slope."""
    n = len(y)
    if n < 3:
        raise ValueError("At least three observations are required")

    # S = sum_{i<j} sign(y[j]-y[i])
    S = int(sum(int(np.sign(y[j] - y[i]))
                for i in range(n - 1) for j in range(i + 1, n)))

    _, counts = np.unique(y, return_counts=True)
    correction = sum(int(t * (t - 1) * (2 * t + 5)) for t in counts if t > 1)
    var_S = (n * (n - 1) * (2 * n + 5) - correction) / 18.0
    if var_S <= 0:
        raise ValueError("Degenerate variance in the Mann–Kendall test")

    # Normal approximation with continuity correction
    Z = float((S - np.sign(S)) / np.sqrt(var_S)) if S != 0 else 0.0
    P = float(2.0 * norm.sf(abs(Z)))

    # Theil–Sen estimator (equivalent to Sen's median pairwise slope)
    sen = theilslopes(y, years, alpha=0.95)
    return {
        "n": n, "S": S, "Var(S)": float(var_S),
        "Z": Z, "P_two_sided": P,
        "Sen_slope_pct_per_yr": float(sen.slope),
        "Sen_intercept": float(sen.intercept),
        "Sen_95CI_low": float(sen.low_slope),
        "Sen_95CI_high": float(sen.high_slope),
        "significant_P_lt_0.05": bool(P < 0.05),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Wildfire_data.xlsx", type=Path)
    parser.add_argument("--output-dir", default=".", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    workbook = SpreadsheetFile.import_xlsx(Blob.load(str(args.input)))
    sheet = workbook.worksheets.get_item("Sheet1")
    # A1:J25 includes two header rows and 23 annual records.
    grid = sheet.get_range("A1:J25").values
    headers = grid[1]
    source = grid[2:]

    results = []
    for latitude, category, col in SERIES:
        if headers[col] != category:
            raise ValueError(
                f"Unexpected header in column {col + 1}: "
                f"expected {category!r}; got {headers[col]!r}"
            )
        pairs = [(int(row[0]), float(row[col]) * 100.0)
                 for row in source
                 if isinstance(row[0], (int, float))
                 and isinstance(row[col], (int, float))
                 and np.isfinite(row[col])]

        record = {"Latitude": latitude, "Category": category,
                  "Excel_column": chr(ord("A") + col)}
        if not pairs:
            record.update({
                "Status": "No data in workbook", "n": 0,
                "S": "", "Var(S)": "", "Z": "", "P_two_sided": "",
                "Sen_slope_pct_per_yr": "", "Sen_intercept": "",
                "Sen_95CI_low": "", "Sen_95CI_high": "",
                "significant_P_lt_0.05": "",
            })
        else:
            years = np.array([year for year, _ in pairs], dtype=float)
            y = np.array([val for _, val in pairs], dtype=float)
            record.update({"Status": "Analyzed", **mk_sen(years, y)})
            if len(pairs) != 23 or int(years.min()) != 2001 or int(years.max()) != 2023:
                print(f"WARNING: {latitude}, {category} has an incomplete/changed year range")
        results.append(record)

    out_csv = args.output_dir / "wildfire_all_latitudes_mk_sen_results.csv"
    fields = ["Latitude", "Category", "Excel_column", "Status",
              "n", "S", "Var(S)", "Z", "P_two_sided",
              "Sen_slope_pct_per_yr", "Sen_intercept", "Sen_95CI_low",
              "Sen_95CI_high", "significant_P_lt_0.05"]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    valid = [r for r in results if r["Status"] == "Analyzed"]
    print("2001–2023 | normalized series multiplied by 100 | two-sided MK, alpha=0.05")
    print("Latitude | Category | n | S | Z | P | Sen (%/yr) | 95% CI")
    for r in valid:
        print(f"{r['Latitude']} | {r['Category']} | {r['n']} | {r['S']} | "
              f"{r['Z']:+.3f} | {r['P_two_sided']:.6f} | "
              f"{r['Sen_slope_pct_per_yr']:+.4f} | "
              f"[{r['Sen_95CI_low']:+.4f}, {r['Sen_95CI_high']:+.4f}]")
    for r in results:
        if r["Status"] != "Analyzed":
            print(f"NOT ESTIMATED: {r['Latitude']} {r['Category']} — source column empty")

    # One stand-alone forest plot showing all available series; no separate subplots.
    fig, ax = plt.subplots(figsize=(10.2, 5.9))
    labels = [f"{r['Latitude']} | {r['Category']}"
              + (" *" if r["significant_P_lt_0.05"] else "")
              for r in valid]
    centers = np.array([r["Sen_slope_pct_per_yr"] for r in valid])
    lower = np.array([r["Sen_95CI_low"] for r in valid])
    upper = np.array([r["Sen_95CI_high"] for r in valid])
    yy = np.arange(len(valid))[::-1]
    ax.errorbar(centers, yy, xerr=np.vstack([centers - lower, upper - centers]),
                fmt="o", capsize=4, markersize=6)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(yy, labels)
    ax.set_xlabel("Sen's slope (% yr$^{-1}$; normalized index × 100)")
    ax.set_title("Annual normalized burned-area trends by latitude and frozen-ground category\n"
                 "2001–2023 | MK test and Sen's slope (95% CI)")
    ax.grid(axis="x", alpha=0.20)
    ax.text(0, -0.15, "* MK P < 0.05; blank low-latitude PF categories excluded.",
            ha="left", va="top", transform=ax.transAxes, fontsize=9)
    fig.subplots_adjust(left=0.32, bottom=0.22, right=0.96, top=0.84)
    out_fig = args.output_dir / "wildfire_all_latitudes_mk_sen_forest.png"
    fig.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_csv}\nSaved: {out_fig}")


if __name__ == "__main__":
    main()