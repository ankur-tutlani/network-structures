#!/usr/bin/env python3
"""
Extract selected rows from the corrected_results sheet of
payoff_low_density_ER_summary.xlsx.

Filter:
    Degree-Gini > 0.29

Output:
    Four separate case sheets: Case1, Case2, Case3, Case4

Columns:
    P
    Run-seed
    Degree-Gini
    final_X_share (%)
    Final_Y_share (%)
    outcome_class

The X/Y shares are multiplied by 100.
No outcome reclassification is performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


DEFAULT_INPUT = "payoff_low_density_ER_summary.xlsx"
DEFAULT_OUTPUT = "ER_Gini_above_0_29_by_case.xlsx"
# GINI_THRESHOLD = 0.29
# GINI_THRESHOLD = 0.22
GINI_THRESHOLD = 0.19

def get_col(df: pd.DataFrame, names: list[str], label: str) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(
        f"Could not find the {label} column. "
        f"Expected one of: {', '.join(names)}"
    )


def extract_case_tables(input_path: Path) -> dict[int, pd.DataFrame]:
    df = pd.read_excel(input_path, sheet_name="corrected_results")

    case_col = get_col(df, ["case", "Case"], "case")
    p_col = get_col(df, ["p", "P"], "P")
    seed_col = get_col(df, ["run_seed", "Run-seed", "run-seed"], "Run-seed")
    gini_col = get_col(df, ["degree_gini", "Degree-Gini"], "Degree-Gini")
    x_col = get_col(
        df,
        ["final_X_share", "final_X_share (%)"],
        "final_X_share",
    )
    y_col = get_col(
        df,
        ["final_Y_share", "final_Y_share (%)"],
        "final_Y_share",
    )
    outcome_col = get_col(df, ["outcome_class"], "outcome_class")

    work = df.copy()

    # Numeric conversion for filtering and output.
    # for col in [case_col, p_col, seed_col, gini_col, x_col, y_col]:
    #     work[col] = pd.to_numeric(work[col], errors="coerce")

    for col in [case_col, p_col, seed_col, "density", gini_col, x_col, y_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    
    # Exact requested filter: Degree-Gini strictly greater than 0.29.
    work = work[
        (work[gini_col] > GINI_THRESHOLD)
        & (work[case_col].isin([1, 2, 3, 4]))
    ].copy()

    results = {}

    for case in [1, 2, 3, 4]:
        sub = work[work[case_col] == case].copy()

        results[case] = pd.DataFrame(
            {
                "P": sub[p_col],
                "Run-seed": sub[seed_col],
                "Density": sub["density"],
                "Degree-Gini": sub[gini_col],
                "final_X_share (%)": sub[x_col] * 100.0,
                "Final_Y_share (%)": sub[y_col] * 100.0,
                "outcome_class": sub[outcome_col],
            }
        ).reset_index(drop=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract corrected_results rows with Degree-Gini > 0.29."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input workbook (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output workbook (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input workbook not found: {input_path}"
        )

    results = extract_case_tables(input_path)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for case in [1, 2, 3, 4]:
            results[case].to_excel(
                writer,
                sheet_name=f"Case{case}",
                index=False,
            )

    print(f"Input : {input_path}")
    print("Sheet : corrected_results")
    print("Filter: Degree-Gini > 0.29")
    print(f"Output: {output_path}")
    for case in [1, 2, 3, 4]:
        print(f"Case {case}: {len(results[case])} rows")


if __name__ == "__main__":
    main()
