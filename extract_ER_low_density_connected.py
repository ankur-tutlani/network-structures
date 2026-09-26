#!/usr/bin/env python3
"""
Extract all connected ER N=20 baseline-payoff network runs with density <= 0.20
for Cases 1-4.

No simulations are rerun.

Outputs:
    ER_low_density_connected_all.csv
    ER_low_density_connected_all.xlsx

The Excel workbook contains:
    All_records
    Case1
    Case2
    Case3
    Case4
"""

from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


BASELINE_CONDITION = "baseline_0.9_1.1"
DENSITY_MAX = 0.20


def normalize_bool(series: pd.Series, field_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
    )

    if normalized.isna().any():
        bad_values = (
            series[normalized.isna()]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Could not interpret some values in {field_name}: {bad_values}"
        )

    return normalized.astype("boolean")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input workbook not found: {path}")

    df = pd.read_excel(path, sheet_name="corrected_results")

    required = [
        "experiment",
        "network",
        "N",
        "case",
        "condition",
        "run_seed",
        "p",
        "density",
        "degree_gini",
        "clustering",
        "diameter",
        "mean_degree",
        "max_degree",
        "max_degree_to_mean_degree",
        "is_connected",
        "final_X_share",
        "final_Y_share",
        "outcome_class_revised",
        "deviates_from_analytical_share_benchmark",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in corrected_results: {missing}"
        )

    # Only the payoff experiment, ER, N=20, baseline payoff condition.
    df = df[
        (df["experiment"] == "payoff")
        & (df["network"] == "ER")
        & (df["N"] == 20)
        & (df["condition"] == BASELINE_CONDITION)
    ].copy()

    if df.empty:
        raise ValueError(
            "No ER N=20 baseline payoff rows were found."
        )

    # Normalize connectedness before filtering.
    df["_connected_bool"] = normalize_bool(
        df["is_connected"], "is_connected"
    )

    # Realized graph density, not the ER generation probability p.
    df = df[
        (df["density"] >= -1e-12)
        & (df["density"] <= DENSITY_MAX + 1e-12)
        & (df["_connected_bool"])
    ].copy()

    if df.empty:
        raise ValueError(
            "No connected ER baseline runs were found with density <= 0.20."
        )

    # Confirm all four cases have at least one selected record.
    expected_cases = {1, 2, 3, 4}
    found_cases = set(df["case"].astype(int).unique())
    missing_cases = expected_cases - found_cases
    if missing_cases:
        raise ValueError(
            "No records were found for Cases "
            f"{sorted(missing_cases)} under the requested filter."
        )

    # Ensure one row per case/network realization.
    duplicate_mask = df.duplicated(
        subset=[
            "experiment",
            "network",
            "N",
            "condition",
            "case",
            "run_seed",
        ],
        keep=False,
    )
    if duplicate_mask.any():
        bad = df.loc[
            duplicate_mask,
            [
                "experiment",
                "network",
                "N",
                "condition",
                "case",
                "run_seed",
            ],
        ].drop_duplicates()

        raise ValueError(
            "Duplicate case/run records found after filtering:\n"
            + bad.to_string(index=False)
        )

    df["final_X_percent"] = 100.0 * df["final_X_share"]
    df["final_Y_percent"] = 100.0 * df["final_Y_share"]
    df["outcome_class"] = df["outcome_class_revised"]
    df["outcome_differs"] = df[
        "deviates_from_analytical_share_benchmark"
    ].astype(int)

    columns = [
        "case",
        "run_seed",
        "p",
        "density",
        "degree_gini",
        "clustering",
        "diameter",
        "mean_degree",
        "max_degree",
        "max_degree_to_mean_degree",
        "is_connected",
        "final_X_percent",
        "final_Y_percent",
        "outcome_class",
        "outcome_differs",
        "final_X_share",
        "final_Y_share",
        "condition",
        "N",
    ]

    return (
        df[columns]
        .sort_values(
            ["case", "density", "degree_gini", "run_seed"]
        )
        .reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract all connected ER N=20 baseline-payoff runs "
            "with density <= 0.20 for Cases 1-4."
        )
    )

    parser.add_argument(
        "--input",
        default="payoff_revised/revised_summary.xlsx",
        help="Revised payoff workbook containing corrected_results.",
    )
    parser.add_argument(
        "--outdir",
        default="ER_low_density_connected",
        help="Output directory.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)

    csv_path = outdir / "ER_low_density_connected_all.csv"
    xlsx_path = outdir / "ER_low_density_connected_all.xlsx"

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All_records", index=False)

        for case in [1, 2, 3, 4]:
            df[df["case"] == case].to_excel(
                writer,
                sheet_name=f"Case{case}",
                index=False,
            )

    print(f"Input: {input_path}")
    print(f"Total selected records: {len(df)}")
    print(
        "Records by case: "
        + ", ".join(
            f"Case {case}={len(df[df['case'] == case])}"
            for case in [1, 2, 3, 4]
        )
    )
    print(f"CSV: {csv_path}")
    print(f"Excel: {xlsx_path}")


if __name__ == "__main__":
    main()
