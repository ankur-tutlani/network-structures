#!/usr/bin/env python3
"""
Section 3 - Sensitivity analysis.

Inputs:
    Separate v7 revised_summary.xlsx files in their existing folders:
        payoff_revised/revised_summary.xlsx
        initial_revised/revised_summary.xlsx
        size_revised/revised_summary.xlsx

The script does not assume the folders must be renamed or moved.
Pass different paths with --payoff, --initial, and --size if needed.

Outputs:
    12 tables:
        4 cases x 3 sensitivity experiments.

Each table contains:
    Network
    Scenario
    X final %
    Y final %
    SD final X %
    SD final Y %
    Deviation %

The baseline/alternative condition labels come from the existing v11/v7
condition names. No simulations are rerun.
"""

from pathlib import Path
import argparse

import pandas as pd


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_excel(path, sheet_name="final_share_summary")


def load_deviation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_excel(path, sheet_name="deviation_summary")


def combine_experiment(path: Path) -> pd.DataFrame:
    shares = load_summary(path)
    dev = load_deviation(path)

    keys = [
        c for c in ["experiment", "network", "case", "condition", "payoff_delta"]
        if c in shares.columns and c in dev.columns
    ]

    out = shares.merge(
        dev[keys + ["deviation_pct"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    out = out.rename(
        columns={
            "mean_final_X_share": "Mean final X%",
            "mean_final_Y_share": "Mean final Y%",
            "sd_final_X_share": "SD final X%",
            "sd_final_Y_share": "SD final Y%",
            "deviation_pct": "Deviation %",
        }
    )

    return out


def validate_experiment_input(df: pd.DataFrame, experiment: str) -> None:
    expected = {
        "payoff": {"baseline_0.9_1.1", "alternative_0.8_1.2", "alternative_0.5_1.5"},
        "initial": {"X50_Y50", "X30_Y70", "X70_Y30"},
        "size": {"N20", "N50", "N100"},
    }[experiment]
    observed = set(df["condition"].dropna().unique())
    missing = expected - observed
    if missing:
        raise ValueError(f"{experiment}: missing expected scenarios: {sorted(missing)}")
    if not {"ER", "BA"}.issubset(set(df["network"].dropna().unique())):
        raise ValueError(f"{experiment}: expected both ER and BA results.")


def normalize_scenario_labels(
    df: pd.DataFrame,
    experiment: str,
) -> pd.DataFrame:
    out = df.copy()

    if experiment == "payoff":
        mapping = {
            "baseline_0.9_1.1": "Baseline 0.9–1.1",
            "alternative_0.8_1.2": "Alternative 0.8–1.2",
            "alternative_0.5_1.5": "Alternative 0.5–1.5",
        }
    elif experiment == "initial":
        mapping = {
            "X50_Y50": "Baseline X50–Y50",
            "X30_Y70": "Alternative X30–Y70",
            "X70_Y30": "Alternative X70–Y30",
        }
    elif experiment == "size":
        mapping = {
            "N20": "Baseline N20",
            "N50": "Alternative N50",
            "N100": "Alternative N100",
        }
    else:
        mapping = {}

    out["Scenario"] = out["condition"].map(mapping).fillna(out["condition"])

    return out


def build_case_table(df: pd.DataFrame, case: int) -> pd.DataFrame:
    sub = df[df["case"] == case].copy()

    keep = [
          "network",
        "Scenario",
        "Mean final X%",
        "Mean final Y%",
        "SD final X%",
        "SD final Y%",
        "Deviation %",
    ]
    # return sub[keep].sort_values("Scenario").reset_index(drop=True)
    return sub[keep].sort_values(["network", "Scenario"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--payoff",
        default="payoff_revised/revised_summary.xlsx",
    )
    parser.add_argument(
        "--initial",
        default="initial_revised/revised_summary.xlsx",
    )
    parser.add_argument(
        "--size",
        default="size_revised/revised_summary.xlsx",
    )
    parser.add_argument(
        "--outdir",
        default="section3_sensitivity_analysis",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sources = {
        "Payoff": (Path(args.payoff), "payoff"),
        "Initial_state": (Path(args.initial), "initial"),
        "Network_size": (Path(args.size), "size"),
    }

    tables = {}

    for label, (path, experiment) in sources.items():
        df = combine_experiment(path)
        validate_experiment_input(df, experiment)
        df = normalize_scenario_labels(df, experiment)

        for case in [1, 2, 3, 4]:
            tables[f"{label}_C{case}"] = build_case_table(df, case)

    with pd.ExcelWriter(
        outdir / "section3_sensitivity_analysis.xlsx",
        engine="openpyxl",
    ) as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
            table.to_csv(outdir / f"{name}.csv", index=False,encoding="utf-8-sig")

    print(f"Output workbook: {outdir / 'section3_sensitivity_analysis.xlsx'}")
    print(f"Tables written: {len(tables)}")


if __name__ == "__main__":
    main()
