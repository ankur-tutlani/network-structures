#!/usr/bin/env python3
"""
Section 5 - Power-law analysis.

Input:
    Existing v7 revised_summary.xlsx from the baseline static payoff analysis.
    Default:
        payoff_revised/revised_summary.xlsx

Question:
    What is the relationship between degree Gini and estimated power-law alpha?

Outputs:
    1. Spearman correlation table for ER, BA, and pooled networks.
    2. Scatterplot for ER.
    3. Scatterplot for BA.
    4. Combined scatterplot.
    5. Excel workbook containing the correlation results and the underlying
       network-level data.

Important:
    This is a descriptive supplementary analysis. The alpha estimate comes
    from the existing power-law fitting procedure and should not be treated
    as proof of a power law for N=20 networks.
"""

from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


def load_data(path: Path,experiment: str = "payoff",
    n_value: int = 20,) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_excel(
        path,
        sheet_name="network_realizations",
    )

    required = [
        "network",
        "run_seed",
        "degree_gini",
        "powerlaw_alpha",
        "fat_tailedness",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # if "condition" in df.columns:
    #     baseline_condition = "baseline_0.9_1.1"
    #     conditions = sorted(df["condition"].dropna().unique().tolist())
    #     if baseline_condition not in conditions:
    #         raise ValueError(
    #             f"Baseline condition {baseline_condition!r} not found; available: {conditions}"
    #         )
    #     df = df[df["condition"] == baseline_condition].copy()

    # if "N" not in df.columns:
    #     raise ValueError("network_realizations must include N for the N=20 validation.")
    # n_values = sorted(df["N"].dropna().unique().tolist())
    # if n_values != [20]:
    #     raise ValueError(f"Section 5 expects baseline N=20 networks only; found N values: {n_values}")

    # if "experiment" in df.columns:
    #     experiments = sorted(df["experiment"].dropna().unique().tolist())
    #     if experiments != ["payoff"]:
    #         raise ValueError(f"Section 5 expects the payoff experiment; found: {experiments}")

    if "N" not in df.columns:
        raise ValueError(
            "network_realizations must include N."
        )

    if "experiment" not in df.columns:
        raise ValueError(
            "network_realizations must include experiment."
        )

    df = df[
        (df["experiment"].astype(str) == experiment)
        & (
            pd.to_numeric(df["N"], errors="coerce")
            == n_value
        )
    ].copy()

    if df.empty:
        raise ValueError(
            f"No rows found for experiment={experiment!r}, "
            f"N={n_value}."
        )
    
    dedupe = [
        c for c in ["experiment", "network", "N", "run_seed"]
        if c in df.columns
    ]

    return df.drop_duplicates(subset=dedupe).copy()


def correlation_for(df: pd.DataFrame, label: str) -> dict:
    tmp = df[["degree_gini", "powerlaw_alpha"]].dropna()
    total = len(df)
    missing_alpha = int(df["powerlaw_alpha"].isna().sum())
    missing_pair = int(len(df) - len(tmp))

    if len(tmp) < 3 or tmp["degree_gini"].nunique() < 2 or tmp["powerlaw_alpha"].nunique() < 2:
        rho = math.nan
        pval = math.nan
    else:
        rho, pval = spearmanr(
            tmp["degree_gini"],
            tmp["powerlaw_alpha"],
        )

    return {
        "group": label,
        "n_total_networks": total,
        "n_valid_gini_alpha": len(tmp),
        "n_excluded_missing_alpha": missing_alpha,
        "n_excluded_missing_pair": missing_pair,
        "spearman_rho": rho,
        "p_value": pval,
    }


def make_plot(df: pd.DataFrame, title: str, out_path: Path) -> None:
    tmp = df[["degree_gini", "powerlaw_alpha"]].dropna()

    plt.figure(figsize=(7, 5))
    plt.scatter(
        tmp["degree_gini"],
        tmp["powerlaw_alpha"],
        alpha=0.70,
    )

    plt.xlabel("Degree Gini")
    plt.ylabel("Power-law alpha")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="payoff_revised/revised_summary.xlsx",
    )
    parser.add_argument(
        "--outdir",
        default="section5_powerlaw_analysis",
    )
    parser.add_argument(
    "--experiment",
    default="payoff",
    help="Experiment: payoff, initial, or size.",
)

    parser.add_argument(
        "--N",
        type=int,
        default=20,
        dest="n_value",
        help="Network size: 20, 50, or 100.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # df = load_data(input_path)
    df = load_data(
    input_path,
    experiment=args.experiment,
    n_value=args.n_value,
)

    correlations = pd.DataFrame(
        [
            correlation_for(df[df["network"] == "ER"], "ER"),
            correlation_for(df[df["network"] == "BA"], "BA"),
            correlation_for(df, "All networks"),
        ]
    )

    with pd.ExcelWriter(
        outdir / "section5_powerlaw_analysis.xlsx",
        engine="openpyxl",
    ) as writer:
        correlations.to_excel(
            writer,
            sheet_name="Spearman_correlation",
            index=False,
        )
        df.to_excel(
            writer,
            sheet_name="Network_level_data",
            index=False,
        )
        pd.DataFrame({
            "Item": ["Scope", "Fat-tailness use"],
            "Definition": [
                "Baseline payoff experiment, N=20, network-level observations.",
                "fat_tailedness is retained as a descriptive field but is not used as a selection or inference variable in the Gini-alpha correlation.",
            ],
        }).to_excel(writer, sheet_name="definitions", index=False)

    make_plot(
        df[df["network"] == "ER"],
        "ER: degree Gini vs power-law alpha",
        figdir / "ER_gini_vs_alpha.png",
    )
    make_plot(
        df[df["network"] == "BA"],
        "BA: degree Gini vs power-law alpha",
        figdir / "BA_gini_vs_alpha.png",
    )
    make_plot(
        df,
        "All networks: degree Gini vs power-law alpha",
        figdir / "All_networks_gini_vs_alpha.png",
    )

    print(f"Output workbook: {outdir / 'section5_powerlaw_analysis.xlsx'}")
    print(f"Figures: {figdir}")


if __name__ == "__main__":
    main()
