#!/usr/bin/env python3
"""
Section 1 - Network characteristics.

Input:
    A v7 revised_summary.xlsx from the baseline static payoff analysis.
    Default:
        payoff_revised/revised_summary.xlsx

Purpose:
    1. ER: Spearman correlations for p, density, clustering, diameter,
       and degree Gini.
    2. BA: Spearman correlations for m1/m2, density, clustering,
       diameter, degree Gini, and hub measures.
    3. Produce scatter plots and square correlation cross-tabulations.
    4. Keep network characteristics separate from norm-evolution outcomes.

Output folder:
    section1_network_characteristics/
        section1_network_characteristics.xlsx
        figures/
            ER_structural_relationships.png
            BA_structural_relationships.png

The script does not rerun simulations.
"""

from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


def read_network_realizations(path: Path,experiment: str = "payoff",
    n_value: int = 20,) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input workbook not found: {path}")

    df = pd.read_excel(path, sheet_name="network_realizations")

    required = [
        "network", "N", "run_seed", "p", "density", "clustering",
        "diameter", "degree_gini", "m1", "m2", "mean_degree",
        "max_degree", "max_degree_to_mean_degree"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in network_realizations: {missing}"
        )

        # Explicitly validate the intended payoff baseline network set.
    if "experiment" not in df.columns:
        raise ValueError("The network_realizations sheet must contain 'experiment'.")

    if "N" not in df.columns:
        raise ValueError("The network_realizations sheet must contain 'N'.")

    # experiments = sorted(df["experiment"].dropna().astype(str).unique().tolist())
    # if experiments != ["payoff"]:
    #     raise ValueError(
    #         f"Section 1 expects payoff-experiment networks only. "
    #         f"Found experiments: {experiments}"
    #     )

    # if not (df["N"] == 20).all():
    #     found_N = sorted(df["N"].dropna().unique().tolist())
    #     raise ValueError(
    #         f"Section 1 expects N=20 networks only. Found N values: {found_N}"
    #     )

    # df = df[(df["experiment"] == "payoff") & (df["N"] == 20)].copy()

    # df = df.copy()

    df = df[
    (df["experiment"].astype(str) == experiment)
    & (pd.to_numeric(df["N"], errors="coerce") == n_value)
].copy()


    # One row per starting network.
    dedupe = [c for c in ["experiment", "network", "N", "run_seed"]
              if c in df.columns]
    df = df.drop_duplicates(subset=dedupe).copy()

    return df


def spearman_rows(df: pd.DataFrame, network_name: str,
                  relationships: list[tuple[str, str]]) -> pd.DataFrame:
    sub = df[df["network"] == network_name].copy()
    rows = []

    for x, y in relationships:
        if x not in sub.columns or y not in sub.columns:
            rows.append({
                "network": network_name, "x_variable": x, "y_variable": y,
                "n": 0, "spearman_rho": math.nan, "p_value": math.nan,
                "status": "missing column",
            })
            continue

        tmp = sub[[x, y]].dropna()
        if len(tmp) < 3 or tmp[x].nunique() < 2 or tmp[y].nunique() < 2:
            rho = math.nan
            pval = math.nan
            status = "insufficient variation"
        else:
            rho, pval = spearmanr(tmp[x], tmp[y])
            status = "ok"

        rows.append({
            "network": network_name,
            "x_variable": x,
            "y_variable": y,
            "n": len(tmp),
            "spearman_rho": rho,
            "p_value": pval,
            "status": status,
        })

    return pd.DataFrame(rows)


def expected_direction(network_name: str, x: str, y: str) -> str:
    explicit = {
        ("ER", "p", "density"): "Positive",
        ("ER", "p", "clustering"): "Positive",
        ("ER", "p", "diameter"): "Negative",
        ("ER", "density", "clustering"): "Positive",
        ("ER", "density", "diameter"): "Negative",
        # ("ER", "density", "degree_gini"): "Negative",
        # ("BA", "m2", "density"): "Positive",
        # ("BA", "m2", "mean_degree"): "Positive",
        # ("BA", "m2", "degree_gini"): "Negative",
        # ("BA", "m2", "max_degree"): "Positive",
        # ("BA", "m2", "max_degree_to_mean_degree"): "Negative",
    }
    return explicit.get((network_name, x, y), "Not pre-specified")


def add_direction_annotations(corr: pd.DataFrame) -> pd.DataFrame:
    out = corr.copy()
    out["expected_direction"] = [
        expected_direction(r["network"], r["x_variable"], r["y_variable"])
        for _, r in out.iterrows()
    ]

    def matches(row):
        if row["status"] != "ok" or row["expected_direction"] == "Not pre-specified":
            return pd.NA
        if pd.isna(row["spearman_rho"]):
            return pd.NA
        if row["spearman_rho"] == 0:
            return "No direction"
        return (row["spearman_rho"] > 0 and row["expected_direction"] == "Positive") or (
            row["spearman_rho"] < 0 and row["expected_direction"] == "Negative"
        )

    out["direction_matches_expectation"] = out.apply(matches, axis=1)
    return out


def make_correlation_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    vars_ = sorted(set(corr["x_variable"]).union(corr["y_variable"]))
    mat = pd.DataFrame(pd.NA, index=vars_, columns=vars_, dtype="object")
    for v in vars_:
        mat.loc[v, v] = 1.0
    for _, row in corr.iterrows():
        if row["status"] == "ok" and not pd.isna(row["spearman_rho"]):
            x, y, rho = row["x_variable"], row["y_variable"], float(row["spearman_rho"])
            if x in mat.index and y in mat.columns:
                mat.loc[x, y] = rho
                mat.loc[y, x] = rho
    mat.index.name = "Variable"
    return mat.reset_index()


def plot_relationship_grid(
    df: pd.DataFrame,
    network_name: str,
    relationships: list[tuple[str, str]],
    out_path: Path,
) -> None:
    sub = df[df["network"] == network_name].copy()

    n = len(relationships)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(15, 4.5 * rows),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, (x, y) in zip(axes, relationships):
        if x not in sub.columns or y not in sub.columns:
            ax.text(0.5, 0.5, "Missing variable", ha="center", va="center")
            ax.set_axis_off()
            continue

        tmp = sub[[x, y]].dropna()
        ax.scatter(tmp[x], tmp[y], alpha=0.70)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{x} vs {y}")

    for ax in axes[len(relationships):]:
        ax.set_axis_off()

    # fig.suptitle(
    #     f"{network_name} network characteristics",
    #     fontsize=14,
    # )
    # fig.tight_layout()

    fig.suptitle(
    f"{network_name} network characteristics",
    fontsize=14,
    y=0.995,
)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="payoff_revised/revised_summary.xlsx",
        help="v7 revised_summary.xlsx for the baseline static payoff experiment.",
    )
    parser.add_argument(
        "--outdir",
        default="section1_network_characteristics",
    )

    parser.add_argument(
    "--experiment",
    default="payoff",
    help="Experiment to analyse: payoff, initial, or size."
)

    parser.add_argument(
        "--N",
        type=int,
        default=20,
        help="Network size to analyse: 20, 50, or 100."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # df = read_network_realizations(input_path)
    df = read_network_realizations(
        input_path,
        experiment=args.experiment,
        n_value=args.N,
    )

    er_variables = [
        "p",
        "density",
        "clustering",
        "diameter",
        "degree_gini",
    ]

    ba_variables = [
        "p",
        "m1",
        "m2",
        "density",
        "clustering",
        "diameter",
        "degree_gini",
        "max_degree",
        "max_degree_to_mean_degree",
        "mean_degree",
    ]

    # Compute every unique pairwise Spearman correlation among the
    # variables specified for each network family.
    from itertools import combinations

    er_relationships = list(combinations(er_variables, 2))
    ba_relationships = list(combinations(ba_variables, 2))

    er_corr = spearman_rows(df, "ER", er_relationships)
    ba_corr = spearman_rows(df, "BA", ba_relationships)
    correlations = add_direction_annotations(
        pd.concat([er_corr, ba_corr], ignore_index=True)
    )
    er_matrix = make_correlation_matrix(er_corr)
    ba_matrix = make_correlation_matrix(ba_corr)

    er_data = df[df["network"] == "ER"].copy()
    ba_data = df[df["network"] == "BA"].copy()

    with pd.ExcelWriter(
        outdir / "section1_network_characteristics.xlsx",
        engine="openpyxl",
    ) as writer:
        correlations.to_excel(writer, sheet_name="Spearman_details", index=False)
        er_matrix.to_excel(writer, sheet_name="ER_correlation_matrix", index=False)
        ba_matrix.to_excel(writer, sheet_name="BA_correlation_matrix", index=False)
        er_data.to_excel(writer, sheet_name="ER_network_data", index=False)
        ba_data.to_excel(writer, sheet_name="BA_network_data", index=False)

    def plot_in_chunks(network_name, relationships, stem, chunk_size=9):
        for chunk_no, start_idx in enumerate(range(0, len(relationships), chunk_size), start=1):
            chunk = relationships[start_idx:start_idx + chunk_size]
            suffix = "" if len(relationships) <= chunk_size else f"_{chunk_no}"
            plot_relationship_grid(
                df,
                network_name,
                chunk,
                figdir / f"{stem}{suffix}.png",
            )

    plot_in_chunks("ER", er_relationships, "ER_structural_relationships")
    plot_in_chunks("BA", ba_relationships, "BA_structural_relationships")

    print(f"Input: {input_path}")
    print(f"Output workbook: {outdir / 'section1_network_characteristics.xlsx'}")
    print(f"Figures: {figdir}")


if __name__ == "__main__":
    main()
