#!/usr/bin/env python3
"""
Section 2 - Baseline norm-evolution results by density and heterogeneity.

Input:
    A v7 revised_summary.xlsx from the baseline static payoff experiment.
    Default:
        payoff_revised/revised_summary.xlsx

Main output:
    8 tables total:
        Case 1: ER + BA
        Case 2: ER + BA
        Case 3: ER + BA
        Case 4: ER + BA

Main table definitions follow the agreed structure.

ER:
    Density band
    Connected networks (%)
    Mean degree Gini
    N runs
    Deviation %
    X_only (count)
    Y_only (count)
    coexistence (count)
    coexistence_non_50_50 (count)
    mixed_50_50 (count)
    X_only
    Y_only
    coexistence
    coexistence_non_50_50
    mixed_50_50

BA:
    Density band
    Mean degree Gini
    Mean max degree / mean degree
    N runs
    Deviation %
    X_only (count)
    Y_only (count)
    coexistence (count)
    coexistence_non_50_50 (count)
    mixed_50_50 (count)
    X_only
    Y_only
    coexistence
    coexistence_non_50_50
    mixed_50_50

Additional supporting sheet:
    density_heterogeneity_support
        A density-band x heterogeneity-band view of deviation rates.
        This is supplementary to the 8 main tables.

No simulations are rerun.
"""

from pathlib import Path
import argparse
import pandas as pd


DENSITY_BANDS = ["0.00–0.20", "0.20–0.40", "0.40–0.60", "0.60–0.80", "0.80–1.00"]


def load_data(path: Path,condition=None) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="corrected_results")
  

    required = [
        "network", "case", "condition", "density", "is_connected",
        "mean_degree", "degree_std", "degree_gini",
        "final_X_share", "final_Y_share",
        "outcome_class_revised",
        "deviates_from_analytical_share_benchmark",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Explicitly select the baseline payoff condition.  This makes the
    # script robust to a workbook that also contains sensitivity conditions.
    # baseline_condition = "baseline_0.9_1.1"
    available = sorted(df["condition"].dropna().unique().tolist())
    if condition not in available:
        raise ValueError(
            f"Baseline condition {condition!r} was not found. "
            f"Available conditions: {available}"
        )
    if condition is not None:
        df = df[df["condition"] == condition].copy()
    # df = df[df["condition"] == baseline_condition].copy()

    # Use five equal-width density bands over the valid [0, 1] density range.
    df["density_band"] = pd.cut(
        df["density"],
        bins=[-1e-12, 0.20, 0.40, 0.60, 0.80, 1.000001],
        labels=DENSITY_BANDS,
        include_lowest=True,
        right=True,
    )

    # Mean degree Gini tertiles within each network and condition.
    def tertile(s):
        try:
            return pd.qcut(
                s,
                q=3,
                labels=["Low", "Medium", "High"],
                duplicates="drop",
            )
        except ValueError:
            return pd.Series(
                ["Insufficient variation"] * len(s),
                index=s.index,
                dtype="object",
            )

    df["heterogeneity_band"] = (
        df.groupby(["network"], observed=True, dropna=False)["degree_gini"]
        .transform(tertile)
    )

    return df


def connected_pct(s: pd.Series) -> float:
    normalized = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        })
    )
    if normalized.isna().any():
        raise ValueError(
            "is_connected contains values that could not be interpreted "
            "as True/False, 1/0, or Yes/No."
        )
    return 100.0 * normalized.mean()

def _add_outcome_counts(out: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    """Add counts of each final population-share outcome by density band."""
    outcome_labels = [
        "X_only",
        "Y_only",
        "coexistence",
        "coexistence_non_50_50",
        "mixed_50_50",
    ]

    counts = (
        sub.groupby(
            ["density_band", "outcome_class_revised"],
            observed=True,
            dropna=False,
        )
        .size()
        .unstack(fill_value=0)
    )

    counts = counts.reindex(columns=outcome_labels, fill_value=0)
    counts = counts.reset_index()

    rename_map = {label: label for label in outcome_labels}
    counts = counts.rename(columns=rename_map)

    out = out.merge(counts, on="density_band", how="left")
    for label in outcome_labels:
        if label not in out.columns:
            out[label] = 0
        out[label] = out[label].fillna(0).astype(int)

    return out


def build_er_table(df: pd.DataFrame, case: int) -> pd.DataFrame:
    sub = df[(df["network"] == "ER") & (df["case"] == case)].copy()
    sub = sub.drop_duplicates(
        subset=[c for c in ["network", "N", "run_seed"] if c in sub.columns]
    )

    out = (
        sub.groupby("density_band", observed=True, as_index=False)
        .agg(
            **{
                "Connected networks (%)": ("is_connected", connected_pct),
                "Mean degree Gini": ("degree_gini", "mean"),
                "N runs": ("run_seed", "size"),
                "Deviation %": (
                    "deviates_from_analytical_share_benchmark",
                    lambda s: 100.0 * s.mean(),
                ),
            }
        )
    )

    out = _add_outcome_counts(out, sub)
    out.insert(0, "Density band", out.pop("density_band"))

    return out[
        [
            "Density band",
            "Connected networks (%)",
            "Mean degree Gini",
            "N runs",
            "Deviation %",
            "X_only",
            "Y_only",
            "coexistence",
            "coexistence_non_50_50",
            "mixed_50_50",
        ]
    ]


def build_ba_table(df: pd.DataFrame, case: int) -> pd.DataFrame:
    sub = df[(df["network"] == "BA") & (df["case"] == case)].copy()
    sub = sub.drop_duplicates(
        subset=[c for c in ["network", "N", "run_seed"] if c in sub.columns]
    )

    out = (
        sub.groupby("density_band", observed=True, as_index=False)
        .agg(
            **{
                "Mean degree Gini": ("degree_gini", "mean"),
                "Mean max degree / mean degree": (
                    "max_degree_to_mean_degree",
                    "mean",
                ),
                "N runs": ("run_seed", "size"),
                "Deviation %": (
                    "deviates_from_analytical_share_benchmark",
                    lambda s: 100.0 * s.mean(),
                ),
            }
        )
    )

    out = _add_outcome_counts(out, sub)
    out.insert(0, "Density band", out.pop("density_band"))

    return out[
        [
            "Density band",
            "Mean degree Gini",
            "Mean max degree / mean degree",
            "N runs",
            "Deviation %",
            "X_only",
            "Y_only",
            "coexistence",
            "coexistence_non_50_50",
            "mixed_50_50",
        ]
    ]

def build_supporting_density_heterogeneity(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.drop_duplicates(
        subset=[c for c in ["network", "N", "run_seed", "case"] if c in df.columns]
    ).copy()
    return (
        sub.groupby(
            ["network", "case", "density_band", "heterogeneity_band"],
            observed=True,
            dropna=False,
            as_index=False,
        )
        .agg(
            n_runs=("run_seed", "size"),
            deviation_pct=(
                "deviates_from_analytical_share_benchmark",
                lambda s: 100.0 * s.mean(),
            ),
            mean_degree_gini=("degree_gini", "mean"),
            mean_final_X_share=("final_X_share", "mean"),
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="payoff_revised/revised_summary.xlsx",
        help="v7 revised_summary.xlsx for the baseline static payoff experiment.",
    )
    parser.add_argument(
        "--outdir",
        default="section2_baseline_density_heterogeneity",
    )
    parser.add_argument(
    "--condition",
    default="baseline_0.9_1.1",
)
    args = parser.parse_args()

    input_path = Path(args.input)
    condition=args.condition
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path,condition)

    tables = {}
    for case in [1, 2, 3, 4]:
        tables[f"Case{case}_ER"] = build_er_table(df, case)
        tables[f"Case{case}_BA"] = build_ba_table(df, case)

    support = build_supporting_density_heterogeneity(df)

    heterogeneity_definition = pd.DataFrame({
        "Item": [
            "Main-table heterogeneity",
            "Supporting heterogeneity band",
            "Band construction",
            "Density intervals",
        ],
        "Definition": [
            "Mean degree Gini within each density band.",
            "Relative Low/Medium/High classification of network-level degree Gini.",
            "Tertiles computed separately for ER and BA across the baseline networks.",
            "0.00–0.20 means density <= 0.20; 0.20–0.40 means 0.20 < density <= 0.40; 0.40–0.60 means 0.40 < density <= 0.60; 0.60–0.80 means 0.60 < density <= 0.80; 0.80–1.00 means 0.80 < density <= 1.00.",
        ],
    })

    with pd.ExcelWriter(
        outdir / "section2_baseline_density_heterogeneity.xlsx",
        engine="openpyxl",
    ) as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
        support.to_excel(
            writer,
            sheet_name="density_heterogeneity_support",
            index=False,
        )
        heterogeneity_definition.to_excel(
            writer,
            sheet_name="definitions",
            index=False,
        )

    # Also save each main table as CSV for easy inspection.
    for name, table in tables.items():
        # table.to_csv(outdir / f"{name}.csv", index=False)
        table.to_csv(
            outdir / f"{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    # support.to_csv(outdir / "density_heterogeneity_support.csv", index=False)
    support.to_csv(
        outdir / "density_heterogeneity_support.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Input: {input_path}")
    print(f"Output workbook: {outdir / 'section2_baseline_density_heterogeneity.xlsx'}")
    print(f"Main tables written: {len(tables)}")
    print(f"Supporting table: {outdir / 'density_heterogeneity_support.csv'}")


if __name__ == "__main__":
    main()
