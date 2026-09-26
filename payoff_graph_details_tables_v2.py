#!/usr/bin/env python3
"""
Create two detailed tables for selected payoff-experiment appendix networks.

Inputs:
    graph_selection_manifest.csv
    payoff revised_summary.xlsx

Outputs:
    payoff_graph_details_ER.csv
    payoff_graph_details_BA.csv
    payoff_graph_details.xlsx

The script does not rerun simulations. It uses the graph-selection manifest
to identify the selected runs, then retrieves full network and case-level
information from the revised payoff workbook.

ER columns:
    P, density, Gini, clustering, diameter, connectedness,
    run_seed, final X %, final Y %, outcome

BA columns:
    P, density, m1, m2, Gini, hub ratio, clustering, diameter,
    connectedness, run_seed, final X %, final Y %, outcome

"outcome" is:
    Differs    -> deviates_from_analytical_share_benchmark is True
    Does not differ -> otherwise

Default inputs are intended for the payoff experiment only.
"""

from pathlib import Path
import argparse
import pandas as pd


def normalize_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")

    mapped = (
        s.astype(str)
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
    if mapped.isna().any():
        raise ValueError(
            "The connectedness/deviation field contains values that could "
            "not be interpreted as Boolean."
        )
    return mapped.astype("boolean")


def load_inputs(manifest_path: Path, workbook_path: Path):
    manifest = pd.read_csv(manifest_path)

    required_manifest = [
        "condition",
        "network",
        "case",
        "run_seed",
    ]
    missing = [c for c in required_manifest if c not in manifest.columns]
    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {missing}"
        )

    # This script is for the payoff experiment only.
    # The payoff experiment has exactly these three conditions.
    payoff_conditions = {
        "baseline_0.9_1.1",
        "alternative_0.8_1.2",
        "alternative_0.5_1.5",
    }
    conditions = set(
        manifest["condition"].dropna().astype(str).unique().tolist()
    )
    unexpected = sorted(conditions - payoff_conditions)
    if unexpected:
        raise ValueError(
            "The supplied graph_selection_manifest.csv does not appear to "
            "be the payoff-experiment manifest. Unexpected conditions: "
            f"{unexpected}"
        )

    network_df = pd.read_excel(
        workbook_path,
        sheet_name="network_realizations",
    )
    results_df = pd.read_excel(
        workbook_path,
        sheet_name="corrected_results",
    )

    return manifest, network_df, results_df


def build_tables(manifest, network_df, results_df):
    # ---------------------------------------------------------------
    # Structural network information
    # ---------------------------------------------------------------
    # In the paired payoff experiment, the same network is reused across
    # payoff conditions. Therefore, condition is NOT part of structural
    # network identity. The revised post-processor deduplicates payoff
    # network_realizations by experiment + network + N + run_seed.
    structural_cols = [
        "experiment",
        "network",
        "N",
        "run_seed",
        "p",
        "m1",
        "m2",
        "density",
        "clustering",
        "diameter",
        "is_connected",
        "degree_gini",
        "max_degree_to_mean_degree",
    ]
    missing_structural = [
        c for c in structural_cols if c not in network_df.columns
    ]
    if missing_structural:
        raise ValueError(
            "network_realizations is missing required columns: "
            f"{missing_structural}"
        )

    # Restrict explicitly to payoff, N=20 network realizations.
    structural = network_df[
        (network_df["experiment"] == "payoff")
        & (network_df["N"].astype(int) == 20)
    ][structural_cols].copy()

    structural_key = ["network", "N", "run_seed"]

    # Detect conflicting structural records before deduplication.
    structural_value_cols = [
        "p",
        "m1",
        "m2",
        "density",
        "clustering",
        "diameter",
        "is_connected",
        "degree_gini",
        "max_degree_to_mean_degree",
    ]
    conflict_rows = []
    for key, group in structural.groupby(structural_key, dropna=False):
        for col in structural_value_cols:
            if group[col].drop_duplicates().shape[0] > 1:
                conflict_rows.append((*key, col))
    if conflict_rows:
        preview = pd.DataFrame(
            conflict_rows,
            columns=structural_key + ["conflicting_field"],
        ).head(10)
        raise ValueError(
            "Conflicting structural values were found for the same paired "
            "payoff network realization. First conflicts:\n"
            f"{preview.to_string(index=False)}"
        )

    structural = structural.drop_duplicates(subset=structural_key).copy()

    # ---------------------------------------------------------------
    # Case-level final outcome information
    # ---------------------------------------------------------------
    result_cols = [
        "condition",
        "network",
        "case",
        "run_seed",
        "final_X_share",
        "final_Y_share",
        "outcome_class_revised",
        "deviates_from_analytical_share_benchmark",
    ]
    missing_results = [c for c in result_cols if c not in results_df.columns]
    if missing_results:
        raise ValueError(
            "corrected_results is missing required columns: "
            f"{missing_results}"
        )

    results = results_df[
        result_cols
    ].copy()

    result_key = ["condition", "network", "case", "run_seed"]

    # Detect conflicting case-level records before deduplication.
    result_value_cols = [
        "final_X_share",
        "final_Y_share",
        "outcome_class_revised",
        "deviates_from_analytical_share_benchmark",
    ]
    conflict_rows = []
    for key, group in results.groupby(result_key, dropna=False):
        for col in result_value_cols:
            if group[col].drop_duplicates().shape[0] > 1:
                conflict_rows.append((*key, col))
    if conflict_rows:
        preview = pd.DataFrame(
            conflict_rows,
            columns=result_key + ["conflicting_field"],
        ).head(10)
        raise ValueError(
            "Conflicting corrected_results records were found for the same "
            "payoff case. First conflicts:\n"
            f"{preview.to_string(index=False)}"
        )

    results = results.drop_duplicates(subset=result_key).copy()

    # ---------------------------------------------------------------
    # Selected rows from the payoff graph-selection manifest
    # ---------------------------------------------------------------
    selected = manifest[
        ["condition", "network", "case", "run_seed"]
    ].drop_duplicates().copy()

    # Payoff experiment uses N=20.
    selected["N"] = 20

    # Structural join excludes condition because the graph is paired across
    # all payoff conditions.
    merged = selected.merge(
        structural,
        on=["network", "N", "run_seed"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_struct"),
    )

    # Case-level outcome join retains condition and case.
    merged = merged.merge(
        results,
        on=["condition", "network", "case", "run_seed"],
        how="left",
        validate="one_to_one",
    )

    # Fail loudly if a selected row could not be matched.
    match_cols = [
        "p",
        "density",
        "clustering",
        "diameter",
        "is_connected",
        "degree_gini",
        "max_degree_to_mean_degree",
        "final_X_share",
        "final_Y_share",
        "outcome_class_revised",
        "deviates_from_analytical_share_benchmark",
    ]
    if merged[match_cols].isna().any().any():
        bad = merged[
            merged[match_cols].isna().any(axis=1)
        ][["condition", "network", "case", "run_seed"]]
        raise ValueError(
            "Some selected payoff-manifest rows could not be matched "
            "completely to the revised payoff workbook:\n"
            f"{bad.to_string(index=False)}"
        )

    merged["connected_bool"] = normalize_bool(merged["is_connected"])
    merged["deviates_bool"] = normalize_bool(
        merged["deviates_from_analytical_share_benchmark"]
    )

    merged["connectedness"] = merged["connected_bool"].map(
        {True: "Yes", False: "No"}
    )
    merged["outcome"] = merged["deviates_bool"].map(
        {True: "Differs", False: "Does not differ"}
    )

    merged["P"] = merged["p"]
    merged["Gini"] = merged["degree_gini"]
    merged["hub ratio"] = merged["max_degree_to_mean_degree"]
    merged["final X %"] = 100.0 * merged["final_X_share"]
    merged["final Y %"] = 100.0 * merged["final_Y_share"]

    er = merged[merged["network"] == "ER"].copy()
    ba = merged[merged["network"] == "BA"].copy()

    er = er[
        [
            "condition",
            "case",
            "P",
            "density",
            "Gini",
            "clustering",
            "diameter",
            "connectedness",
            "run_seed",
            "final X %",
            "final Y %",
            "outcome",
            "outcome_class_revised",
        ]
    ].sort_values(
        ["condition", "case", "run_seed"]
    ).reset_index(drop=True)

    ba = ba[
        [
            "condition",
            "case",
            "P",
            "density",
            "m1",
            "m2",
            "Gini",
            "hub ratio",
            "clustering",
            "diameter",
            "connectedness",
            "run_seed",
            "final X %",
            "final Y %",
            "outcome",
            "outcome_class_revised",
        ]
    ].sort_values(
        ["condition", "case", "run_seed"]
    ).reset_index(drop=True)

    return er, ba


def main():
    parser = argparse.ArgumentParser(
        description="Create detailed ER and BA payoff appendix network tables."
    )
    parser.add_argument(
        "--manifest",
        default="graph_selection_manifest.csv",
        help="Payoff-experiment graph_selection_manifest.csv.",
    )
    parser.add_argument(
        "--input",
        default="payoff_revised/revised_summary.xlsx",
        help="Revised payoff workbook containing network_realizations and corrected_results.",
    )
    parser.add_argument(
        "--outdir",
        default="payoff_graph_details",
        help="Output directory.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    workbook_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    manifest, network_df, results_df = load_inputs(
        manifest_path, workbook_path
    )
    er, ba = build_tables(manifest, network_df, results_df)

    er_path = outdir / "payoff_graph_details_ER.csv"
    ba_path = outdir / "payoff_graph_details_BA.csv"
    xlsx_path = outdir / "payoff_graph_details.xlsx"

    er.to_csv(er_path, index=False, encoding="utf-8-sig")
    ba.to_csv(ba_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        er.to_excel(writer, sheet_name="ER", index=False)
        ba.to_excel(writer, sheet_name="BA", index=False)

    print(f"Manifest: {manifest_path}")
    print(f"Workbook: {workbook_path}")
    print(f"ER rows: {len(er)}")
    print(f"BA rows: {len(ba)}")
    print(f"ER CSV: {er_path}")
    print(f"BA CSV: {ba_path}")
    print(f"Excel workbook: {xlsx_path}")


if __name__ == "__main__":
    main()
