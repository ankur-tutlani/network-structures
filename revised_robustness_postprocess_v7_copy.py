#!/usr/bin/env python
"""
Revised post-processing for the existing robustness-experiment CSV files.

IMPORTANT:
- This script does NOT rerun any simulations.
- It keeps the original case numbering and simulated matrices.
- It recalculates the population-share outcome classification.
- It can process payoff, initial-state, and network-size CSVs.
- It can generate appendix network figures from the SAVED initial/final states
  and saved edge lists.

Population-share benchmark used here:
  Case 1: Y-only is the analytical population-share outcome.
  Case 2: X-only is the analytical population-share outcome.
  Case 3: X-only, Y-only, and 50/50 are treated as analytically consistent.
  Case 4: coexistence is treated as population-share consistent; exact 50/50
           is labelled separately. This is a population-share approximation,
           NOT a complete node-level Nash-equilibrium test.

Appendix graph selection:
  The figures are deliberately selected to demonstrate deviations from the
  analytical population-share outcome, not average/median outcomes.

  Case 1: strongest available X persistence, preferentially sparse network.
  Case 2: strongest available Y persistence, preferentially sparse network.
  Case 3: strongest non-50/50 coexistence, measured by distance from the
           nearest benchmark share {0, 0.5, 1}.
  Case 4: homogeneous X-only and/or Y-only deviations where available.

A user-supplied layout seed is required for reproducible node positions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# from artifact_tool import Workbook, SpreadsheetFile


def classify_population_share(
    case: int, final_x_count: int, n: int
) -> tuple[bool, str]:
    """Return (deviates_from_analytical_share_benchmark, outcome_class)."""

    # Case 1: Y strictly dominant; population-share benchmark = Y-only.
    if case == 1:
        if final_x_count == 0:
            return False, "Y_only"
        if final_x_count == n:
            return True, "X_only"
        return True, "coexistence"

    # Case 2: X strictly dominant; population-share benchmark = X-only.
    if case == 2:
        if final_x_count == n:
            return False, "X_only"
        if final_x_count == 0:
            return True, "Y_only"
        return True, "coexistence"

    # Case 3: coordination game in the population-share interpretation:
    # X-only, Y-only, and 50/50 are benchmark-consistent.
    if case == 3:
        if final_x_count == 0:
            return False, "Y_only"
        if final_x_count == n:
            return False, "X_only"
        if 2 * final_x_count == n:
            return False, "mixed_50_50"
        return True, "coexistence_non_50_50"

    # Case 4: aggregate population-share interpretation:
    # coexistence is treated as consistent; homogeneous outcomes are deviations.
    # Exact 50/50 is labelled separately but is also benchmark-consistent.
    if case == 4:
        if final_x_count == 0:
            return True, "Y_only"
        if final_x_count == n:
            return True, "X_only"
        if 2 * final_x_count == n:
            return False, "mixed_50_50"
        return False, "coexistence"

    raise ValueError(f"Unknown case: {case}")



def add_parameter_ranges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "p" in out.columns:
        out["p_range"] = pd.cut(
            out["p"],
            bins=[-1e-9, 0.30, 0.80, 1.00001],
            labels=["0.00–0.30", "0.31–0.80", ">0.80"],
            include_lowest=True,
            right=True,
        )

    if "density" in out.columns:
        out["density_band"] = pd.cut(
            out["density"],
            bins=[-1e-12, 0.10, 0.20, 0.30, 0.50, 1.000001],
            labels=[
                "0.00–0.10",
                "0.11–0.20",
                "0.21–0.30",
                "0.31–0.50",
                ">0.50",
            ],
            include_lowest=True,
            right=True,
        )

    def m_range(series: pd.Series, n: int) -> pd.Series:
        if n == 20:
            bins = [0, 5, 10, 15, 19]
            labels = ["1–5", "6–10", "11–15", "16–19"]
        elif n == 50:
            bins = [0, 10, 20, 30, 40, 49]
            labels = ["1–10", "11–20", "21–30", "31–40", "41–49"]
        elif n == 100:
            bins = [0, 20, 40, 60, 80, 99]
            labels = ["1–20", "21–40", "41–60", "61–80", "81–99"]
        else:
            return pd.Series(
                [None] * len(series),
                index=series.index,
                dtype="object",
            )

        return pd.cut(
            series,
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True,
        )

    if "N" in out.columns:
        if "m1" in out.columns:
            out["m1_range"] = pd.Series(
                [None] * len(out),
                index=out.index,
                dtype="object",
            )
        if "m2" in out.columns:
            out["m2_range"] = pd.Series(
                [None] * len(out),
                index=out.index,
                dtype="object",
            )

        for n_value in sorted(out["N"].dropna().unique()):
            mask = out["N"] == n_value

            if "m1" in out.columns:
                out.loc[mask, "m1_range"] = m_range(
                    out.loc[mask, "m1"],
                    int(n_value),
                )

            if "m2" in out.columns:
                out.loc[mask, "m2_range"] = m_range(
                    out.loc[mask, "m2"],
                    int(n_value),
                )

    if {"network", "condition", "degree_gini"}.issubset(out.columns):
        def tertile(series: pd.Series) -> pd.Series:
            try:
                return pd.qcut(
                    series,
                    q=3,
                    labels=["Low", "Medium", "High"],
                    duplicates="drop",
                )
            except ValueError:
                return pd.Series(
                    ["Insufficient variation"] * len(series),
                    index=series.index,
                    dtype="object",
                )

        out["heterogeneity_band"] = (
            out.groupby(
                ["network", "condition"],
                observed=True,
                dropna=False,
            )["degree_gini"]
            .transform(tertile)
        )

    return out




def prepare_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    if "accepted" not in df.columns:
        raise ValueError("Input CSV must contain an 'accepted' column.")

    accepted_values = (
        df["accepted"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    accepted_mask = accepted_values.isin({"true", "1", "yes"})

    accepted = df[accepted_mask].copy()

    # Keep old classification only as an audit field; never use it in summaries.
    accepted = accepted.rename(
        columns={
            "outcome_class": "outcome_class_original",
            "differs_from_analytical": (
                "differs_from_analytical_original"
            ),
        }
    )

    required = {
        "case",
        "N",
        "final_X_count",
        "run_seed",
        "network",
        "condition",
    }
    missing = required - set(accepted.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    accepted["case"] = accepted["case"].astype(int)
    accepted["N"] = accepted["N"].astype(int)
    accepted["final_X_count"] = accepted["final_X_count"].astype(int)

    if "final_Y_count" not in accepted.columns:
        accepted["final_Y_count"] = (
            accepted["N"] - accepted["final_X_count"]
        )
    else:
        accepted["final_Y_count"] = accepted["final_Y_count"].astype(int)

    if (
        (accepted["final_X_count"] < 0).any()
        or (accepted["final_X_count"] > accepted["N"]).any()
    ):
        raise ValueError("Some final_X_count values are outside [0, N].")

    if (
        (accepted["final_Y_count"] < 0).any()
        or (accepted["final_Y_count"] > accepted["N"]).any()
    ):
        raise ValueError("Some final_Y_count values are outside [0, N].")

    if (
        accepted["final_X_count"] + accepted["final_Y_count"]
        != accepted["N"]
    ).any():
        bad = accepted[
            accepted["final_X_count"] + accepted["final_Y_count"]
            != accepted["N"]
        ].head(10)
        raise ValueError(
            "final_X_count + final_Y_count != N in saved results. "
            f"First bad rows: {bad.index.tolist()}"
        )

    accepted["final_X_share"] = (
        accepted["final_X_count"] / accepted["N"]
    )
    accepted["final_Y_share"] = (
        accepted["final_Y_count"] / accepted["N"]
    )

    classified = accepted.apply(
        lambda row: classify_population_share(
            int(row["case"]),
            int(row["final_X_count"]),
            int(row["N"]),
        ),
        axis=1,
        result_type="expand",
    )
    classified.columns = [
        "deviates_from_analytical_share_benchmark",
        "outcome_class_revised",
    ]

    accepted = pd.concat([accepted, classified], axis=1)
    accepted["deviates_from_analytical_share_benchmark"] = (
        accepted["deviates_from_analytical_share_benchmark"].astype(int)
    )

    # Validate saved strategy-state JSON when available.
    problems = []

    def validate_state_json(
        raw_state,
        n: int,
        expected_x: int | None = None,
        expected_y: int | None = None,
        label: str = "state",
    ) -> None:
        if pd.isna(raw_state) or not str(raw_state).strip():
            return

        try:
            state = json.loads(raw_state)
        except Exception as exc:
            problems.append(f"{label}: invalid JSON ({exc})")
            return

        if not isinstance(state, list):
            problems.append(f"{label}: JSON value is not a list")
            return

        if len(state) != n:
            problems.append(f"{label}: state length {len(state)} != N {n}")
            return

        invalid_values = set()
        converted = []
        for value in state:
            try:
                ivalue = int(value)
            except Exception:
                invalid_values.add(str(value))
                continue
            if ivalue not in {0, 1}:
                invalid_values.add(ivalue)
            converted.append(ivalue)

        if invalid_values:
            problems.append(
                f"{label}: invalid strategy values {sorted(invalid_values)}"
            )
            return

        actual_x = sum(v == 0 for v in converted)
        actual_y = sum(v == 1 for v in converted)

        if expected_x is not None and actual_x != expected_x:
            problems.append(
                f"{label}: JSON X count {actual_x} != saved X count {expected_x}"
            )
        if expected_y is not None and actual_y != expected_y:
            problems.append(
                f"{label}: JSON Y count {actual_y} != saved Y count {expected_y}"
            )

    # Validate final and initial states independently.
    for idx, row in accepted.iterrows():
        n = int(row["N"])

        if "final_state_json" in accepted.columns:
            validate_state_json(
                row.get("final_state_json"),
                n=n,
                expected_x=int(row["final_X_count"]),
                expected_y=int(row["final_Y_count"]),
                label=f"{idx}: final_state_json",
            )

        if "initial_state_json" in accepted.columns:
            validate_state_json(
                row.get("initial_state_json"),
                n=n,
                label=f"{idx}: initial_state_json",
            )

    if problems:
        raise ValueError(
            "Saved strategy-state validation failed:\n"
            + "\n".join(problems[:10])
        )

    if "final_state_json" not in accepted.columns:
        accepted["final_state_json"] = None
    if "initial_state_json" not in accepted.columns:
        accepted["initial_state_json"] = None
    if "edge_list_json" not in accepted.columns:
        accepted["edge_list_json"] = None

    return add_parameter_ranges(accepted)




def build_summaries(
    full_df: pd.DataFrame,
    accepted_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    group_cols = [
        c
        for c in ["experiment", "network", "case", "condition", "payoff_delta"]
        if c in accepted_df.columns
    ]

    final_share_summary = (
        accepted_df.groupby(group_cols, as_index=False, observed=True)
        .agg(
            n_runs=("run_seed", "size"),
            mean_final_X_share=("final_X_share", "mean"),
            median_final_X_share=("final_X_share", "median"),
            sd_final_X_share=("final_X_share", "std"),
            mean_final_Y_share=("final_Y_share", "mean"),
            median_final_Y_share=("final_Y_share", "median"),
            sd_final_Y_share=("final_Y_share", "std"),
        )
    )

    deviation_summary = (
        accepted_df.groupby(group_cols, as_index=False, observed=True)
        .agg(
            n_runs=("run_seed", "size"),
            n_deviating=(
                "deviates_from_analytical_share_benchmark",
                "sum",
            ),
        )
    )
    deviation_summary["deviation_pct"] = (
        100.0
        * deviation_summary["n_deviating"]
        / deviation_summary["n_runs"]
    )

    outcome_counts = (
        accepted_df.groupby(
            group_cols + ["outcome_class_revised"],
            as_index=False,
            observed=True,
        )
        .size()
        .rename(columns={"size": "n"})
    )
    outcome_counts["pct"] = (
        100.0
        * outcome_counts["n"]
        / outcome_counts.groupby(group_cols, observed=True)["n"]
        .transform("sum")
    )

    outcome_composition = (
        outcome_counts.pivot_table(
            index=group_cols,
            columns="outcome_class_revised",
            values="pct",
            fill_value=0.0,
        )
        .reset_index()
    )

    subgroup_cols = [
        c
        for c in [
            "experiment",
            "network",
            "case",
            "condition",
            "p_range",
            "m1_range",
            "m2_range",
            "density_band",
            "heterogeneity_band",
            "fat_tailedness",
        ]
        if c in accepted_df.columns
    ]

    parameter_subgroups = (
        accepted_df.groupby(
            subgroup_cols,
            dropna=False,
            observed=True,
            as_index=False,
        )
        .agg(
            n_runs=("run_seed", "size"),
            deviation_pct=(
                "deviates_from_analytical_share_benchmark",
                lambda s: 100.0 * s.mean(),
            ),
            mean_final_X_share=("final_X_share", "mean"),
            median_final_X_share=("final_X_share", "median"),
        )
    ) if subgroup_cols else pd.DataFrame()

    structural_group_cols = [
        c
        for c in [
            "experiment",
            "network",
            "case",
            "condition",
            "density_band",
            "heterogeneity_band",
        ]
        if c in accepted_df.columns
    ]

    if structural_group_cols:
        agg_dict = {
            "n_runs": ("run_seed", "size"),
            "deviation_pct": (
                "deviates_from_analytical_share_benchmark",
                lambda s: 100.0 * s.mean(),
            ),
            "mean_final_X_share": ("final_X_share", "mean"),
        }
        if "degree_gini" in accepted_df.columns:
            agg_dict["mean_degree_gini"] = ("degree_gini", "mean")

        density_heterogeneity = (
            accepted_df.groupby(
                structural_group_cols,
                dropna=False,
                observed=True,
                as_index=False,
            )
            .agg(**agg_dict)
        )
    else:
        density_heterogeneity = pd.DataFrame()

    different_runs = accepted_df[
        accepted_df["deviates_from_analytical_share_benchmark"] == 1
    ].copy()

    # Candidate-level accounting: accepted cases are repeated once per case.
    # Deduplicate to one row per network-generation attempt before counting.
    candidate_cols = [
        c
        for c in [
            "experiment",
            "network",
            "condition",
            "N",
            "payoff_delta",
            "attempt_number",
            "run_seed",
            "accepted",
            "failure_reason",
        ]
        if c in full_df.columns
    ]

    candidate_level = full_df[candidate_cols].copy()
    candidate_dedupe_cols = [
        c
        for c in [
            "experiment",
            "network",
            "condition",
            "N",
            "attempt_number",
        ]
        if c in candidate_level.columns
    ]
    candidate_level = candidate_level.drop_duplicates(
        subset=candidate_dedupe_cols
    ).copy()

    candidate_level["accepted_bool"] = (
        candidate_level["accepted"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )

    attempts_group_cols = [
        c
        for c in [
            "experiment",
            "network",
            "condition",
            "N",
            "payoff_delta",
        ]
        if c in candidate_level.columns
    ]

    attempts_summary = (
        candidate_level.groupby(
            attempts_group_cols,
            as_index=False,
            observed=True,
        )
        .agg(
            candidate_attempts=("attempt_number", "size"),
            accepted_realizations=("accepted_bool", "sum"),
        )
    )
    attempts_summary["rejected_candidates"] = (
        attempts_summary["candidate_attempts"]
        - attempts_summary["accepted_realizations"]
    )
    attempts_summary["acceptance_rate_pct"] = (
        100.0
        * attempts_summary["accepted_realizations"]
        / attempts_summary["candidate_attempts"]
    )

    # Experiment-level unique candidate accounting. For paired payoff/initial
    # experiments, condition is intentionally excluded so paired candidates
    # are counted once across conditions.
    global_candidate_cols = [
        c
        for c in ["experiment", "network", "N", "attempt_number"]
        if c in candidate_level.columns
    ]
    global_candidates = candidate_level.drop_duplicates(
        subset=global_candidate_cols
    ).copy()

    global_attempt_group_cols = [
        c for c in ["experiment", "network", "N"]
        if c in global_candidates.columns
    ]
    experiment_attempts_summary = (
        global_candidates.groupby(
            global_attempt_group_cols,
            as_index=False,
            observed=True,
        )
        .agg(
            unique_candidate_attempts=("attempt_number", "size"),
            unique_accepted_realizations=("accepted_bool", "sum"),
        )
    )
    experiment_attempts_summary["unique_rejected_candidates"] = (
        experiment_attempts_summary["unique_candidate_attempts"]
        - experiment_attempts_summary["unique_accepted_realizations"]
    )
    experiment_attempts_summary["unique_acceptance_rate_pct"] = (
        100.0
        * experiment_attempts_summary["unique_accepted_realizations"]
        / experiment_attempts_summary["unique_candidate_attempts"]
    )

    rejected = candidate_level[
        ~candidate_level["accepted_bool"]
    ].copy()

    if "failure_reason" in rejected.columns:
        reject_group_cols = [
            c
            for c in [
                "experiment",
                "network",
                "condition",
                "N",
                "payoff_delta",
                "failure_reason",
            ]
            if c in rejected.columns
        ]
        rejected_candidates = (
            rejected.groupby(
                reject_group_cols,
                as_index=False,
                observed=True,
            )
            .size()
            .rename(columns={"size": "n_rejected"})
        )
    else:
        rejected_candidates = pd.DataFrame()

    network_cols = [
        c
        for c in [
            "experiment",
            "network",
            "condition",
            "N",
            "run_seed",
            "accepted_realization_number",
            "attempt_number",
            "p",
            "m1",
            "m2",
            "density",
            "clustering",
            "diameter",
            "is_connected",
            "fat_tailedness",
            "powerlaw_alpha",
            "powerlaw_xmin",
            "mean_degree",
            "degree_std",
            "degree_gini",
            "max_degree",
            "max_degree_to_mean_degree",
            "isolates",
            "n_edges",
            "edge_list_json",
        ]
        if c in accepted_df.columns
    ]

    if (
        "experiment" in accepted_df.columns
        and accepted_df["experiment"].isin(["payoff", "initial"]).all()
    ):
        # Paired payoff/initial experiments reuse the same network across
        # conditions, so condition is not part of unique network identity.
        dedupe_cols = [
            c
            for c in ["experiment", "network", "N", "run_seed"]
            if c in accepted_df.columns
        ]
    else:
        dedupe_cols = [
            c
            for c in [
                "experiment",
                "network",
                "condition",
                "N",
                "run_seed",
            ]
            if c in accepted_df.columns
        ]

    network_realizations = (
        accepted_df[network_cols]
        .drop_duplicates(subset=dedupe_cols)
        .copy()
    )

    classification_audit_cols = [
        c
        for c in [
            "experiment",
            "network",
            "condition",
            "N",
            "run_seed",
            "case",
            "final_X_count",
            "final_Y_count",
            "final_X_share",
            "outcome_class_original",
            "differs_from_analytical_original",
            "outcome_class_revised",
            "deviates_from_analytical_share_benchmark",
        ]
        if c in accepted_df.columns
    ]
    classification_audit = accepted_df[classification_audit_cols].copy()

    return {
        "final_share_summary": final_share_summary,
        "deviation_summary": deviation_summary,
        "outcome_composition": outcome_composition,
        "parameter_subgroups": parameter_subgroups,
        "density_heterogeneity": density_heterogeneity,
        "different_runs": different_runs,
        "attempts_summary": attempts_summary,
        "experiment_attempts_summary": experiment_attempts_summary,
        "rejected_candidates": rejected_candidates,
        "network_realizations": network_realizations,
        "classification_audit": classification_audit,
        "corrected_results": accepted_df,
    }


def _excel_value(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


# def write_excel(summaries: dict[str, pd.DataFrame], output_xlsx: Path):
#     wb = Workbook.create()

#     def col_letter(num: int) -> str:
#         s = ""
#         while num:
#             num, rem = divmod(num - 1, 26)
#             s = chr(65 + rem) + s
#         return s

#     for sheet_name, frame in summaries.items():
#         sheet = wb.worksheets.add(sheet_name[:31])
#         values = [list(frame.columns)] + [
#             [_excel_value(v) for v in row]
#             for row in frame.itertuples(index=False, name=None)
#         ]

#         if not values:
#             continue

#         nrows = len(values)
#         ncols = len(values[0])
#         end_col = col_letter(ncols)

#         sheet.get_range(
#             f"A1:{end_col}{nrows}"
#         ).values = values
#         sheet.get_range(
#             f"A1:{end_col}1"
#         ).format = {
#             "font": {"bold": True},
#             "wrap_text": True,
#         }
#         sheet.get_range(
#             f"A1:{end_col}{nrows}"
#         ).format.wrap_text = True
#         sheet.get_range(
#             f"A1:{end_col}{nrows}"
#         ).format.autofit_columns()
#         sheet.freeze_panes.freeze_rows(1)

#     SpreadsheetFile.export_xlsx(wb).save(output_xlsx)

def write_excel(
    summaries: dict[str, pd.DataFrame],
    output_xlsx: Path,
):
    with pd.ExcelWriter(
        output_xlsx,
        engine="openpyxl"
    ) as writer:

        for sheet_name, frame in summaries.items():
            frame.to_excel(
                writer,
                sheet_name=sheet_name[:31],
                index=False,
            )

            ws = writer.sheets[sheet_name[:31]]

            # Freeze header row
            ws.freeze_panes = "A2"

            # Bold header
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)

            # Reasonable column widths
            for column_cells in ws.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter

                for cell in column_cells:
                    value = "" if cell.value is None else str(cell.value)
                    max_length = max(max_length, len(value))

                ws.column_dimensions[column_letter].width = min(
                    max(max_length + 2, 10),
                    40,
                )



def parse_graph_row(row: pd.Series):
    required = ["edge_list_json", "initial_state_json", "final_state_json"]
    for col in required:
        if col not in row.index or pd.isna(row[col]) or not row[col]:
            raise ValueError(
                f"Selected run {row.get('run_seed')} does not contain {col}."
            )

    edges = json.loads(row["edge_list_json"])
    initial_state = json.loads(row["initial_state_json"])
    final_state = json.loads(row["final_state_json"])

    n = int(row["N"])
    if not isinstance(edges, list):
        raise ValueError("edge_list_json must contain a list of edges.")

    normalized_edges = []
    seen_edges = set()
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError(
                f"Invalid edge record in run {row.get('run_seed')}: {edge}"
            )
        u, v = int(edge[0]), int(edge[1])
        if not (0 <= u < n) or not (0 <= v < n):
            raise ValueError(
                f"Edge endpoint outside 0..{n-1}: ({u}, {v})"
            )
        if u == v:
            raise ValueError(
                f"Unexpected self-loop in run {row.get('run_seed')}: ({u}, {v})"
            )
        canonical = tuple(sorted((u, v)))
        if canonical in seen_edges:
            raise ValueError(
                f"Duplicate edge in run {row.get('run_seed')}: {canonical}"
            )
        seen_edges.add(canonical)
        normalized_edges.append(canonical)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(normalized_edges)

    return G, initial_state, final_state


# def draw_network_pair(
#     row: pd.Series,
#     output_path: Path,
#     layout_seed: int,
# ):
#     G, initial_state, final_state = parse_graph_row(row)

#     if len(initial_state) != int(row["N"]) or len(final_state) != int(row["N"]):
#         raise ValueError(
#             f"State length mismatch for run {row.get('run_seed')}."
#         )

#     pos = nx.spring_layout(G, seed=layout_seed)

#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#     for ax, state, title in [
#         (
#             axes[0],
#             initial_state,
#             f"Initial state\n"
#             f"X={sum(s == 0 for s in initial_state)}, "
#             f"Y={sum(s == 1 for s in initial_state)}",
#         ),
#         (
#             axes[1],
#             final_state,
#             f"After 50 periods\n"
#             f"X={sum(s == 0 for s in final_state)}, "
#             f"Y={sum(s == 1 for s in final_state)}",
#         ),
#     ]:
#         nx.draw_networkx_edges(
#             G,
#             pos=pos,
#             ax=ax,
#             alpha=0.7,
#         )

#         x_nodes = [
#             i for i, s in enumerate(state) if int(s) == 0
#         ]
#         y_nodes = [
#             i for i, s in enumerate(state) if int(s) == 1
#         ]

#         nx.draw_networkx_nodes(
#             G,
#             pos=pos,
#             ax=ax,
#             nodelist=x_nodes,
#             node_size=280,
#             node_shape="o",
#         )
#         nx.draw_networkx_nodes(
#             G,
#             pos=pos,
#             ax=ax,
#             nodelist=y_nodes,
#             node_size=280,
#             node_shape="s",
#         )
#         nx.draw_networkx_labels(
#             G,
#             pos=pos,
#             ax=ax,
#             font_size=7,
#         )

#         ax.set_title(title)
#         ax.axis("off")

#     fig.suptitle(
#         f"{row['network']} | Case {int(row['case'])} | "
#         f"{row['condition']} | p={row.get('p', np.nan):.2f} | "
#         f"density={row.get('density', np.nan):.2f} | "
#         f"seed={int(row['run_seed'])}",
#         fontsize=11,
#     )
#     fig.tight_layout()
#     fig.savefig(
#         output_path,
#         dpi=300,
#         bbox_inches="tight",
#     )
#     plt.close(fig)


def draw_network_pair(
    row: pd.Series,
    output_path: Path,
    layout_seed: int,
):
    G, initial_state, final_state = parse_graph_row(row)

    n = int(row["N"])

    if len(initial_state) != n:
        raise ValueError(
            f"Initial state length {len(initial_state)} != N {n}"
        )

    if len(final_state) != n:
        raise ValueError(
            f"Final state length {len(final_state)} != N {n}"
        )

    # One fixed layout for BOTH panels.
    pos = nx.spring_layout(
        G,
        seed=layout_seed,
    )

    # ---------------------------------------------------------
    # FIXED STRATEGY COLORS
    # ---------------------------------------------------------
    # X is ALWAYS this color.
    # Y is ALWAYS this color.
    #
    # Do not change these between initial and final panels.
    X_COLOR = "tab:blue"
    Y_COLOR = "tab:orange"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    states = [
        (
            axes[0],
            initial_state,
            "Initial state",
        ),
        (
            axes[1],
            final_state,
            "After 50 periods",
        ),
    ]

    for ax, state, panel_title in states:

        # Draw the exact saved network topology.
        nx.draw_networkx_edges(
            G,
            pos=pos,
            ax=ax,
            alpha=0.7,
        )

        # Identify X and Y nodes.
        x_nodes = [
            node
            for node in G.nodes()
            if int(state[node]) == 0
        ]

        y_nodes = [
            node
            for node in G.nodes()
            if int(state[node]) == 1
        ]

        # X nodes ALWAYS use X_COLOR.
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            ax=ax,
            nodelist=x_nodes,
            node_color=X_COLOR,
            node_size=500,
            node_shape="o",
        )

        # Y nodes ALWAYS use Y_COLOR.
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            ax=ax,
            nodelist=y_nodes,
            node_color=Y_COLOR,
            node_size=500,
            node_shape="o",
        )

        # Label nodes with X or Y, NOT node numbers.
        labels = {
            node: "X" if int(state[node]) == 0 else "Y"
            for node in G.nodes()
        }

        nx.draw_networkx_labels(
            G,
            pos=pos,
            labels=labels,
            ax=ax,
            font_size=9,
            font_weight="bold",
            font_color="black",
        )

        x_count = len(x_nodes)
        y_count = len(y_nodes)

        ax.set_title(
            f"{panel_title}\n"
            f"X={x_count}, Y={y_count}"
        )

        ax.axis("off")

    # One legend shared by both panels.
    # from matplotlib.lines import Line2D

    # legend_handles = [
    #     Line2D(
    #         [0],
    #         [0],
    #         marker="o",
    #         color="w",
    #         markerfacecolor=X_COLOR,
    #         markersize=10,
    #         label="X",
    #     ),
    #     Line2D(
    #         [0],
    #         [0],
    #         marker="o",
    #         color="w",
    #         markerfacecolor=Y_COLOR,
    #         markersize=10,
    #         label="Y",
    #     ),
    # ]

    # fig.legend(
    #     handles=legend_handles,
    #     loc="upper center",
    #     ncol=2,
    #     frameon=False,
    # )

    fig.suptitle(
        f"{row['network']} | Case {int(row['case'])} | "
        f"{row['condition']} | "
        f"p={row.get('p', np.nan):.2f} | "
        f"density={row.get('density', np.nan):.2f} | "
        f"run seed={int(row['run_seed'])}",
        fontsize=11,
    )

    # fig.tight_layout(
    #     rect=[0, 0, 1, 0.92]
    # )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def select_deviation_examples(
    df: pd.DataFrame,
    network: str,
    case: int,
    condition: str,
    max_case4_examples: int = 2,
) -> list[pd.Series]:
    """
    Select actual simulated runs that strongly demonstrate deviations.

    Rules are fixed and transparent before examining individual figures.
    These are illustrative, not statistically representative.
    """
    sub = df[
        (df["network"] == network)
        & (df["case"] == case)
        & (df["condition"] == condition)
    ].copy()

    if sub.empty:
        return []

    # Prefer sparse networks for the Cases 1/2 deviation illustrations because
    # the paper's mechanism concerns difficulty of convergence in sparse graphs.
    sparse = sub[sub["density"] <= 0.20].copy()
    candidate_pool = sparse if not sparse.empty else sub.copy()

    if case == 1:
        # Analytical population benchmark: X share = 0.
        # Pick the strongest persistence of X.
        deviations = candidate_pool[
            candidate_pool["final_X_count"] > 0
        ].copy()

        if deviations.empty:
            return []

        deviations = deviations.sort_values(
            ["final_X_share", "density", "run_seed"],
            ascending=[False, True, True],
        )

        row = deviations.iloc[0]
        row = row.copy()
        row["selection_reason"] = (
            "strongest available X persistence; "
            "sparse-network preference"
        )
        return [row]

    if case == 2:
        # Analytical population benchmark: X share = 1.
        # Pick the strongest persistence of Y.
        deviations = candidate_pool[
            candidate_pool["final_X_count"] < candidate_pool["N"]
        ].copy()

        if deviations.empty:
            return []

        deviations = deviations.sort_values(
            ["final_X_share", "density", "run_seed"],
            ascending=[True, True, True],
        )

        row = deviations.iloc[0]
        row = row.copy()
        row["selection_reason"] = (
            "strongest available Y persistence; "
            "sparse-network preference"
        )
        return [row]

    if case == 3:
        # Benchmark shares: 0, 0.5, 1.
        # Select strongest non-benchmark coexistence.
        deviations = sub[
            (sub["final_X_count"] > 0)
            & (sub["final_X_count"] < sub["N"])
            & ((2 * sub["final_X_count"]) != sub["N"])
        ].copy()

        if deviations.empty:
            return []

        deviations["distance_from_benchmark"] = np.minimum.reduce(
            [
                deviations["final_X_share"],
                (deviations["final_X_share"] - 0.5).abs(),
                1.0 - deviations["final_X_share"],
            ]
        )

        deviations = deviations.sort_values(
            ["distance_from_benchmark", "run_seed"],
            ascending=[False, True],
        )

        row = deviations.iloc[0]
        row = row.copy()
        row["selection_reason"] = (
            "strongest non-50/50 coexistence deviation"
        )
        return [row]

    if case == 4:
        # Population-share benchmark: coexistence.
        # Select clear homogeneous deviations in both directions where possible.
        rows = []

        x_only = sub[sub["final_X_count"] == sub["N"]].copy()
        y_only = sub[sub["final_X_count"] == 0].copy()

        if not x_only.empty:
            x_row = x_only.sort_values(
                ["density", "run_seed"],
                # ascending=[False, True],
                ascending=[True, True],
            ).iloc[0].copy()
            x_row["selection_reason"] = (
                "X-only final state: clear deviation "
                "from coexistence benchmark"
            )
            rows.append(x_row)

        if not y_only.empty and len(rows) < max_case4_examples:
            y_row = y_only.sort_values(
                ["density", "run_seed"],
                # ascending=[False, True],
                ascending=[True, True],
            ).iloc[0].copy()
            y_row["selection_reason"] = (
                "Y-only final state: clear deviation "
                "from coexistence benchmark"
            )
            rows.append(y_row)

        return rows

    return []


def generate_appendix_graphs(
    df: pd.DataFrame,
    output_dir: Path,
    layout_seed: int,
    conditions: list[str],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for condition in conditions:
        for network in ["ER", "BA"]:
            for case in [1, 2, 3, 4]:
                selected = select_deviation_examples(
                    df=df,
                    network=network,
                    case=case,
                    condition=condition,
                )

                for i, row in enumerate(selected, start=1):
                    safe_condition = (
                        str(condition)
                        .replace("/", "_")
                        .replace(" ", "_")
                    )

                    filename = (
                        f"{network}_case{case}_{safe_condition}"
                        f"_example{i}_seed{int(row['run_seed'])}.png"
                    )

                    output_path = output_dir / filename

                    draw_network_pair(
                        row,
                        output_path=output_path,
                        layout_seed=layout_seed,
                    )

                    manifest.append(
                        {
                            "condition": condition,
                            "network": network,
                            "case": case,
                            "example_number": i,
                            "run_seed": int(row["run_seed"]),
                            "p": float(row["p"])
                            if pd.notna(row.get("p"))
                            else np.nan,
                            "density": float(row["density"])
                            if pd.notna(row.get("density"))
                            else np.nan,
                            "final_X_share": float(row["final_X_share"]),
                            "final_Y_share": float(row["final_Y_share"]),
                            "outcome_class": row[
                                "outcome_class_revised"
                            ],
                            "selection_reason": row.get(
                                "selection_reason", ""
                            ),
                            "figure": filename,
                        }
                    )

    pd.DataFrame(manifest).to_csv(
        output_dir / "graph_selection_manifest.csv",
        index=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Correct and summarize existing robustness results and "
            "generate evidence-based appendix network graphs."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Existing raw CSV: payoff_all_attempts(3).csv, "
            "initial_all_attempts(1).csv, or size_all_attempts(1).csv"
        ),
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help=(
            "User-supplied seed for reproducible node layout in figures. "
            "It does not rerun or alter simulations."
        ),
    )

    # parser.add_argument(
    #     "--condition",
    #     default=None,
    #     help="Generate figures for one named condition.",
    # )

    # parser.add_argument(
    #     "--all-conditions",
    #     action="store_true",
    #     help="Generate figures for every condition in the CSV.",
    # )
    parser.add_argument(
    "--all-conditions",
    action="store_true",
    help="Include all conditions in the summary calculations.",
)

    parser.add_argument(
        "--graph-condition",
        default=None,
        help=(
            "Condition for which appendix network graphs should be generated. "
            "If omitted, no graphs are generated."
        ),
    )

    args = parser.parse_args()

    # if args.condition and args.all_conditions:
    #     raise ValueError(
    #         "Use either --condition or --all-conditions, not both."
    #     )

    input_path = Path(args.input)
    output_dir = Path(args.outdir)

    full_df = pd.read_csv(input_path)
    df = prepare_data(full_df)

    output_dir.mkdir(parents=True, exist_ok=True)

    corrected_csv = (
        output_dir / "corrected_accepted_results.csv"
    )
    df.to_csv(corrected_csv, index=False)

    # Also write a full audit file containing every original candidate row.
    # Revised outcome fields are attached only where an accepted case-level
    # simulation exists; rejected candidate rows retain their original data.
    merge_keys = [
        c for c in [
            "experiment",
            "network",
            "condition",
            "N",
            "run_seed",
            "case",
        ]
        if c in full_df.columns and c in df.columns
    ]
    # Attach only genuinely new classification fields. The original full
    # CSV already contains final_X_share/final_Y_share, so merging them again
    # would create confusing _x/_y duplicate columns.
    attach_cols = [
        c for c in [
            "outcome_class_revised",
            "deviates_from_analytical_share_benchmark",
        ]
        if c in df.columns
    ]

    if merge_keys:
        audit_values = df[merge_keys + attach_cols].copy()
        audit_values = audit_values.drop_duplicates(subset=merge_keys)

        full_df_for_audit = full_df.rename(
            columns={
                "outcome_class": "outcome_class_original",
                "differs_from_analytical": (
                    "differs_from_analytical_original"
                ),
            }
        )

        all_attempts_corrected = full_df_for_audit.merge(
            audit_values,
            on=merge_keys,
            how="left",
            validate="one_to_one",
        )
    else:
        all_attempts_corrected = full_df.rename(
            columns={
                "outcome_class": "outcome_class_original",
                "differs_from_analytical": (
                    "differs_from_analytical_original"
                ),
            }
        )

    all_attempts_csv = output_dir / "corrected_all_attempts.csv"
    all_attempts_corrected.to_csv(all_attempts_csv, index=False)

    summaries = build_summaries(full_df, df)
    output_xlsx = output_dir / "revised_summary.xlsx"
    write_excel(summaries, output_xlsx)

    # available_conditions = sorted(
    #     df["condition"].dropna().unique().tolist()
    # )

    # if args.all_conditions:
    #     conditions = available_conditions
    # elif args.condition:
    #     if args.condition not in available_conditions:
    #         raise ValueError(
    #             f"Condition '{args.condition}' not found. "
    #             f"Available: {available_conditions}"
    #         )
    #     conditions = [args.condition]
    # else:
    #     # One explicit condition is required so that figure selection is
    #     # intentional and not silently based on a default.
    #     conditions = []

    available_conditions = sorted(
    df["condition"].dropna().unique().tolist()
)

    # Summary calculations always use the full input CSV.
    # --all-conditions is retained as an explicit command-line choice.
    if not args.all_conditions:
        raise ValueError(
            "Use --all-conditions so that the revised workbook contains "
            "all robustness conditions."
        )

    # Graphs are independently controlled by --graph-condition.
    if args.graph_condition is not None:
        if args.graph_condition not in available_conditions:
            raise ValueError(
                f"Graph condition '{args.graph_condition}' not found. "
                f"Available: {available_conditions}"
            )
        graph_conditions = [args.graph_condition]
    else:
        graph_conditions = []

    graph_dir = output_dir / "appendix_network_graphs"
    # if conditions:
    #     generate_appendix_graphs(
    #         df=df,
    #         output_dir=graph_dir,
    #         layout_seed=args.seed,
    #         conditions=conditions,
    #     )
    if graph_conditions:
        generate_appendix_graphs(
            df=df,
            output_dir=graph_dir,
            layout_seed=args.seed,
            conditions=graph_conditions,
        )

    print(f"Corrected accepted CSV: {corrected_csv}")
    print(f"Corrected all-attempts CSV: {all_attempts_csv}")
    print(f"Revised Excel: {output_xlsx}")
    print(f"Accepted simulations processed: {len(df)}")
    print(f"Layout seed: {args.seed}")

    if graph_conditions:
        print(f"Graph condition: {graph_conditions}")
        print(f"Network figures: {graph_dir}")
    else:
        print("No appendix network graphs generated.")

    # if conditions:
    #     print(f"Graph conditions: {conditions}")
    #     print(f"Network figures: {graph_dir}")
    # else:
    #     print(
    #         "No graphs generated. Supply --condition or "
    #         "--all-conditions if appendix figures are needed."
    #     )


if __name__ == "__main__":
    main()
