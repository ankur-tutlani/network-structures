#!/usr/bin/env python3
"""
Targeted ER low-density connected-network payoff experiment.

Purpose:
    Generate additional ER network realizations specifically for:
        - N = 20
        - baseline payoff condition: 0.9 / 1.1
        - realized ER graph density < 0.20
        - connected graphs only

The script keeps trying until the requested number of valid network
realizations is reached. Each accepted network is reused across all four
cases with the same initial 50/50 allocation.

The original strategy-update function perturbed_response4_v2 is unchanged.

Important:
    Power-law fitting is completely disabled in this targeted experiment.
    The output retains the same power-law-related columns for compatibility,
    but they are written as:
        fat_tailedness = "DISABLED"
        powerlaw_alpha = NaN
        powerlaw_xmin = NaN

Outcome classification uses the corrected population-share logic from the
revised post-processing workflow:
    Case 1: Y-only is benchmark-consistent.
    Case 2: X-only is benchmark-consistent.
    Case 3: X-only, Y-only, and exact 50/50 are benchmark-consistent.
    Case 4: coexistence and exact 50/50 are benchmark-consistent;
            homogeneous X-only/Y-only outcomes are deviations.

Outputs are compatible in structure with the earlier payoff experiment:
    - payoff_low_density_ER_all_attempts.csv
    - payoff_low_density_ER_summary.xlsx

The summary workbook contains:
    aggregate_case_results
    parameter_subgroups
    density_heterogeneity
    structural_coverage
    network_realizations
    strategy_snapshots
    attempts_summary
    outcomes
    different_runs
    rejected_attempts
    corrected_results

For this targeted experiment, all accepted network realizations satisfy
density < 0.20 and connectedness = True by construction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from Payofffunction import perturbed_response4_v2


X = 0
Y = 1
ALL_CASES = [1, 2, 3, 4]

BASELINE_CONDITION = "N100"
PAYOFF_DELTA = 0.1
N = 100
INITIAL_X_SHARE = 0.50
PERIODS = 50
DENSITY_LIMIT = 0.20


def build_matrix(case: int, delta: float) -> np.ndarray:
    lo, hi = 1.0 - delta, 1.0 + delta

    if case == 1:
        return np.array([[1.0, lo], [hi, 1.0]])
    if case == 2:
        return np.array([[1.0, hi], [lo, 1.0]])
    if case == 3:
        return np.array([[1.0, lo], [lo, 1.0]])
    if case == 4:
        return np.array([[1.0, hi], [hi, 1.0]])

    raise ValueError(f"Unknown case: {case}")


def classify_outcome(
    case: int,
    final_x_count: int,
    n: int,
) -> Tuple[bool, str]:
    """Corrected population-share classification."""

    if case == 1:
        if final_x_count == 0:
            return False, "Y_only"
        if final_x_count == n:
            return True, "X_only"
        return True, "coexistence"

    if case == 2:
        if final_x_count == n:
            return False, "X_only"
        if final_x_count == 0:
            return True, "Y_only"
        return True, "coexistence"

    if case == 3:
        if final_x_count == 0:
            return False, "Y_only"
        if final_x_count == n:
            return False, "X_only"
        if 2 * final_x_count == n:
            return False, "mixed_50_50"
        return True, "coexistence_non_50_50"

    if case == 4:
        if final_x_count == 0:
            return True, "Y_only"
        if final_x_count == n:
            return True, "X_only"
        if 2 * final_x_count == n:
            return False, "mixed_50_50"
        return False, "coexistence"

    raise ValueError(f"Unknown case: {case}")


def degree_gini(degrees: List[int]) -> float:
    x = np.sort(np.asarray(degrees, dtype=float))

    if len(x) == 0:
        return math.nan

    total = np.sum(x)

    if total == 0:
        return 0.0

    n = len(x)

    return float(
        (n + 1 - 2 * np.sum(np.cumsum(x)) / total) / n
    )


def network_metrics(G: nx.Graph) -> Dict:
    """
    Calculate structural metrics without any power-law fitting.
    """
    degrees = [int(d) for _, d in G.degree()]

    try:
        diameter = float(nx.diameter(G))
    except nx.NetworkXError:
        diameter = math.nan

    mean_degree = float(np.mean(degrees)) if degrees else 0.0
    max_degree = max(degrees) if degrees else 0

    return {
        "density": float(nx.density(G)),
        "clustering": float(nx.average_clustering(G)),
        "diameter": diameter,
        "is_connected": bool(nx.is_connected(G)),
        "fat_tailedness": "DISABLED",
        "powerlaw_alpha": math.nan,
        "powerlaw_xmin": math.nan,
        "mean_degree": mean_degree,
        "degree_std": float(np.std(degrees)) if degrees else 0.0,
        "max_degree": max_degree,
        "degree_gini": degree_gini(degrees),
        "max_degree_to_mean_degree": (
            max_degree / mean_degree if mean_degree > 0 else math.nan
        ),
        "isolates": int(nx.number_of_isolates(G)),
        "n_edges": int(G.number_of_edges()),
    }


def draw_er_candidate(
    n: int,
    run_seed: int,
) -> Tuple[float, Optional[nx.Graph], str]:
    """
    Draw an ER graph exactly as in the earlier experiment.

    A candidate is valid only if:
        density < 0.20
        connected = True
    """
    rng = random.Random(run_seed)
    p = round(rng.uniform(0.0, 1.0), 2)

    G = nx.erdos_renyi_graph(
        n=n,
        p=p,
        seed=run_seed,
    )

    density = float(nx.density(G))

    if not nx.is_connected(G):
        return p, None, "rejected_disconnected"

    if density >= DENSITY_LIMIT:
        return p, None, "rejected_density_ge_0.20"

    return p, G, "accepted"


def make_initial_state(
    n: int,
    x_share: float,
    run_seed: int,
) -> List[int]:
    x_count = int(round(x_share * n))

    state = [X] * x_count + [Y] * (n - x_count)

    random.seed(run_seed)

    return random.sample(state, len(state))


def serialize_edges(
    edges: List[Tuple[int, int]],
) -> str:
    return json.dumps(
        [[int(u), int(v)] for u, v in edges],
        separators=(",", ":"),
    )


def serialize_state(state: List[int]) -> str:
    return json.dumps(
        [int(s) for s in state],
        separators=(",", ":"),
    )


def simulate_case(
    graph: nx.Graph,
    edges: List[Tuple[int, int]],
    initial_state: List[int],
    case: int,
    n: int,
    payoff_delta: float,
    periods: int,
    run_seed: int,
) -> Dict:
    """
    Run the unchanged strategy-update process on the accepted static graph.
    """
    numpy_seed = run_seed % (2**32 - 1)

    random.seed(run_seed)
    np.random.seed(numpy_seed)

    matrix = build_matrix(case, payoff_delta)

    state = pd.DataFrame(
        {
            "agent_no": list(range(n)),
            "strategy": list(initial_state),
        }
    )

    for _ in range(periods):
        payoffs = []

        for agent in range(n):
            incident = [
                edge
                for edge in edges
                if edge[0] == agent or edge[1] == agent
            ]

            neighbours = list(
                {node for edge in incident for node in edge}
            )

            if agent in neighbours:
                neighbours.remove(agent)

            if not neighbours:
                payoffs.append(None)
                continue

            current_strategy = int(
                state.loc[
                    state["agent_no"] == agent,
                    "strategy",
                ].values[0]
            )

            payoff = 0.0

            for neighbour in neighbours:
                neighbour_strategy = int(
                    state.loc[
                        state["agent_no"] == neighbour,
                        "strategy",
                    ].values[0]
                )

                payoff += matrix[
                    current_strategy,
                    neighbour_strategy,
                ]

            payoffs.append(payoff)

        state_with_payoff = pd.DataFrame(
            {
                "agent_no": list(range(n)),
                "strategy": state["strategy"],
                "payoff": payoffs,
            }
        )

        new_state = perturbed_response4_v2(
            n,
            state_with_payoff,
            edges,
            [],
            0,
        )

        state = pd.DataFrame(
            {
                "agent_no": list(range(n)),
                "strategy": new_state,
            }
        )

    final_x_count = int(
        (state["strategy"] == X).sum()
    )

    final_y_count = int(
        (state["strategy"] == Y).sum()
    )

    differs, outcome_class = classify_outcome(
        case,
        final_x_count,
        n,
    )

    final_state = [
        int(s)
        for s in state["strategy"].tolist()
    ]

    return {
        "final_X_count": final_x_count,
        "final_Y_count": final_y_count,
        "final_X_share": final_x_count / n,
        "final_Y_share": final_y_count / n,
        "outcome_class": outcome_class,
        "differs_from_analytical": int(differs),
        "initial_state_json": serialize_state(initial_state),
        "final_state_json": serialize_state(final_state),
    }


def run_experiment(
    target_realizations: int,
    master_seed: int,
) -> pd.DataFrame:
    """
    Keep drawing ER candidates until target_realizations are accepted.

    Each accepted graph is reused across:
        - baseline payoff condition
        - Cases 1-4
        - the same initial 50/50 allocation
    """
    rows = []

    accepted_realizations = 0
    attempt_number = 0

    while accepted_realizations < target_realizations:
        attempt_number += 1
        run_seed = master_seed + attempt_number

        p, G, status = draw_er_candidate(
            N,
            run_seed,
        )

        if status != "accepted":
            rows.append(
                {
                    "experiment": "payoff_low_density_ER",
                    "condition": BASELINE_CONDITION,
                    "network": "ER",
                    "N": N,
                    "initial_X_share": INITIAL_X_SHARE,
                    "payoff_delta": PAYOFF_DELTA,
                    "attempt_number": attempt_number,
                    "run_seed": run_seed,
                    "accepted": False,
                    "accepted_realization_number": math.nan,
                    "case": math.nan,
                    "p": p,
                    "m1": math.nan,
                    "m2": math.nan,
                    "failure_reason": status,
                }
            )
            continue

        accepted_realizations += 1

        graph_stats = network_metrics(G)
        edges = list(G.edges)

        # Same initial allocation for all four cases.
        initial_state = make_initial_state(
            N,
            INITIAL_X_SHARE,
            run_seed,
        )

        for case in ALL_CASES:
            results = simulate_case(
                graph=G,
                edges=edges,
                initial_state=initial_state,
                case=case,
                n=N,
                payoff_delta=PAYOFF_DELTA,
                periods=PERIODS,
                run_seed=run_seed,
            )

            rows.append(
                {
                    "experiment": "payoff_low_density_ER",
                    "condition": BASELINE_CONDITION,
                    "network": "ER",
                    "N": N,
                    "initial_X_share": INITIAL_X_SHARE,
                    "payoff_delta": PAYOFF_DELTA,
                    "attempt_number": attempt_number,
                    "run_seed": run_seed,
                    "accepted": True,
                    "accepted_realization_number": accepted_realizations,
                    "case": case,
                    "p": p,
                    "m1": math.nan,
                    "m2": math.nan,
                    "failure_reason": "",
                    "edge_list_json": serialize_edges(edges),
                    **graph_stats,
                    **results,
                }
            )

        if (
            accepted_realizations == 1
            or accepted_realizations % 10 == 0
            or accepted_realizations == target_realizations
        ):
            print(
                f"Accepted valid ER networks: "
                f"{accepted_realizations}/{target_realizations} "
                f"(attempts={attempt_number})"
            )

    return pd.DataFrame(rows)


def add_parameter_ranges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["p_range"] = pd.cut(
        out["p"],
        bins=[-1e-9, 0.30, 0.80, 1.00001],
        labels=["0.00–0.30", "0.31–0.80", ">0.80"],
        include_lowest=True,
        right=True,
    )

    out["density_band"] = pd.cut(
        out["density"],
        bins=[-1e-12, 0.20],
        labels=["[0.00, 0.20)"],
        include_lowest=True,
        right=False,
    )

    out["m1_range"] = pd.Series(
        [None] * len(out),
        index=out.index,
        dtype="object",
    )
    out["m2_range"] = pd.Series(
        [None] * len(out),
        index=out.index,
        dtype="object",
    )

    out["heterogeneity_band"] = pd.NA

    return out


def make_summary_tables(
    all_attempts: pd.DataFrame,
):
    accepted = all_attempts[
        all_attempts["accepted"] == True
    ].copy()

    aggregate = (
        accepted
        .groupby(
            ["network", "case", "condition"],
            observed=True,
            dropna=False,
        )
        .agg(
            total_case_runs=("differs_from_analytical", "size"),
            different_runs=("differs_from_analytical", "sum"),
            mean_density=("density", "mean"),
            mean_clustering=("clustering", "mean"),
            median_diameter=("diameter", "median"),
            mean_degree_gini=("degree_gini", "mean"),
            mean_max_degree=("max_degree", "mean"),
            mean_hub_ratio=(
                "max_degree_to_mean_degree",
                "mean",
            ),
        )
        .reset_index()
    )

    aggregate["difference_rate_percent"] = (
        100
        * aggregate["different_runs"]
        / aggregate["total_case_runs"]
    ).round(1)

    binned = add_parameter_ranges(accepted)

    subgroup = (
        binned
        .groupby(
            [
                "network",
                "case",
                "condition",
                "p_range",
                "m1_range",
                "m2_range",
                "fat_tailedness",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            runs_in_group=(
                "differs_from_analytical",
                "size",
            ),
            different_runs=(
                "differs_from_analytical",
                "sum",
            ),
            mean_density=("density", "mean"),
            mean_clustering=("clustering", "mean"),
            median_diameter=("diameter", "median"),
            mean_degree_gini=("degree_gini", "mean"),
            mean_max_degree=("max_degree", "mean"),
            mean_hub_ratio=(
                "max_degree_to_mean_degree",
                "mean",
            ),
        )
        .reset_index()
    )

    subgroup["difference_rate_percent"] = (
        100
        * subgroup["different_runs"]
        / subgroup["runs_in_group"]
    ).round(1)

    density_heterogeneity = (
        binned
        .groupby(
            [
                "network",
                "case",
                "condition",
                "density_band",
                "heterogeneity_band",
                "fat_tailedness",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            runs_in_group=(
                "differs_from_analytical",
                "size",
            ),
            different_runs=(
                "differs_from_analytical",
                "sum",
            ),
            mean_density=("density", "mean"),
            mean_degree_gini=("degree_gini", "mean"),
            mean_max_degree=("max_degree", "mean"),
            mean_hub_ratio=(
                "max_degree_to_mean_degree",
                "mean",
            ),
        )
        .reset_index()
    )

    if not density_heterogeneity.empty:
        density_heterogeneity["difference_rate_percent"] = (
            100
            * density_heterogeneity["different_runs"]
            / density_heterogeneity["runs_in_group"]
        ).round(1)
    else:
        density_heterogeneity[
            "difference_rate_percent"
        ] = pd.Series(dtype=float)

    structural_coverage = (
        accepted[
            [
                "experiment",
                "condition",
                "network",
                "attempt_number",
                "run_seed",
                "accepted_realization_number",
                "N",
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
                "max_degree",
                "degree_gini",
                "max_degree_to_mean_degree",
                "isolates",
                "n_edges",
                "edge_list_json",
            ]
        ]
        .drop_duplicates(
            subset=[
                "experiment",
                "network",
                "N",
                "run_seed",
            ]
        )
        .reset_index(drop=True)
    )

    outcomes = (
        accepted
        .groupby(
            [
                "network",
                "case",
                "condition",
                "outcome_class",
            ],
            observed=True,
            dropna=False,
        )
        .size()
        .reset_index(name="runs")
    )

    different_runs = accepted[
        accepted["differs_from_analytical"] == 1
    ].copy()

    strategy_snapshots = accepted[
        [
            "experiment",
            "condition",
            "network",
            "N",
            "attempt_number",
            "run_seed",
            "accepted_realization_number",
            "case",
            "p",
            "m1",
            "m2",
            "initial_X_share",
            "payoff_delta",
            "edge_list_json",
            "initial_state_json",
            "final_state_json",
        ]
    ].copy()

    candidate_level = (
        all_attempts[
            [
                "experiment",
                "condition",
                "network",
                "attempt_number",
                "run_seed",
                "accepted",
                "failure_reason",
            ]
        ]
        .drop_duplicates(
            subset=[
                "experiment",
                "condition",
                "network",
                "attempt_number",
            ]
        )
    )

    attempts_summary = (
        candidate_level
        .groupby(
            ["network", "condition"],
            observed=True,
            dropna=False,
        )
        .agg(
            total_candidate_attempts=(
                "attempt_number",
                "count",
            ),
            accepted_realizations=("accepted", "sum"),
            rejected_candidates=(
                "accepted",
                lambda s: int((~s).sum()),
            ),
        )
        .reset_index()
    )

    attempts_summary["acceptance_rate_percent"] = (
        100
        * attempts_summary["accepted_realizations"]
        / attempts_summary["total_candidate_attempts"]
    ).round(1)

    rejected_attempts = (
        candidate_level[
            candidate_level["accepted"] == False
        ]
        .groupby(
            ["network", "condition", "failure_reason"],
            observed=True,
            dropna=False,
        )
        .size()
        .reset_index(name="rejected_candidates")
    )

    corrected_results = accepted.copy()

    classification_audit = accepted[
        [
            "experiment",
            "network",
            "condition",
            "N",
            "run_seed",
            "case",
            "final_X_count",
            "final_Y_count",
            "final_X_share",
            "outcome_class",
            "differs_from_analytical",
        ]
    ].copy()

    return {
        "aggregate_case_results": aggregate,
        "parameter_subgroups": subgroup,
        "density_heterogeneity": density_heterogeneity,
        "structural_coverage": structural_coverage,
        "network_realizations": structural_coverage.copy(),
        "strategy_snapshots": strategy_snapshots,
        "attempts_summary": attempts_summary,
        "outcomes": outcomes,
        "different_runs": different_runs,
        "rejected_attempts": rejected_attempts,
        "corrected_results": corrected_results,
        "classification_audit": classification_audit,
    }


def write_outputs(
    all_attempts: pd.DataFrame,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = (
        output_dir
        / "payoff_low_density_ER_all_attempts.csv"
    )
    summary_xlsx = (
        output_dir
        / "payoff_low_density_ER_summary.xlsx"
    )

    all_attempts.to_csv(
        raw_csv,
        index=False,
        encoding="utf-8-sig",
    )

    summaries = make_summary_tables(all_attempts)

    with pd.ExcelWriter(
        summary_xlsx,
        engine="openpyxl",
    ) as writer:
        for sheet_name, frame in summaries.items():
            frame.to_excel(
                writer,
                sheet_name=sheet_name[:31],
                index=False,
            )

    return raw_csv, summary_xlsx


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate additional connected ER N=20 networks with "
            "realized density < 0.20 for all four payoff cases."
        )
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help=(
            "Number of valid accepted ER network realizations. "
            "Each is reused across all four cases."
        ),
    )

    parser.add_argument(
        "--master-seed",
        type=int,
        default=20260916,
    )

    parser.add_argument(
        "--out",
        default="payoff_low_density_ER",
    )

    args = parser.parse_args()

    if args.runs < 50:
        parser.error(
            "--runs must be at least 50 to satisfy the requested "
            "minimum of 50 valid records for every case."
        )

    all_attempts = run_experiment(
        target_realizations=args.runs,
        master_seed=args.master_seed,
    )

    raw_csv, summary_xlsx = write_outputs(
        all_attempts=all_attempts,
        output_dir=Path(args.out),
    )

    accepted = all_attempts[
        all_attempts["accepted"] == True
    ].copy()

    accepted_networks = (
        accepted[
            [
                "network",
                "N",
                "run_seed",
                "accepted_realization_number",
            ]
        ]
        .drop_duplicates()
    )

    print()
    print(
        "Valid accepted network realizations: "
        f"{len(accepted_networks)}"
    )
    print(
        "Valid accepted case-level simulations: "
        f"{len(accepted)}"
    )
    print(
        "Expected case-level simulations: "
        f"{args.runs * 4}"
    )

    print()
    print(f"Raw CSV:   {raw_csv}")
    print(f"Summary:   {summary_xlsx}")


if __name__ == "__main__":
    main()
