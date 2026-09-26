
from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from Payofffunction import perturbed_response4_v2

X = 0
Y = 1
ALL_CASES = [1, 2, 3, 4]


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


def classify_outcome(case: int, final_x_count: int, n: int) -> Tuple[bool, str]:
    final_y_count = n - final_x_count
    if case == 1:
        if final_y_count == n:
            return False, "Y_only"
        if final_x_count == n:
            return True, "X_only"
        return True, "coexistence"
    if case == 2:
        if final_x_count == n:
            return False, "X_only"
        if final_y_count == n:
            return True, "Y_only"
        return True, "coexistence"
    if case == 3:
        if final_x_count == 0:
            return True, "Y_only"
        if final_x_count == n:
            return True, "X_only"
        return False, "coexistence"
    if case == 4:
        if final_x_count == 0:
            return False, "Y_only"
        if final_x_count == n:
            return False, "X_only"
        if 2 * final_x_count == n:
            return False, "mixed_50_50"
        return True, "coexistence_non_50_50"
    raise ValueError(f"Unknown case: {case}")


def degree_gini(degrees: List[int]) -> float:
    x = np.sort(np.asarray(degrees, dtype=float))
    if len(x) == 0:
        return math.nan
    total = np.sum(x)
    if total == 0:
        return 0.0
    n = len(x)
    return float((n + 1 - 2 * np.sum(np.cumsum(x)) / total) / n)


def fat_tail(degrees: List[int]) -> Tuple[str, float, float]:
    try:
        import powerlaw
    except ImportError as exc:
        raise ImportError(
            "powerlaw is required. Install with: pip install powerlaw"
        ) from exc

    positive = [int(d) for d in degrees if d > 0]
    if len(positive) < 5 or len(set(positive)) < 2:
        return "NA", math.nan, math.nan

    # fit = powerlaw.Fit(positive, discrete=True, verbose=False)
    # alpha = float(fit.power_law.alpha)
    # xmin = float(fit.power_law.xmin)
    try:
        fit = powerlaw.Fit(
            positive,
            discrete=True,
            verbose=False,
        )
        alpha = float(fit.power_law.alpha)
        xmin = float(fit.power_law.xmin)

    except Exception as exc:
        warnings.warn(
            f"Power-law fit failed for degree sequence: {exc}",
            RuntimeWarning,
        )
        return "NA", math.nan, math.nan

    if alpha < 2:
        label = "Heavy"
    elif alpha < 3:
        label = "Medium"
    else:
        label = "Low"

    return label, alpha, xmin


def network_metrics(G: nx.Graph) -> Dict:
    degrees = [int(d) for _, d in G.degree()]
    label, alpha, xmin = fat_tail(degrees)

    try:
        diameter = float(nx.diameter(G))
    except nx.NetworkXError:
        # diameter = -1.0
        diameter = math.nan


    mean_degree = float(np.mean(degrees)) if degrees else 0.0
    max_degree = max(degrees) if degrees else 0

    return {
        "density": float(nx.density(G)),
        "clustering": float(nx.average_clustering(G)),
        "diameter": diameter,
        "is_connected": bool(nx.is_connected(G)),
        "fat_tailedness": label,
        "powerlaw_alpha": alpha,
        "powerlaw_xmin": xmin,
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


def draw_candidate(
    network: str,
    n: int,
    run_seed: int,
) -> Tuple[float, Optional[int], Optional[int], Optional[nx.Graph], str]:
    rng = random.Random(run_seed)
    p = round(rng.uniform(0.0, 1.0), 2)

    if network == "ER":
        G = nx.erdos_renyi_graph(n=n, p=p, seed=run_seed)
        return p, None, None, G, "accepted"

    if network == "BA":
        # m2 = rng.randint(1, n - 1)
        # m1 = rng.randint(m2, n - 1)
        m2 = rng.randint(1, n - 1)
        m1 = rng.randint(max(2, m2), n - 1)

        initial_graph = nx.erdos_renyi_graph(
            n=m1, p=p, seed=run_seed
        )

        if not nx.is_connected(initial_graph):
            return (
                p, m1, m2, None,
                "rejected_disconnected_initial_graph",
            )

        G = nx.barabasi_albert_graph(
            n=n,
            m=m2,
            seed=run_seed,
            initial_graph=initial_graph,
        )
        return p, m1, m2, G, "accepted"

    raise ValueError(f"Unknown network: {network}")


def make_initial_state(n: int, x_share: float, run_seed: int) -> List[int]:
    x_count = int(round(x_share * n))
    state = [X] * x_count + [Y] * (n - x_count)
    random.seed(run_seed)
    return random.sample(state, len(state))



def serialize_edges(edges: List[Tuple[int, int]]) -> str:
    """Store the exact network edge list as compact JSON."""
    return json.dumps([[int(u), int(v)] for u, v in edges], separators=(",", ":"))


def serialize_state(state: List[int]) -> str:
    """Store the node-strategy vector as compact JSON."""
    return json.dumps([int(s) for s in state], separators=(",", ":"))


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

    numpy_seed = run_seed % (2**32 - 1)
    random.seed(run_seed)
    # np.random.seed(run_seed)
    np.random.seed(numpy_seed)

    matrix = build_matrix(case, payoff_delta)
    state = pd.DataFrame(
        {"agent_no": list(range(n)), "strategy": list(initial_state)}
    )

    for _ in range(periods):
        payoffs = []

        for agent in range(n):
            incident = [
                edge for edge in edges
                if edge[0] == agent or edge[1] == agent
            ]
            neighbours = list({node for edge in incident for node in edge})

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
                payoff += matrix[current_strategy, neighbour_strategy]

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

    final_x_count = int((state["strategy"] == X).sum())
    final_y_count = int((state["strategy"] == Y).sum())
    differs, outcome_class = classify_outcome(case, final_x_count, n)

    final_state = [int(s) for s in state["strategy"].tolist()]

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


def experiment_conditions(experiment: str):
    if experiment == "payoff":
        return [
            ("baseline_0.9_1.1", 0.1, 20, 0.50, ALL_CASES),
            ("alternative_0.8_1.2", 0.2, 20, 0.50, ALL_CASES),
            ("alternative_0.5_1.5", 0.5, 20, 0.50, ALL_CASES),
        ]
    if experiment == "initial":
        return [
            # ("X30_Y70", 0.1, 20, 0.30, [3, 4]),
            # ("X50_Y50", 0.1, 20, 0.50, [3, 4]),
            # ("X70_Y30", 0.1, 20, 0.70, [3, 4]),

            ("X30_Y70", 0.1, 20, 0.30, ALL_CASES),
            ("X50_Y50", 0.1, 20, 0.50, ALL_CASES),
            ("X70_Y30", 0.1, 20, 0.70, ALL_CASES),
        ]
    if experiment == "size":
        return [
            # ("N20", 0.1, 20, 0.50, [3, 4]),
            # ("N50", 0.1, 50, 0.50, [3, 4]),
            # ("N100", 0.1, 100, 0.50, [3, 4]),

            ("N20", 0.1, 20, 0.50, ALL_CASES),
            ("N50", 0.1, 50, 0.50, ALL_CASES),
            ("N100", 0.1, 100, 0.50, ALL_CASES),
        ]
    if experiment == "structure":
        return [
            ("structure_baseline", 0.1, 20, 0.50, []),
        ]
    raise ValueError(f"Unknown experiment: {experiment}")


def run_experiment(
    experiment: str,
    target_realizations: int,
    master_seed: int,
) -> pd.DataFrame:
    """
    Run reviewer experiments.

    Paired design:
    - payoff sensitivity: same accepted network + same initial allocation
      across all payoff conditions; only payoff_delta changes.
    - initial-state sensitivity: same accepted network across all initial
      share conditions; only x_share changes.
    - size sensitivity: separate network realizations for each N because N
      changes the graph itself.
    - structure: network generation only.

    Rejected BA candidates are recorded as candidate-level rows.
    """
    conditions = experiment_conditions(experiment)
    rows = []
    condition_index = 0

    # ------------------------------------------------------------------
    # Paired payoff sensitivity:
    # generate each accepted network once per network type and reuse it
    # across the three payoff conditions.
    # ------------------------------------------------------------------
    if experiment == "payoff":
        for network in ["ER", "BA"]:
            accepted_realizations = 0
            attempt_number = 0

            while accepted_realizations < target_realizations:
                attempt_number += 1
                network_index = 0 if network == "ER" else 1

                run_seed = (
                    master_seed
                    + network_index * 100_000
                    + attempt_number
                )

                p, m1, m2, G, status = draw_candidate(
                    network, 20, run_seed
                )

                if status != "accepted":
                    for condition, payoff_delta, n, x_share, cases in conditions:
                        rows.append(
                            {
                                "experiment": experiment,
                                "condition": condition,
                                "network": network,
                                "N": n,
                                "initial_X_share": x_share,
                                "payoff_delta": payoff_delta,
                                "attempt_number": attempt_number,
                                "run_seed": run_seed,
                                "accepted": False,
                                "accepted_realization_number": math.nan,
                                "case": math.nan,
                                "p": p,
                                "m1": m1 if m1 is not None else math.nan,
                                "m2": m2 if m2 is not None else math.nan,
                                "failure_reason": status,
                            }
                        )
                    continue

                accepted_realizations += 1
                graph_stats = network_metrics(G)
                edges = list(G.edges)

                # Same initial allocation for all payoff conditions.
                initial_state = make_initial_state(
                    20, 0.50, run_seed
                )

                for condition, payoff_delta, n, x_share, cases in conditions:
                    for case in cases:
                        results = simulate_case(
                            graph=G,
                            edges=edges,
                            initial_state=initial_state,
                            case=case,
                            n=n,
                            payoff_delta=payoff_delta,
                            periods=50,
                            run_seed=run_seed,
                        )

                        rows.append(
                            {
                                "experiment": experiment,
                                "condition": condition,
                                "network": network,
                                "N": n,
                                "initial_X_share": x_share,
                                "payoff_delta": payoff_delta,
                                "attempt_number": attempt_number,
                                "run_seed": run_seed,
                                "accepted": True,
                                "accepted_realization_number": accepted_realizations,
                                "case": case,
                                "p": p,
                                "m1": m1 if m1 is not None else math.nan,
                                "m2": m2 if m2 is not None else math.nan,
                                "failure_reason": "",
                                "edge_list_json": serialize_edges(edges),
                                **graph_stats,
                                **results,
                            }
                        )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Paired initial-state sensitivity:
    # same accepted network realization is reused across 30/70, 50/50,
    # and 70/30. Only x_share changes.
    # ------------------------------------------------------------------
    if experiment == "initial":
        for network in ["ER", "BA"]:
            accepted_realizations = 0
            attempt_number = 0

            while accepted_realizations < target_realizations:
                attempt_number += 1
                network_index = 0 if network == "ER" else 1

                # Independent network seed namespace for the initial-state
                # experiment, while remaining paired across its conditions.
                run_seed = (
                    master_seed
                    + 10_000_000
                    + network_index * 100_000
                    + attempt_number
                )

                # All initial-state conditions use N=20 and payoff_delta=0.1.
                p, m1, m2, G, status = draw_candidate(
                    network, 20, run_seed
                )

                if status != "accepted":
                    for condition, payoff_delta, n, x_share, cases in conditions:
                        rows.append(
                            {
                                "experiment": experiment,
                                "condition": condition,
                                "network": network,
                                "N": n,
                                "initial_X_share": x_share,
                                "payoff_delta": payoff_delta,
                                "attempt_number": attempt_number,
                                "run_seed": run_seed,
                                "accepted": False,
                                "accepted_realization_number": math.nan,
                                "case": math.nan,
                                "p": p,
                                "m1": m1 if m1 is not None else math.nan,
                                "m2": m2 if m2 is not None else math.nan,
                                "failure_reason": status,
                            }
                        )
                    continue

                accepted_realizations += 1
                graph_stats = network_metrics(G)
                edges = list(G.edges)

                for condition, payoff_delta, n, x_share, cases in conditions:
                    # Initial allocation changes, so generate a distinct
                    # state from the same network for each x_share.
                    initial_state = make_initial_state(
                        n, x_share, run_seed
                    )

                    for case in cases:
                        # Use the same run_seed for the deterministic
                        # simulation machinery; the initial_state itself
                        # differs by x_share.
                        results = simulate_case(
                            graph=G,
                            edges=edges,
                            initial_state=initial_state,
                            case=case,
                            n=n,
                            payoff_delta=payoff_delta,
                            periods=50,
                            run_seed=run_seed,
                        )

                        rows.append(
                            {
                                "experiment": experiment,
                                "condition": condition,
                                "network": network,
                                "N": n,
                                "initial_X_share": x_share,
                                "payoff_delta": payoff_delta,
                                "attempt_number": attempt_number,
                                "run_seed": run_seed,
                                "accepted": True,
                                "accepted_realization_number": accepted_realizations,
                                "case": case,
                                "p": p,
                                "m1": m1 if m1 is not None else math.nan,
                                "m2": m2 if m2 is not None else math.nan,
                                "failure_reason": "",
                                "edge_list_json": serialize_edges(edges),
                                **graph_stats,
                                **results,
                            }
                        )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Existing behavior for size and structure experiments.
    # ------------------------------------------------------------------
    for condition, payoff_delta, n, x_share, cases in conditions:
        for network in ["ER", "BA"]:
            accepted_realizations = 0
            attempt_number = 0

            while accepted_realizations < target_realizations:
                attempt_number += 1
                network_index = 0 if network == "ER" else 1

                run_seed = (
                    master_seed
                    + condition_index * 1_000_000
                    + network_index * 100_000
                    + attempt_number
                )

                p, m1, m2, G, status = draw_candidate(
                    network, n, run_seed
                )

                if status != "accepted":
                    rows.append(
                        {
                            "experiment": experiment,
                            "condition": condition,
                            "network": network,
                            "N": n,
                            "initial_X_share": x_share,
                            "payoff_delta": payoff_delta,
                            "attempt_number": attempt_number,
                            "run_seed": run_seed,
                            "accepted": False,
                            "accepted_realization_number": math.nan,
                            "case": math.nan,
                            "p": p,
                            "m1": m1 if m1 is not None else math.nan,
                            "m2": m2 if m2 is not None else math.nan,
                            "failure_reason": status,
                        }
                    )
                    continue

                accepted_realizations += 1
                graph_stats = network_metrics(G)
                edges = list(G.edges)
                initial_state = make_initial_state(
                    n, x_share, run_seed
                )

                # Structure-only experiment: stop after network creation.
                if not cases:
                    rows.append(
                        {
                            "experiment": experiment,
                            "condition": condition,
                            "network": network,
                            "N": n,
                            "initial_X_share": x_share,
                            "payoff_delta": payoff_delta,
                            "attempt_number": attempt_number,
                            "run_seed": run_seed,
                            "accepted": True,
                            "accepted_realization_number": accepted_realizations,
                            "case": math.nan,
                            "p": p,
                            "m1": m1 if m1 is not None else math.nan,
                            "m2": m2 if m2 is not None else math.nan,
                            "failure_reason": "",
                            "edge_list_json": serialize_edges(edges),
                            **graph_stats,
                        }
                    )
                    continue

                for case in cases:
                    results = simulate_case(
                        graph=G,
                        edges=edges,
                        initial_state=initial_state,
                        case=case,
                        n=n,
                        payoff_delta=payoff_delta,
                        periods=50,
                        run_seed=run_seed,
                    )

                    rows.append(
                        {
                            "experiment": experiment,
                            "condition": condition,
                            "network": network,
                            "N": n,
                            "initial_X_share": x_share,
                            "payoff_delta": payoff_delta,
                            "attempt_number": attempt_number,
                            "run_seed": run_seed,
                            "accepted": True,
                            "accepted_realization_number": accepted_realizations,
                            "case": case,
                            "p": p,
                            "m1": m1 if m1 is not None else math.nan,
                            "m2": m2 if m2 is not None else math.nan,
                            "failure_reason": "",
                            "edge_list_json": serialize_edges(edges),
                            **graph_stats,
                            **results,
                        }
                    )

            # Preserve condition-based seed namespaces for size/structure.
        condition_index += 1

    return pd.DataFrame(rows)


# def add_parameter_ranges(df: pd.DataFrame) -> pd.DataFrame:
#     out = df.copy()
#     out["p_range"] = pd.cut(
#         out["p"],
#         bins=[-1e-9, 0.30, 0.80, 1.00001],
#         labels=["0.00–0.30", "0.31–0.80", ">0.80"],
#         include_lowest=True,
#     )
#     out["m1_range"] = pd.cut(
#         out["m1"],
#         bins=[0, 5, 10, 15, 19, 1000],
#         labels=["1–5", "6–10", "11–15", "16–19", "20+"],
#         include_lowest=True,
#     )
#     out["m2_range"] = pd.cut(
#         out["m2"],
#         bins=[0, 5, 10, 15, 19, 1000],
#         labels=["1–5", "6–10", "11–15", "16–19", "20+"],
#         include_lowest=True,
#     )
#     return out


def add_parameter_ranges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["p_range"] = pd.cut(
        out["p"],
        bins=[-1e-9, 0.30, 0.80, 1.00001],
        labels=["0.00–0.30", "0.31–0.80", ">0.80"],
        include_lowest=True,
        right=True
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
            return pd.Series(np.nan, index=series.index)

        return pd.cut(
            series,
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True,
        )

    # out["m1_range"] = np.nan
    # out["m2_range"] = np.nan
    out["m1_range"] = pd.Series(index=out.index, dtype="object")
    out["m2_range"] = pd.Series(index=out.index, dtype="object")

    for n_value in sorted(out["N"].dropna().unique()):
        mask = out["N"] == n_value
        out.loc[mask, "m1_range"] = m_range(
            out.loc[mask, "m1"], int(n_value)
        )
        out.loc[mask, "m2_range"] = m_range(
            out.loc[mask, "m2"], int(n_value)
        )

    return out

def add_structural_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

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


def make_summary_tables(all_attempts: pd.DataFrame):
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
            mean_hub_ratio=("max_degree_to_mean_degree", "mean"),
            median_alpha=("powerlaw_alpha", "median"),
        )
        .reset_index()
    )

    aggregate["difference_rate_percent"] = (
        100 * aggregate["different_runs"]
        / aggregate["total_case_runs"]
    ).round(1)

    binned = add_parameter_ranges(accepted)
    binned = add_structural_bins(binned)

    subgroup = (
        binned
        .groupby(
            [
                "network", "case", "condition",
                "p_range", "m1_range", "m2_range",
                "fat_tailedness",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            runs_in_group=("differs_from_analytical", "size"),
            different_runs=("differs_from_analytical", "sum"),
            mean_density=("density", "mean"),
            mean_clustering=("clustering", "mean"),
            median_diameter=("diameter", "median"),
            mean_degree_gini=("degree_gini", "mean"),
            mean_max_degree=("max_degree", "mean"),
            mean_hub_ratio=("max_degree_to_mean_degree", "mean"),
            median_alpha=("powerlaw_alpha", "median"),
        )
        .reset_index()
    )

    subgroup["difference_rate_percent"] = (
        100 * subgroup["different_runs"] / subgroup["runs_in_group"]
    ).round(1)

    density_heterogeneity = make_density_heterogeneity_outcomes(accepted)
    structural_coverage = make_structural_coverage(accepted)
    structural_matrix = make_structural_matrix(accepted)

    candidate_level = (
        all_attempts[
            [
                "experiment", "condition", "network",
                "attempt_number", "run_seed",
                "accepted", "failure_reason",
            ]
        ]
        .drop_duplicates(
            subset=[
                "experiment", "condition",
                "network", "attempt_number",
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
            total_candidate_attempts=("attempt_number", "count"),
            accepted_realizations=("accepted", "sum"),
            rejected_candidates=(
                "accepted",
                lambda s: int((~s).sum()),
            ),
        )
        .reset_index()
    )

    attempts_summary["acceptance_rate_percent"] = (
        100 * attempts_summary["accepted_realizations"]
        / attempts_summary["total_candidate_attempts"]
    ).round(1)

    outcomes = (
        accepted
        .groupby(
            ["network", "case", "condition", "outcome_class"],
            observed=True,
            dropna=False,
        )
        .size()
        .reset_index(name="runs")
    )

    rejected_summary = (
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

    different_runs = accepted[
        accepted["differs_from_analytical"] == 1
    ].copy()

    strategy_snapshots = make_strategy_snapshots(accepted)

    return (
        aggregate,
        subgroup,
        density_heterogeneity,
        structural_matrix,
        structural_coverage,
        strategy_snapshots,
        attempts_summary,
        outcomes,
        different_runs,
        rejected_summary,
    )



def make_strategy_snapshots(accepted: pd.DataFrame) -> pd.DataFrame:
    """Return one row per accepted case with exact initial/final strategy vectors."""
    cols = [
        "experiment", "condition", "network", "N",
        "attempt_number", "run_seed", "accepted_realization_number",
        "case", "p", "m1", "m2",
        "initial_X_share", "payoff_delta",
        "edge_list_json", "initial_state_json", "final_state_json",
    ]
    return accepted[cols].copy()


def make_structural_coverage(accepted: pd.DataFrame) -> pd.DataFrame:
    """
    Network-level structural table.

    For paired payoff/initial experiments, one underlying network should
    appear only once because its structure is identical across conditions.
    For other experiments, condition remains part of the unique key.
    """
    d = make_structural_coverage_for_condition(accepted)
    if d.empty:
        return d

    if d["experiment"].iat[0] in {"payoff", "initial"}:
        d = (
            d.drop_duplicates(
                subset=[
                    "experiment",
                    "network",
                    "attempt_number",
                ]
            )
            .copy()
        )

    return d


def make_structural_matrix(accepted: pd.DataFrame) -> pd.DataFrame:
    """
    Structural coverage matrix.

    For paired payoff/initial experiments, the same network is intentionally
    evaluated under multiple conditions. Therefore:
      - retain the condition label in this condition-specific matrix;
      - count each underlying network once per condition, not once per case.
    For non-paired experiments, condition is still part of the unique key.
    """
    d = make_structural_coverage_for_condition(accepted)

    return (
        d.groupby(
            [
                "network",
                "condition",
                "density_band",
                "heterogeneity_band",
            ],
            observed=True,
            dropna=False,
        )
        .size()
        .reset_index(name="realizations")
    )


def make_structural_coverage_for_condition(
    accepted: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a structural-coverage table with one row per network realization
    per condition.

    This is distinct from make_structural_coverage(), whose purpose is the
    network-level table and therefore removes paired-condition duplicates.
    """
    cols = [
        "experiment", "condition", "network",
        "attempt_number", "run_seed",
        "accepted_realization_number", "N",
        "p", "m1", "m2", "density",
        "clustering", "diameter", "is_connected",
        "fat_tailedness", "powerlaw_alpha",
        "powerlaw_xmin", "mean_degree", "degree_std",
        "max_degree", "degree_gini",
        "max_degree_to_mean_degree", "isolates", "n_edges",
        "edge_list_json",
    ]

    d = (
        accepted[cols]
        .drop_duplicates(
            subset=[
                "experiment",
                "condition",
                "network",
                "attempt_number",
            ]
        )
        .copy()
    )

    return add_structural_bins(d)


def make_density_heterogeneity_outcomes(
    accepted: pd.DataFrame,
) -> pd.DataFrame:
    d = add_structural_bins(accepted)

    result = (
        d.groupby(
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
            runs_in_group=("differs_from_analytical", "size"),
            different_runs=("differs_from_analytical", "sum"),
            mean_density=("density", "mean"),
            mean_degree_gini=("degree_gini", "mean"),
            mean_max_degree=("max_degree", "mean"),
            mean_hub_ratio=("max_degree_to_mean_degree", "mean"),
            median_alpha=("powerlaw_alpha", "median"),
        )
        .reset_index()
    )

    result["difference_rate_percent"] = (
        100 * result["different_runs"]
        / result["runs_in_group"]
    ).round(1)

    return result


def write_outputs(
    all_attempts: pd.DataFrame,
    output_dir: Path,
    experiment: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = output_dir / f"{experiment}_all_attempts.csv"
    summary_xlsx = output_dir / f"{experiment}_summary.xlsx"

    all_attempts.to_csv(raw_csv, index=False)

    # Structure-only runs do not contain case-level outcome columns.
    # Therefore they must not be passed through the evolutionary summaries.
    if experiment == "structure":
        accepted = all_attempts[
            all_attempts["accepted"] == True
        ].copy()

        structural_coverage = make_structural_coverage(accepted)
        structural_matrix = make_structural_matrix(accepted)

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
                total_candidate_attempts=("attempt_number", "count"),
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

        rejected_summary = (
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

        with pd.ExcelWriter(
            summary_xlsx,
            engine="openpyxl",
        ) as writer:
            structural_matrix.to_excel(
                writer,
                sheet_name="structural_coverage",
                index=False,
            )
            structural_coverage.to_excel(
                writer,
                sheet_name="network_realizations",
                index=False,
            )
            attempts_summary.to_excel(
                writer,
                sheet_name="attempts_summary",
                index=False,
            )
            rejected_summary.to_excel(
                writer,
                sheet_name="rejected_attempts",
                index=False,
            )

        return raw_csv, summary_xlsx

    (
        aggregate,
        subgroup,
        density_heterogeneity,
        structural_matrix,
        structural_coverage,
        strategy_snapshots,
        attempts_summary,
        outcomes,
        different_runs,
        rejected_summary,
    ) = make_summary_tables(all_attempts)

    with pd.ExcelWriter(
        summary_xlsx,
        engine="openpyxl",
    ) as writer:
        aggregate.to_excel(
            writer,
            sheet_name="aggregate_case_results",
            index=False,
        )
        subgroup.to_excel(
            writer,
            sheet_name="parameter_subgroups",
            index=False,
        )
        density_heterogeneity.to_excel(
            writer,
            sheet_name="density_heterogeneity",
            index=False,
        )
        structural_matrix.to_excel(
            writer,
            sheet_name="structural_coverage",
            index=False,
        )
        structural_coverage.to_excel(
            writer,
            sheet_name="network_realizations",
            index=False,
        )
        strategy_snapshots.to_excel(
            writer,
            sheet_name="strategy_snapshots",
            index=False,
        )
        attempts_summary.to_excel(
            writer,
            sheet_name="attempts_summary",
            index=False,
        )
        outcomes.to_excel(
            writer,
            sheet_name="outcomes",
            index=False,
        )
        different_runs.to_excel(
            writer,
            sheet_name="different_runs",
            index=False,
        )
        rejected_summary.to_excel(
            writer,
            sheet_name="rejected_attempts",
            index=False,
        )

    return raw_csv, summary_xlsx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["payoff", "initial", "size", "structure"],
        default="payoff",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Accepted network realizations per experiment cell.",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=20260908,
    )
    parser.add_argument(
        "--out",
        default="reviewer_results",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be a positive integer.")

    all_attempts = run_experiment(
        experiment=args.experiment,
        target_realizations=args.runs,
        master_seed=args.master_seed,
    )

    raw_csv, summary_xlsx = write_outputs(
        all_attempts=all_attempts,
        output_dir=Path(args.out),
        experiment=args.experiment,
    )

    if args.experiment in {"payoff", "initial"}:
        # These experiments duplicate each underlying network candidate
        # across three paired conditions. Count the network only once.
        candidate_attempts = len(
            all_attempts[
                ["experiment", "network", "attempt_number"]
            ].drop_duplicates()
        )
        accepted_realizations = len(
            all_attempts[all_attempts["accepted"] == True][
                [
                    "experiment", "network",
                    "accepted_realization_number",
                ]
            ].drop_duplicates()
        )
    else:
        candidate_attempts = len(
            all_attempts[
                [
                    "experiment", "condition",
                    "network", "attempt_number",
                ]
            ].drop_duplicates()
        )
        accepted_realizations = len(
            all_attempts[all_attempts["accepted"] == True][
                [
                    "experiment", "condition",
                    "network", "accepted_realization_number",
                ]
            ].drop_duplicates()
        )

    print()
    print(f"Candidate attempts:              {candidate_attempts}")
    print(f"Accepted realizations:           {accepted_realizations}")
    print(
        "Rejected candidates:             "
        f"{candidate_attempts - accepted_realizations}"
    )

    if args.experiment != "structure":
        print(
            "Accepted case-level simulations: "
            f"{int((all_attempts['accepted'] == True).sum())}"
        )
        different_case_rows = int(
            (
                (all_attempts["accepted"] == True)
                & (all_attempts["differs_from_analytical"] == 1)
            ).sum()
        )
        print(
            "Different case-level runs:       "
            f"{different_case_rows}"
        )

    print()
    print(f"Raw CSV:   {raw_csv}")
    print(f"Summary:   {summary_xlsx}")


if __name__ == "__main__":
    main()
