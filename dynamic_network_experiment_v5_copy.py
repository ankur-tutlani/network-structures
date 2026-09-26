
from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Payofffunction import perturbed_response4_v2


X = 0
Y = 1
ALL_CASES = [1, 2, 3, 4]


def build_matrix(case: int, delta: float) -> np.ndarray:
    """Same game matrices as the validated v11/static code."""
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
    """
    Population-share benchmark.

    IMPORTANT:
    This is not a node-level Nash-equilibrium test.

    Case 1: analytical population outcome is Y-only.
    Case 2: analytical population outcome is X-only.
    Case 3: X-only, Y-only, and 50/50 are benchmark-consistent.
    Case 4: coexistence is population-share-consistent; 50/50 is
             labelled separately. Because the asymmetric pure
             equilibria cannot be verified from aggregate shares,
             this is an approximation.
    """

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


def fat_tail(
    degrees: List[int],
) -> Tuple[str, float, float]:
    try:
        import powerlaw
    except ImportError as exc:
        raise ImportError(
            "powerlaw is required. Install with: pip install powerlaw"
        ) from exc

    positive = [
        int(d)
        for d in degrees
        if d > 0
    ]

    if len(positive) < 5 or len(set(positive)) < 2:
        return "NA", math.nan, math.nan

    try:
        fit = powerlaw.Fit(
            positive,
            discrete=True,
            verbose=False,
        )

        alpha = float(
            fit.power_law.alpha
        )

        xmin = float(
            fit.power_law.xmin
        )

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
    degrees = [
        int(d)
        for _, d in G.degree()
    ]

    label, alpha, xmin = fat_tail(degrees)

    if nx.is_connected(G):
        try:
            diameter = float(
                nx.diameter(G)
            )
        except nx.NetworkXError:
            diameter = math.nan
    else:
        diameter = math.nan

    mean_degree = (
        float(np.mean(degrees))
        if degrees
        else 0.0
    )

    max_degree = (
        max(degrees)
        if degrees
        else 0
    )

    return {
        "density": float(
            nx.density(G)
        ),
        "clustering": float(
            nx.average_clustering(G)
        ),
        "diameter": diameter,
        "is_connected": bool(
            nx.is_connected(G)
        ),
        "fat_tailedness": label,
        "powerlaw_alpha": alpha,
        "powerlaw_xmin": xmin,
        "mean_degree": mean_degree,
        "degree_std": float(
            np.std(degrees)
        ) if degrees else 0.0,
        "max_degree": max_degree,
        "degree_gini": degree_gini(
            degrees
        ),
        "max_degree_to_mean_degree": (
            max_degree / mean_degree
            if mean_degree > 0
            else math.nan
        ),
        "isolates": int(
            nx.number_of_isolates(G)
        ),
        "n_edges": int(
            G.number_of_edges()
        ),
    }


def draw_candidate(
    network: str,
    n: int,
    run_seed: int,
) -> Tuple[
    float,
    Optional[int],
    Optional[int],
    Optional[nx.Graph],
    str,
]:
    """
    Exact network-generation logic used by the validated static v11 code.

    The same run_seed therefore gives the same starting topology as the
    corresponding static payoff experiment.
    """
    rng = random.Random(
        run_seed
    )

    p = round(
        rng.uniform(0.0, 1.0),
        2,
    )

    if network == "ER":
        G = nx.erdos_renyi_graph(
            n=n,
            p=p,
            seed=run_seed,
        )

        # Dynamic rewiring is defined only for connected social networks.
        # Reject disconnected ER candidates at the network-generation stage
        # so accepted dynamic runs can actually undergo degree-preserving
        # connected rewiring.
        if not nx.is_connected(G):
            return (
                p,
                None,
                None,
                None,
                "rejected_disconnected_er",
            )

        return (
            p,
            None,
            None,
            G,
            "accepted",
        )

    if network == "BA":
        m2 = rng.randint(
            1,
            n - 1,
        )

        m1 = rng.randint(
            max(2, m2),
            n - 1,
        )

        initial_graph = (
            nx.erdos_renyi_graph(
                n=m1,
                p=p,
                seed=run_seed,
            )
        )

        if not nx.is_connected(
            initial_graph
        ):
            return (
                p,
                m1,
                m2,
                None,
                "rejected_disconnected_initial_graph",
            )

        G = nx.barabasi_albert_graph(
            n=n,
            m=m2,
            seed=run_seed,
            initial_graph=initial_graph,
        )

        return (
            p,
            m1,
            m2,
            G,
            "accepted",
        )

    raise ValueError(
        f"Unknown network: {network}"
    )


def make_initial_state(
    n: int,
    x_share: float,
    run_seed: int,
) -> List[int]:
    x_count = int(
        round(x_share * n)
    )

    state = (
        [X] * x_count
        + [Y] * (n - x_count)
    )

    random.seed(run_seed)

    return random.sample(
        state,
        len(state),
    )


def serialize_edges(
    edges: List[Tuple[int, int]],
) -> str:
    return json.dumps(
        [
            [int(u), int(v)]
            for u, v in edges
        ],
        separators=(",", ":"),
    )


def serialize_state(
    state: List[int],
) -> str:
    return json.dumps(
        [
            int(s)
            for s in state
        ],
        separators=(",", ":"),
    )


def rewire_degree_preserving(
    G: nx.Graph,
    rewiring_rate: float,
    rng: random.Random,
) -> Tuple[int, float]:
    """
    Random degree-preserving rewiring by double-edge swaps.

    rewiring_rate is the approximate fraction of existing edges whose
    endpoints are changed per period.

    Each successful double-edge swap removes two existing edges and adds
    two new edges, preserving:
      - number of nodes
      - number of edges
      - degree sequence

    Returns:
      successful_changed_edges, achieved_turnover_rate

    The function deliberately uses a local RNG so that topology rewiring
    does not consume the random stream used by perturbed_response4_v2().
    """

    if rewiring_rate <= 0.0:
        return 0, 0.0

    if rewiring_rate >= 1.0:
        raise ValueError(
            "rewiring_rate must be < 1.0"
        )

    edge_count = G.number_of_edges()

    if edge_count < 2:
        return 0, 0.0

    target_changed_edges = int(
        round(
            rewiring_rate
            * edge_count
        )
    )

    # A double-edge swap changes exactly two edges.
    target_swaps = max(
        1,
        math.ceil(
            target_changed_edges / 2
        ),
    )

    successful_swaps = 0
    max_attempts = max(
        100,
        50 * target_swaps,
    )

    attempts = 0

    while (
        successful_swaps < target_swaps
        and attempts < max_attempts
    ):
        attempts += 1

        edges = list(
            G.edges()
        )

        if len(edges) < 2:
            break

        (a, b), (c, d) = rng.sample(
            edges,
            2,
        )

        # Four distinct nodes avoid self-loops in the proposed edges.
        if len({
            a, b, c, d
        }) < 4:
            continue

        if rng.random() < 0.5:
            candidate_1 = (
                tuple(sorted((a, c)))
            )
            candidate_2 = (
                tuple(sorted((b, d)))
            )
        else:
            candidate_1 = (
                tuple(sorted((a, d)))
            )
            candidate_2 = (
                tuple(sorted((b, c)))
            )

        old_1 = tuple(
            sorted((a, b))
        )
        old_2 = tuple(
            sorted((c, d))
        )

        # Do not recreate either old edge and do not create an existing
        # edge that was not one of the two edges being replaced.
        if candidate_1 in {
            old_1,
            old_2,
        } or candidate_2 in {
            old_1,
            old_2,
        }:
            continue

        if G.has_edge(
            *candidate_1
        ) or G.has_edge(
            *candidate_2
        ):
            continue

        # Apply the swap temporarily. Keep the network connected so the
        # dynamic extension retains the connected-network assumption of the
        # accepted static candidates.
        G.remove_edge(
            *old_1
        )
        G.remove_edge(
            *old_2
        )

        G.add_edge(
            *candidate_1
        )
        G.add_edge(
            *candidate_2
        )

        if not nx.is_connected(G):
            # Revert the swap.
            G.remove_edge(
                *candidate_1
            )
            G.remove_edge(
                *candidate_2
            )
            G.add_edge(
                *old_1
            )
            G.add_edge(
                *old_2
            )
            continue

        successful_swaps += 1

    changed_edges = (
        2 * successful_swaps
    )

    achieved_rate = (
        changed_edges / edge_count
        if edge_count > 0
        else 0.0
    )

    return (
        changed_edges,
        achieved_rate,
    )


def simulate_dynamic_case(
    initial_graph: nx.Graph,
    initial_state: List[int],
    case: int,
    n: int,
    payoff_delta: float,
    periods: int,
    run_seed: int,
    rewiring_rate: float,
    initial_metrics
) -> Dict:
    """
    Dynamic-network simulation.

    Sequence within each period:
      1. Calculate payoffs on the current network.
      2. Update strategies using the existing best-response function.
      3. Rewire the network for the next period.

    This isolates the effect of network turnover while preserving the
    original behavioral update function.
    """

    numpy_seed = (
        run_seed
        % (2**32 - 1)
    )

    random.seed(run_seed)
    np.random.seed(
        numpy_seed
    )

    # Local topology RNG. It does not alter the behavior RNG stream.
    rewire_seed = (
        run_seed
        + 50_000_003
        + int(
            round(
                rewiring_rate
                * 1_000_000
            )
        )
    )

    rewire_rng = random.Random(
        rewire_seed
    )

    G = initial_graph.copy()

    edges = list(
        G.edges()
    )

    state = pd.DataFrame(
        {
            "agent_no": list(
                range(n)
            ),
            "strategy": list(
                initial_state
            ),
        }
    )

    x_share_trajectory = [
        float(
            sum(
                int(s) == X
                for s in initial_state
            )
            / n
        )
    ]

    # initial_metrics = network_metrics(G)

    degree_gini_trajectory = [
        initial_metrics["degree_gini"]
    ] * (periods + 1)

    # degree_gini_trajectory = [
    #     network_metrics(G)[
    #         "degree_gini"
    #     ]
    # ]

    edge_turnover_trajectory = []

    total_changed_edges = 0
    periods_with_rewiring = 0

    for period in range(
        periods
    ):
        payoffs = []

        # Use the current graph topology.
        for agent in range(n):

            incident = [
                edge
                for edge in edges
                if (
                    edge[0] == agent
                    or edge[1] == agent
                )
            ]

            neighbours = list(
                {
                    node
                    for edge in incident
                    for node in edge
                }
            )

            if agent in neighbours:
                neighbours.remove(
                    agent
                )

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

                payoff += (
                    build_matrix(
                        case,
                        payoff_delta,
                    )[current_strategy, neighbour_strategy]
                )

            payoffs.append(payoff)

        state_with_payoff = pd.DataFrame(
            {
                "agent_no": list(
                    range(n)
                ),
                "strategy": state[
                    "strategy"
                ],
                "payoff": payoffs,
            }
        )

        # EXACT behavioral update function from the existing model.
        new_state = perturbed_response4_v2(
            n,
            state_with_payoff,
            edges,
            [],
            0,
        )

        state = pd.DataFrame(
            {
                "agent_no": list(
                    range(n)
                ),
                "strategy": new_state,
            }
        )

        # Rewire after strategy updating so rewiring changes the topology
        # used in the next period.
        changed_edges, turnover_rate = (
            rewire_degree_preserving(
                G,
                rewiring_rate,
                rewire_rng,
            )
        )

        edges = list(
            G.edges()
        )

        total_changed_edges += (
            changed_edges
        )

        if changed_edges > 0:
            periods_with_rewiring += 1

        edge_turnover_trajectory.append(
            float(turnover_rate)
        )

        x_share_trajectory.append(
            float(
                (
                    state["strategy"] == X
                ).sum()
                / n
            )
        )

        # degree_gini_trajectory.append(
        #     network_metrics(G)[
        #         "degree_gini"
        #     ]
        # )

    final_x_count = int(
        (
            state["strategy"] == X
        ).sum()
    )

    final_y_count = int(
        (
            state["strategy"] == Y
        ).sum()
    )

    deviates, outcome_class = (
        classify_outcome(
            case,
            final_x_count,
            n,
        )
    )

    final_state = [
        int(s)
        for s in state[
            "strategy"
        ].tolist()
    ]

    # initial_metrics = network_metrics(
    #     initial_graph
    # )
    # final_metrics = network_metrics(
    #     G
    # )
    final_metrics = initial_metrics.copy()

    final_metrics["clustering"] = nx.average_clustering(G)
    final_metrics["is_connected"] = nx.is_connected(G)

    if nx.is_connected(G):
        try:
            final_metrics["diameter"] = float(nx.diameter(G))
        except nx.NetworkXError:
            final_metrics["diameter"] = math.nan
    else:
        final_metrics["diameter"] = math.nan

    mean_turnover = (
        float(
            np.mean(
                edge_turnover_trajectory
            )
        )
        if edge_turnover_trajectory
        else 0.0
    )

    return {
        "final_X_count": final_x_count,
        "final_Y_count": final_y_count,
        "final_X_share": final_x_count / n,
        "final_Y_share": final_y_count / n,
        "outcome_class": outcome_class,
        "deviates_from_analytical_share_benchmark": int(
            deviates
        ),
        "initial_state_json": serialize_state(
            initial_state
        ),
        "final_state_json": serialize_state(
            final_state
        ),
        # Dynamic topology is stored explicitly at both endpoints.
        # edge_list_json is retained as an initial-topology compatibility
        # alias for older tooling; it does NOT replace the final topology.
        "edge_list_json": serialize_edges(
            list(
                initial_graph.edges()
            )
        ),
        "initial_edge_list_json": serialize_edges(
            list(
                initial_graph.edges()
            )
        ),
        "final_edge_list_json": serialize_edges(
            list(
                G.edges()
            )
        ),
        "x_share_trajectory_json": json.dumps(
            x_share_trajectory,
            separators=(",", ":"),
        ),
        "degree_gini_trajectory_json": json.dumps(
            degree_gini_trajectory,
            separators=(",", ":"),
        ),
        "edge_turnover_trajectory_json": json.dumps(
            edge_turnover_trajectory,
            separators=(",", ":"),
        ),
        "mean_edge_turnover_rate": mean_turnover,
        "total_changed_edges": int(
            total_changed_edges
        ),
        "periods_with_rewiring": int(
            periods_with_rewiring
        ),
        "initial_density": initial_metrics[
            "density"
        ],
        "final_density": final_metrics[
            "density"
        ],
        "initial_degree_gini": initial_metrics[
            "degree_gini"
        ],
        "final_degree_gini": final_metrics[
            "degree_gini"
        ],
        "initial_diameter": initial_metrics[
            "diameter"
        ],
        "final_diameter": final_metrics[
            "diameter"
        ],
        "initial_clustering": initial_metrics[
            "clustering"
        ],
        "final_clustering": final_metrics[
            "clustering"
        ],
        "initial_is_connected": initial_metrics[
            "is_connected"
        ],
        "final_is_connected": final_metrics[
            "is_connected"
        ],
        "final_mean_degree": final_metrics[
            "mean_degree"
        ],
        "final_max_degree": final_metrics[
            "max_degree"
        ],
    }


def parse_dynamic_graph_row(row: pd.Series):
    required = [
        "initial_edge_list_json",
        "final_edge_list_json",
        "initial_state_json",
        "final_state_json",
    ]
    for col in required:
        if col not in row.index or pd.isna(row[col]) or not row[col]:
            raise ValueError(
                f"Selected run {row.get('run_seed')} does not contain {col}."
            )

    n = int(row["N"])
    initial_edges = json.loads(row["initial_edge_list_json"])
    final_edges = json.loads(row["final_edge_list_json"])
    initial_state = json.loads(row["initial_state_json"])
    final_state = json.loads(row["final_state_json"])

    if len(initial_state) != n or len(final_state) != n:
        raise ValueError(
            f"State length mismatch for run {row.get('run_seed')}."
        )

    def make_graph(edges):
        if not isinstance(edges, list):
            raise ValueError("Edge list JSON must contain a list of edges.")

        G = nx.Graph()
        G.add_nodes_from(range(n))
        seen = set()

        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                raise ValueError(f"Invalid edge record: {edge}")
            u, v = int(edge[0]), int(edge[1])
            if not (0 <= u < n and 0 <= v < n) or u == v:
                raise ValueError(f"Invalid edge endpoints: ({u}, {v})")
            e = tuple(sorted((u, v)))
            if e in seen:
                raise ValueError(f"Duplicate edge: {e}")
            seen.add(e)
            G.add_edge(*e)

        return G

    return (
        make_graph(initial_edges),
        make_graph(final_edges),
        [int(s) for s in initial_state],
        [int(s) for s in final_state],
    )


def select_dynamic_deviation_examples(
    df: pd.DataFrame,
    network: str,
    case: int,
    condition: str,
) -> list[pd.Series]:
    sub = df[
        (df["network"] == network)
        & (df["case"] == case)
        & (df["condition"] == condition)
        & (df["accepted"] == True)
    ].copy()

    if sub.empty:
        return []

        # Measure how different the final topology is from the initial topology.
    # This is used only as a secondary criterion for appendix-figure selection.
    def edge_distance(row):
        try:
            initial_edges = {
                tuple(sorted((int(e[0]), int(e[1]))))
                for e in json.loads(row["initial_edge_list_json"])
            }
            final_edges = {
                tuple(sorted((int(e[0]), int(e[1]))))
                for e in json.loads(row["final_edge_list_json"])
            }

            union_edges = initial_edges | final_edges
            intersection_edges = initial_edges & final_edges

            if not union_edges:
                return 0.0

            return 1.0 - (
                len(intersection_edges) / len(union_edges)
            )

        except Exception:
            return 0.0

    sub["edge_distance"] = sub.apply(edge_distance, axis=1)

    sparse = sub[sub["initial_density"] <= 0.20].copy()
    candidate_pool = sparse if not sparse.empty else sub

    if case == 1:
        d = candidate_pool[candidate_pool["final_X_count"] > 0].copy()
        if d.empty:
            return []
        # row = d.sort_values(
        #     ["final_X_share", "initial_density", "run_seed"],
        #     ascending=[False, True, True],
        # ).iloc[0].copy()

        row = d.sort_values(
    ["final_X_share", "edge_distance", "initial_density", "run_seed"],
    ascending=[False, False, True, True],
).iloc[0].copy()
        
        row["selection_reason"] = "strongest available X persistence; sparse-network preference"
        return [row]

    if case == 2:
        d = candidate_pool[candidate_pool["final_X_count"] < candidate_pool["N"]].copy()
        if d.empty:
            return []
        # row = d.sort_values(
        #     ["final_X_share", "initial_density", "run_seed"],
        #     ascending=[True, True, True],
        # ).iloc[0].copy()

        row = d.sort_values(
    ["final_X_share", "edge_distance", "initial_density", "run_seed"],
    ascending=[True, False, True, True],
).iloc[0].copy()


        row["selection_reason"] = "strongest available Y persistence; sparse-network preference"
        return [row]

    if case == 3:
        d = sub[
            (sub["final_X_count"] > 0)
            & (sub["final_X_count"] < sub["N"])
            & ((2 * sub["final_X_count"]) != sub["N"])
        ].copy()
        if d.empty:
            return []
        d["distance_from_benchmark"] = np.minimum.reduce(
            [
                d["final_X_share"],
                (d["final_X_share"] - 0.5).abs(),
                1.0 - d["final_X_share"],
            ]
        )
        # row = d.sort_values(
        #     ["distance_from_benchmark", "run_seed"],
        #     ascending=[False, True],
        # ).iloc[0].copy()

        row = d.sort_values(
            ["distance_from_benchmark", "edge_distance", "run_seed"],
            ascending=[False, False, True],
        ).iloc[0].copy()

        row["selection_reason"] = "strongest non-50/50 coexistence deviation"
        return [row]

    if case == 4:
        rows = []
        x_only = sub[sub["final_X_count"] == sub["N"]].copy()
        y_only = sub[sub["final_X_count"] == 0].copy()

        if not x_only.empty:
            # row = x_only.sort_values(
            #     ["initial_density", "run_seed"],
            #     ascending=[False, True],
            # ).iloc[0].copy()

            row = x_only.sort_values(
    ["initial_density", "edge_distance", "run_seed"],
    ascending=[True, False, True],
).iloc[0].copy()
            
            row["selection_reason"] = "X-only final state: clear deviation from coexistence benchmark"
            rows.append(row)

        if not y_only.empty:
            # row = y_only.sort_values(
            #     ["initial_density", "run_seed"],
            #     ascending=[False, True],
            # ).iloc[0].copy()

            row = y_only.sort_values(
    ["initial_density", "edge_distance", "run_seed"],
    ascending=[True, False, True],
).iloc[0].copy()
            
            row["selection_reason"] = "Y-only final state: clear deviation from coexistence benchmark"
            rows.append(row)

        return rows[:2]

    return []


def draw_dynamic_network_pair(
    row: pd.Series,
    output_path: Path,
    layout_seed: int,
):
    G_initial, G_final, initial_state, final_state = parse_dynamic_graph_row(row)

    # Same node positions in both panels, based on the union of initial/final edges.
    G_layout = nx.Graph()
    G_layout.add_nodes_from(G_initial.nodes())
    G_layout.add_edges_from(G_initial.edges())
    G_layout.add_edges_from(G_final.edges())
    pos = nx.spring_layout(G_layout, seed=layout_seed)

    X_COLOR = "tab:blue"
    Y_COLOR = "tab:orange"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, G, state, title in [
        (axes[0], G_initial, initial_state, "Initial network/state"),
        (axes[1], G_final, final_state, "Final network/state"),
    ]:
        nx.draw_networkx_edges(G, pos=pos, ax=ax, alpha=0.7)

        x_nodes = [node for node in G.nodes() if int(state[node]) == X]
        y_nodes = [node for node in G.nodes() if int(state[node]) == Y]

        nx.draw_networkx_nodes(
            G, pos=pos, ax=ax, nodelist=x_nodes,
            node_color=X_COLOR, node_size=500, node_shape="o",
        )
        nx.draw_networkx_nodes(
            G, pos=pos, ax=ax, nodelist=y_nodes,
            node_color=Y_COLOR, node_size=500, node_shape="o",
        )

        labels = {node: ("X" if int(state[node]) == X else "Y") for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos=pos, labels=labels, ax=ax,
            font_size=9, font_weight="bold", font_color="black",
        )

        ax.set_title(
            f"{title}\nX={sum(int(s) == X for s in state)}, "
            f"Y={sum(int(s) == Y for s in state)}"
        )
        ax.axis("off")

    fig.suptitle(
        f"{row['network']} | Case {int(row['case'])} | {row['condition']} | "
        f"p={row.get('p', np.nan):.2f} | run seed={int(row['run_seed'])}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_appendix_figures(
    df: pd.DataFrame,
    output_dir: Path,
    condition: str,
    layout_seed: int,
):
    figure_dir = output_dir / "appendix_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for network in ["ER", "BA"]:
        for case in [1, 2, 3, 4]:
            selected = select_dynamic_deviation_examples(
                df, network, case, condition
            )

            for i, row in enumerate(selected, start=1):
                filename = (
                    f"{network}_case{case}_{condition}_example{i}_"
                    f"seed{int(row['run_seed'])}.png"
                )
                draw_dynamic_network_pair(
                    row, figure_dir / filename, layout_seed
                )
                manifest.append({
                    "condition": condition,
                    "network": network,
                    "case": case,
                    "example_number": i,
                    "run_seed": int(row["run_seed"]),
                    "p": float(row["p"]),
                    "initial_density": float(row["initial_density"]),
                    "final_density": float(row["final_density"]),
                    "final_X_share": float(row["final_X_share"]),
                    "final_Y_share": float(row["final_Y_share"]),
                    "outcome_class": row["outcome_class"],
                    "selection_reason": row.get("selection_reason", ""),
                    "figure": filename,
                })

    pd.DataFrame(manifest).to_csv(
        output_dir / "appendix_graph_selection_manifest.csv",
        index=False,
    )

    return figure_dir, pd.DataFrame(manifest)


def experiment_conditions(
    rewiring_rates: List[float],
):
    conditions = []

    for rate in rewiring_rates:
        if rate < 0 or rate >= 1:
            raise ValueError(
                "Each rewiring rate must satisfy 0 <= rate < 1."
            )

        if rate == 0:
            label = "static_control_q0.00"
        else:
            label = (
                f"dynamic_q{rate:.2f}"
            )

        conditions.append(
            (
                label,
                rate,
                0.1,   # baseline payoff differential
                20,    # baseline N
                0.50,  # baseline initial share
                ALL_CASES,
            )
        )

    return conditions


def run_experiment(
    target_realizations: int,
    master_seed: int,
    rewiring_rates: List[float],
) -> pd.DataFrame:
    """
    Paired dynamic-network experiment.

    The same starting ER/BA network and the same initial allocation are
    reused across all rewiring-rate conditions. Thus q=0.00 is a direct
    static control and should reproduce the corresponding static baseline
    dynamics when the same master seed is used.
    """

    conditions = experiment_conditions(
        rewiring_rates
    )

    rows = []

    for network in [
        "ER",
        "BA",
    ]:
        accepted_realizations = 0
        attempt_number = 0

        while (
            accepted_realizations
            < target_realizations
        ):
            attempt_number += 1

            network_index = (
                0
                if network == "ER"
                else 1
            )

            # IMPORTANT:
            # This deliberately matches the baseline static-payoff seed
            # namespace, so the q=0 control can be checked against the
            # existing static baseline results.
            run_seed = (
                master_seed
                + network_index * 100_000
                + attempt_number
            )

            p, m1, m2, G, status = (
                draw_candidate(
                    network,
                    20,
                    run_seed,
                )
            )

            if status != "accepted":
                for (
                    condition,
                    rate,
                    payoff_delta,
                    n,
                    x_share,
                    cases,
                ) in conditions:
                    rows.append(
                        {
                            "experiment": "dynamic",
                            "condition": condition,
                            "network": network,
                            "N": n,
                            "initial_X_share": x_share,
                            "payoff_delta": payoff_delta,
                            "rewiring_rate": rate,
                            "attempt_number": attempt_number,
                            "run_seed": run_seed,
                            "accepted": False,
                            "accepted_realization_number": math.nan,
                            "case": math.nan,
                            "p": p,
                            "m1": (
                                m1
                                if m1 is not None
                                else math.nan
                            ),
                            "m2": (
                                m2
                                if m2 is not None
                                else math.nan
                            ),
                            "failure_reason": status,
                        }
                    )

                continue

            accepted_realizations += 1

            graph_stats = network_metrics(G)
            initial_state = make_initial_state(
                20,
                0.50,
                run_seed,
            )

            for (
                condition,
                rate,
                payoff_delta,
                n,
                x_share,
                cases,
            ) in conditions:

                for case in cases:

                    results = simulate_dynamic_case(
                        initial_graph=G,
                        initial_state=initial_state,
                        initial_metrics=graph_stats,
                        case=case,
                        n=n,
                        payoff_delta=payoff_delta,
                        periods=50,
                        run_seed=run_seed,
                        rewiring_rate=rate,
                    )

                    rows.append(
                        {
                            "experiment": "dynamic",
                            "condition": condition,
                            "network": network,
                            "N": n,
                            "initial_X_share": x_share,
                            "payoff_delta": payoff_delta,
                            "rewiring_rate": rate,
                            "attempt_number": attempt_number,
                            "run_seed": run_seed,
                            "accepted": True,
                            "accepted_realization_number": accepted_realizations,
                            "case": case,
                            "p": p,
                            "m1": (
                                m1
                                if m1 is not None
                                else math.nan
                            ),
                            "m2": (
                                m2
                                if m2 is not None
                                else math.nan
                            ),
                            "failure_reason": "",
                            **graph_stats,
                            **results,
                        }
                    )

    return pd.DataFrame(rows)


def add_ranges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["p_range"] = pd.cut(
        out["p"],
        bins=[
            -1e-9,
            0.30,
            0.80,
            1.00001,
        ],
        labels=[
            "0.00–0.30",
            "0.31–0.80",
            ">0.80",
        ],
        include_lowest=True,
        right=True,
    )

    return out


def build_summaries(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    accepted = df[
        df["accepted"] == True
    ].copy()

    group_cols = [
        "network",
        "case",
        "condition",
        "rewiring_rate",
    ]

    aggregate = (
        accepted.groupby(
            group_cols,
            observed=True,
            dropna=False,
            as_index=False,
        )
        .agg(
            total_case_runs=(
                "run_seed",
                "size",
            ),
            different_runs=(
                "deviates_from_analytical_share_benchmark",
                "sum",
            ),
            mean_final_X_share=(
                "final_X_share",
                "mean",
            ),
            sd_final_X_share=(
                "final_X_share",
                "std",
            ),
            mean_edge_turnover_rate=(
                "mean_edge_turnover_rate",
                "mean",
            ),
            mean_initial_degree_gini=(
                "initial_degree_gini",
                "mean",
            ),
            mean_final_degree_gini=(
                "final_degree_gini",
                "mean",
            ),
            mean_initial_diameter=(
            "initial_diameter",
            "mean",
            ),
            mean_final_diameter=(
                "final_diameter",
                "mean",
            ),
        )
    )

    aggregate["difference_rate_percent"] = (
        100.0
        * aggregate["different_runs"]
        / aggregate["total_case_runs"]
    ).round(1)

    aggregate["mean_final_X_share"] = (
        aggregate["mean_final_X_share"].round(4)
    )

    outcome_counts = (
        accepted.groupby(
            group_cols + [
                "outcome_class"
            ],
            observed=True,
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(
            columns={"size": "runs"}
        )
    )

    outcome_counts["percent"] = (
        100.0
        * outcome_counts["runs"]
        / outcome_counts.groupby(
            group_cols,
            observed=True,
        )["runs"].transform("sum")
    ).round(1)

    # Build a clean static-vs-dynamic comparison by run seed.
    q0 = accepted[
        accepted["rewiring_rate"] == 0.0
    ][
        [
            "network",
            "case",
            "run_seed",
            "final_X_share",
            "outcome_class",
            "deviates_from_analytical_share_benchmark",
        ]
    ].rename(
        columns={
            "final_X_share": "q0_final_X_share",
            "outcome_class": "q0_outcome_class",
            "deviates_from_analytical_share_benchmark":
                "q0_deviates",
        }
    )

    dynamic = accepted[
        accepted["rewiring_rate"] > 0.0
    ][
        [
            "network",
            "case",
            "run_seed",
            "condition",
            "rewiring_rate",
            "final_X_share",
            "outcome_class",
            "deviates_from_analytical_share_benchmark",
            "mean_edge_turnover_rate",
        ]
    ].rename(
        columns={
            "final_X_share": "dynamic_final_X_share",
            "outcome_class": "dynamic_outcome_class",
            "deviates_from_analytical_share_benchmark":
                "dynamic_deviates",
            "mean_edge_turnover_rate":
                "dynamic_mean_edge_turnover_rate",
        }
    )

    static_dynamic_comparison = dynamic.merge(
        q0,
        on=[
            "network",
            "case",
            "run_seed",
        ],
        how="left",
        validate="many_to_one",
    )

    static_dynamic_comparison[
        "change_in_final_X_share"
    ] = (
        static_dynamic_comparison[
            "dynamic_final_X_share"
        ]
        - static_dynamic_comparison[
            "q0_final_X_share"
        ]
    )

    return {
        "aggregate_case_results": aggregate,
        "outcomes": outcome_counts,
        "static_dynamic_comparison": static_dynamic_comparison,
        "accepted_results": accepted,
    }


def write_outputs(
    df: pd.DataFrame,
    output_dir: Path,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_csv = (
        output_dir
        / "dynamic_all_attempts.csv"
    )

    summary_xlsx = (
        output_dir
        / "dynamic_summary.xlsx"
    )

    df.to_csv(
        raw_csv,
        index=False,
    )

    summaries = build_summaries(
        df
    )

    with pd.ExcelWriter(
        summary_xlsx,
        engine="openpyxl",
    ) as writer:

        for name, frame in summaries.items():
            frame.to_excel(
                writer,
                sheet_name=name[:31],
                index=False,
            )

    return (
        raw_csv,
        summary_xlsx,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Controlled dynamic-network extension of the validated "
            "static migration-game model."
        )
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help=(
            "Accepted ER and BA starting networks per network type."
        ),
    )

    parser.add_argument(
        "--master-seed",
        type=int,
        default=20260908,
        help=(
            "Master seed. The default matches the baseline static "
            "payoff experiment so q=0 can be validated directly."
        ),
    )

    parser.add_argument(
        "--rewiring-rates",
        type=float,
        nargs="+",
        default=[0.0, 0.10],
        help=(
            "One or more per-period edge-turnover rates. "
            "The default runs the static control (q=0) and dynamic case (q=0.10). "
            "Example: --rewiring-rates 0.0 0.10 0.25"
        ),
    )

    parser.add_argument(
        "--out",
        default="dynamic_results",
        help="Output directory.",
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error(
            "--runs must be a positive integer."
        )

    if any(
        rate < 0 or rate >= 1
        for rate in args.rewiring_rates
    ):
        parser.error(
            "Each rewiring rate must satisfy 0 <= rate < 1."
        )

    rewiring_rates = []

    for rate in args.rewiring_rates:
        rate = float(rate)

        if rate not in rewiring_rates:
            rewiring_rates.append(rate)

    all_attempts = run_experiment(
        target_realizations=args.runs,
        master_seed=args.master_seed,
        rewiring_rates=rewiring_rates,
    )

    output_dir = Path(args.out)

    raw_csv, summary_xlsx = write_outputs(
        df=all_attempts,
        output_dir=output_dir,
    )

    figure_condition = None
    positive_rates = [r for r in rewiring_rates if r > 0]
    if positive_rates:
        figure_condition = f"dynamic_q{positive_rates[0]:.2f}"
    elif 0.0 in rewiring_rates:
        figure_condition = "static_control_q0.00"

    print(
    f"Appendix figure condition:       "
    f"{figure_condition if figure_condition is not None else 'None'}"
)
    print(
        "Appendix figures are generated for one rewiring condition only; "
        "all requested rewiring rates are retained in the numerical results."
    )

    if figure_condition is not None:
        figure_dir, manifest = generate_appendix_figures(
            df=all_attempts,
            output_dir=output_dir,
            condition=figure_condition,
            layout_seed=args.master_seed,
        )
        print(
            f"Appendix figures:                 {figure_dir}"
        )
        print(
            f"Appendix figures generated:       {len(manifest)}"
        )

    accepted_mask = (
        all_attempts["accepted"] == True
    )

    accepted_rows = all_attempts[
        accepted_mask
    ]

    candidate_attempts = len(
        all_attempts[
            [
                "experiment",
                "network",
                "attempt_number",
            ]
        ].drop_duplicates()
    )

    accepted_networks = len(
        accepted_rows[
            [
                "experiment",
                "network",
                "accepted_realization_number",
            ]
        ].drop_duplicates()
    )

    accepted_case_runs = len(
        accepted_rows
    )

    different_case_runs = int(
        accepted_rows[
            "deviates_from_analytical_share_benchmark"
        ].sum()
    )

    print()
    print(
        f"Candidate attempts:              "
        f"{candidate_attempts}"
    )
    print(
        f"Accepted starting networks:      "
        f"{accepted_networks}"
    )
    print(
        f"Rejected candidates:             "
        f"{candidate_attempts - accepted_networks}"
    )
    print(
        f"Accepted case-level simulations: "
        f"{accepted_case_runs}"
    )
    print(
        f"Different case-level runs:       "
        f"{different_case_runs}"
    )
    print(
        "Rewiring rates:                  "
        f"{rewiring_rates}"
    )

    print()
    print(
        f"Raw CSV:   {raw_csv}"
    )
    print(
        f"Summary:   {summary_xlsx}"
    )


if __name__ == "__main__":
    main()
