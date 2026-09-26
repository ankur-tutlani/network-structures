#!/usr/bin/env python3
"""
BA hub-strategy analysis for the baseline payoff experiment.

For each case (1-4) and each density band, this script:
1. Uses only accepted BA, N=20, baseline payoff runs.
2. Reconstructs the saved static network from edge_list_json.
3. Identifies all hub agents as the nodes tied at the maximum degree.
4. Uses the same hub set for the initial and final states because the network is static.
5. For each run, calculates the fraction of hubs using X and Y.
6. Sums those run-level fractions across the density band.

Therefore, for every density band:
    Initial X + Initial Y = N runs
    Final X + Final Y = N runs

The main tables contain:
    Density band
    N runs
    Mean distinct hub agents
    Initial strategy X
    Initial strategy Y
    Final strategy X
    Final strategy Y

Strategy columns are equivalent run counts. A run with two hubs, one X and
one Y, contributes 0.5 to X and 0.5 to Y.

No simulations are rerun.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import networkx as nx
import pandas as pd


DENSITY_BANDS = [
    "0.00–0.20",
    "0.20–0.40",
    "0.40–0.60",
    "0.60–0.80",
    "0.80–1.00",
]
DENSITY_BINS = [-1e-12, 0.20, 0.40, 0.60, 0.80, 1.000001]


def parse_json_list(value, field_name: str, run_seed) -> list:
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(f"Run {run_seed} does not contain {field_name}.")
    try:
        parsed = json.loads(value)
    except Exception as exc:
        raise ValueError(
            f"Run {run_seed} has invalid JSON in {field_name}: {exc}"
        ) from exc
    if not isinstance(parsed, list):
        raise ValueError(
            f"Run {run_seed} field {field_name} must contain a JSON list."
        )
    return parsed


def build_graph(edge_list_json, n: int, run_seed) -> nx.Graph:
    edges = parse_json_list(edge_list_json, "edge_list_json", run_seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    seen = set()
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError(f"Run {run_seed} contains invalid edge: {edge}")
        u, v = int(edge[0]), int(edge[1])
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(
                f"Run {run_seed} has edge outside 0..{n-1}: {(u, v)}"
            )
        if u == v:
            raise ValueError(f"Run {run_seed} contains self-loop: {(u, v)}")
        key = tuple(sorted((u, v)))
        if key in seen:
            raise ValueError(f"Run {run_seed} contains duplicate edge: {key}")
        seen.add(key)
        graph.add_edge(u, v)

    return graph


def parse_state(value, n: int, field_name: str, run_seed) -> list[int]:
    state = parse_json_list(value, field_name, run_seed)
    if len(state) != n:
        raise ValueError(
            f"Run {run_seed} {field_name} length {len(state)} != N={n}."
        )

    result = []
    for item in state:
        try:
            value_int = int(item)
        except Exception as exc:
            raise ValueError(
                f"Run {run_seed}: invalid strategy {item!r} in {field_name}."
            ) from exc
        if value_int not in (0, 1):
            raise ValueError(
                f"Run {run_seed}: strategy {value_int} outside {{0,1}}."
            )
        result.append(value_int)
    return result


def load_data(path: Path,experiment=None,condition=None,N=None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input workbook not found: {path}")

### temp fix for N100 issue json parsing
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_excel(path, sheet_name="corrected_results")

    required = [
        "experiment",
        "network",
        "N",
        "case",
        "condition",
        "run_seed",
        "density",
        "edge_list_json",
        "initial_state_json",
        "final_state_json",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # df = df[
    #     (df["experiment"] == "payoff")
    #     & (df["network"] == "BA")
    #     & (df["N"] == 20)
    #     & (df["condition"] == "baseline_0.9_1.1")
    # ].copy()

    df = df[
        (df["experiment"] == experiment)
        & (df["network"] == "BA")
        & (df["N"] == N)
        & (df["condition"] == condition)
    ].copy()

    if df.empty:
        raise ValueError(
            "No BA N=20 baseline payoff rows were found in corrected_results."
        )

    expected_cases = {1, 2, 3, 4}
    found_cases = set(df["case"].astype(int).unique())

    if found_cases != expected_cases:
        raise ValueError(
            f"Expected Cases 1-4 in the baseline BA results, "
            f"but found: {sorted(found_cases)}"
        )


    duplicate_mask = df.duplicated(
        # subset=["case", "run_seed"],
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
            duplicate_mask, ["case", "run_seed"]
        ].drop_duplicates()
        raise ValueError(
            "Duplicate BA baseline rows found for the same case/run_seed:\n"
            + bad.to_string(index=False)
        )

    df["density_band"] = pd.cut(
        df["density"],
        bins=DENSITY_BINS,
        labels=DENSITY_BANDS,
        include_lowest=True,
        right=True,
    )

    if df["density_band"].isna().any():
        bad = df.loc[
            df["density_band"].isna(),
            ["case", "run_seed", "density"],
        ]
        raise ValueError(
            "Some BA baseline runs have density values outside [0,1]:\n"
            + bad.to_string(index=False)
        )

    return df


def analyze_run(row: pd.Series) -> dict:
    n = int(row["N"])
    run_seed = row["run_seed"]

    graph = build_graph(row["edge_list_json"], n, run_seed)
    initial = parse_state(
        row["initial_state_json"], n, "initial_state_json", run_seed
    )
    final = parse_state(
        row["final_state_json"], n, "final_state_json", run_seed
    )

    degrees = dict(graph.degree())
    max_degree = max(degrees.values())
    hubs = [
        node for node, degree in degrees.items()
        if degree == max_degree
    ]
    if not hubs:
        raise ValueError(f"Run {run_seed} has no hub agents.")

    n_hubs = len(hubs)

    initial_x = sum(initial[node] == 0 for node in hubs)
    initial_y = sum(initial[node] == 1 for node in hubs)
    final_x = sum(final[node] == 0 for node in hubs)
    final_y = sum(final[node] == 1 for node in hubs)

    return {
        "case": int(row["case"]),
        "run_seed": run_seed,
        "density": float(row["density"]),
        "density_band": row["density_band"],
        "n_hubs": n_hubs,
        "max_degree": max_degree,
        "initial_x_fraction": initial_x / n_hubs,
        "initial_y_fraction": initial_y / n_hubs,
        "final_x_fraction": final_x / n_hubs,
        "final_y_fraction": final_y / n_hubs,
    }


def build_case_table(run_df: pd.DataFrame, case: int) -> pd.DataFrame:
    sub = run_df[run_df["case"] == case].copy()

    table = (
        sub.groupby("density_band", observed=True, as_index=False)
        .agg(
            **{
                "N runs": ("run_seed", "size"),
                # "Mean distinct hub agents": ("n_hubs", "mean"),
                "Mean number of tied maximum-degree hubs": ("n_hubs", "mean"),
                "Mean maximum degree": ("max_degree", "mean"),
                "Initial strategy X": ("initial_x_fraction", "sum"),
                "Initial strategy Y": ("initial_y_fraction", "sum"),
                "Final strategy X": ("final_x_fraction", "sum"),
                "Final strategy Y": ("final_y_fraction", "sum"),
            }
        )
    )

    table.insert(0, "Density band", table.pop("density_band"))

    initial_total = table["Initial strategy X"] + table["Initial strategy Y"]
    final_total = table["Final strategy X"] + table["Final strategy Y"]

    if (
        (initial_total.sub(table["N runs"]).abs() >= 1e-9).any()
        or (final_total.sub(table["N runs"]).abs() >= 1e-9).any()
    ):
        raise ValueError(
            f"Hub strategy counts do not reconcile to N runs for Case {case}."
        )

    return table


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BA hub strategies by density band for baseline payoff results."
    )
    parser.add_argument(
        "--input",
        default="payoff_revised/revised_summary.xlsx",
        help="Revised payoff workbook containing corrected_results.",
    )
    parser.add_argument(
        "--outdir",
        default="section2_BA_hub_strategy",
        help="Output directory.",
    )
    parser.add_argument(
                "--condition",
                default="baseline_0.9_1.1",
                help="",
            )
    parser.add_argument(
                "--N",
                type=int,
                default=20,
                help="",
            )
    parser.add_argument(
                "--experiment",
                default="payoff",
                help="",
            )
    args = parser.parse_args()

    input_path = Path(args.input)
    condition=args.condition
    N=args.N
    experiment=args.experiment
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path,experiment,condition,N)
    run_df = pd.DataFrame(
        [analyze_run(row) for _, row in df.iterrows()]
    )

    tables = {
        f"Case{case}": build_case_table(run_df, case)
        for case in [1, 2, 3, 4]
    }

    workbook_path = outdir / "section2_BA_hub_strategy.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
        run_df.to_excel(writer, sheet_name="run_level_audit", index=False)

    for name, table in tables.items():
        table.to_csv(
            outdir / f"{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    run_df.to_csv(
        outdir / "run_level_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Input: {input_path}")
    print(f"Output workbook: {workbook_path}")
    for name, table in tables.items():
        print(f"{name}: {len(table)} density-band rows")
    print(f"Audit rows: {len(run_df)}")


if __name__ == "__main__":
    main()
