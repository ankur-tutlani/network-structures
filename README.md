# Network Structures

This repository contains research code for studying how network structure shapes evolutionary-game outcomes. It simulates strategy evolution on Erdős–Rényi (ER) and Barabási–Albert (BA) networks, measures structural properties such as density, clustering, diameter, and degree heterogeneity, and analyzes how those properties relate to final strategy shares and outcomes.

The project also includes R scripts for examining classical evolutionary dynamics in two-strategy games. These provide a well-mixed theoretical reference for selected payoff matrices; they are currently separate from the Python network simulations rather than a single automated, head-to-head comparison pipeline.

## Research Questions

- How do ER and BA network structures relate to the strategies that persist or spread?
- How do payoff asymmetry, initial strategy shares, network size, and dynamic rewiring affect final outcomes?
- Are network-level properties such as degree heterogeneity or clustering associated with differences from well-mixed analytical benchmarks?
- How do classical evolutionary dynamics behave for the same kinds of two-strategy payoff matrices?

## Model Overview

The Python simulation workflows generate ER or BA graphs and assign strategies to nodes. Each agent's payoff is accumulated from interactions with its network neighbors. Strategy updates use a local perturbed-best-response rule: an agent compares its payoff with those in its local neighborhood and may adopt the strategy of a highest-payoff neighbor. Updates are performed over simulation time steps. This is a network-based stochastic process, distinct from the deterministic population-share dynamics in the R scripts.

The paired simulation workflow varies several factors:

- **Network family:** ER random graphs and BA preferential-attachment graphs.
- **Payoffs:** baseline and alternative payoff asymmetries across four game cases.
- **Initial composition:** X shares of 30%, 50%, or 70% in the initial-state experiment.
- **Population size:** N = 20, 50, or 100 in the size experiment.
- **Network structure:** graph metrics include realized density, clustering, diameter, connectedness, degree statistics, degree Gini, and power-law fit summaries where applicable.

The scripts save run-level data and summary tables so results can be grouped by network family, game case, condition, and structural characteristics. Check each simulation's docstring and `--help` output for the specific design and output schema; some specialized scripts use different settings.

## Evolutionary-Dynamics Reference Models

`Phasediagram.R` uses the `EvolutionaryGames` package to plot two-strategy phase diagrams for **Replicator**, **Brown-von Neumann-Nash (BNN)**, **Smith**, and **Logit** dynamics. It varies payoff matrices and, for Logit dynamics, the parameter `eta`.

`Jacobian.R` defines Replicator and Logit dynamics for a two-strategy payoff matrix, approximates the Jacobian near an equilibrium by finite differences, and reports its eigenvalues. This is a local stability calculation for the specified equilibrium.

These R analyses help interpret payoff-driven evolutionary tendencies under well-mixed population assumptions. They do not currently consume the Python simulation results or reproduce the network-local update rule; comparisons between the two are interpretive rather than a built-in matched experiment.

## Requirements

- Conda for the Python environment
- Python 3.11
- R 4.3.2 for the R scripts
- RStudio is optional and can be used to restore the R project library

## Setup

From the repository root, create and activate the Conda environment:

```sh
conda env create --file environment.yml
conda activate myenv
```

To restore the R packages, open this repository in RStudio and run this in the Console:

```r
renv::restore()
```

The R lockfile records R 4.3.2. Install that R version separately for the closest match when restoring the project library.

## Example Workflows

Run the paired payoff simulation from the repository root:

```sh
python reviewer_batch_v11_paired_payoff_initial_copy.py --experiment payoff --runs 10 --master-seed 20260908 --out reviewer_results
```

Post-process its generated CSV, keeping all experiment conditions:

```sh
python revised_robustness_postprocess_v7_copy.py --input reviewer_results/payoff_all_attempts.csv --outdir payoff_revised --seed 42 --all-conditions
```

Analyze network characteristics from the revised workbook:

```sh
python section1_network_characteristics_v4_copy.py --input payoff_revised/revised_summary.xlsx --experiment payoff --N 20 --outdir section1_network_characteristics
```

Each script can be run with `--help` to see its command-line options. Most analysis scripts require simulation CSV or Excel workbooks as inputs. Those datasets are not included in this repository.

## Repository Contents

- `reviewer_batch_v11_paired_payoff_initial_copy.py`: paired payoff and initial-state simulations, plus size and structure experiment modes.
- `dynamic_network_experiment_v5_copy.py`: dynamic-network experiment.
- `payoff_low_density_ER_targeted_copy.py`: targeted low-density ER simulations.
- `revised_robustness_postprocess_v7_copy.py`: classification, audit, and summary outputs from simulation CSV data.
- `section1_network_characteristics_v4_copy.py` through `section5_powerlaw_analysis_v2_copy.py`: analysis of revised simulation workbooks.
- `extract_*.py` and `payoff_graph_details_tables_v2.py`: focused data extraction and table-generation scripts.
- `Payofffunction.py` and `NetworkProperties.py`: shared Python helper modules.
- `Jacobian.R` and `Phasediagram.R`: R scripts for evolutionary dynamics and phase diagrams.
- `environment.yml` and `renv.lock`: Python and R dependency records.

## Dependencies and Outputs

`environment.yml` is the recommended Conda setup and pins the Python dependencies. `pyproject.toml` records the same Python dependency versions for Python tooling and package metadata.

Simulation and analysis scripts write CSV, Excel, and figure outputs to their selected output directories. Keep large datasets and generated results out of Git unless you specifically intend to publish them. The repository does not currently include an automated test suite.