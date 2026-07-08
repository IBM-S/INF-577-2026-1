# inf577-project

**Preference-Selective Activation in Pareto HyperNetworks for Multi-Objective Set Covering: A Case Study on Optimal Camera Placement**

A single Pareto HyperNetwork (PHN) generates the weights of a GraphSAGE conditioned on a continuous preference vector `r = (r_coverage, r_cost)`, learning the entire Pareto front of the bi-objective Optimal Camera Placement (OCP) problem with one model. Beyond measuring Pareto-front quality with the hypervolume indicator against exact AMPL+Gurobi baselines, this project investigates whether the hypernetwork exhibits **preference-selective activation** — i.e. whether distinct preferences engage structurally different internal pathways.

## AI declaration

For this project, generative AI tools were used only to assist with code generation. All generated code was reviewed and tested by the author. All architecture and design decisions were made and studied by the author, with contributions from academic colleagues.

## Setup

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Requires Python 3.10+, PyTorch 2.x and PyTorch Geometric 2.8.

## Datasets

Three sources of OCP instances are used. Raw sources go under `data/raw/`; running
`scripts/instances/build_instances.py` converts them into standardized AMPL `.dat`
instances (coverage radius `R` and pre-installed count baked in) under `data/instances/`.

**p-Hub Median benchmark** — 1,000 instances, sourced from https://github.com/aleksandra-gro/Dataset. Each instance merges 100 demand points and 50 candidate locations into a unified set of 150 points. Pre-installed cameras are drawn uniformly from {5, ..., 20} and the coverage radius `R` is sampled from (0.05, 0.20). Raw format: one folder per instance with the original coordinate CSVs.

Only 3 sample instances (`wsc_0`, `wsc_1`, `wsc_2`) are kept under version control at every
pipeline stage (`raw/`, `instances/`, `graphs/`, `baselines/`) — there is no point pushing all
1,000 instances (and their AMPL runs) to GitHub. Download the full benchmark from the source
repo above and regenerate the rest locally with `scripts/instances/build_instances.py`.

```
data/raw/phub/
    wsc_0/Input/coordinates_branches_0.csv, coordinates_hubs_0.csv, ...
    wsc_1/Input/...
    ...  (1000 instances)

data/instances/phub/
    wsc_0_R0p1455_pre13.dat
    ...  (1000 instances, AMPL-ready)
```

**Clustering basic benchmark** — 9 files with diverse 2D spatial distributions: s1–s4 (5,000 points each), a1 (3,000), a2 (5,250), a3 (7,500), dim2 (1,351), unbalance (6,500). Download from http://cs.uef.fi/sipu/datasets/

```
data/raw/clustering/
    s1.txt  s2.txt  s3.txt  s4.txt
    a1.txt  a2.txt  a3.txt
    dim2.txt  unbalance.txt

data/instances/clustering/
    s1_R28000_pre9.dat
    ...  (9 instances, AMPL-ready)
```

**CDMX (real-world)** — 16 Mexico City districts built from high-impact crime data, fixed coverage radius `R = 200 m`. Raw files are already in AMPL format (no conversion needed — `build_instances.py` just copies them into `data/instances/cdmx/`).

```
data/raw/cdmx/
    cam_11410_TLALPAN.dat
    cam_12319_COYOACAN.dat
    ...  (16 districts)
```

## Data partitioning

The test set exclusively contains unseen spatial distributions to rigorously evaluate out-of-distribution generalization.

| Split      | Source              | Instances   | Percentage |
|------------|---------------------|-------------|------------|
| Train      | p-Hub Median        | 800         | 78.05%     |
| Validation | p-Hub Median        | 100         | 9.76%      |
| Test       | p-Hub Median        | 100         | 9.76%      |
| Test       | Clustering + CDMX   | 25 (9+16)   | 2.44%      |
| **Total**  |                     | **1025**    | **100%**   |

## Pipeline

Points are converted to homogeneous proximity graphs (edge iff Euclidean distance < `R`). Each node carries 3 features: `[pre-installed camera flag, crime probability, coverage degree]`. Exact reference Pareto fronts are computed with AMPL+Gurobi; each front point provides a binary target vector, and the training label per preference is selected by the `r`-weighted scalarization.

Run the notebooks in order:

```
notebooks/
    01_pareto_filter.ipynb         Filter dominated points from exact AMPL fronts → data/baselines/clean/
    02_graph_construction.ipynb    Build per-instance proximity graphs → data/graphs/
    03_preference_matching.ipynb   ParetoDataset: ties graphs + preference vectors via dynamic matching
    04_model.ipynb                 PHN + GraphSAGE architecture
    05_train.ipynb                 Training loop → checkpoints_*/
    06_evaluate_hvi.ipynb          Hypervolume + preference sweep → data/results/
    07_xai.ipynb                   Preference-selective activation analysis → data/figures/07_xai/
```

## Architecture

```
r = (r_cov, r_cost) ──► HyperNetwork (MLP 2→100→100) ──► GraphSAGE weights W
                                                            │
graph (x, edge_index) ─────────────────────────────────►   │
                                                            ▼
                        SAGEConv₁ → BN → ReLU → Dropout
                        SAGEConv₂ → BN → ReLU → Dropout
                        Linear → Sigmoid  ──►  p̂ ∈ [0,1]ᴺ  (per-node prob.)
```

At evaluation, 101 uniformly spaced preferences `r^(k) = (k/100, 1−k/100)` are swept; per-node probabilities are thresholded at 0.5 to obtain binary placement decisions.

## Baselines and metrics

- **Pareto-front quality** — Hypervolume indicator (HV) against exact AMPL+Gurobi reference fronts.
- **Selective activation (XAI)** — per-neuron selectivity score `s_n = |a_n^(1,0) − a_n^(0,1)|` and layer-level cosine similarity between the two extreme preferences `r=(1,0)` and `r=(0,1)`, contrasted against an **untrained** hypernetwork as a strict control baseline.

## Project structure

Data is organized by pipeline stage; each stage keeps the same `phub` / `clustering` / `cdmx` split.

```
data/
    raw/                   original benchmark sources, untouched
        phub/  clustering/  cdmx/
    instances/             standardized AMPL .dat instances (R + n_pre baked in)
        phub/  clustering/  cdmx/
    graphs/                PyG Data objects (.pt) + split.json (train/val/test)
        phub/  clustering/  cdmx/
    baselines/             exact AMPL+Gurobi reference Pareto fronts
        raw/               per-instance run_1..run_N solver outputs
        clean/             filtered non-dominated fronts (objectives.npy, solutions.npy, pareto_clean.txt)
    figures/               diagnostic plots, one subfolder per notebook/script
    results/               final tables/figures for the report

scripts/
    instances/             build_instances.py (raw → data/instances/), gen_instance_lists.py
    plotting/               plot_instances.py, plot_experiments.py (instance visualization)

ampl/                      AMPL model (mo_location_model.mod, mo_drp_location.run),
                           batch runner (run_all_instances_ampl_camaras_inf_577.sh),
                           instance_lists.sh (generated instance ordering)
notebooks/                 01 → 07 pipeline (see above)
checkpoints_*/             trained model checkpoints, one folder per experiment (gitignored)
informe/                   paper/report source (gitignored)
requirements.txt
LICENSE
```

## Future Work

Some possible directions:

- Confirm preference-selective specialization and use it for **targeted pruning** of the hypernetwork to reduce the well-known PHN memory overhead.
- Extend beyond two objectives to higher-dimensional preference spaces.

## Author

Ignacio Muñoz-Sánchez — Universidad Técnica Federico Santa María. INF577 course project.
