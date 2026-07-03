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

Three sources of OCP instances are used. Place them under `data/` as shown below.

**p-Hub Median benchmark** — 1,000 training/validation/test instances. Each instance merges 100 demand points and 50 candidate locations into a unified set of 150 points. Pre-installed cameras are drawn uniformly from {5, ..., 20} and the coverage radius `R` is sampled from (0.05, 0.20).

```
data/phub_median/
    wsc_0.txt
    wsc_1.txt
    ...  (1000 instances)
```

**Clustering basic benchmark** — 9 files with diverse 2D spatial distributions: S1–S4 (5,000 points each), A1 (3,000), A2 (5,250), A3 (7,500), Dim-low (1,351), Unbalance (6,500). Download from http://cs.uef.fi/sipu/datasets/

```
data/clustering/
    S1.txt  S2.txt  S3.txt  S4.txt
    A1.txt  A2.txt  A3.txt
    Dim-low.txt  Unbalance.txt
```

**CDMX (real-world)** — 16 Mexico City districts built from high-impact crime data, fixed coverage radius `R = 200 m` (e.g. Tlalpan: 11,410 points / 177,295 edges / 804 pre-installed cameras; Coyoacan: 12,319 points / 266,498 edges / 973 pre-installed cameras).

```
data/cdmx/
    tlalpan.txt
    coyoacan.txt
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
    02_pareto_filter.ipynb        Filter dominated points from exact fronts
    03_graph_construction.ipynb   Build per-instance proximity graphs
    04_dataset.ipynb              PyG dataset + train/val/test splits (split.json)
    05_model.ipynb                PHN + GraphSAGE architecture
    06_train.ipynb                Training loop
    07_evaluate.ipynb             Hypervolume + 101-point preference sweep
    08_xai.ipynb                  Preference-selective activation analysis
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

```
data/
    phub_median/                  raw p-Hub Median instances
    clustering/                   raw clustering benchmark files
    cdmx/                         raw CDMX district instances
graphs/                           processed PyG graphs + split.json
frentes_pareto_resultados/        exact reference Pareto fronts (AMPL+Gurobi)
checkpoints/                      trained models
images/                           generated figures
notebooks/                        02 → 08 pipeline (see above)
requirements.txt
LICENSE
```

## Future Work

Some possible directions:

- Confirm preference-selective specialization and use it for **targeted pruning** of the hypernetwork to reduce the well-known PHN memory overhead.
- Extend beyond two objectives to higher-dimensional preference spaces.

## Author

Ignacio Muñoz-Sánchez — Universidad Técnica Federico Santa María. INF577 course project.
