# SC4020 Group 15 — Similarity Search

A benchmark of exact and approximate nearest-neighbour (k-NN) search methods on three
standard vector datasets. Every method is built, queried and scored through the same
interface, so their speed, memory use and accuracy can be compared directly.

## Methods

| Name (CLI)    | Method                              | Library        | Exact? | Parameters swept                  |
|---------------|-------------------------------------|----------------|--------|-----------------------------------|
| `linear_scan` | Brute-force search (baseline)       | NumPy          | Yes    | –                                 |
| `kd_tree`     | KD-tree                             | scikit-learn   | Yes    | `leaf_size`                       |
| `ball_tree`   | Ball tree                           | scikit-learn   | Yes    | `leaf_size`                       |
| `annoy`       | Random projection forest            | Annoy          | No     | `n_trees`, `search_k`             |
| `ivf`         | Inverted file (`IndexIVFFlat`)      | Faiss          | No     | `nlist`, `nprobe`                 |
| `pq`          | Product quantisation (`IndexPQ`)    | Faiss          | No     | `m`, `nbits`, symmetric/asymmetric |
| `ivf_pq`      | IVF + residual PQ (`IndexIVFPQ`)    | Faiss          | No     | `nlist`, `nprobe`, `m`, `nbits`   |

Each method's parameter grid is the `PARAM_GRID` at the top of its file in [methods/](methods/).

## Datasets

| Name    | Vectors                        | Dim | Queries | Distance                          |
|---------|--------------------------------|-----|---------|-----------------------------------|
| `sift`  | SIFT1M, 1,000,000 base         | 128 | 1,000   | Euclidean                         |
| `gist`  | GIST1M, 1,000,000 base         | 960 | 1,000   | Euclidean                         |
| `glove` | GloVe 6B, 399,000 base         | 50  | 1,000   | Euclidean on unit vectors (= cosine) |

The datasets are not committed. Download them and place them as follows:

```
datasets/
├── sift/   sift_base.fvecs  sift_query.fvecs  sift_groundtruth.ivecs  sift_learn.fvecs
├── gist/   gist_base.fvecs  gist_query.fvecs  gist_groundtruth.ivecs  gist_learn.fvecs
└── glove/  glove.6B.50d.txt
```

- SIFT1M and GIST1M: [TEXMEX corpus](http://corpus-texmex.irisa.fr/) (`sift.tar.gz`, `gist.tar.gz`)
- GloVe: [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) (`glove.6B.zip`, only the 50-d file is used)

GloVe has no query set or ground truth, so [data.py](data.py) holds out 1,000 random
vectors (fixed seed) as queries and computes the exact neighbours by brute force. The
parsed GloVe array is cached as `datasets/glove/glove.6B.50d.npy` after the first load.


## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run everything from the project root.

```bash
python main.py                                         # all methods, all datasets, k = 1 10 100
python main.py --methods ivf pq --datasets sift        # a subset
python main.py --k 10                                  # a single k
python main.py --max-base 100000 --max-queries 100     # small, fast run for debugging
python main.py --plot-only --methods ivf               # redraw graphs from saved CSVs only
python graph.py ivf pq                                 # same as above, without main.py
```

| Option          | Default        | Meaning                                    |
|-----------------|----------------|--------------------------------------------|
| `--methods`     | all            | Methods to run                             |
| `--datasets`    | all            | Datasets to run on                         |
| `--k`           | `1 10 100`     | Neighbours returned per query              |
| `--max-base`    | all            | Use only the first N base vectors          |
| `--max-queries` | `1000`         | Use only the first N queries               |
| `--out-dir`     | `results/`     | Where the CSVs are written                 |
| `--fig-dir`     | `figures/`     | Where the graphs are written               |
| `--plot-only`   | off            | Skip the runs and only redraw graphs       |
| `--skip-ablation` | off          | Skip the ablation study when IVF-PQ is selected |

Each index is built once per parameter setting and then searched once per `k`. Results
are written to `results/<method>.csv` after each method finishes. Existing rows are kept,
and a row is only replaced when the dataset, `k` and parameters all match, so new settings
can be added without re-running old ones.

### IVF-PQ ablation

[experiments/ivf_pq_ablation.py](experiments/ivf_pq_ablation.py) runs controlled
ablations around a reference IVF-PQ setting per dataset: component comparison (linear
scan vs IVF vs PQ vs IVF-PQ), single-parameter sweeps of `nprobe`, `nlist`, `m` and
`nbits`, exhaustive search (`nprobe = nlist`), and an `nlist` sweep with a constant
`nprobe/nlist` ratio.

```bash
python -m experiments.ivf_pq_ablation
python -m experiments.ivf_pq_ablation --datasets sift gist glove --k 10
python -m experiments.ivf_pq_ablation --experiments nprobe m --threads 1
python -m experiments.ivf_pq_ablation --plot-only --datasets sift gist glove
```

`main.py` runs the ablation study on all three datasets by default, using `k=1, 10, 100`. The standalone ablation script defaults to SIFT and `k=10`.
Use `--datasets gist glove` to select GIST and GloVe. 

Results go to `results/ivf_pq_ablation/<dataset>.csv` and graphs to
`figures/ivf_pq_ablation/`.

## Metrics

Every result row records the following, computed in [methods/base.py](methods/base.py):

| Column           | Meaning                                                                     |
|------------------|-----------------------------------------------------------------------------|
| `build_time_s`   | Time to build the index                                                     |
| `index_size_mb`  | Size of everything needed to answer queries, including stored vectors       |
| `query_time_ms`  | Average search time per query                                               |
| `qps`            | Queries per second                                                          |
| `recall`         | Recall@k: fraction of the true k nearest neighbours returned (order ignored) |
| `ndcg`           | NDCG@k: order-aware accuracy, 1.0 only for the right neighbours in the right order |
| `distance_ratio` | Distance to the k-th returned neighbour ÷ distance to the true k-th neighbour (1.0 is perfect) |

## Project structure

```
├── main.py                  # entry point: runs methods, saves CSVs, draws graphs
├── data.py                  # dataset loaders and brute-force ground truth
├── graph.py                 # one plot_<method>() per method, reads results/*.csv
├── methods/
│   ├── base.py              # BaseANN interface and the shared evaluate() + metrics
│   ├── linear_scan.py
│   ├── kd_tree.py
│   ├── ball_tree.py
│   ├── annoy_index.py
│   ├── ivf.py
│   ├── pq.py
│   └── ivf_pq.py
├── experiments/
│   └── ivf_pq_ablation.py   # controlled IVF-PQ ablation study
├── results/                 # one CSV per method
├── figures/                 # graphs, one sub-folder per method
└── requirements.txt
```
