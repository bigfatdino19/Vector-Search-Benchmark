"""Entry point: runs every similarity search method on every dataset and saves the results.

Each method gets its own file, results/<method>.csv, with one row per dataset and
parameter setting in that method's PARAM_GRID.

Usage (from the project root):
    python main.py                                  # all methods, all datasets
    python main.py --methods kd_tree --datasets sift
    python main.py --max-base 100000 --max-queries 100   # small, fast run for debugging
"""
import argparse
import csv
from pathlib import Path

from methods import annoy_index, ball_tree, ivf_pq, kd_tree, linear_scan

# Name on the command line -> the method file. Each file exposes run(), which
# loads the data itself (via data.py) and returns one result row per parameter setting.
METHODS = {
    "linear_scan": linear_scan,
    "kd_tree": kd_tree,
    "ball_tree": ball_tree,
    "annoy": annoy_index,
    "ivf_pq": ivf_pq,
}
# Dataset names understood by data.load_dataset().
DATASETS = ["sift", "gist", "glove"]

# Columns of each output CSV, in order. The method's own parameters (e.g. leaf_size) are
# inserted as separate columns between the two groups. Must match the keys returned by
# evaluate() in methods/base.py.
ID_FIELDS = ["dataset", "method"]
RESULT_FIELDS = ["n_base", "n_queries", "dim", "k",
                 "build_time_s", "index_size_mb", "query_time_ms", "qps", "recall", "ndcg",
                 "one_recall_at_k", "distance_ratio"]


def parse_args():
    """Read the command-line options. Every option has a default, so none are required."""
    parser = argparse.ArgumentParser(description="Compare similarity search methods.")
    # Which datasets / methods to run; default is all of them.
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    # k in k-nearest-neighbours: how many neighbours each method returns per query.
    parser.add_argument("--k", type=int, default=10, help="neighbours per query")
    # Optional limits for quick test runs; None means use everything.
    parser.add_argument("--max-base", type=int, default=None, help="use only the first N base vectors")
    parser.add_argument("--max-queries", type=int, default=1000, help="use only the first N queries")
    # Folder the per-method CSVs are written to.
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def save_results(method, dataset, results, out_dir):
    """Write one method's rows for one dataset into <out_dir>/<method>.csv.

    Rows already in the file for other datasets are kept, so datasets can be run one at a
    time; rows for `dataset` are replaced by the new ones.
    """
    path = out_dir / f"{method}.csv"
    kept = []
    if path.exists():
        with open(path, newline="") as f:
            kept = [row for row in csv.DictReader(f) if row["dataset"] != dataset]
    # Spread the params dict into one column per parameter so it is easy to filter and plot.
    new = [{**{key: value for key, value in row.items() if key != "params"}, **row["params"]}
           for row in results]
    rows = kept + new
    param_fields = list(dict.fromkeys(key for row in rows for key in row
                                      if key not in ID_FIELDS + RESULT_FIELDS))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ID_FIELDS + param_fields + RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    args = parse_args()
    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        for name in args.methods:
            try:
                # Hand over to the method file; it loads the data and evaluates its PARAM_GRID.
                results = METHODS[name].run(dataset, k=args.k, max_base=args.max_base,
                                            max_queries=args.max_queries)
            except NotImplementedError:
                # build()/query() are still stubs, so skip instead of crashing the whole run.
                print(f"{name:<60} not implemented, skipped")
                continue
            for row in results:
                size = "n/a" if row["index_size_mb"] is None else f"{row['index_size_mb']}MB"
                print(f"{row['method'] + ' ' + str(row['params']):<60} build={row['build_time_s']}s "
                      f"size={size} qps={row['qps']} recall@{args.k}={row['recall']} "
                      f"ndcg@{args.k}={row['ndcg']} 1-recall@{args.k}={row['one_recall_at_k']} "
                      f"dist_ratio={row['distance_ratio']}")
            # Saved straight away, so a crash in a later (slow) run does not lose these results.
            # TODO: Write code to plot graphs later
            path = save_results(name, dataset, results, args.out_dir)
            print(f"saved {len(results)} rows to {path}")


if __name__ == "__main__":
    main()
