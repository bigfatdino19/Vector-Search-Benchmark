"""Entry point: runs every similarity search method on every dataset and saves the results.

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

# Columns of the output CSV, in order. Must match the keys returned by evaluate() in methods/base.py.
FIELDS = ["dataset", "method", "params", "n_base", "n_queries", "dim", "k",
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
    # Where the results CSV is written.
    parser.add_argument("--out", type=Path, default=Path("results") / "results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []  # one dict per (dataset, method, parameter setting)

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
            rows.extend(results)

    # Save everything to one CSV (overwrites the previous file) for tables and plots in the report.
    # TODO: Write code to plot graphs later
    if rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
