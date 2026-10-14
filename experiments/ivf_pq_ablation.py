"""Controlled IVF-PQ ablations, independent of the regular benchmark grid."""
import argparse
import csv
from pathlib import Path
import time
import warnings

import faiss
import numpy as np

from data import load_dataset
from methods.base import recall_at_k, ndcg_at_k, distance_ratio
from methods.ivf import IVF
from methods.ivf_pq import IVFPQ, M_VALUES
from methods.linear_scan import LinearScan
from methods.pq import PQ

EXPERIMENTS = ("components", "nprobe", "nlist", "m", "nbits", "exhaustive", "nlist_ratio")
METRICS = {
    "recall": "Recall", "query_time_ms": "Query time (ms/query)",
    "index_size_mb": "Index size (MB)", "build_time_s": "Build time (s)",
}
REFERENCE = {
    "sift": dict(nlist=1024, m=64, nbits=8, nprobe=8),
    "gist": dict(nlist=1024, m=240, nbits=8, nprobe=64),
    "glove": dict(nlist=1024, m=25, nbits=8, nprobe=16),
}


def configurations(dataset, nlist=1024):
    """Return (experiment, displayed value, method class, params) tuples."""
    reference = {**REFERENCE[dataset], "nlist": nlist}
    reference["nprobe"] = min(reference["nprobe"], nlist)
    coarse = {key: reference[key] for key in ("nlist", "nprobe")}
    compression = {key: reference[key] for key in ("m", "nbits")}
    settings = [
        ("components", "Linear scan", LinearScan, {}),
        ("components", "IVF", IVF, coarse),
        ("components", "PQ", PQ, {**compression, "symmetric": False}),
        ("components", "IVF-PQ", IVFPQ, reference.copy()),
    ]
    sweeps = {
        "nprobe": [1, 4, 8, 16, 32, 64],
        "nlist": sorted({256, 1024, 4096, nlist}),
        "m": sorted({*M_VALUES[dataset], reference["m"]}),
        "nbits": [4, 6, 8],
    }
    for parameter, values in sweeps.items():
        for value in values:
            params = {**reference, parameter: value}
            if params["nprobe"] <= params["nlist"]:
                settings.append((parameter, value, IVFPQ, params))
    for cls, label, params in [(IVF, "IVF", coarse), (IVFPQ, "IVF-PQ", reference)]:
        settings.append(("exhaustive", label, cls, {**params, "nprobe": nlist}))
    # Companion to the fixed-nprobe nlist sweep: hold the searched fraction constant.
    for value in sweeps["nlist"]:
        settings.append(("nlist_ratio", value, IVFPQ,
                         {**reference, "nlist": value,
                          "nprobe": max(1, round(value * reference["nprobe"] / nlist))}))
    return settings


def measure(index, base, queries, truth, ks):
    """Build once; warm up and measure each k using the common accuracy metrics."""
    start = time.perf_counter()
    index.build(base)
    build_time = time.perf_counter() - start
    size = index.index_size()
    for k in ks:
        index.query(queries[:min(10, len(queries))], k)
        start = time.perf_counter()
        predicted = index.query(queries, k)
        elapsed = time.perf_counter() - start
        yield {
            "method": index.name, "k": k,
            "n_base": len(base), "n_queries": len(queries), "dim": base.shape[1],
            "build_time_s": build_time, "index_size_mb": size / 1e6,
            "query_time_ms": elapsed * 1000 / len(queries), "qps": len(queries) / elapsed,
            "recall": recall_at_k(predicted, truth, k),
            "ndcg": ndcg_at_k(predicted, truth, k),
            "distance_ratio": distance_ratio(predicted, truth, base, queries, k),
        }


def run_study(dataset, args):
    settings = [s for s in configurations(dataset, args.reference_nlist)
                if s[0] in args.experiments]
    base, queries, truth = load_dataset(dataset, args.max_base, args.max_queries)
    if not len(queries) or not all(1 <= k <= min(len(base), truth.shape[1]) for k in args.k):
        raise ValueError("Need nonempty queries and k within the base and ground-truth sizes")
    # Check the whole study before writing output or beginning expensive training.
    for _, _, _, params in settings:
        required = max(params.get("nlist", 1), 2 ** params.get("nbits", 0))
        if len(base) < required:
            raise ValueError(f"Need at least {required} base vectors for {params}; got {len(base)}")
    largest = max(max(p.get("nlist", 1), 2 ** p.get("nbits", 0))
                  for _, _, _, p in settings)
    if len(base) < 39 * largest:
        warnings.warn("Small training set relative to the number of centroids; "
                      "use this run for debugging and a larger base for the report.")
    path = args.out_dir / f"{dataset}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = None
        for experiment, value, cls, params in settings:
            print(f"{dataset}: {experiment}={value}", flush=True)
            for result in measure(cls(**params), base, queries, truth, args.k):
                row = {"dataset": dataset, "experiment": experiment, "value": value,
                       "threads": args.threads,
                       **{key: params.get(key, "") for key in ("nlist", "nprobe", "m", "nbits")},
                       **result}
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
    return path


def plot_results(path, fig_dir):
    """Plot the saved measurements for each configuration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    saved = []
    for experiment in dict.fromkeys(row["experiment"] for row in rows):
        for k in sorted({int(row["k"]) for row in rows}):
            selected = [r for r in rows if r["experiment"] == experiment and int(r["k"]) == k]
            values = list(dict.fromkeys(r["value"] for r in selected))
            categorical = experiment in ("components", "exhaustive")
            if not categorical:
                values.sort(key=float)
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
            for ax, (metric, label) in zip(axes.flat, METRICS.items()):
                samples = [np.array([float(r[metric]) for r in selected if r["value"] == v])
                           for v in values]
                means = [s.mean() for s in samples]
                if categorical:
                    ax.bar(values, means)
                else:
                    ax.plot([float(v) for v in values], means, marker="o")
                    ax.set_xticks([float(v) for v in values], values)
                ax.set_xlabel("nlist (constant nprobe/nlist ratio)" if experiment == "nlist_ratio"
                              else experiment)
                ax.set_ylabel(f"Recall@{k}" if metric == "recall" else label)
                ax.grid(axis="y", alpha=0.25)
                if metric == "recall":
                    ax.set_ylim(0, 1.05)
            fig.suptitle(f"{path.stem.upper()} — {experiment}, k={k}")
            target = fig_dir / f"{path.stem}_{experiment}_k{k}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=180)
            plt.close(fig)
            saved.append(target)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(M_VALUES), default=["sift"])
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--k", type=int, nargs="+", default=[10])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reference-nlist", type=int, default=1024)
    parser.add_argument("--max-base", type=int)
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=Path("results/ivf_pq_ablation"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures/ivf_pq_ablation"))
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if (args.threads < 1 or args.reference_nlist < 8
            or args.max_queries < 1 or (args.max_base is not None and args.max_base < 100)):
        parser.error("threads/max-queries must be positive; reference-nlist >= 8; max-base >= 100")
    faiss.omp_set_num_threads(args.threads)
    for dataset in args.datasets:
        path = args.out_dir / f"{dataset}.csv" if args.plot_only else run_study(dataset, args)
        print(f"Results: {path}")
        for saved in plot_results(path, args.fig_dir):
            print(f"Plot: {saved}")


if __name__ == "__main__":
    main()
