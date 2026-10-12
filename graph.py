"""Graphs for the report, drawn from the per-method CSVs that main.py writes to results/.

There is one plot_<method>() function per method. main.py calls plot() after a run, and the
file can also be run on its own to redraw from the saved CSVs without re-running anything:

    python graph.py            # every method
    python graph.py ivf pq     # only these methods

Every graph is one chart that answers one question, with one line per dataset.
IVF-PQ uses scatter plots: each point is a parameter setting, coloured by dataset.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to files; no window needed
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
FIG_DIR = Path("figures")

# Default k for plots that show one neighbour count.
REPORT_K = 10

# Each dataset keeps the same label, colour and marker in every graph.
DATASETS = {
    "sift": ("SIFT1M (128-d)", "#2a78d6", "o"),
    "gist": ("GIST1M (960-d)", "#eb6834", "s"),
    "glove": ("GloVe (50-d)", "#1baf7a", "^"),
}

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "axes.axisbelow": True, "legend.frameon": False,
})


# ---------------------------------------------------------------- shared helpers

def _number(value):
    """CSV cells are strings; turn them back into int/float where possible ('' -> None)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value or None
    return int(number) if number.is_integer() else number


def load_results(method, results_dir=RESULTS_DIR, k=REPORT_K):
    """Read numeric CSV rows at one k, or all k values when k=None."""
    path = Path(results_dir) / f"{method}.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(f)]
    return rows if k is None else [row for row in rows if row["k"] == k]


def add_speedup(rows, results_dir=RESULTS_DIR):
    """Add row["speedup"]: the method's QPS divided by linear scan's QPS on the same dataset.

    Returns False (and adds nothing) if linear scan has no saved result for one of the datasets.
    """
    baseline = {row["dataset"]: row["qps"] for row in load_results("linear_scan", results_dir)}
    if any(row["dataset"] not in baseline for row in rows):
        return False
    for row in rows:
        row["speedup"] = row["qps"] / baseline[row["dataset"]]
    return True


def line_chart(rows, x, y, xlabel, ylabel, title, path, logy=False, hline=None):
    """One line per dataset of row[y] against row[x]; x is drawn on a log-2 axis.

    hline  optional (value, label): a dashed horizontal reference line
    """
    fig, ax = plt.subplots(figsize=(7.5, 4), layout="constrained")
    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        label, colour, marker = DATASETS[dataset]
        points = sorted((row[x], row[y]) for row in rows if row["dataset"] == dataset)
        ax.plot(*zip(*points), color=colour, marker=marker, linewidth=2, markersize=7, label=label)
    if hline:
        ax.axhline(hline[0], color="#898781", linestyle="--", linewidth=1.2, label=hline[1])
    ticks = sorted({row[x] for row in rows})
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
    ax.minorticks_off()
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    fig.legend(loc="outside right upper")  # beside the chart, so it never covers a line
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return Path(path)


def plot_linear_scan(results_dir=RESULTS_DIR, fig_dir=FIG_DIR):
    """Grouped query-time bars for all saved k; one size/build bar per dataset."""
    rows = load_results("linear_scan", results_dir, k=None)
    if not rows:
        return []
    datasets = [dataset for dataset in DATASETS if any(r["dataset"] == dataset for r in rows)]
    ks = sorted({row["k"] for row in rows})
    by_dataset = {dataset: next(r for r in rows if r["dataset"] == dataset)
                  for dataset in datasets}
    labels = [f"{dataset.upper()} ({by_dataset[dataset]['dim']}-d)\n"
              f"n={by_dataset[dataset]['n_base']:,}" for dataset in datasets]
    saved = []
    for metric, ylabel, filename in [
        ("query_time_ms", "Query time (ms/query)", "query_time"),
        ("index_size_mb", "Index size (MB)", "index_size"),
        ("build_time_s", "Build time (s)", "build_time"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        if metric == "query_time_ms":
            width = 0.8 / len(ks)
            hatches = ["", "//", "xx", "..", "\\\\"]
            for j, k in enumerate(ks):
                points = [(i, row) for i, dataset in enumerate(datasets)
                          for row in rows if row["dataset"] == dataset and row["k"] == k]
                ax.bar([i + (j - (len(ks) - 1) / 2) * width for i, row in points],
                       [row[metric] for i, row in points], width=width,
                       color=[DATASETS[row["dataset"]][1] for i, row in points],
                       hatch=hatches[j % len(hatches)], edgecolor="#333333",
                       linewidth=0.5, label=f"k={k}")
            ax.legend(title="Neighbours")
        else:
            # Build time and size are repeated for each k in the benchmark CSV.
            points = [(i, by_dataset[d]) for i, d in enumerate(datasets)
                      if by_dataset[d].get(metric) is not None]
            ax.bar([i for i, row in points], [row[metric] for i, row in points],
                   color=[DATASETS[row["dataset"]][1] for i, row in points], width=0.65)
        ax.set_xticks(range(len(datasets)), labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Linear scan: {ylabel.split(' (')[0].lower()}",
                     loc="left", fontweight="bold")
        ax.set_ylim(bottom=0)
        path = Path(fig_dir) / "linear_scan" / f"linear_scan_{filename}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)
        saved.append(path)
    return saved


def plot_kd_tree(results_dir=RESULTS_DIR, fig_dir=FIG_DIR):
    raise NotImplementedError


def plot_ball_tree(results_dir=RESULTS_DIR, fig_dir=FIG_DIR):
    raise NotImplementedError


def annoy_speed_vs_accuracy(rows, path, trees=(10, 50, 200), target=0.95):
    """One panel per dataset: queries per second against recall, one line per n_trees.

    Each point is one search_k (labelled), so a line shows the speed/accuracy tradeoff of
    searching more candidates. The dashed line is the target recall; the circled point is the
    fastest setting that reaches it (none is circled if no setting does).
    """
    datasets = [d for d in DATASETS if any(row["dataset"] == d for row in rows)]
    markers = dict(zip(trees, "o^s"))
    shades = dict(zip(trees, (0.45, 0.7, 1.0)))   # more trees = darker line
    # search_k = -1 means "Annoy's default", which has no fixed value, so it is left out.
    rows = [row for row in rows if row["search_k"] > 0 and row["n_trees"] in trees]
    ticks = {500: "500", 5000: "5k", 50000: "50k"}  # only these points get a label

    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 4.4), sharey=True, layout="constrained")
    for ax, dataset in zip(axes, datasets):
        name, colour, _ = DATASETS[dataset]
        shown = [row for row in rows if row["dataset"] == dataset]
        for n in trees:
            points = sorted((row["recall"], row["qps"], row["search_k"])
                            for row in shown if row["n_trees"] == n)
            ax.plot([p[0] for p in points], [p[1] for p in points], color=colour, alpha=shades[n],
                    marker=markers[n], linewidth=2, markersize=7, label=f"n_trees = {n}")
            for recall, qps, search_k in points:
                if search_k in ticks:
                    ax.annotate(ticks[search_k], (recall, qps), textcoords="offset points",
                                xytext=(5, 5), fontsize=8, color="#555555")
        ax.axvline(target, color="#898781", linestyle="--", linewidth=1.2)
        reached = [row for row in shown if row["recall"] >= target]
        if reached:
            best = max(reached, key=lambda row: row["qps"])
            ax.scatter([best["recall"]], [best["qps"]], s=420, facecolors="none",
                       edgecolors="black", linewidths=1.6, zorder=5)
        else:
            ax.text(0.03, 0.04, f"no setting reaches recall {target}", transform=ax.transAxes,
                    fontsize=8, color="#555555")
        ax.set_yscale("log")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(f"Recall@{REPORT_K}")
        ax.set_title(name, loc="left", fontweight="bold")
        handles, labels = ax.get_legend_handles_labels()
        if reached:
            # A fixed-size handle; the circle drawn on the plot is too big for the legend.
            handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=11,
                                      markerfacecolor="none", markeredgecolor="black",
                                      markeredgewidth=1.6))
            labels.append(f"fastest at recall {target}")
        ax.legend(handles, labels, loc="upper right", fontsize=8)
    axes[0].set_ylabel("Queries per second")
    fig.suptitle(f"Annoy: speed against accuracy (k = {REPORT_K}; point labels are search_k, "
                 f"dashed line is recall {target})", fontweight="bold")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return Path(path)


def plot_annoy(results_dir=RESULTS_DIR, fig_dir=FIG_DIR):
    """Five charts: the speed/accuracy tradeoff, then what more trees and more candidates cost and buy."""
    rows = load_results("annoy", results_dir)
    if not rows:
        return []
    fig_dir = Path(fig_dir)
    # search_k = -1 means Annoy's default (n_trees * k); it has no fixed value, so the
    # n_trees charts use only those rows and the search_k charts leave them out.
    default_k = [row for row in rows if row["search_k"] == -1]
    # The search_k charts hold n_trees fixed at this value.
    fixed_trees = 50
    fixed = [row for row in rows if row["n_trees"] == fixed_trees and row["search_k"] > 0]
    return [
        annoy_speed_vs_accuracy(rows, fig_dir / "annoy_speed_vs_accuracy.png"),
        line_chart(default_k, "n_trees", "recall", "n_trees", f"Recall@{REPORT_K}",
                   "Annoy: recall vs number of trees",
                   fig_dir / "annoy_recall_vs_trees.png"),
        line_chart(default_k, "n_trees", "build_time_s", "n_trees", "Build time (s)",
                   "Annoy: build time vs number of trees",
                   fig_dir / "annoy_build_time_vs_trees.png", logy=True),
        line_chart(fixed, "search_k", "recall", "search_k (candidates checked)", f"Recall@{REPORT_K}",
                   f"Annoy: recall vs search_k (n_trees={fixed_trees})",
                   fig_dir / "annoy_recall_vs_search_k.png"),
        line_chart(fixed, "search_k", "qps", "search_k (candidates checked)", "Queries per second",
                   f"Annoy: QPS vs search_k (n_trees={fixed_trees})",
                   fig_dir / "annoy_qps_vs_search_k.png", logy=True),
    ]


def plot_ivf_pq(results_dir=RESULTS_DIR, fig_dir=FIG_DIR):
    """Plot the regular IVF-PQ grid at REPORT_K; every point is one setting.

    Scatter plots keep configurations with different nlist/m/nbits/nprobe
    separate instead of connecting them into an arbitrary parameter curve.
    """
    rows = load_results("ivf_pq", results_dir)
    if not rows:
        return []
    saved = []
    charts = [
        ("qps", "Queries per second (higher is better)", "recall_vs_qps", True),
        ("index_size_mb", "Index size (MB)", "recall_vs_size", False),
        ("build_time_s", "Build time (s)", "recall_vs_build_time", False),
    ]
    for x, xlabel, filename, logx in charts:
        fig, ax = plt.subplots(figsize=(7.5, 4), layout="constrained")
        for dataset in dict.fromkeys(row["dataset"] for row in rows):
            label, colour, marker = DATASETS[dataset]
            points = [row for row in rows if row["dataset"] == dataset
                      and isinstance(row.get(x), (int, float))
                      and isinstance(row.get("recall"), (int, float))
                      and (not logx or row[x] > 0)]
            if points:
                ax.scatter([row[x] for row in points], [row["recall"] for row in points],
                           color=colour, marker=marker, s=45, alpha=0.75, label=label)
        if logx:
            ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Recall@{REPORT_K}")
        ax.set_title("IVF-PQ: accuracy and " + {
            "qps": "search throughput", "index_size_mb": "storage",
            "build_time_s": "build cost",
        }[x], loc="left", fontweight="bold")
        ax.text(0.01, 0.02, "Each point = one parameter setting", transform=ax.transAxes,
                fontsize=8, color="#555555")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside right upper")
        path = Path(fig_dir) / "ivf_pq" / f"ivf_pq_{filename}_k{REPORT_K}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)
        saved.append(path)
    return saved


# Method name (as used by main.py and the CSV file names) -> its graph function.
GRAPHS = {
    "linear_scan": plot_linear_scan,
    "kd_tree": plot_kd_tree,
    "ball_tree": plot_ball_tree,
    "annoy": plot_annoy,
    "ivf_pq": plot_ivf_pq,
}


def plot(method, results_dir=RESULTS_DIR, fig_dir=FIG_DIR):
    """Draw every graph for one method and report what happened. Called by main.py."""
    try:
        saved = GRAPHS[method](results_dir, fig_dir)
    except NotImplementedError:
        print(f"{method}: no graphs written yet, skipped")
        return []
    if not saved:
        print(f"{method}: no results file in {results_dir}, skipped")
    for path in saved:
        print(f"{method}: saved {path}")
    return saved


if __name__ == "__main__":
    for name in sys.argv[1:] or GRAPHS:
        plot(name)
