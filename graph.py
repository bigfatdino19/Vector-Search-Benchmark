"""Graphs for the report, drawn from the per-method CSVs that main.py writes to results/.

There is one plot_<method>() function per method. main.py calls plot() after a run, and the
file can also be run on its own to redraw from the saved CSVs without re-running anything:

    python graph.py            # every method
    python graph.py ivf pq     # only these methods

Every graph is one chart that answers one question, with one line per dataset.
The two exact trees share a comparison chart per measure, one panel per dataset.
IVF-PQ uses scatter plots: each point is a parameter setting, coloured by dataset.
"""
import copy
import csv
import sys
from pathlib import Path

import matplotlib
import matplotlib.colors
import matplotlib.patches

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


def line_chart(rows, x, y, xlabel, ylabel, title, path, logy=False, hline=None, ylim=None):
    """One line per dataset of row[y] against row[x]; x is drawn on a log-2 axis.

    hline     optional (value, label): a dashed horizontal reference line
    ylim      optional (low, high): a fixed y range instead of the automatic one
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
    if ylim:
        ax.set_ylim(*ylim)
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


# ---------------------------------------------------------------- KD-tree and ball tree

TREE_LABELS = {"kd_tree": "KD-tree", "ball_tree": "Ball tree"}
TREE_XLABEL = "leaf_size (largest number of vectors in one leaf)"
# Both trees are exact, so every leaf_size reaches full recall (up to distance ties) and the
# best setting is simply the fastest one. The floor only guards against a damaged result.
TREE_MIN_RECALL = 0.95
# Line shade and marker of each tree when both are drawn in one panel (tree_comparison_chart).
TREE_STYLES = {"kd_tree": (1.0, "o"), "ball_tree": (0.5, "^")}


def _tree_rows(method, results_dir):
    """Rows of one tree method at REPORT_K, in dataset order, with an index_overhead column.

    index_overhead  index size divided by the size of the raw float32 vectors (n_base * dim * 4
                    bytes), so the three datasets share one axis. 1.0 means the index is exactly
                    as big as the data; the excess is the tree's node and bound arrays.
    """
    rows = load_results(method, results_dir)
    rows.sort(key=lambda row: list(DATASETS).index(row["dataset"]))
    for row in rows:
        row["index_overhead"] = row["index_size_mb"] / (row["n_base"] * row["dim"] * 4 / 1e6)
    return rows


def tree_comparison_chart(rows_by_method, metric, ylabel, title, path, hline=None):
    """row[metric] against leaf_size for both trees, one panel per dataset.

    rows_by_method  {method: rows}; the KD-tree is drawn in the dataset's full colour and the
                    ball tree in a lighter shade, with its own marker (TREE_STYLES)
    Each panel has its own y axis, because the datasets differ by a factor of 50 or more.
    """
    datasets = [dataset for dataset in DATASETS
                if any(row["dataset"] == dataset for rows in rows_by_method.values() for row in rows)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4.2),
                             layout="constrained", squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        label, colour, _ = DATASETS[dataset]
        for method, rows in rows_by_method.items():
            weight, marker = TREE_STYLES[method]
            points = sorted((row["leaf_size"], row[metric]) for row in rows if row["dataset"] == dataset)
            ax.plot(*zip(*points), color=_shade(colour, weight), marker=marker, linewidth=2,
                    markersize=7, label=TREE_LABELS[method])
        if hline:
            ax.axhline(hline[0], color="#898781", linestyle="--", linewidth=1.2, label=hline[1])
        ticks = sorted({row["leaf_size"] for rows in rows_by_method.values() for row in rows})
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
        ax.minorticks_off()
        ax.set_xlabel("leaf_size")
        ax.set_title(label, loc="left", fontweight="bold")
    # One legend beside the panels, so it never covers a line. The colour differs per panel,
    # so the legend shows the first panel's entries in grey (as in pq_param_chart).
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles = [copy.copy(handle) for handle in handles]
    greys = [_shade("#3d3c39", TREE_STYLES[method][0]) for method in rows_by_method] + ["#898781"]
    for handle, grey in zip(handles, greys):
        handle.set_color(grey)
        handle.set_markeredgecolor(grey)
    axes[0][0].set_ylabel(ylabel)
    fig.legend(handles, labels, loc="outside right upper", fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return Path(path)


def _plot_tree(method, results_dir, fig_dir):
    """One tree method: every measure against leaf_size, then its fastest setting per dataset."""
    rows = _tree_rows(method, results_dir)
    if not rows:
        return []
    has_speedup = add_speedup(rows, results_dir)
    label = TREE_LABELS[method]
    fig_dir = Path(fig_dir) / method  # each tree gets its own sub-folder
    saved = [
        line_chart(rows, "leaf_size", "qps", TREE_XLABEL, "Queries per second",
                   f"{label}: QPS against leaf_size (k = {REPORT_K})",
                   fig_dir / f"{method}_qps_vs_leaf_size.png", logy=True),
        line_chart(rows, "leaf_size", "build_time_s", TREE_XLABEL, "Build time (s)",
                   f"{label}: build time against leaf_size",
                   fig_dir / f"{method}_build_time_vs_leaf_size.png", logy=True),
        line_chart(rows, "leaf_size", "index_overhead", TREE_XLABEL, "Index size / raw data size",
                   f"{label}: index overhead against leaf_size",
                   fig_dir / f"{method}_index_overhead_vs_leaf_size.png", hline=(1.0, "raw data")),
        # Exact search: the lines sit at 1.0, so the y range is fixed to keep tie noise flat.
        line_chart(rows, "leaf_size", "recall", TREE_XLABEL, f"Recall@{REPORT_K}",
                   f"{label}: recall@{REPORT_K} against leaf_size",
                   fig_dir / f"{method}_recall_vs_leaf_size.png", hline=(1.0, "exact"), ylim=(0.95, 1.01)),
    ]
    if has_speedup:
        saved.append(line_chart(
            rows, "leaf_size", "speedup", TREE_XLABEL, "Speed-up over linear scan (x)",
            f"{label}: speed-up over linear scan (k = {REPORT_K})",
            fig_dir / f"{method}_speedup_vs_leaf_size.png", logy=True, hline=(1.0, "linear scan")))
    best = best_rows(rows, TREE_MIN_RECALL)
    saved.append(best_chart(best, has_speedup, fig_dir / f"{method}_best.png", ("leaf_size",),
                            f"{label}: fastest leaf_size per dataset (k = {REPORT_K})"))
    return saved


def _plot_tree_comparison(results_dir, fig_dir):
    """KD-tree against ball tree on every dataset; drawn only when both result files exist."""
    rows_by_method = {method: _tree_rows(method, results_dir) for method in TREE_LABELS}
    if not all(rows_by_method.values()):
        return []
    fig_dir = Path(fig_dir) / "trees"  # shared by both trees
    return [
        tree_comparison_chart(rows_by_method, "qps", "Queries per second",
                              f"KD-tree against ball tree: QPS (k = {REPORT_K})",
                              fig_dir / "trees_qps_vs_leaf_size.png"),
        tree_comparison_chart(rows_by_method, "build_time_s", "Build time (s)",
                              "KD-tree against ball tree: build time",
                              fig_dir / "trees_build_time_vs_leaf_size.png"),
        tree_comparison_chart(rows_by_method, "index_overhead", "Index size / raw data size",
                              "KD-tree against ball tree: index overhead",
                              fig_dir / "trees_index_overhead_vs_leaf_size.png", hline=(1.0, "raw data")),
    ]


def plot_kd_tree(results_dir=RESULTS_DIR, fig_dir=FIG_DIR):
    """KD-tree graphs, plus the comparison with the ball tree (either tree keeps it current)."""
    return _plot_tree("kd_tree", results_dir, fig_dir) + _plot_tree_comparison(results_dir, fig_dir)


def plot_ball_tree(results_dir=RESULTS_DIR, fig_dir=FIG_DIR):
    """Ball tree graphs, plus the comparison with the KD-tree (either tree keeps it current)."""
    return _plot_tree("ball_tree", results_dir, fig_dir) + _plot_tree_comparison(results_dir, fig_dir)


def plot_annoy(results_dir=RESULTS_DIR, fig_dir=FIG_DIR):
    """Four charts: what more trees cost and buy, and what searching more candidates costs and buys."""
    rows = load_results("annoy", results_dir)
    if not rows:
        return []
    fig_dir = Path(fig_dir) / "annoy"
    # search_k = -1 means Annoy's default (n_trees * k); it has no fixed value, so the
    # n_trees charts use only those rows and the search_k charts leave them out.
    default_k = [row for row in rows if row["search_k"] == -1]
    # The search_k charts hold n_trees fixed at this value.
    fixed_trees = 50
    fixed = [row for row in rows if row["n_trees"] == fixed_trees and row["search_k"] > 0]
    fixed_text = f"n_trees = {fixed_trees}, k = {REPORT_K}"
    return [
        line_chart(fixed, "search_k", "qps", "search_k (candidates checked per query)",
                   "Queries per second", f"Annoy: QPS against search_k ({fixed_text})",
                   fig_dir / "annoy_qps_vs_search_k.png", logy=True),
        line_chart(fixed, "search_k", "recall", "search_k (candidates checked per query)",
                   f"Recall@{REPORT_K}", f"Annoy: recall against search_k ({fixed_text})",
                   fig_dir / "annoy_recall_vs_search_k.png"),
        line_chart(default_k, "n_trees", "recall", "n_trees", f"Recall@{REPORT_K}",
                   f"Annoy: recall against n_trees (default search_k, k = {REPORT_K})",
                   fig_dir / "annoy_recall_vs_trees.png"),
        line_chart(default_k, "n_trees", "build_time_s", "n_trees", "Build time (s)",
                   f"Annoy: build time against n_trees (k = {REPORT_K})",
                   fig_dir / "annoy_build_time_vs_trees.png", logy=True),
    ]


# ---------------------------------------------------------------- IVF

# One parameter is varied in each IVF line chart while the other stays at this value.
IVF_FIXED = {"nlist": 1024, "nprobe": 8}
IVF_PARAM_LABELS = {"nlist": "nlist (number of clusters)", "nprobe": "nprobe (clusters searched per query)"}
# The best setting per dataset is the fastest one that reaches this recall.
IVF_MIN_RECALL = 0.95

# Result column -> (axis label, title text, log y-axis, dashed reference line or None).
IVF_METRICS = {
    "build_time_s": ("Build time (s)", "build time", True, None),
    "query_time_ms": ("Query time (ms per query)", "query time", True, None),
    "qps": ("Queries per second", "QPS", True, None),
    "recall": (f"Recall@{REPORT_K}", f"recall@{REPORT_K}", False, (1.0, "exact")),
    "ndcg": (f"NDCG@{REPORT_K}", f"NDCG@{REPORT_K}", False, (1.0, "exact")),
    "distance_ratio": (f"Distance ratio@{REPORT_K}", f"distance ratio@{REPORT_K}", False, (1.0, "exact")),
}


def best_rows(rows, min_recall):
    """The best row per dataset: the highest QPS among settings with recall >= min_recall.

    If no setting on a dataset reaches min_recall, its most accurate setting is used instead.
    """
    best = {}
    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        mine = [row for row in rows if row["dataset"] == dataset]
        good = [row for row in mine if row["recall"] >= min_recall]
        best[dataset] = (max(good, key=lambda row: row["qps"]) if good
                         else max(mine, key=lambda row: row["recall"]))
    return best


def _shade(colour, weight):
    """Mix a colour with white: weight 1 keeps the colour, smaller values are lighter."""
    rgb = matplotlib.colors.to_rgb(colour)
    return tuple(1 - weight * (1 - channel) for channel in rgb)


def tradeoff_chart(rows, best, path, series, point, title, y="qps", ylabel="Queries per second",
                   min_recall=None, legend_loc="lower left"):
    """Recall against row[y], one panel per dataset and one line per value of row[series].

    Each point is labelled with its row[point]; the row in best[dataset] is circled.
    min_recall  optional recall target, drawn as a dashed vertical line
    legend_loc  corner of each panel that the lines leave empty
    """
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    values = sorted({row[series] for row in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4.2), sharey=True,
                             layout="constrained", squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        label, colour, _ = DATASETS[dataset]
        for i, (value, marker) in enumerate(zip(values, "o^sDvP")):
            shade = _shade(colour, 0.45 + 0.55 * i / max(1, len(values) - 1))
            points = sorted((row[point], row["recall"], row[y]) for row in rows
                            if row["dataset"] == dataset and row[series] == value)
            ax.plot([p[1] for p in points], [p[2] for p in points], color=shade, marker=marker,
                    linewidth=2, markersize=7, label=f"{series} = {value}")
            for name, recall, height in points:
                ax.annotate(f"{name}", (recall, height), textcoords="offset points", xytext=(7, 6),
                            fontsize=8, color="#52514e")
        chosen = best[dataset]
        ax.plot(chosen["recall"], chosen[y], marker="o", markersize=16, markerfacecolor="none",
                markeredgecolor="#0b0b0b", markeredgewidth=1.5, linestyle="none", label="best setting")
        if min_recall is not None:
            ax.axvline(min_recall, color="#898781", linestyle="--", linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(f"Recall@{REPORT_K}")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.legend(loc=legend_loc, fontsize=8)
    axes[0][0].set_ylabel(ylabel)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return Path(path)


def _count_label(value):
    """Bar label for a count such as QPS: no decimals from 100 up, one decimal below (1.6, not 2)."""
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}"


def best_chart(best, has_speedup, path, params, title):
    """Bar charts comparing each dataset's best setting, one panel per measure.

    params  the parameter columns named under each bar, e.g. ("nlist", "nprobe")
    """
    panels = [("qps", "Queries per second", _count_label), ("recall", f"Recall@{REPORT_K}", "{:.3f}"),
              ("build_time_s", "Build time (s)", "{:.1f}"), ("index_size_mb", "Index size (MB)", "{:,.0f}")]
    if has_speedup:
        panels.insert(1, ("speedup", "Speed-up over linear scan (x)", "{:.3g}x"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.8), layout="constrained")
    names = ["\n".join([DATASETS[dataset][0].split(" (")[0]] + [f"{param} {row[param]}" for param in params])
             for dataset, row in best.items()]
    colours = [DATASETS[dataset][1] for dataset in best]
    for ax, (key, ylabel, fmt) in zip(axes, panels):
        bars = ax.bar(names, [row[key] for row in best.values()], color=colours, width=0.6)
        ax.bar_label(bars, labels=[fmt(row[key]) if callable(fmt) else fmt.format(row[key])
                                   for row in best.values()], padding=2, fontsize=9)
        ax.set_title(ylabel, loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=8)
        ax.margins(y=0.12)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return Path(path)


def plot_ivf(results_dir=RESULTS_DIR, fig_dir=FIG_DIR):
    """IVF graphs: every measure against nprobe and against nlist, the trade-off, and the best settings."""
    rows = load_results("ivf", results_dir)
    if not rows:
        return []
    has_speedup = add_speedup(rows, results_dir)
    rows.sort(key=lambda row: list(DATASETS).index(row["dataset"]))  # same dataset order in every graph
    fig_dir = Path(fig_dir) / "ivf"  # IVF graphs get their own sub-folder
    saved = []
    # Vary one parameter and hold the other at its IVF_FIXED value.
    for varied, fixed in (("nprobe", "nlist"), ("nlist", "nprobe")):
        subset = [row for row in rows if row[fixed] == IVF_FIXED[fixed]]
        for metric, (ylabel, name, logy, hline) in IVF_METRICS.items():
            # k is already in the name of the accuracy measures; a longer title would run into the legend.
            k_note = "" if "@" in name else f", k = {REPORT_K}"
            saved.append(line_chart(
                subset, varied, metric, IVF_PARAM_LABELS[varied], ylabel,
                f"IVF: {name} against {varied} ({fixed} = {IVF_FIXED[fixed]}{k_note})",
                fig_dir / f"ivf_{metric}_vs_{varied}.png", logy=logy, hline=hline))
    best = best_rows(rows, IVF_MIN_RECALL)
    saved.append(tradeoff_chart(
        rows, best, fig_dir / "ivf_tradeoff.png", "nlist", "nprobe",
        f"IVF: speed against accuracy (k = {REPORT_K}; point labels are nprobe, "
        f"dashed line is recall {IVF_MIN_RECALL})", min_recall=IVF_MIN_RECALL))
    saved.append(best_chart(
        best, has_speedup, fig_dir / "ivf_best.png", ("nlist", "nprobe"),
        f"IVF: best setting per dataset (fastest with recall@{REPORT_K} >= {IVF_MIN_RECALL})"))
    return saved


# ---------------------------------------------------------------- PQ

# The recommended (m, nbits) per dataset, with asymmetric search. PQ has no single best setting:
# a longer code is more accurate but slower and larger, so this is a judgement, explained in the report.
PQ_RECOMMENDED = {"sift": (32, 8), "gist": (240, 8), "glove": (25, 8)}

# Same measures as IVF, plus index size: the code length (m * nbits) changes it, unlike IVF's parameters.
PQ_METRICS = {**IVF_METRICS, "index_size_mb": ("Index size (MB)", "index size", True, None)}
# Measures where linear scan's value on the same dataset is drawn as a reference line.
PQ_BASELINE_METRICS = ("query_time_ms", "qps", "index_size_mb")


def pq_param_chart(rows, metric, ylabel, title, path, logy=False, exact=False, baseline=None):
    """row[metric] against m, one panel per dataset (their m values differ) and one line per nbits.

    Asymmetric search only; symmetric search has its own graph (pq_symmetric_chart).
    exact     draw a dashed line at 1.0, the value an exact search would give
    baseline  optional {dataset: value}: linear scan's value, drawn as a dashed line
    """
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4.2), sharey=True,
                             layout="constrained", squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        label, colour, _ = DATASETS[dataset]
        mine = [row for row in rows if row["dataset"] == dataset]
        nbits_values = sorted({row["nbits"] for row in mine})
        for i, (nbits, marker) in enumerate(zip(nbits_values, "o^sDvP")):
            shade = _shade(colour, 0.45 + 0.55 * i / max(1, len(nbits_values) - 1))
            points = sorted((row["m"], row[metric]) for row in mine
                            if row["nbits"] == nbits and not row["symmetric"])
            ax.plot(*zip(*points), color=shade, marker=marker, linewidth=2, markersize=7,
                    label=f"nbits = {nbits}")
        if exact:
            ax.axhline(1.0, color="#898781", linestyle="--", linewidth=1.2, label="exact")
        elif baseline and dataset in baseline:
            ax.axhline(baseline[dataset], color="#898781", linestyle="--", linewidth=1.2, label="linear scan")
        ticks = sorted({row["m"] for row in mine})
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
        ax.minorticks_off()
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("m (sub-vectors per vector)")
        ax.set_title(label, loc="left", fontweight="bold")
    # One legend beside the panels, so it never covers a line. The colour differs per panel,
    # so the legend shows the first panel's entries in grey.
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles = [copy.copy(handle) for handle in handles]
    n_lines = sum(label.startswith("nbits") for label in labels)
    greys = [_shade("#3d3c39", 0.45 + 0.55 * i / max(1, n_lines - 1)) for i in range(n_lines)]
    for handle, grey in zip(handles, greys + ["#898781"]):
        handle.set_color(grey)
        handle.set_markeredgecolor(grey)
    axes[0][0].set_ylabel(ylabel)
    fig.legend(handles, labels, loc="outside right upper", fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return Path(path)


def pq_symmetric_chart(rows, path):
    """Accuracy of asymmetric against symmetric search, at the one setting run both ways per dataset.

    One panel per accuracy measure, with a pair of bars per dataset.
    """
    pairs = {}  # dataset -> (asymmetric row, symmetric row) at the same m and nbits
    for sym in rows:
        if sym["symmetric"]:
            asym = next(row for row in rows if not row["symmetric"] and
                        (row["dataset"], row["m"], row["nbits"]) == (sym["dataset"], sym["m"], sym["nbits"]))
            pairs[sym["dataset"]] = (asym, sym)
    panels = [("recall", f"Recall@{REPORT_K}", "{:.3f}"), ("ndcg", f"NDCG@{REPORT_K}", "{:.3f}"),
              ("distance_ratio", f"Distance ratio@{REPORT_K}", "{:.3f}")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4), layout="constrained")
    names = [f"{DATASETS[dataset][0].split(' (')[0]}\nm {asym['m']}, nbits {asym['nbits']}"
             for dataset, (asym, _) in pairs.items()]
    colours = [DATASETS[dataset][1] for dataset in pairs]
    width = 0.38
    for ax, (key, ylabel, fmt) in zip(axes, panels):
        for i, (search, weight) in enumerate((("asymmetric", 1.0), ("symmetric", 0.45))):
            values = [pair[i][key] for pair in pairs.values()]
            positions = [x + (i - 0.5) * (width + 0.03) for x in range(len(pairs))]
            bars = ax.bar(positions, values, width=width, color=[_shade(colour, weight) for colour in colours])
            ax.bar_label(bars, labels=[fmt.format(value) for value in values], padding=2, fontsize=8)
        if key == "distance_ratio":
            ax.set_ylim(bottom=1.0)  # 1.0 is a perfect result, so the bars show the excess over it
        ax.set_xticks(range(len(pairs)), names)
        ax.set_title(ylabel, loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=8)
        ax.margins(y=0.12)
    legend = [matplotlib.patches.Patch(color=_shade("#3d3c39", weight), label=search)
              for search, weight in (("asymmetric", 1.0), ("symmetric", 0.45))]
    fig.legend(handles=legend, loc="outside right upper", fontsize=9)
    fig.suptitle(f"PQ: asymmetric against symmetric search (k = {REPORT_K})", x=0.01, ha="left",
                 fontweight="bold")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return Path(path)


def plot_pq(results_dir=RESULTS_DIR, fig_dir=FIG_DIR):
    """PQ graphs: every measure against m for each nbits, the trade-offs, and the best settings."""
    rows = load_results("pq", results_dir)
    if not rows:
        return []
    has_speedup = add_speedup(rows, results_dir)
    rows.sort(key=lambda row: list(DATASETS).index(row["dataset"]))  # same dataset order in every graph
    for row in rows:
        row["symmetric"] = row["symmetric"] == "True"  # the CSV stores the flag as text
    linear_scan = {row["dataset"]: row for row in load_results("linear_scan", results_dir)}
    fig_dir = Path(fig_dir) / "pq"  # PQ graphs get their own sub-folder
    # Symmetric search is left out of every graph except its own comparison with asymmetric search.
    asymmetric = [row for row in rows if not row["symmetric"]]
    saved = []
    for metric, (ylabel, name, logy, hline) in PQ_METRICS.items():
        baseline = ({dataset: row[metric] for dataset, row in linear_scan.items()}
                    if metric in PQ_BASELINE_METRICS else None)
        k_note = "" if "@" in name else f" (k = {REPORT_K})"
        saved.append(pq_param_chart(
            asymmetric, metric, ylabel, f"PQ: {name} against m{k_note}", fig_dir / f"pq_{metric}_vs_m.png",
            logy=logy, exact=hline is not None, baseline=baseline))
    saved.append(pq_symmetric_chart(rows, fig_dir / "pq_symmetric_vs_asymmetric.png"))
    best = {row["dataset"]: row for row in asymmetric
            if (row["m"], row["nbits"]) == PQ_RECOMMENDED.get(row["dataset"])}
    saved.append(tradeoff_chart(
        asymmetric, best, fig_dir / "pq_tradeoff_qps.png", "nbits", "m",
        f"PQ: speed against accuracy (k = {REPORT_K}; point labels are m)"))
    saved.append(tradeoff_chart(
        asymmetric, best, fig_dir / "pq_tradeoff_size.png", "nbits", "m",
        f"PQ: index size against accuracy (k = {REPORT_K}; point labels are m)",
        y="index_size_mb", ylabel="Index size (MB)", legend_loc="lower right"))
    saved.append(best_chart(
        best, has_speedup, fig_dir / "pq_best.png", ("m", "nbits"),
        f"PQ: recommended setting per dataset (k = {REPORT_K})"))
    return saved


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


def plot_ivf_pq_ablation(path, fig_dir=FIG_DIR / "ivf_pq_ablation"):
    """Plot the saved measurements for each configuration."""
    path, fig_dir = Path(path), Path(fig_dir)
    metrics = {
        "recall": "Recall", "query_time_ms": "Query time (ms/query)",
        "index_size_mb": "Index size (MB)", "build_time_s": "Build time (s)",
    }

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
            for ax, (metric, label) in zip(axes.flat, metrics.items()):
                samples = [[float(r[metric]) for r in selected if r["value"] == v]
                           for v in values]
                means = [sum(sample) / len(sample) for sample in samples]
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


# Method name (as used by main.py and the CSV file names) -> its graph function.
GRAPHS = {
    "linear_scan": plot_linear_scan,
    "kd_tree": plot_kd_tree,
    "ball_tree": plot_ball_tree,
    "annoy": plot_annoy,
    "ivf": plot_ivf,
    "pq": plot_pq,
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
