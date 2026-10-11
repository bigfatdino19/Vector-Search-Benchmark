"""Graphs for the report, drawn from the per-method CSVs that main.py writes to results/.

There is one plot_<method>() function per method. main.py calls plot() after a run, and the
file can also be run on its own to redraw from the saved CSVs without re-running anything:

    python graph.py            # every method
    python graph.py ivf pq     # only these methods

Every graph is one chart that answers one question, with one line per dataset.
"""
import csv
import sys
from pathlib import Path

import matplotlib
import matplotlib.colors

matplotlib.use("Agg")  # draw straight to files; no window needed
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
FIG_DIR = Path("figures")

# All graphs are drawn for this k.
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
    """Rows of results/<method>.csv at one k, as dicts with numeric values ([] if no file)."""
    path = Path(results_dir) / f"{method}.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(f)]
    return [row for row in rows if row["k"] == k]


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
    raise NotImplementedError


def plot_kd_tree(results_dir=RESULTS_DIR, fig_dir=FIG_DIR):
    raise NotImplementedError


def plot_ball_tree(results_dir=RESULTS_DIR, fig_dir=FIG_DIR):
    raise NotImplementedError


def plot_annoy(results_dir=RESULTS_DIR, fig_dir=FIG_DIR):
    raise NotImplementedError


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


def best_ivf_rows(rows, min_recall=IVF_MIN_RECALL):
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


def ivf_tradeoff_chart(rows, best, path):
    """Recall against QPS, one panel per dataset and one line per nlist (points are nprobe values)."""
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    nlists = sorted({row["nlist"] for row in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4.2), sharey=True,
                             layout="constrained", squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        label, colour, _ = DATASETS[dataset]
        for i, (nlist, marker) in enumerate(zip(nlists, "o^sDvP")):
            shade = _shade(colour, 0.45 + 0.55 * i / max(1, len(nlists) - 1))
            points = sorted((row["nprobe"], row["recall"], row["qps"]) for row in rows
                            if row["dataset"] == dataset and row["nlist"] == nlist)
            ax.plot([p[1] for p in points], [p[2] for p in points], color=shade, marker=marker,
                    linewidth=2, markersize=7, label=f"nlist = {nlist}")
            for nprobe, recall, qps in points:
                ax.annotate(f"{nprobe}", (recall, qps), textcoords="offset points", xytext=(7, 6),
                            fontsize=8, color="#52514e")
        chosen = best[dataset]
        ax.plot(chosen["recall"], chosen["qps"], marker="o", markersize=16, markerfacecolor="none",
                markeredgecolor="#0b0b0b", markeredgewidth=1.5, linestyle="none", label="best setting")
        ax.axvline(IVF_MIN_RECALL, color="#898781", linestyle="--", linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(f"Recall@{REPORT_K}")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.legend(loc="lower left", fontsize=8)
    axes[0][0].set_ylabel("Queries per second")
    fig.suptitle(f"IVF: speed against accuracy (k = {REPORT_K}; point labels are nprobe, "
                 f"dashed line is recall {IVF_MIN_RECALL})", x=0.01, ha="left", fontweight="bold")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return Path(path)


def ivf_best_chart(best, has_speedup, path):
    """Bar charts comparing each dataset's best setting, one panel per measure."""
    panels = [("qps", "Queries per second", "{:,.0f}"), ("recall", f"Recall@{REPORT_K}", "{:.3f}"),
              ("build_time_s", "Build time (s)", "{:.1f}"), ("index_size_mb", "Index size (MB)", "{:,.0f}")]
    if has_speedup:
        panels.insert(1, ("speedup", "Speed-up over linear scan (x)", "{:.0f}x"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.8), layout="constrained")
    names = [f"{DATASETS[dataset][0].split(' (')[0]}\nnlist {row['nlist']}\nnprobe {row['nprobe']}"
             for dataset, row in best.items()]
    colours = [DATASETS[dataset][1] for dataset in best]
    for ax, (key, ylabel, fmt) in zip(axes, panels):
        bars = ax.bar(names, [row[key] for row in best.values()], color=colours, width=0.6)
        ax.bar_label(bars, labels=[fmt.format(row[key]) for row in best.values()], padding=2, fontsize=9)
        ax.set_title(ylabel, loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=8)
        ax.margins(y=0.12)
    fig.suptitle(f"IVF: best setting per dataset (fastest with recall@{REPORT_K} >= {IVF_MIN_RECALL})",
                 x=0.01, ha="left", fontweight="bold")
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
    best = best_ivf_rows(rows)
    saved.append(ivf_tradeoff_chart(rows, best, fig_dir / "ivf_tradeoff.png"))
    saved.append(ivf_best_chart(best, has_speedup, fig_dir / "ivf_best.png"))
    return saved


def plot_ivf_pq(results_dir=RESULTS_DIR, fig_dir=FIG_DIR):
    raise NotImplementedError


# Method name (as used by main.py and the CSV file names) -> its graph function.
GRAPHS = {
    "linear_scan": plot_linear_scan,
    "kd_tree": plot_kd_tree,
    "ball_tree": plot_ball_tree,
    "annoy": plot_annoy,
    "ivf": plot_ivf,
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
