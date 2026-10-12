"""Graphs for the report, drawn from the per-method CSVs that main.py writes to results/.

There is one plot_<method>() function per method. main.py calls plot() after a run, and the
file can also be run on its own to redraw from the saved CSVs without re-running anything:

    python graph.py            # every method
    python graph.py ivf pq     # only these methods

Every graph is one chart that answers one question, with one line per dataset.
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


def best_chart(best, has_speedup, path, params, title):
    """Bar charts comparing each dataset's best setting, one panel per measure.

    params  the parameter columns named under each bar, e.g. ("nlist", "nprobe")
    """
    panels = [("qps", "Queries per second", "{:,.0f}"), ("recall", f"Recall@{REPORT_K}", "{:.3f}"),
              ("build_time_s", "Build time (s)", "{:.1f}"), ("index_size_mb", "Index size (MB)", "{:,.0f}")]
    if has_speedup:
        panels.insert(1, ("speedup", "Speed-up over linear scan (x)", "{:.3g}x"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.8), layout="constrained")
    names = ["\n".join([DATASETS[dataset][0].split(" (")[0]] + [f"{param} {row[param]}" for param in params])
             for dataset, row in best.items()]
    colours = [DATASETS[dataset][1] for dataset in best]
    for ax, (key, ylabel, fmt) in zip(axes, panels):
        bars = ax.bar(names, [row[key] for row in best.values()], color=colours, width=0.6)
        ax.bar_label(bars, labels=[fmt.format(row[key]) for row in best.values()], padding=2, fontsize=9)
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
    raise NotImplementedError


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
