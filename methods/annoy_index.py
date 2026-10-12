import csv
import os
import tempfile
import time
from pathlib import Path

from annoy import AnnoyIndex

from data import load_dataset

from .base import BaseANN, evaluate

# Parameter settings to sweep; each dict is passed to Annoy() as keyword arguments.
PARAM_GRID = [{"n_trees": t, "search_k": s}
              for t in (5, 10, 20, 50, 100, 200)
              for s in (-1, 500, 1000, 2000, 5000, 10000, 20000, 50000)]



class Annoy(BaseANN):
    """Annoy (random projection forest). Params: n_trees, search_k."""

    name = "annoy"

    def build(self, data):
        self.dim = data.shape[1]
        self.index = AnnoyIndex(self.dim, "euclidean")
        for i, vec in enumerate(data):
            self.index.add_item(i, vec)
        self.index.build(self.params["n_trees"])

    def query(self, queries, k):
        search_k = self.params["search_k"]
        results = []
        for q in queries:
            neighbors = self.index.get_nns_by_vector(q.tolist(), k, search_k=search_k)
            if len(neighbors) < k:
                # Not enough candidates found; pad with -1 so it counts as a miss.
                neighbors = neighbors + [-1] * (k - len(neighbors))
            results.append(neighbors)
        return results

    def index_size(self):
        with tempfile.NamedTemporaryFile(suffix=".ann", delete=False) as f:
            path = f.name
        try:
            self.index.save(path)
            return os.path.getsize(path)
        finally:
            os.remove(path)


def _already_run(dataset, n_base, n_queries, ks, results_dir):
    """(n_trees, search_k) settings already saved in results/annoy.csv for every k in `ks`."""
    path = Path(results_dir) / "annoy.csv"
    if not path.exists():
        return set()
    seen = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if (row["dataset"] == dataset and int(row["n_base"]) == n_base
                    and int(row["n_queries"]) == n_queries):
                setting = (int(float(row["n_trees"])), int(float(row["search_k"])))
                seen.setdefault(setting, set()).add(int(float(row["k"])))
    return {setting for setting, saved in seen.items() if set(ks) <= saved}


def run(dataset, ks=(1, 10, 100), max_base=None, max_queries=1000, results_dir="results"):
    """Evaluate the settings in PARAM_GRID on `dataset`; returns one result row per setting and k.

    Settings already saved in results_dir (same dataset size and every k) are skipped, and each
    n_trees is built only once and then searched with each of its search_k values.
    """
    base, queries, truth = load_dataset(dataset, max_base, max_queries)
    done = _already_run(dataset, len(base), len(queries), ks, results_dir)
    pending = [p for p in PARAM_GRID if (p["n_trees"], p["search_k"]) not in done]
    print(f"annoy: {len(PARAM_GRID) - len(pending)} of {len(PARAM_GRID)} settings already saved, "
          f"running {len(pending)}")
    rows = []
    for n_trees in dict.fromkeys(p["n_trees"] for p in pending):
        settings = [p for p in pending if p["n_trees"] == n_trees]
        index = Annoy(**settings[0])
        prebuilt = None
        for params in settings:
            index.params = params
            if prebuilt is None:
                # First setting of this n_trees: evaluate() builds the index and times it.
                start = time.perf_counter()
                index.build(base)
                prebuilt = (time.perf_counter() - start, index.index_size())
            rows += [{"dataset": dataset, **row}
                     for row in evaluate(index, base, queries, truth, ks, prebuilt=prebuilt)]
    return rows
