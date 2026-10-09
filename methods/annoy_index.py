import os
import tempfile

from annoy import AnnoyIndex

from data import load_dataset

from .base import BaseANN, evaluate

# Parameter settings to sweep; each dict is passed to Annoy() as keyword arguments.
PARAM_GRID = [{"n_trees": t, "search_k": s}
              for t in (10, 50, 100)
              for s in (-1, 1000, 5000, 20000)]


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


def run(dataset, ks=(1, 10, 100), max_base=None, max_queries=1000):
    """Evaluate every setting in PARAM_GRID on `dataset`; returns one result row per setting and k."""
    base, queries, truth = load_dataset(dataset, max_base, max_queries)
    return [{"dataset": dataset, **row}
            for params in PARAM_GRID
            for row in evaluate(Annoy(**params), base, queries, truth, ks)]
