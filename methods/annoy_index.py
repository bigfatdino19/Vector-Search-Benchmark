from data import load_dataset

from .base import BaseANN, evaluate

# Parameter settings to sweep; each dict is passed to Annoy() as keyword arguments.
PARAM_GRID = [{"n_trees": 10, "search_k": -1}]


class Annoy(BaseANN):
    """Annoy (random projection forest). Params: n_trees, search_k."""

    name = "annoy"

    def build(self, data):
        raise NotImplementedError

    def query(self, queries, k):
        raise NotImplementedError


def run(dataset, ks=(1, 10, 100), max_base=None, max_queries=1000):
    """Evaluate every setting in PARAM_GRID on `dataset`; returns one result row per setting and k."""
    base, queries, truth = load_dataset(dataset, max_base, max_queries)
    return [{"dataset": dataset, **row}
            for params in PARAM_GRID
            for row in evaluate(Annoy(**params), base, queries, truth, ks)]
