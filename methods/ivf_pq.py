from data import load_dataset

from .base import BaseANN, evaluate

# Parameter settings to sweep; each dict is passed to IVFPQ() as keyword arguments.
PARAM_GRID = [{"nlist": 1024, "m": 8, "nbits": 8, "nprobe": 16}]


class IVFPQ(BaseANN):
    """Inverted file index with product quantisation. Params: nlist, m, nbits, nprobe."""

    name = "ivf_pq"

    def build(self, data):
        raise NotImplementedError

    def query(self, queries, k):
        raise NotImplementedError


def run(dataset, k=10, max_base=None, max_queries=1000):
    """Evaluate every setting in PARAM_GRID on `dataset`; returns one result row per setting."""
    base, queries, truth = load_dataset(dataset, max_base, max_queries)
    return [{"dataset": dataset, **evaluate(IVFPQ(**params), base, queries, truth, k)}
            for params in PARAM_GRID]
