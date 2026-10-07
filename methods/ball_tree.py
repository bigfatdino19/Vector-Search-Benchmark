from data import load_dataset

from .base import BaseANN, evaluate

# Parameter settings to sweep; each dict is passed to BallTree() as keyword arguments.
PARAM_GRID = [{"leaf_size": 40}]


class BallTree(BaseANN):
    """Ball tree. Params: leaf_size."""

    name = "ball_tree"

    def build(self, data):
        raise NotImplementedError

    def query(self, queries, k):
        raise NotImplementedError


def run(dataset, k=10, max_base=None, max_queries=1000):
    """Evaluate every setting in PARAM_GRID on `dataset`; returns one result row per setting."""
    base, queries, truth = load_dataset(dataset, max_base, max_queries)
    return [{"dataset": dataset, **evaluate(BallTree(**params), base, queries, truth, k)}
            for params in PARAM_GRID]
