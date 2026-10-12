# The float32 tree keeps the vectors as loaded; the public KDTree copies them to float64.
from sklearn.neighbors._kd_tree import KDTree32

from data import GT_K, N_QUERIES, load_dataset

from .base import BaseANN, evaluate

# Parameter settings to sweep; each dict is passed to KDTree() as keyword arguments.
#   leaf_size  largest number of vectors in one leaf; splitting stops below it
LEAF_SIZES = [5, 10, 20, 40, 80, 160]

PARAM_GRID = [{"leaf_size": leaf_size} for leaf_size in LEAF_SIZES]


class KDTree(BaseANN):
    """KD-tree (scikit-learn, float32), exact search by Euclidean distance. Params: leaf_size.

    The tree cuts the space one coordinate at a time into axis-aligned boxes. A query
    walks down to its box, then visits every other box that can hold a closer vector.
    """

    name = "kd_tree"

    def build(self, data):
        self.tree = KDTree32(data, leaf_size=self.params["leaf_size"], metric="euclidean")

    def query(self, queries, k):
        return self.tree.query(queries, k=k, return_distance=False)

    def index_size(self):
        # The stored vectors plus the index, node and bound arrays of the tree.
        return sum(array.nbytes for array in self.tree.get_arrays())


def run(dataset, ks=(1, 10, GT_K), max_base=None, max_queries=N_QUERIES):
    """Evaluate every setting in PARAM_GRID on `dataset`; returns one result row per setting and k."""
    base, queries, truth = load_dataset(dataset, max_base, max_queries)
    return [{"dataset": dataset, **row}
            for params in PARAM_GRID
            for row in evaluate(KDTree(**params), base, queries, truth, ks)]
