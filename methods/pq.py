import faiss
import numpy as np

from data import load_dataset

from .base import BaseANN, evaluate

# Parameter settings to sweep; each dict is passed to PQ() as keyword arguments.
#   m      number of sub-vectors each vector is cut into; it must divide the dimension,
#          so the settings are listed per dataset (sift 128-d, gist 960-d, glove 50-d)
#   nbits  bits per sub-vector code; each sub-vector is replaced by one of 2**nbits centroids
# A vector is stored in m * nbits / 8 bytes (shown beside each setting).
# Each list has three groups:
#   1. vary m with nbits fixed at 8
#   2. vary nbits with m fixed at the middle value (its nbits=8 point is already in group 1)
#   3. a setting with the same bytes per vector as the middle one, but twice the sub-vectors
#      at half the bits, to compare the two ways of spending the same memory
# The first setting of group 2 also matches the bytes of the smallest setting in group 1.
PARAM_GRID = {
    "sift": [
        {"m": 8, "nbits": 8},     # 8 bytes
        {"m": 16, "nbits": 8},    # 16 bytes
        {"m": 32, "nbits": 8},    # 32 bytes
        {"m": 16, "nbits": 4},    # 8 bytes
        {"m": 16, "nbits": 6},    # 12 bytes
        {"m": 32, "nbits": 4},    # 16 bytes
    ],
    "gist": [
        {"m": 60, "nbits": 8},    # 60 bytes
        {"m": 120, "nbits": 8},   # 120 bytes
        {"m": 240, "nbits": 8},   # 240 bytes
        {"m": 120, "nbits": 4},   # 60 bytes
        {"m": 120, "nbits": 6},   # 90 bytes
        {"m": 240, "nbits": 4},   # 120 bytes
    ],
    "glove": [
        {"m": 5, "nbits": 8},     # 5 bytes
        {"m": 10, "nbits": 8},    # 10 bytes
        {"m": 25, "nbits": 8},    # 25 bytes
        {"m": 10, "nbits": 4},    # 5 bytes
        {"m": 10, "nbits": 6},    # 7.5 bytes
        {"m": 25, "nbits": 4},    # 12.5 bytes
    ],
}


class PQ(BaseANN):
    """Product quantisation without an inverted file (faiss IndexPQ). Params: m, nbits.

    Every base vector is compressed to m small codes. A query is compared with all base
    vectors, but using distances estimated from the codes instead of the full vectors.
    """

    name = "pq"

    def build(self, data):
        data = np.ascontiguousarray(data, dtype=np.float32)
        dim = data.shape[1]
        if dim % self.params["m"]:
            raise ValueError(f"m={self.params['m']} must divide the dimension {dim}")
        self.index = faiss.IndexPQ(dim, self.params["m"], self.params["nbits"], faiss.METRIC_L2)
        self.index.train(data)  # k-means per sub-vector to learn the codebooks
        self.index.add(data)    # encode every base vector; only the codes are stored

    def query(self, queries, k):
        _, ids = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return ids

    def index_size(self):
        return faiss.serialize_index(self.index).nbytes


def run(dataset, ks=(1, 10, 100), max_base=None, max_queries=1000):
    """Evaluate every setting in PARAM_GRID[dataset]; returns one result row per setting and k."""
    base, queries, truth = load_dataset(dataset, max_base, max_queries)
    return [{"dataset": dataset, **row}
            for params in PARAM_GRID[dataset]
            for row in evaluate(PQ(**params), base, queries, truth, ks)]
