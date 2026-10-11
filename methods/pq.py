import faiss
import numpy as np

from data import load_dataset

from .base import BaseANN, evaluate

# Parameter settings to sweep; each dict is passed to PQ() as keyword arguments.
#   m      number of sub-vectors each vector is cut into; it must divide the dimension,
#          so the settings are listed per dataset (sift 128-d, gist 960-d, glove 50-d)
#   nbits  bits per sub-vector code; each sub-vector is replaced by one of 2**nbits centroids
#   symmetric  optional, default False. False = asymmetric search: the query stays uncompressed
#              and only the base vectors are codes. True = symmetric search: the query is
#              compressed too, so distances are between two codes. faiss supports this
#              only with nbits = 8.
# A vector is stored in m * nbits / 8 bytes.
#
# Every m is combined with every nbits, all with asymmetric search. The most accurate
# combination (largest m, nbits = 8) is also run with symmetric search for comparison.
M_VALUES = {"sift": [8, 16, 32], "gist": [60, 120, 240], "glove": [5, 10, 25]}
NBITS_VALUES = [4, 8]

PARAM_GRID = {
    dataset: [{"m": m, "nbits": nbits, "symmetric": False} for m in ms for nbits in NBITS_VALUES]
             + [{"m": max(ms), "nbits": 8, "symmetric": True}]
    for dataset, ms in M_VALUES.items()
}


class PQ(BaseANN):
    """Product quantisation without an inverted file (faiss IndexPQ). Params: m, nbits, symmetric.

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
        if self.params.get("symmetric", False):
            if self.params["nbits"] != 8:
                raise ValueError("symmetric search needs nbits = 8")
            self.index.search_type = faiss.IndexPQ.ST_SDC
            self.index.pq.compute_sdc_table()  # centroid-to-centroid distances used by this mode
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
