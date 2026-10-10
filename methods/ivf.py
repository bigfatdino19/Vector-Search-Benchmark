import faiss
import numpy as np

from data import load_dataset

from .base import BaseANN, evaluate

# Parameter settings to sweep; each dict is passed to IVF() as keyword arguments.
#   nlist   number of clusters the base vectors are split into
#   nprobe  number of nearest clusters searched per query (nprobe = nlist is an exact search)
# Every nlist is combined with every nprobe. One extra setting searches all the clusters
# (nprobe = nlist): it returns the exact neighbours, so it shows what clustering costs
# when it saves no work.
NLIST_VALUES = [256, 1024, 4096]
NPROBE_VALUES = [1, 8, 64]

PARAM_GRID = [{"nlist": nlist, "nprobe": nprobe} for nlist in NLIST_VALUES for nprobe in NPROBE_VALUES]


class IVF(BaseANN):
    """Inverted file index without compression (faiss IndexIVFFlat). Params: nlist, nprobe.

    k-means splits the base vectors into nlist clusters. A query is compared only with the
    vectors in its nprobe nearest clusters, using exact Euclidean distances on the full vectors.
    """

    name = "ivf"

    def build(self, data):
        data = np.ascontiguousarray(data, dtype=np.float32)
        dim = data.shape[1]
        # The coarse quantiser finds the nearest cluster centres by exact search.
        self.quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(self.quantizer, dim, self.params["nlist"], faiss.METRIC_L2)
        self.index.train(data)  # k-means to learn the cluster centres
        self.index.add(data)    # assign every base vector to its nearest centre
        self.index.nprobe = self.params["nprobe"]

    def query(self, queries, k):
        # Queries that find fewer than k vectors in the probed clusters are padded with -1.
        _, ids = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return ids

    def index_size(self):
        return faiss.serialize_index(self.index).nbytes


def run(dataset, ks=(1, 10, 100), max_base=None, max_queries=1000):
    """Evaluate every setting in PARAM_GRID on `dataset`; returns one result row per setting and k."""
    base, queries, truth = load_dataset(dataset, max_base, max_queries)
    return [{"dataset": dataset, **row}
            for params in PARAM_GRID
            for row in evaluate(IVF(**params), base, queries, truth, ks)]
