import faiss
import numpy as np

from data import load_dataset

from .base import BaseANN, evaluate

# Parameter settings to sweep; each dict is passed to IVFPQ() as keyword arguments.
# nlist / nprobe control clustering and how many clusters a query searches.
# m / nbits control compression; m must divide the dataset's dimension.
NLIST_VALUES = [256, 1024, 4096]
NPROBE_VALUES = [1, 4, 8, 16, 32, 64]
M_VALUES = {"sift": [8, 16, 32, 64], "gist": [60, 120, 240], "glove": [5, 10, 25]}
NBITS_VALUES = [4, 6, 8]

PARAM_GRID = {
    dataset: [{"nlist": nlist, "m": m, "nbits": nbits, "nprobe": nprobe}
              for nlist in NLIST_VALUES
              for m in ms
              for nbits in NBITS_VALUES
              for nprobe in NPROBE_VALUES]
    for dataset, ms in M_VALUES.items()
}


class IVFPQ(BaseANN):
    """Inverted file with residual product quantisation (faiss IndexIVFPQ).

    Params: nlist, m, nbits, nprobe. Each vector is assigned to a coarse cluster,
    then its residual from that centre is stored as m small codes. Queries search
    only nprobe clusters using approximate distances to the compressed vectors.
    Searching all clusters still gives approximate distances because of PQ.
    """

    name = "ivf_pq"

    def build(self, data):
        data = np.ascontiguousarray(data, dtype=np.float32)
        dim = data.shape[1]
        if dim % self.params["m"]:
            raise ValueError(f"m={self.params['m']} must divide the dimension {dim}")
        self.quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFPQ(
            self.quantizer, dim, self.params["nlist"], self.params["m"],
            self.params["nbits"], faiss.METRIC_L2)
        self.index.train(data)  # learn coarse centres and residual PQ codebooks
        self.index.add(data)    # store vector IDs and compressed residuals per cluster
        self.index.nprobe = self.params["nprobe"]

    def query(self, queries, k):
        # Faiss pads with -1 when the probed clusters contain fewer than k vectors.
        _, ids = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return ids

    def index_size(self):
        return faiss.serialize_index(self.index).nbytes


def run(dataset, ks=(1, 10, 100), max_base=None, max_queries=1000):
    """Evaluate PARAM_GRID[dataset]; returns one result row per setting and k."""
    base, queries, truth = load_dataset(dataset, max_base, max_queries)
    return [{"dataset": dataset, **row}
            for params in PARAM_GRID[dataset]
            for row in evaluate(IVFPQ(**params), base, queries, truth, ks)]
