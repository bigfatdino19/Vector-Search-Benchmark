"""Shared pieces for all method files: the common interface and the evaluation code."""
import time

import numpy as np


class BaseANN:
    """Common interface every method implements so they can be evaluated identically."""

    name = "base"  # overridden in each subclass; appears in the results CSV

    def __init__(self, **params):
        # One entry of the method's PARAM_GRID, e.g. {"leaf_size": 40}.
        # Read values in build()/query() with self.params["leaf_size"].
        self.params = params

    def build(self, data):
        """Build the index over `data`, a float32 array of shape (n, d)."""
        raise NotImplementedError

    def query(self, queries, k):
        """Return an int array of shape (n_queries, k): indices into `data` of the
        k nearest neighbours of each query, closest first."""
        raise NotImplementedError

    def index_size(self):
        """Return the size of the built index in bytes, or None if not measured.

        Count everything the method needs to answer queries, including any copy of the
        vectors it keeps, so the numbers are comparable across methods. For example:
            raw NumPy data   self.data.nbytes
            scikit-learn     len(pickle.dumps(self.tree))
            faiss            faiss.serialize_index(self.index).nbytes
            annoy            save the index to a temporary file and take os.path.getsize()
        """
        return None


def recall_at_k(predicted, truth, k):
    """Not order-aware. Fraction of the true k nearest neighbours that were returned, averaged over all queries.

    1.0 means every query got exactly the right k neighbours (order within the k does not matter).
    """
    hits = sum(len(set(p[:k]) & set(t[:k])) for p, t in zip(predicted, truth))
    return hits / (len(truth) * k)


def one_recall_at_r(predicted, truth, r):
    """1-Recall@R: fraction of queries whose true nearest neighbour is among the first R results."""
    hits = sum(t[0] in p[:r] for p, t in zip(predicted, truth))
    return hits / len(truth)


def distance_ratio(predicted, truth, base, queries, k):
    """Distance to the k-th returned neighbour divided by the distance to the true k-th
    neighbour, averaged over all queries.

    1.0 is perfect; larger values show how much further away the returned neighbours are
    than the correct ones. Queries are left out of the average when the true distance is 0
    (the ratio is undefined) or when the method returned fewer than k results (marked by -1).
    """
    true_dist = np.linalg.norm(queries - base[truth[:, k - 1]], axis=1)
    # The k-th returned neighbour is the furthest of the k, whatever order they came back in.
    pred_dist = np.zeros(len(queries), dtype=np.float32)
    for j in range(k):
        ids = predicted[:, j]
        dist = np.linalg.norm(queries - base[ids], axis=1)
        pred_dist = np.maximum(pred_dist, np.where(ids < 0, np.inf, dist))
    valid = (true_dist > 0) & np.isfinite(pred_dist)
    return float((pred_dist[valid] / true_dist[valid]).mean()) if valid.any() else float("nan")


def ndcg_at_k(predicted, truth, k):
    """Order-aware accuracy in [0, 1], averaged over all queries (NDCG@k).

    The true nearest neighbour has relevance k, the second k-1, ... the k-th 1, anything else 0.
    Each returned neighbour adds relevance / log2(position + 1), so correct results count for
    more near the top of the list. Dividing by the score of the perfect list gives 1.0 only
    when the right k neighbours are returned in the right order.
    """
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = (np.arange(k, 0, -1) * discounts).sum()
    total = 0.0
    for p, t in zip(predicted, truth):
        relevance = {idx: k - rank for rank, idx in enumerate(t[:k])}
        total += sum(relevance.get(idx, 0) * d for idx, d in zip(p[:k], discounts))
    return total / (len(truth) * ideal)


def evaluate(index, base, queries, truth, ks):
    """Build one index, then time and score a search for every k in `ks`.

    Returns one result row per k. The index is built only once, so build time and
    index size are the same in every row; query time and accuracy are measured per k.
    """
    ks = [int(k) for k in np.atleast_1d(ks)]
    # truth only stores GT_K neighbours per query, so recall cannot be computed beyond that.
    if not all(1 <= k <= truth.shape[1] for k in ks):
        raise ValueError(f"k must be between 1 and {truth.shape[1]}")

    # Index construction time.
    start = time.perf_counter()
    index.build(base)
    build_time = time.perf_counter() - start
    size = index.index_size()

    rows = []
    for k in ks:
        # Search time for all queries in one call.
        start = time.perf_counter()
        predicted = np.asarray(index.query(queries, k))
        query_time = time.perf_counter() - start

        # One result row; the method file adds "dataset" and main.py writes it to the CSV.
        rows.append({
            "method": index.name,
            "params": index.params,
            "n_base": len(base),
            "n_queries": len(queries),
            "dim": base.shape[1],
            "k": k,
            "build_time_s": round(build_time, 4),
            "index_size_mb": None if size is None else round(size / 1e6, 2),
            "query_time_ms": round(1000 * query_time / len(queries), 4),  # average time per query
            "qps": round(len(queries) / query_time, 1),                   # queries per second
            "recall": round(recall_at_k(predicted, truth, k), 4),
            "ndcg": round(ndcg_at_k(predicted, truth, k), 4),
            "one_recall_at_k": round(one_recall_at_r(predicted, truth, k), 4),  # 1-Recall@R, R = k
            "distance_ratio": round(distance_ratio(predicted, truth, base, queries, k), 4),
        })
    return rows
