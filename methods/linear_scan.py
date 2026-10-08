import numpy as np

from data import load_dataset

from .base import BaseANN, evaluate

# Parameter settings to sweep; each dict is passed to LinearScan() as keyword arguments.
PARAM_GRID = [{}]


class LinearScan(BaseANN):
    """Exact brute-force search; the baseline every other method is compared against."""

    name = "linear_scan"

    def build(self, data):
        """Keep base vectors and precompute squared norms for distance ranking."""
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2 or not all(data.shape):
            raise ValueError("data must be a non-empty 2-D array")
        for start in range(0, len(data), 8192):
            if not np.isfinite(data[start:start + 8192]).all():
                raise ValueError("data must contain only finite values")
        self.data = data
        self.base_squared = np.empty(len(data), dtype=np.float32)
        for start in range(0, len(data), 8192):
            block = data[start:start + 8192]
            self.base_squared[start:start + len(block)] = (block ** 2).sum(axis=1)

    def query(self, queries, k):
        """Scan every base vector and return the k closest IDs per query."""
        if not hasattr(self, "data"):
            raise RuntimeError("call build() before query()")
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.data.shape[1]:
            raise ValueError("queries must be a 2-D array matching the base dimensions")
        if not np.isfinite(queries).all():
            raise ValueError("queries must contain only finite values")
        if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(self.data):
            raise ValueError(f"k must be an integer between 1 and {len(self.data)}")

        neighbours = np.empty((len(queries), k), dtype=np.int64)
        for query_start in range(0, len(queries), 100):
            q = queries[query_start:query_start + 100]
            best_scores = np.empty((len(q), 0), dtype=np.float32)
            best_ids = np.empty((len(q), 0), dtype=np.int64)
            for start in range(0, len(self.data), 8192):
                block = self.data[start:start + 8192]
                # Same method as brute_force_neighbours in data.py: 
                # ||q-b||^2 = ||q||^2 + ||b||^2 - 2 q.b.
                # Omit ||q||^2: it is constant across base vectors for each query.
                scores = self.base_squared[start:start + len(block)] - 2 * (q @ block.T)
                count = min(k, len(block))
                local = np.argpartition(scores, count - 1, axis=1)[:, :count]
                candidate_scores = np.concatenate(
                    (best_scores, np.take_along_axis(scores, local, axis=1)), axis=1)
                candidate_ids = np.concatenate((best_ids, local + start), axis=1)
                # Keep the best k across every block scanned so far.
                count = min(k, candidate_scores.shape[1])
                keep = np.argpartition(candidate_scores, count - 1, axis=1)[:, :count]
                best_scores = np.take_along_axis(candidate_scores, keep, axis=1)
                best_ids = np.take_along_axis(candidate_ids, keep, axis=1)
            order = np.argsort(best_scores, axis=1, kind="stable")
            neighbours[query_start:query_start + len(q)] = np.take_along_axis(
                best_ids, order, axis=1)
        return neighbours

    def index_size(self):
        """Count stored vectors and the precomputed squared norms."""
        return self.data.nbytes + self.base_squared.nbytes


def run(dataset, k=10, max_base=None, max_queries=1000):
    """Evaluate every setting in PARAM_GRID on `dataset`; returns one result row per setting."""
    base, queries, truth = load_dataset(dataset, max_base, max_queries)
    return [{"dataset": dataset, **evaluate(LinearScan(**params), base, queries, truth, k)}
            for params in PARAM_GRID]
