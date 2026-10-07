"""Dataset loading for the similarity search methods.

The method files call load_dataset(name), which returns three arrays:
    base     vectors to index,           float32, shape (n_base, dim)
    queries  vectors to search for,      float32, shape (n_queries, dim)
    truth    correct answers,            int,     shape (n_queries, GT_K)
             truth[i] = indices into base of the true nearest neighbours of queries[i], closest first

Expected files (not committed to git):
    datasets/sift/sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs, sift_learn.fvecs
    datasets/gist/gist_base.fvecs, gist_query.fvecs, gist_groundtruth.ivecs, gist_learn.fvecs
    datasets/glove/glove.6B.50d.txt
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "datasets"
GT_K = 100            # true neighbours stored per query, so recall@k works for any k <= 100
N_QUERIES = 1000      # queries used per dataset, the same for all three so results are comparable


def _read_vecs(path, dtype, count=None):
    """Read a .fvecs (dtype float32) or .ivecs (dtype int32) file into a 2-D array.

    Every vector is stored as a 4-byte int giving its dimension d, followed by d 4-byte values.
    `count` reads only the first N vectors.
    """
    # Memory-map the file so only the rows that are kept get copied into RAM
    # (GIST's base file is 3.8 GB, so reading it whole and then copying would need twice that).
    raw = np.memmap(path, dtype=np.int32, mode="r")
    dim = int(raw[0])
    # One row per vector, drop the leading dimension column, then reinterpret the bytes as dtype.
    return np.ascontiguousarray(raw.reshape(-1, dim + 1)[:count, 1:]).view(dtype)


def _load_texmex(name, max_base, max_queries):
    """Shared loader for SIFT1M and GIST1M, which use the same file layout and naming."""
    folder = DATA_DIR / name
    base = _read_vecs(folder / f"{name}_base.fvecs", np.float32, max_base)
    queries = _read_vecs(folder / f"{name}_query.fvecs", np.float32, max_queries)
    truth = _read_vecs(folder / f"{name}_groundtruth.ivecs", np.int32, max_queries)
    if truth.max() >= len(base):
        # The shipped ground truth refers to the full base set; with a truncated base set
        # it points at vectors that were dropped, so recompute it.
        truth = brute_force_neighbours(base, queries, GT_K)
    return base, queries, truth


def brute_force_neighbours(base, queries, k, batch_size=100):
    """Exact k nearest neighbours by Euclidean distance, used as ground truth."""
    # ||q - b||^2 = ||q||^2 + ||b||^2 - 2 q.b ; ||q||^2 is the same for every b, so it
    # does not change the ranking and is left out.
    base_sq = (base ** 2).sum(axis=1)
    out = np.empty((len(queries), k), dtype=np.int64)
    # Work in batches so the (batch, n_base) distance matrix fits in memory.
    for start in range(0, len(queries), batch_size):
        q = queries[start:start + batch_size]
        dist = base_sq - 2 * q @ base.T
        # argpartition finds the k smallest without sorting the whole row; then sort just those k.
        top = np.argpartition(dist, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(dist, top, axis=1), axis=1)
        out[start:start + len(q)] = np.take_along_axis(top, order, axis=1)
    return out


def load_sift(max_base=None, max_queries=N_QUERIES):
    """SIFT1M: 1,000,000 base vectors, 128 dimensions, Euclidean distance.

    The file has 10,000 queries; by default only the first N_QUERIES are used.
    """
    return _load_texmex("sift", max_base, max_queries)


def load_sift_learn(max_vectors=None):
    """Separate SIFT training vectors, for methods that train a quantiser (e.g. IVF+PQ)."""
    return _read_vecs(DATA_DIR / "sift" / "sift_learn.fvecs", np.float32, max_vectors)


def load_gist(max_base=None, max_queries=N_QUERIES):
    """GIST1M: 1,000,000 base vectors and 1,000 queries, 960 dimensions, Euclidean distance.

    The full base set takes about 3.8 GB of RAM.
    """
    return _load_texmex("gist", max_base, max_queries)


def load_gist_learn(max_vectors=None):
    """Separate GIST training vectors (500,000), for methods that train a quantiser (e.g. IVF+PQ)."""
    return _read_vecs(DATA_DIR / "gist" / "gist_learn.fvecs", np.float32, max_vectors)


def load_glove(dim=50, max_base=None, max_queries=N_QUERIES, seed=0):
    """GloVe 6B word vectors: 400,000 words, split into 399,000 base vectors and 1,000 queries."""
    folder = DATA_DIR / "glove"
    # Parsing the text file is slow, so the parsed array is saved as .npy and reused next time.
    cache = folder / f"glove.6B.{dim}d.npy"
    if cache.exists():
        vectors = np.load(cache)
    else:
        with open(folder / f"glove.6B.{dim}d.txt", encoding="utf-8") as f:
            # Each line is "word v1 v2 ... vd"; keep only the numbers.
            vectors = np.array([line.rstrip().split(" ")[1:] for line in f], dtype=np.float32)
        np.save(cache, vectors)

    # Unit-normalise so that Euclidean nearest neighbours match cosine similarity.
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    # GloVe has no query set, so hold out N_QUERIES random vectors as queries.
    # The seed is fixed so every method sees the same split.
    order = np.random.default_rng(seed).permutation(len(vectors))
    queries = vectors[order[:N_QUERIES]][:max_queries]
    base = np.ascontiguousarray(vectors[order[N_QUERIES:]][:max_base])
    # No ground truth is shipped either, so compute it by exact search.
    truth = brute_force_neighbours(base, queries, GT_K)
    return base, queries, truth


# Dataset name -> loader function.
DATASETS = {"sift": load_sift, "gist": load_gist, "glove": load_glove}


@lru_cache(maxsize=1)
def load_dataset(name, max_base=None, max_queries=N_QUERIES):
    """Return (base, queries, truth) for a dataset name.

    base and queries are float32 arrays of shape (n, d); truth holds the indices
    of the GT_K exact nearest base vectors of each query, closest first. The last
    result is cached so every method reuses the same arrays without reloading.
    """
    return DATASETS[name](max_base=max_base, max_queries=max_queries)
