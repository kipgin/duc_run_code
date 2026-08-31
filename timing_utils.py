"""
timing_utils.py
----------------
Recreates HW4 §0.3 deliverable 4 (the time table).

WMD: per-query time = time to compute one query document's exact W1
     distance to every document of the lookup corpus (the split's
     training set), looped over queries (so parallelization over pairs
     doesn't blur query boundaries). Pooled mean +- SD over every test
     query of every split. We time W1 only (W2 differs only by squaring
     the cost matrix -- essentially identical runtime, per the HW4
     text).

FSW (per m): (i) pre-embedding time -- embed the whole lookup corpus,
     once, timed as a single block. (ii) per-query time -- embed the
     query doc + compute its n Euclidean distances to the pre-embedded
     corpus, pooled mean +- SD over the same queries as WMD.

A few warm-up queries are run and discarded before measuring, per the
HW4 instruction.
"""
import time
import numpy as np

from cache_utils import machine_name
from wmd_distances import _cost_matrix, _ot_distance
from fsw_embed import make_fsw_module, euclidean_distance_matrix


def time_wmd_per_query(X_train, W_train, X_queries, W_queries,
                        n_warmup=3):
    """Loop over queries; each query's full block of distances to the
    n_train lookup-corpus documents is timed as one unit."""

    def one_query(Xq, Wq):
        t0 = time.perf_counter()
        for Xt, Wt in zip(X_train, W_train):
            C = _cost_matrix(Xq, Xt, squared=False)
            _ot_distance(Wq, Wt, C)
        return time.perf_counter() - t0

    for i in range(min(n_warmup, len(X_queries))):
        one_query(X_queries[i], W_queries[i])

    times = [one_query(Xq, Wq) for Xq, Wq in zip(X_queries, W_queries)]
    return np.array(times)


def time_fsw(X_train, W_train, X_queries, W_queries, m, d_in=300,
             seed=0, n_warmup=3):
    """Returns (pre_embedding_seconds, per_query_times array)."""
    module = make_fsw_module(d_in=d_in, d_out=m, seed=seed)

    import torch
    def embed_one(X, W):
        with torch.no_grad():
            v = module(torch.from_numpy(X).float(), torch.from_numpy(W).float())
        return v.numpy()

    t0 = time.perf_counter()
    corpus_vecs = np.stack([embed_one(X, W) for X, W in zip(X_train, W_train)])
    pre_embedding_seconds = time.perf_counter() - t0

    def one_query(Xq, Wq):
        t0 = time.perf_counter()
        qvec = embed_one(Xq, Wq)[None, :]
        euclidean_distance_matrix(qvec, corpus_vecs)
        return time.perf_counter() - t0

    for i in range(min(n_warmup, len(X_queries))):
        one_query(X_queries[i], W_queries[i])

    times = [one_query(Xq, Wq) for Xq, Wq in zip(X_queries, W_queries)]
    return pre_embedding_seconds, np.array(times)


def pool_over_splits(per_split_time_arrays):
    """Concatenate per-query time arrays across all splits and report
    mean +- SD, per HW4's 'time every test query of every split, pooled'."""
    pooled = np.concatenate(per_split_time_arrays)
    return {
        "mean_seconds": float(pooled.mean()),
        "std_seconds": float(pooled.std(ddof=0)),
        "n_queries": int(pooled.size),
        "machine": machine_name(),
    }
