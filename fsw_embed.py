"""
fsw_embed.py
------------
Recreates HW3's FSW usage rules:
  1. torch.manual_seed(0) before creating the module.
  2. FSWEmbedding(d_in=d, d_out=m, minimize_slice_coherence=True,
     frequency_init='even') -- ALWAYS these arguments, so results are
     reproducible/comparable.
  3. One embedding module per experiment: create it once, embed every
     document in the corpus with that same module.
  4. Cache the VECTORS, not their distances (distances from vectors are
     instant) -- embeddings/{dataset}/fasttext_nbow_fsw{m}.npy

HW4 reuses m=500,1000,2000 from the HW3 cache and adds m=250.
"""
import time
import numpy as np

from cache_utils import emb_path, save_timing_increment, dist_dir


def make_fsw_module(d_in, d_out, seed=0):
    import torch
    torch.manual_seed(seed)
    from fswlib import FSWEmbedding
    return FSWEmbedding(d_in=d_in, d_out=d_out,
                         minimize_slice_coherence=True,
                         frequency_init="even")


def embed_corpus(module, X_list, W_list, batch_report_every=500):
    """Embed every document with the SAME module instance. Returns an
    (n_docs, m) float32 array. A plain loop, as HW3 allows."""
    import torch
    vecs = []
    t0 = time.perf_counter()
    for i, (X, W) in enumerate(zip(X_list, W_list)):
        xt = torch.from_numpy(X).float()
        wt = torch.from_numpy(W).float()
        with torch.no_grad():
            v = module(xt, wt)
        vecs.append(v.numpy())
        if (i + 1) % batch_report_every == 0:
            print(f"  embedded {i + 1}/{len(X_list)} docs "
                  f"({time.perf_counter() - t0:.1f}s so far)")
    elapsed = time.perf_counter() - t0
    return np.stack(vecs).astype(np.float32), elapsed


def embed_and_cache(dataset, embedding, weights, m, X_list, W_list,
                     d_in=300, seed=0, force=False):
    """Load cached FSW vectors for this (dataset, m) if present, else
    embed the whole corpus once and cache it. Returns (vectors, was_cached).
    """
    path = emb_path(dataset, embedding, weights, m)
    import os
    if os.path.exists(path) and not force:
        return np.load(path), True

    print(f"[{dataset}] embedding {len(X_list)} docs with FSW m={m} "
          f"(new -- not in cache)")
    module = make_fsw_module(d_in=d_in, d_out=m, seed=seed)
    vecs, elapsed = embed_corpus(module, X_list, W_list)
    np.save(path, vecs)

    # Record embedding cost in the same timing.json convention, filed
    # under a pseudo distance-folder so it's not lost.
    folder = dist_dir(dataset, embedding, weights, f"fsw{m}")
    save_timing_increment(folder, elapsed, len(X_list), num_workers=1)
    return vecs, False


def euclidean_distance_matrix(A, B=None):
    """Instant: Euclidean distances between rows of A (and B, or A vs
    itself if B is None). Used to turn cached FSW vectors into the
    'FSW distance matrix' HW4 §0.2 says needs no separate storage."""
    if B is None:
        B = A
    A2 = np.sum(A * A, axis=1)[:, None]
    B2 = np.sum(B * B, axis=1)[None, :]
    D2 = A2 + B2 - 2.0 * A @ B.T
    np.clip(D2, 0.0, None, out=D2)
    D = np.sqrt(D2)
    if B is A:  # symmetric: self-distances are exactly 0, not ~1e-3
        np.fill_diagonal(D, 0.0)
    return D
