"""
wmd_distances.py
-----------------
Exact optimal-transport (WMD) distances, recreating HW1 Task 2 and the
"Addition to HW1" resumable / progress-bar / cache-layout requirement
referenced throughout HW3-HW4.

W1: cost C_kl = ||x_k - x_l||_2               -> ot.emd2(d, d', C)
W2: cost C_kl = ||x_k - x_l||_2^2, sqrt at end -> sqrt(ot.emd2(d, d', C^2))

Matrices are symmetric with zero diagonal for train-train; only i<j is
computed and mirrored. Computation is resumable: progress is checkpointed
to disk periodically so a killed/interrupted job picks up where it left
off, and timing.json accumulates compute cost across resumptions (never
overwritten), per HW3 §0.7.

NOTE: POT (ot) eagerly imports torch unless told otherwise; in spawned
multiprocessing workers a concurrent torch DLL load exhausts the page file
(WinError 1455). These env vars force POT onto its pure-numpy backend and
must be set *before* the first `import ot` (here, at module import time) so
they take effect in the parent and every spawned worker.
"""
import os

os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_JAX", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_CUPY", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")
import time
import numpy as np
from tqdm import tqdm

from cache_utils import dist_dir, write_config, save_timing_increment


def _cost_matrix(Xi, Xj, squared):
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b ; avoids materialising the
    # (n_i, n_j, d) difference tensor, which OOMs on large documents
    # (d=300 makes that tensor ~d x bigger than the (n_i, n_j) result).
    a2 = np.sum(Xi * Xi, axis=1)[:, None]          # (n_i, 1)
    b2 = np.sum(Xj * Xj, axis=1)[None, :]          # (1, n_j)
    C = a2 + b2 - 2.0 * (Xi @ Xj.T)                # (n_i, n_j)
    np.clip(C, 0.0, None, out=C)
    if not squared:
        np.sqrt(C, out=C)
    return C


def _ot_distance(Wi, Wj, C):
    import ot
    return ot.emd2(Wi.astype(np.float64), Wj.astype(np.float64),
                    C.astype(np.float64))


def _pair_distance(Xi, Wi, Xj, Wj, metric):
    if metric == "w1":
        C = _cost_matrix(Xi, Xj, squared=False)
        return _ot_distance(Wi, Wj, C)
    elif metric == "w2":
        C = _cost_matrix(Xi, Xj, squared=True)
        val = _ot_distance(Wi, Wj, C)
        return float(np.sqrt(max(val, 0.0)))
    else:
        raise ValueError(metric)


# ---- multiprocessing workers (module-level so they pickle) -------------
_WORKER = {}  # per-process global holding the document lists


def _worker_init(X_list_i, W_list_i, X_list_j, W_list_j, metric):
    _WORKER["Xi"] = X_list_i
    _WORKER["Wi"] = W_list_i
    _WORKER["Xj"] = X_list_j
    _WORKER["Wj"] = W_list_j
    _WORKER["metric"] = metric


def _worker_pair(ab):
    a, b = ab
    return (a, b, _pair_distance(_WORKER["Xi"][a], _WORKER["Wi"][a],
                                 _WORKER["Xj"][b], _WORKER["Wj"][b],
                                 _WORKER["metric"]))


def _apply_results(results, D, done_mask, symmetric, n_new_list):
    for a, b, val in results:
        D[a, b] = val
        if symmetric:
            D[b, a] = val
        done_mask[a, b] = True
        n_new_list[0] += 1


def compute_matrix(X_list_i, W_list_i, X_list_j, W_list_j, metric,
                    dataset, embedding, weights, symmetric,
                    checkpoint_every=200, num_workers=1):
    """Compute (or resume) a distance matrix and store it under the
    HW3 cache layout: distances/{dataset}/{embedding}_{weights}_{metric}/

    symmetric=True  -> square train-train matrix (X_list_j is X_list_i);
                        only i<j entries are computed, then mirrored.
    symmetric=False -> rectangular test-train matrix.

    num_workers>1 parallelizes the (dominant, W1) pair computations across
    worker processes while keeping the resumable checkpoint discipline:
    progress is saved to _checkpoint.npz + timing.json after every batch,
    so a killed job picks up where it left off.
    """
    folder = dist_dir(dataset, embedding, weights, metric)
    write_config(folder, dataset, embedding, weights, metric,
                 extra={"symmetric": symmetric})
    w_path = os.path.join(folder, "w.npy")
    ckpt_path = os.path.join(folder, "_checkpoint.npz")

    n_i, n_j = len(X_list_i), len(X_list_j)

    if os.path.exists(w_path) and not os.path.exists(ckpt_path):
        print(f"[{folder}] already complete, loading from disk.")
        return np.load(w_path)

    if os.path.exists(ckpt_path):
        with np.load(ckpt_path) as ck:  # close file handle so it can be removed
            D = np.array(ck["D"])
            done_mask = np.array(ck["done_mask"])
        print(f"[{folder}] resuming: "
              f"{done_mask.sum()}/{done_mask.size} pairs already computed.")
    else:
        D = np.zeros((n_i, n_j), dtype=np.float64)
        done_mask = np.zeros((n_i, n_j), dtype=bool)

    if symmetric:
        pending = [(a, b) for a in range(n_i) for b in range(a + 1, n_j)
                   if not done_mask[a, b]]
    else:
        pending = [(a, b) for a in range(n_i) for b in range(n_j)
                   if not done_mask[a, b]]

    if not pending:
        return _finalize(D, symmetric, w_path, ckpt_path)

    if num_workers > 1 and len(pending) > 1:
        _parallel_compute(pending, D, done_mask, symmetric, num_workers,
                          X_list_i, W_list_i, X_list_j, W_list_j, metric,
                          dataset, embedding, weights, folder, ckpt_path)
    else:
        _serial_compute(pending, D, done_mask, symmetric,
                        X_list_i, W_list_i, X_list_j, W_list_j, metric,
                        dataset, embedding, weights, folder, ckpt_path)

    return _finalize(D, symmetric, w_path, ckpt_path)


def _finalize(D, symmetric, w_path, ckpt_path):
    if symmetric:
        np.fill_diagonal(D, 0.0)
    np.save(w_path, D)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"done, saved w.npy with shape {D.shape}")
    return D


def _serial_compute(pending, D, done_mask, symmetric,
                    X_list_i, W_list_i, X_list_j, W_list_j, metric,
                    dataset, embedding, weights, folder, ckpt_path):
    n_new = 0
    t_start = time.perf_counter()
    last_ckpt = t_start
    for a, b in tqdm(pending, desc=f"{metric} {dataset}"):
        val = _pair_distance(X_list_i[a], W_list_i[a],
                             X_list_j[b], W_list_j[b], metric)
        D[a, b] = val
        if symmetric:
            D[b, a] = val
        done_mask[a, b] = True
        n_new += 1
        now = time.perf_counter()
        if now - last_ckpt > 60:  # checkpoint at least once a minute
            np.savez(ckpt_path, D=D, done_mask=done_mask)
            save_timing_increment(folder, now - t_start, n_new, 1)
            t_start = now
            n_new = 0
            last_ckpt = now
    if n_new > 0:
        save_timing_increment(folder, time.perf_counter() - t_start,
                              n_new, 1)


def _parallel_compute(pending, D, done_mask, symmetric, num_workers,
                      X_list_i, W_list_i, X_list_j, W_list_j, metric,
                      dataset, embedding, weights, folder, ckpt_path):
    from multiprocessing import Pool

    # Checkpoint every few minutes, not every few seconds: writing the full
    # (N,N) _checkpoint.npz too often becomes the bottleneck. A batch is
    # ~num_workers*60 pairs (≈ a minute of work on W1), minimum 5000.
    per_batch = max(num_workers * 60, 5000)
    batches = [pending[i:i + per_batch]
               for i in range(0, len(pending), per_batch)]

    n_new = 0
    t_start = time.perf_counter()
    pool = Pool(num_workers, _worker_init,
                (X_list_i, W_list_i, X_list_j, W_list_j, metric))
    try:
        for bi, batch in enumerate(
                tqdm(batches, desc=f"{metric} {dataset} (p={num_workers})")):
            results = pool.map(_worker_pair, batch)
            for a, b, val in results:
                D[a, b] = val
                if symmetric:
                    D[b, a] = val
                done_mask[a, b] = True
                n_new += 1
            # checkpoint after each batch -> resumable at batch granularity
            np.savez(ckpt_path, D=D, done_mask=done_mask)
            elapsed = time.perf_counter() - t_start
            save_timing_increment(folder, elapsed, n_new, num_workers)
            t_start = time.perf_counter()
            n_new = 0
    finally:
        pool.close()
        pool.join()



def default_num_workers():
    """Number of workers to use for WMD computation. Honour an explicit
    env override, else default to the machine's CPU count (cap at 10 to
    stay gentle on RAM when POT/numba loads in each worker)."""
    v = os.environ.get("WMD_NUM_WORKERS")
    if v:
        return int(v)
    try:
        return min(os.cpu_count() or 1, 4)
    except TypeError:
        return 1


def load_or_compute(dataset, embedding, weights, metric,
                     X_train, W_train, X_test=None, W_test=None,
                     num_workers=None):
    """Convenience wrapper matching HW4 §0.2: 'your caches already hold
    the W2 matrices ... any missing W1 matrix is computed the usual
    way'. Tries to load train-train and test-train matrices; computes
    whichever is missing.
    """
    if num_workers is None:
        num_workers = default_num_workers()
    folder = dist_dir(dataset, embedding, weights, metric)
    w_path = os.path.join(folder, "w.npy")
    if os.path.exists(w_path) and not os.path.exists(
            os.path.join(folder, "_checkpoint.npz")):
        return np.load(w_path)

    if X_test is None:
        return compute_matrix(X_train, W_train, X_train, W_train, metric,
                               dataset, embedding, weights, symmetric=True,
                               num_workers=num_workers)
    else:
        return compute_matrix(X_test, W_test, X_train, W_train, metric,
                               dataset, embedding, weights, symmetric=False,
                               num_workers=num_workers)
