"""
distortion_stats.py
--------------------
Recreates the FSW/W2 ratio statistics from HW3 Task 2/3 (r = FSW(mu,nu) /
W2(mu,nu), sampled over pairs), needed by HW4b §0.2 to build the four
alpha values -- computed fresh from the cached FSW-1000 vectors and the
cached exact-W2 matrix rather than assuming your original HW3 numbers
are still lying around somewhere.

If you DO still have your original HW3 Task 3 distortion numbers for
m=1000, prefer those instead (edit run_hw4b.py's compute_alphas to load
them) -- what's computed here is a faithful re-derivation, not a
replacement, in case they differ due to sampling noise.
"""
import numpy as np


def sample_ratio_stats(V_fsw, D_w2_full, n_pairs=5000, seed=0):
    """
    V_fsw: (n, m) FSW vectors (m=1000 for HW4b).
    D_w2_full: (n, n) exact W2 distance matrix over the same document pool
               (same indexing as V_fsw's rows).
    Returns a dict with min/mean/std/median/p2/p5/p95/p98/max of r, plus
    the three distortion estimates (max/min, P98/P2, P95/P5).
    """
    n = V_fsw.shape[0]
    rng = np.random.RandomState(seed)

    max_possible = n * (n - 1) // 2
    n_pairs = min(n_pairs, max_possible)
    pairs = set()
    while len(pairs) < n_pairs:
        i, j = rng.randint(0, n, size=2)
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    pairs = np.array(list(pairs))
    ii, jj = pairs[:, 0], pairs[:, 1]

    w2 = D_w2_full[ii, jj]
    diff = V_fsw[ii] - V_fsw[jj]
    fsw = np.sqrt(np.sum(diff * diff, axis=1))

    valid = w2 > 1e-12
    r = fsw[valid] / w2[valid]

    stats = {
        "n_pairs": int(r.size),
        "min": float(r.min()), "max": float(r.max()),
        "mean": float(r.mean()), "std": float(r.std()),
        "median": float(np.median(r)),
        "p2": float(np.percentile(r, 2)), "p5": float(np.percentile(r, 5)),
        "p95": float(np.percentile(r, 95)), "p98": float(np.percentile(r, 98)),
    }
    stats["distortion_maxmin"] = stats["max"] / stats["min"]
    stats["distortion_98_2"] = stats["p98"] / stats["p2"]
    stats["distortion_95_5"] = stats["p95"] / stats["p5"]
    return stats


def alpha_values(stats):
    """The four alpha values required by HW4b §0.3, keyed by label."""
    maxmin = stats["distortion_maxmin"]
    p98p2 = stats["distortion_98_2"]
    return {
        "sqrt(P98/P2)": float(np.sqrt(p98p2)),
        "P98/P2": float(p98p2),
        "rmax/rmin": float(maxmin),
        "2*rmax/rmin": float(2 * maxmin),
    }
