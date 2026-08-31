"""
shortlist_rerank.py
--------------------
HW4b's combined classifier: use FSW (m=1000) to build a cheap shortlist
of candidates, then rerank only that shortlist by exact WMD-W2 (looked
up from the cached matrix -- no new exact distances are computed here,
per HW4b §0.3).

Shortlist rule (HW4b §0.1), for query q, neighbor count k, expansion
factor alpha:
    rho = k-th smallest FSW distance from q to the candidate pool
    S(q,k,alpha) = { x : FSW(q,x) <= alpha * rho }
    prediction = kNN vote over S using exact W2 distances, tie broken
                 by decreasing k (same rule as everywhere else in this
                 project -- reuses cv_knn._vote_with_tiebreak).

This module mirrors cv_knn.py's structure 1:1 (outer splits from TR/TE,
5-fold inner CV, k in {1..30}, smallest-k tie rule) so the two are easy
to compare side by side; the only difference is the classifier itself.
"""
import numpy as np
from sklearn.model_selection import KFold

from cv_knn import K_MAX, N_INNER_FOLDS, _vote_with_tiebreak, select_k_star


def shortlist_predict(fsw_row, w2_row, y_pool, k, alpha, order_fsw=None):
    """
    fsw_row, w2_row, y_pool: aligned 1-D arrays over the candidate pool.
    order_fsw: precomputed np.argsort(fsw_row); pass it in across the k
    loop to avoid re-sorting per k, per HW4b's "sort once" note.
    Returns (predicted_label, shortlist_size).
    """
    if order_fsw is None:
        order_fsw = np.argsort(fsw_row)

    rho = fsw_row[order_fsw[k - 1]]
    thresh = alpha * rho
    S = np.where(fsw_row <= thresh)[0]

    order_w2 = np.argsort(w2_row[S])
    sorted_labels = y_pool[S][order_w2]
    pred = _vote_with_tiebreak(np.arange(len(S)), sorted_labels, k)
    return pred, len(S)


def inner_cv_curve_combined(D_fsw_tt, D_w2_tt, y_train, alpha,
                             k_values=range(1, K_MAX + 1),
                             n_folds=N_INNER_FOLDS, random_state=0):
    """Same structure/semantics as cv_knn.inner_cv_curve: mean of
    per-fold mean errors, one curve value per k."""
    n = len(y_train)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    k_values = list(k_values)
    fold_errors = {k: [] for k in k_values}

    for fold_train_idx, fold_val_idx in kf.split(np.arange(n)):
        y_fold_train = y_train[fold_train_idx]
        y_fold_val = y_train[fold_val_idx]
        err_counts = {k: 0 for k in k_values}

        for qi, val_pos in enumerate(fold_val_idx):
            fsw_row = D_fsw_tt[val_pos, fold_train_idx]
            w2_row = D_w2_tt[val_pos, fold_train_idx]
            order_fsw = np.argsort(fsw_row)
            for k in k_values:
                pred, _ = shortlist_predict(fsw_row, w2_row, y_fold_train,
                                             k, alpha, order_fsw)
                if pred != y_fold_val[qi]:
                    err_counts[k] += 1

        for k in k_values:
            fold_errors[k].append(err_counts[k] / len(fold_val_idx))

    cv_errors = {k: float(np.mean(v)) for k, v in fold_errors.items()}
    return cv_errors, fold_errors


def evaluate_one_split_combined(D_fsw_full, D_w2_full, y_full, train_idx,
                                 test_idx, alpha, k_values=range(1, K_MAX + 1),
                                 random_state=0):
    D_fsw_tt = D_fsw_full[np.ix_(train_idx, train_idx)]
    D_w2_tt = D_w2_full[np.ix_(train_idx, train_idx)]
    y_train = y_full[train_idx]
    y_test = y_full[test_idx]

    cv_errors, _ = inner_cv_curve_combined(D_fsw_tt, D_w2_tt, y_train, alpha,
                                            k_values, random_state=random_state)
    k_star = select_k_star(cv_errors)

    D_fsw_te = D_fsw_full[np.ix_(test_idx, train_idx)]
    D_w2_te = D_w2_full[np.ix_(test_idx, train_idx)]

    preds, sizes = [], []
    for i in range(len(test_idx)):
        pred, s = shortlist_predict(D_fsw_te[i], D_w2_te[i], y_train, k_star, alpha)
        preds.append(pred)
        sizes.append(s)
    preds = np.array(preds)
    test_err = float(np.mean(preds != y_test))

    return {"k_star": k_star, "test_error": test_err, "cv_errors": cv_errors,
            "shortlist_sizes": np.array(sizes)}


def evaluate_all_splits_combined(D_fsw_full, D_w2_full, y_full, TR, TE, alpha,
                                  k_values=range(1, K_MAX + 1)):
    """Mirrors cv_knn.evaluate_all_splits, plus shortlist-size stats
    (HW4b deliverable 1's |S| mean/median/p95, pooled over all queries
    of all splits, per HW4's timing-table pooling convention)."""
    per_split = []
    for s, (train_idx, test_idx) in enumerate(zip(TR, TE)):
        res = evaluate_one_split_combined(D_fsw_full, D_w2_full, y_full,
                                           train_idx, test_idx, alpha,
                                           k_values=k_values, random_state=s)
        per_split.append(res)

    test_errors = np.array([r["test_error"] for r in per_split])
    all_sizes = np.concatenate([r["shortlist_sizes"] for r in per_split])

    return {
        "mean_test_error": float(test_errors.mean()),
        "std_test_error": float(test_errors.std(ddof=0)) if len(test_errors) > 1 else 0.0,
        "k_stars": [r["k_star"] for r in per_split],
        "shortlist_mean": float(all_sizes.mean()),
        "shortlist_median": float(np.median(all_sizes)),
        "shortlist_p95": float(np.percentile(all_sizes, 95)),
        "per_split": per_split,
        "n_splits": len(per_split),
    }


def neighbor_coverage(D_fsw_full, D_w2_full, alpha, k=10, n_query_sample=500, seed=0):
    """
    HW4b deliverable 2: fraction of the true top-k W2 neighbors contained
    in S(q,k,alpha), averaged over a sample of query documents, using the
    WHOLE pooled document pool as candidates (every other document; self
    excluded). Not tied to any one outer split -- HW4b doesn't specify
    which split to use for this diagnostic, so it's measured corpus-wide.
    """
    n = D_fsw_full.shape[0]
    rng = np.random.RandomState(seed)
    queries = rng.choice(n, size=min(n_query_sample, n), replace=False)

    fractions = []
    for q in queries:
        others = np.array([i for i in range(n) if i != q])
        fsw_row = D_fsw_full[q, others]
        w2_row = D_w2_full[q, others]

        order_fsw = np.argsort(fsw_row)
        rho = fsw_row[order_fsw[k - 1]]
        S = set(others[fsw_row <= alpha * rho])

        order_w2 = np.argsort(w2_row)
        true_topk = set(others[order_w2[:k]])

        fractions.append(len(true_topk & S) / k)

    return float(np.mean(fractions))
