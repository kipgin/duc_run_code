"""
cv_knn.py
---------
Recreates HW1 Task 3's kNN+CV scheme, updated to the "Addition to HW1"
protocol as specified in HW3/HW4:

  - outer splits: 5 for amazon and classic; 1 for reuters (its single
    official split) -- driven entirely by however many rows TR/TE have.
  - inside each outer split: 5-fold CV on the outer-training documents,
    for k in {1, ..., 30}.
  - k* = the k with lowest mean CV error; ties broken by taking the
    SMALLEST such k ("smallest-k tie rule").
  - with k*, classify the outer-test documents.
  - vote ties (among classes, for a given k) are broken by decreasing k
    (drop the farthest of the k neighbors and re-vote, repeating; at
    k=1 there is a single neighbor so the tie must resolve).
  - test error reported as mean +/- std over outer splits (a single
    number for reuters).

This module is distance-matrix agnostic: it only needs a full (N, N)
pairwise distance matrix over the dataset's document pool, and, for
each outer split, arrays of document indices for train/test drawn from
that same pool (exactly what data_io.load_dataset's TR/TE give).
"""
from collections import Counter
import numpy as np
from sklearn.model_selection import KFold

K_MAX = 30
N_INNER_FOLDS = 5


def _vote_with_tiebreak(order, y_train_sub, k):
    """order: neighbor positions (within y_train_sub) sorted by distance,
    ascending. Vote with the k nearest; on a tie among classes, drop to
    k-1 nearest and re-vote, down to k=1 (which is always decisive)."""
    cur_k = k
    while cur_k >= 1:
        labels = y_train_sub[order[:cur_k]]
        counts = Counter(labels.tolist())
        top = max(counts.values())
        winners = [c for c, cnt in counts.items() if cnt == top]
        if len(winners) == 1:
            return winners[0]
        cur_k -= 1
    # cur_k reached 0 only if k was 0; shouldn't happen since k>=1 always
    # resolves at cur_k==1 (single neighbor => single winner).
    return y_train_sub[order[0]]


def knn_predict_all_k(D_query_train, y_train, k_values):
    """D_query_train: (n_query, n_train) distances.
    Returns preds: dict k -> (n_query,) predicted labels array.
    Vectorized over queries, loop over k (k_values usually 1..30)."""
    order_full = np.argsort(D_query_train, axis=1)  # ascending, full order
    n_query = D_query_train.shape[0]
    preds = {k: np.empty(n_query, dtype=y_train.dtype) for k in k_values}
    for q in range(n_query):
        order_q = order_full[q]
        for k in k_values:
            preds[k][q] = _vote_with_tiebreak(order_q, y_train, k)
    return preds


def inner_cv_curve(D_train_train, y_train, k_values=range(1, K_MAX + 1),
                    n_folds=N_INNER_FOLDS, random_state=0):
    """5-fold CV on the outer-training set. Returns:
      cv_errors: dict k -> mean validation error across the n_folds
      per_fold_errors: dict k -> list of per-fold errors (for plotting
                        one curve per outer split isn't needed here --
                        this IS one outer split's curve, averaged over
                        its inner folds, as HW1's Task 3 step 3 defines
                        "the" CV curve for that split)
    """
    n = len(y_train)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    k_values = list(k_values)
    fold_errors = {k: [] for k in k_values}

    for fold_train_idx, fold_val_idx in kf.split(np.arange(n)):
        D_val_fold = D_train_train[np.ix_(fold_val_idx, fold_train_idx)]
        y_fold_train = y_train[fold_train_idx]
        y_fold_val = y_train[fold_val_idx]
        preds = knn_predict_all_k(D_val_fold, y_fold_train, k_values)
        for k in k_values:
            err = float(np.mean(preds[k] != y_fold_val))
            fold_errors[k].append(err)

    cv_errors = {k: float(np.mean(v)) for k, v in fold_errors.items()}
    return cv_errors, fold_errors


def select_k_star(cv_errors):
    """Smallest-k tie rule: among the k's achieving the minimum mean CV
    error, return the smallest."""
    best_err = min(cv_errors.values())
    tied = [k for k, e in cv_errors.items() if np.isclose(e, best_err)]
    return min(tied)


def evaluate_one_split(D_full, y_full, train_idx, test_idx,
                        k_values=range(1, K_MAX + 1), random_state=0):
    """One outer split, per HW1 Task 3 steps 1-5 (k range and tie rule
    per HW4 §0.1). D_full is the full (N, N) distance matrix over the
    dataset's whole document pool; train_idx/test_idx are index arrays
    into that pool for this split."""
    D_train_train = D_full[np.ix_(train_idx, train_idx)]
    y_train = y_full[train_idx]
    y_test = y_full[test_idx]

    cv_errors, fold_errors = inner_cv_curve(
        D_train_train, y_train, k_values, random_state=random_state)
    k_star = select_k_star(cv_errors)

    D_test_train = D_full[np.ix_(test_idx, train_idx)]
    preds = knn_predict_all_k(D_test_train, y_train, [k_star])
    test_err = float(np.mean(preds[k_star] != y_test))

    return {
        "k_star": k_star,
        "test_error": test_err,
        "cv_errors": cv_errors,   # k -> mean CV error, this split's curve
    }


def evaluate_all_splits(D_full, y_full, TR, TE,
                         k_values=range(1, K_MAX + 1)):
    """TR, TE: lists of index arrays, one per outer split (len 5 for
    amazon/classic, len 1 for reuters). Returns a results dict with
    per-split k*, test errors, CV curves, and the mean+-std summary."""
    per_split = []
    for s, (train_idx, test_idx) in enumerate(zip(TR, TE)):
        res = evaluate_one_split(D_full, y_full, train_idx, test_idx,
                                  k_values=k_values, random_state=s)
        per_split.append(res)

    test_errors = np.array([r["test_error"] for r in per_split])
    summary = {
        "mean_test_error": float(test_errors.mean()),
        "std_test_error": float(test_errors.std(ddof=0)) if len(test_errors) > 1 else 0.0,
        "k_stars": [r["k_star"] for r in per_split],
        "per_split": per_split,
        "n_splits": len(per_split),
    }
    return summary
