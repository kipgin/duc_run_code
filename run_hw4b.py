"""
run_hw4b.py
-----------
HW4b: FSW-1000 shortlist + exact WMD-W2 rerank. Reuses run_hw4.py's
dataset loading (same .mat paths, same fastText-subset config you
already filled in) and the same cache (cache_utils, wmd_distances,
fsw_embed), so nothing about your data/cache setup needs to be
duplicated or re-configured here.

Requires, per dataset (all already produced by run_hw4.py):
  - the cached exact-W2 matrix
  - the cached FSW-1000 vectors
  - results/main_table_{dataset}.csv   (for the two pure-method reference rows)
  - results/time_table_{dataset}.csv   (for the estimated combined time)

Run AFTER run_hw4.py has completed for the datasets you want:
    python run_hw4b.py --dataset amazon
    python run_hw4b.py --dataset classic
    python run_hw4b.py --dataset reuters
    python run_hw4b.py --dataset all

Output:
    results/distortion_stats_{dataset}.json   (fresh FSW-1000/W2 ratio stats)
    results/alpha_values_{dataset}.json       (the 4 alpha values)
    results/hw4b_main_table_{dataset}.csv     (deliverable 1 + shortlist-size
                                                stats + coverage + time estimate)
"""
import os
import sys
import csv
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_hw4 import get_dataset, EMBEDDING, WEIGHTS, RESULTS_DIR  # noqa: E402
from wmd_distances import load_or_compute  # noqa: E402
from fsw_embed import embed_and_cache, euclidean_distance_matrix  # noqa: E402
from cv_knn import K_MAX  # noqa: E402
from distortion_stats import sample_ratio_stats, alpha_values  # noqa: E402
from shortlist_rerank import evaluate_all_splits_combined, neighbor_coverage  # noqa: E402

M_SHORTLIST = 1000


def compute_alphas(dataset, V_fsw, D_w2):
    stats = sample_ratio_stats(V_fsw, D_w2, n_pairs=5000, seed=0)
    alphas = alpha_values(stats)
    with open(os.path.join(RESULTS_DIR, f"distortion_stats_{dataset}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(RESULTS_DIR, f"alpha_values_{dataset}.json"), "w") as f:
        json.dump(alphas, f, indent=2)
    return alphas


def _read_csv_dict(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def load_reference_row(dataset, distance_name):
    rows = _read_csv_dict(os.path.join(RESULTS_DIR, f"main_table_{dataset}.csv"))
    if rows is None:
        return None
    for r in rows:
        if r["distance"] == distance_name:
            return r
    return None


def load_timing_row(dataset, row_name):
    rows = _read_csv_dict(os.path.join(RESULTS_DIR, f"time_table_{dataset}.csv"))
    if rows is None:
        return None
    for r in rows:
        if r["row"] == row_name:
            return r
    return None


def estimate_combined_time(dataset, TR, shortlist_mean):
    """t_combined ~= t_FSW_query + |S| * (t_WMD_query / n), per HW4b §0.4.
    n (lookup-corpus size) isn't a column in time_table_*.csv, so it's
    recomputed here as the average outer-train size across TR, matching
    how the pooled WMD/FSW timings in time_table were themselves pooled
    across the same splits."""
    wmd_row = load_timing_row(dataset, "WMD-W1")
    fsw_row = load_timing_row(dataset, f"FSW-{M_SHORTLIST}")
    if wmd_row is None or fsw_row is None:
        return None
    n_lookup = float(np.mean([len(idx) for idx in TR]))
    t_per_solve = float(wmd_row["per_query_mean_s"]) / max(n_lookup, 1.0)
    t_fsw_query = float(fsw_row["per_query_mean_s"])
    return t_fsw_query + shortlist_mean * t_per_solve


def run_dataset(dataset):
    print(f"\n===== HW4b: {dataset} =====")
    data = get_dataset(dataset)
    X, W, y, TR, TE = data["X"], data["W"], data["y"], data["TR"], data["TE"]

    D_w2 = load_or_compute(dataset, EMBEDDING, WEIGHTS, "w2", X, W)
    V_fsw, was_cached = embed_and_cache(dataset, EMBEDDING, WEIGHTS, M_SHORTLIST, X, W)
    print(f"[{dataset}] FSW-{M_SHORTLIST} vectors {'(from cache)' if was_cached else '(freshly embedded)'}: "
          f"{V_fsw.shape}")
    D_fsw = euclidean_distance_matrix(V_fsw)

    alphas = compute_alphas(dataset, V_fsw, D_w2)
    print(f"[{dataset}] alpha values: " +
          ", ".join(f"{k}={v:.3f}" for k, v in alphas.items()))

    rows_out = []

    ref_fsw = load_reference_row(dataset, f"FSW-{M_SHORTLIST}")
    if ref_fsw:
        fsw_timing = load_timing_row(dataset, f"FSW-{M_SHORTLIST}")
        rows_out.append({
            "row": f"FSW-{M_SHORTLIST} (pure)", "alpha": "",
            "mean_test_error": ref_fsw["mean_test_error"],
            "std_test_error": ref_fsw["std_test_error"],
            "k_stars_per_split": ref_fsw["k_stars_per_split"],
            "shortlist_mean": "", "shortlist_median": "", "shortlist_p95": "",
            "coverage_top10": "",
            "est_time_s": fsw_timing["per_query_mean_s"] if fsw_timing else "",
        })
    else:
        print(f"  [warn] no FSW-{M_SHORTLIST} row in main_table_{dataset}.csv "
              f"-- run run_hw4.py first for the reference row.")

    for label, alpha in alphas.items():
        res = evaluate_all_splits_combined(D_fsw, D_w2, y, TR, TE, alpha,
                                            k_values=range(1, K_MAX + 1))
        est_time = estimate_combined_time(dataset, TR, res["shortlist_mean"])
        cov = neighbor_coverage(D_fsw, D_w2, alpha, k=10)

        rows_out.append({
            "row": f"combined (alpha={label})", "alpha": f"{alpha:.4f}",
            "mean_test_error": res["mean_test_error"],
            "std_test_error": res["std_test_error"],
            "k_stars_per_split": ";".join(str(k) for k in res["k_stars"]),
            "shortlist_mean": res["shortlist_mean"],
            "shortlist_median": res["shortlist_median"],
            "shortlist_p95": res["shortlist_p95"],
            "coverage_top10": cov,
            "est_time_s": est_time,
        })
        print(f"[{dataset}] alpha={label} ({alpha:.3f}): "
              f"err={res['mean_test_error']*100:.2f}% +/- {res['std_test_error']*100:.2f}%, "
              f"|S| mean={res['shortlist_mean']:.1f} (median {res['shortlist_median']:.1f}, "
              f"p95 {res['shortlist_p95']:.1f}), coverage@10={cov*100:.1f}%, "
              f"est_time={est_time}")

    ref_w2 = load_reference_row(dataset, "WMD-W2")
    if ref_w2:
        wmd_timing = load_timing_row(dataset, "WMD-W1")
        rows_out.append({
            "row": "WMD-W2 (pure)", "alpha": "",
            "mean_test_error": ref_w2["mean_test_error"],
            "std_test_error": ref_w2["std_test_error"],
            "k_stars_per_split": ref_w2["k_stars_per_split"],
            "shortlist_mean": "", "shortlist_median": "", "shortlist_p95": "",
            "coverage_top10": "",
            "est_time_s": wmd_timing["per_query_mean_s"] if wmd_timing else "",
        })
    else:
        print(f"  [warn] no WMD-W2 row in main_table_{dataset}.csv "
              f"-- run run_hw4.py first for the reference row.")

    out_path = os.path.join(RESULTS_DIR, f"hw4b_main_table_{dataset}.csv")
    fieldnames = ["row", "alpha", "mean_test_error", "std_test_error",
                  "k_stars_per_split", "shortlist_mean", "shortlist_median",
                  "shortlist_p95", "coverage_top10", "est_time_s"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[{dataset}] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["amazon", "classic", "reuters", "all"],
                     default="all")
    args = ap.parse_args()
    datasets = ["amazon", "classic", "reuters"] if args.dataset == "all" else [args.dataset]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for d in datasets:
        run_dataset(d)


if __name__ == "__main__":
    main()
