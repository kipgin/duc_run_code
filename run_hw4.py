"""
run_hw4.py
----------
Ties data_io + wmd_distances + fsw_embed + cv_knn + timing_utils
together into the HW4 deliverables:

  1. main table (results/main_table_{dataset}.csv)
  2. CV curves figure (figures/cv_curves_{dataset}.png)
  3. error-vs-m plot (figures/error_vs_m_{dataset}.png)
  4. time table (results/time_table_{dataset}.csv)

EDIT THE CONFIG BLOCK BELOW to point at your real .mat files and your
fastText-subset loader, then run:

    python run_hw4.py --dataset amazon
    python run_hw4.py --dataset classic
    python run_hw4.py --dataset reuters
    python run_hw4.py --dataset all

Expected runtime: dominated by any missing exact-W1 matrices (minutes
to hours depending on corpus size -- resumable, see wmd_distances.py)
and by embedding FSW-250 (minutes, per HW4 §0.2). Everything else
(loading cached matrices/vectors, kNN+CV, timing) is fast.
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Default the cache root to a workspace-local folder unless the user has
# set WMD_CACHE_ROOT to point at an existing HW3 cache.
if not os.environ.get("WMD_CACHE_ROOT"):
    os.environ["WMD_CACHE_ROOT"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_cache")

from cache_utils import CACHE_ROOT
from wmd_distances import load_or_compute, compute_matrix

# Light module-scope values used by multiple functions.
EMBEDDING = "fasttext"
WEIGHTS = "nbow"
M_VALUES = [250, 500, 1000, 2000]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "figures")

# ---------------------------------------------------------------------
# CONFIG -- fill these in for your environment.
# ---------------------------------------------------------------------
_DATA_RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "raw")
# By default we run on the compact 1000-doc subsets (quick validation).
# Pass --full to use the full-size datasets instead.
_SUB_DATASET_MAT_PATHS = {
    "amazon": os.path.join(_DATA_RAW, "amazon_sub.mat"),
    "classic": os.path.join(_DATA_RAW, "classic_sub.mat"),
    "reuters": os.path.join(_DATA_RAW, "reuters_sub.mat"),
}
_FULL_DATASET_MAT_PATHS = {
    "amazon": os.path.join(_DATA_RAW, "amazon-emd_tr_te_split.mat"),
    "classic": os.path.join(_DATA_RAW, "classic-emd_tr_te_split.mat"),
    "reuters": os.path.join(_DATA_RAW, "r8-emd_tr_te3.mat"),
}
DATASET_MAT_PATHS = _SUB_DATASET_MAT_PATHS

def _lazy(name):
    """Import heavy modules on first use instead of at module top, so the
    multiprocessing WMD workers (which re-import this __main__ module as
    __mp_main__) stay light."""
    import importlib
    return importlib.import_module(name)


_FASTTEXT_MODEL_PATH = os.path.join(
    _DATA_RAW, "fasttext", "fasttext-crawl-subwords-300.model")


def fasttext_full_loader():
    """Loads the FULL fastText crawl-300d-2M-subword vectors (2M words,
    trained on Common Crawl, 600B tokens) as a dict-like {word: vector}.
    Downloaded from huggingface.co/fse/fasttext-crawl-subwords-300, which
    is gensim's native-format re-hosting of fastText's crawl-300d-2M-subword
    model -- same word list as crawl-300d-2M.vec."""
    from gensim.models import KeyedVectors
    return KeyedVectors.load(_FASTTEXT_MODEL_PATH)
# ---------------------------------------------------------------------


def get_dataset(dataset):
    remove_stopwords = True  # amazon/classic/reuters all remove stop words
    data_io = _lazy("data_io")
    return data_io.load_dataset(
        mat_path=DATASET_MAT_PATHS[dataset],
        embedding_name=EMBEDDING,
        full_embedding_loader=fasttext_full_loader,
        cache_dir=CACHE_ROOT,
        dataset_name=dataset,
        remove_stopwords=remove_stopwords,
    )


def run_main_table(dataset, data):
    from fsw_embed import embed_and_cache, euclidean_distance_matrix
    from cv_knn import evaluate_all_splits, K_MAX
    X, W, y, TR, TE = data["X"], data["W"], data["y"], data["TR"], data["TE"]

    D_w1 = load_or_compute(dataset, EMBEDDING, WEIGHTS, "w1", X, W)
    D_w2 = load_or_compute(dataset, EMBEDDING, WEIGHTS, "w2", X, W)

    fsw_vectors = {}
    for m in M_VALUES:
        vecs, cached = embed_and_cache(dataset, EMBEDDING, WEIGHTS, m, X, W)
        fsw_vectors[m] = vecs

    rows = {}
    cv_curves = {}

    for name, D in [("WMD-W1", D_w1), ("WMD-W2", D_w2)]:
        res = evaluate_all_splits(D, y, TR, TE, k_values=range(1, K_MAX + 1))
        rows[name] = res
        cv_curves[name] = res

    for m in M_VALUES:
        D_fsw = euclidean_distance_matrix(fsw_vectors[m])
        res = evaluate_all_splits(D_fsw, y, TR, TE, k_values=range(1, K_MAX + 1))
        rows[f"FSW-{m}"] = res
        cv_curves[f"FSW-{m}"] = res

    # ---- write main table CSV ----
    csv_path = os.path.join(RESULTS_DIR, f"main_table_{dataset}.csv")
    with open(csv_path, "w") as f:
        f.write("distance,mean_test_error,std_test_error,k_stars_per_split\n")
        for name, res in rows.items():
            k_stars = ";".join(str(k) for k in res["k_stars"])
            f.write(f"{name},{res['mean_test_error']:.6f},"
                    f"{res['std_test_error']:.6f},{k_stars}\n")
    print(f"[{dataset}] wrote {csv_path}")

    # ---- full per-split results as JSON (feeds figures + report) ----
    json_path = os.path.join(RESULTS_DIR, f"full_results_{dataset}.json")
    serializable = {
        name: {
            "mean_test_error": res["mean_test_error"],
            "std_test_error": res["std_test_error"],
            "k_stars": res["k_stars"],
            "per_split_cv_errors": [r["cv_errors"] for r in res["per_split"]],
            "per_split_test_error": [r["test_error"] for r in res["per_split"]],
        }
        for name, res in rows.items()
    }
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[{dataset}] wrote {json_path}")

    return rows, fsw_vectors


def plot_cv_curves(dataset, rows):
    """Deliverable 2: one figure per dataset, one panel per distance
    row, CV error vs k, one curve per outer split."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(rows.keys())
    n = len(names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)
    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        res = rows[name]
        for s, split_res in enumerate(res["per_split"]):
            ks = sorted(split_res["cv_errors"].keys())
            errs = [split_res["cv_errors"][k] for k in ks]
            ax.plot(ks, errs, label=f"split {s}")
        ax.set_title(name)
        ax.set_xlabel("k")
        ax.set_ylabel("CV error")
        if len(res["per_split"]) <= 6:
            ax.legend(fontsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"CV curves — {dataset}")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"cv_curves_{dataset}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[{dataset}] wrote {path}")


def plot_error_vs_m(dataset, rows):
    """Deliverable 3: test error (mean +- std) vs m, with WMD-W1 and
    WMD-W2 as horizontal reference lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    means = [rows[f"FSW-{m}"]["mean_test_error"] for m in M_VALUES]
    stds = [rows[f"FSW-{m}"]["std_test_error"] for m in M_VALUES]
    ax.errorbar(M_VALUES, means, yerr=stds, marker="o", label="FSW")
    ax.axhline(rows["WMD-W1"]["mean_test_error"], color="tab:red",
               linestyle="--", label="WMD-W1")
    ax.axhline(rows["WMD-W2"]["mean_test_error"], color="tab:green",
               linestyle="--", label="WMD-W2")
    ax.set_xlabel("m (FSW output dimension)")
    ax.set_ylabel("test error")
    ax.set_title(f"Error vs m — {dataset}")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"error_vs_m_{dataset}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[{dataset}] wrote {path}")


def run_time_table(dataset, data):
    """Deliverable 4. Uses the FIRST outer split's train/test as the
    lookup corpus / query pool, pooled over splits as specified."""
    from timing_utils import (time_wmd_per_query, time_fsw, pool_over_splits)
    X, W, TR, TE = data["X"], data["W"], data["TR"], data["TE"]

    wmd_times_per_split = []
    fsw_times_per_split = {m: [] for m in M_VALUES}
    fsw_preembed_per_split = {m: [] for m in M_VALUES}

    for train_idx, test_idx in zip(TR, TE):
        X_train = [X[i] for i in train_idx]
        W_train = [W[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        W_test = [W[i] for i in test_idx]

        wmd_times_per_split.append(
            time_wmd_per_query(X_train, W_train, X_test, W_test))

        for m in M_VALUES:
            pre_t, per_q = time_fsw(X_train, W_train, X_test, W_test, m)
            fsw_preembed_per_split[m].append(pre_t)
            fsw_times_per_split[m].append(per_q)

    wmd_stats = pool_over_splits(wmd_times_per_split)
    fsw_stats = {}
    for m in M_VALUES:
        pooled = pool_over_splits(fsw_times_per_split[m])
        pooled["pre_embedding_seconds_mean"] = float(
            np.mean(fsw_preembed_per_split[m]))
        pooled["pre_embedding_seconds_std"] = float(
            np.std(fsw_preembed_per_split[m]))
        fsw_stats[m] = pooled

    csv_path = os.path.join(RESULTS_DIR, f"time_table_{dataset}.csv")
    with open(csv_path, "w") as f:
        f.write("row,pre_embedding_mean_s,pre_embedding_std_s,"
                "per_query_mean_s,per_query_std_s,n_queries,machine\n")
        f.write(f"WMD-W1,,,{wmd_stats['mean_seconds']:.6f},"
                f"{wmd_stats['std_seconds']:.6f},{wmd_stats['n_queries']},"
                f"{wmd_stats['machine']}\n")
        for m in M_VALUES:
            s = fsw_stats[m]
            f.write(f"FSW-{m},{s['pre_embedding_seconds_mean']:.6f},"
                    f"{s['pre_embedding_seconds_std']:.6f},"
                    f"{s['mean_seconds']:.6f},{s['std_seconds']:.6f},"
                    f"{s['n_queries']},{s['machine']}\n")
    print(f"[{dataset}] wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["amazon", "classic",
                                               "reuters", "all"],
                         default="all")
    parser.add_argument("--skip-timing", action="store_true",
                         help="skip the (slow) timing measurements")
    parser.add_argument("--full", action="store_true",
                         help="use the full-size datasets instead of the "
                              "1000-doc subsets")
    args = parser.parse_args()

    global DATASET_MAT_PATHS
    if args.full:
        DATASET_MAT_PATHS = _FULL_DATASET_MAT_PATHS
        print("Using FULL-size datasets (WMD will take a very long time).")
    else:
        print("Using 1000-doc subsets (pass --full for full-size data).")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    if args.dataset == "all":
        datasets = [d for d in ["amazon", "classic", "reuters"]
                    if os.path.exists(DATASET_MAT_PATHS[d])]
        if not datasets:
            print("No datasets found!")
            return
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        print(f"\n===== {dataset} =====")
        data = get_dataset(dataset)
        rows, _ = run_main_table(dataset, data)
        plot_cv_curves(dataset, rows)
        plot_error_vs_m(dataset, rows)
        if not args.skip_timing:
            run_time_table(dataset, data)


if __name__ == "__main__":
    main()
