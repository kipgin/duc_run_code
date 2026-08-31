"""
make_subsets.py
---------------
Builds small per-dataset subset .mat files (first N docs of the pool) in
the same schema as the full "split" files (words/BOW_X/X/Y/TR/TE),
remapping each outer split's TR/TE 1-indexed doc indices into the
subset's index space. Lets the unmodified run_hw4 pipeline run on a
tractable number of docs (WMD time scales with N^2).

Usage:
    python make_subsets.py --n 1000
    python make_subsets.py --n 1000 --keep-original-indexes  # map back later
"""
import os
import argparse
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")

# dataset name -> (source file, mat key prefix style: 'split')
SOURCES = {
    "amazon": "amazon-emd_tr_te_split.mat",
    "classic": "classic-emd_tr_te_split.mat",
    "reuters": "r8-emd_tr_te3.mat",
}


def load_pool(path):
    """Returns (words, bow, X, Y, TR, TE) with TR/TE 1-indexed, using the
    same 'split' schema the pipeline expects."""
    m = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    if "words" in m:
        return (m["words"], m["BOW_X"], m["X"], m["Y"], m["TR"], m["TE"])
    elif "words_tr" in m:
        w_tr = m["words_tr"]
        w_te = m["words_te"]
        words = np.concatenate([w_tr, w_te], axis=1)

        b_tr = m["BOW_xtr"]
        b_te = m["BOW_xte"]
        bow = np.concatenate([b_tr, b_te], axis=1)

        x_tr = m["xtr"]
        x_te = m["xte"]
        X = np.concatenate([x_tr, x_te], axis=1)

        y_tr = m["ytr"]
        y_te = m["yte"]
        if y_tr.ndim == 1:
            y_tr = y_tr[None, :]
        if y_te.ndim == 1:
            y_te = y_te[None, :]
        Y = np.concatenate([y_tr, y_te], axis=1)

        n_tr = w_tr.shape[1]
        n_te = w_te.shape[1]
        TR = np.arange(1, n_tr + 1)[None, :]
        TE = np.arange(n_tr + 1, n_tr + n_te + 1)[None, :]
        return (words, bow, X, Y, TR, TE)
    else:
        raise KeyError(f"Unknown format in {path}: {list(m.keys())}")


def subset(path, n, out_path):
    words, bow, X, Y, TR, TE = load_pool(path)
    N = words.shape[1]
    n = min(n, N)
    # Evenly space the selected docs across the whole pool. This keeps
    # coverage from every region (e.g. reuters's single split has all
    # train docs first and all test docs last, so first-N would leave the
    # test set empty). amazon/classic also stay well-covered.
    sel = np.unique(np.round(np.linspace(0, N - 1, n)).astype(int))
    n = len(sel)

    sw = np.empty((1, n), dtype=object)
    sb = np.empty((1, n), dtype=object)
    sx = np.empty((1, n), dtype=object)
    for i, j in enumerate(sel):
        sw[0, i] = words[0, j]
        sb[0, i] = bow[0, j]
        sx[0, i] = X[0, j]
    sy = Y[:, sel]

    # remap TR/TE: keep docs that are in the subset, 1-indexed -> 0-indexed
    # -> subset position (0..n-1) -> 1-indexed
    pos = -np.ones(N, dtype=int)
    pos[sel] = np.arange(n)

    def remap(M):
        M0 = np.atleast_2d(np.asarray(M)).astype(int) - 1  # 0-indexed
        out = []
        for row in M0:
            mapped = pos[row]
            out.append((mapped[mapped >= 0] + 1).astype(np.uint16))
        # object array of shape (n_splits,) -- one 1-D row per split
        s = np.empty(len(out), dtype=object)
        s[:] = out
        return s

    sTR = remap(TR)
    sTE = remap(TE)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sio.savemat(out_path, {"words": sw, "BOW_X": sb, "X": sx,
                           "Y": sy, "TR": sTR, "TE": sTE})
    print(f"wrote {out_path}: n={n} (source N={N}) "
          f"TR shapes={[r.shape for r in sTR]} TE shapes={[r.shape for r in sTE]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--prefix", default="_sub")
    args = ap.parse_args()

    for name, fn in SOURCES.items():
        src = os.path.join(RAW, fn)
        if not os.path.exists(src):
            print(f"Skipping {name} (source {src} not found)")
            continue
        out = os.path.join(RAW, f"{name}{args.prefix}.mat")
        subset(src, args.n, out)


if __name__ == "__main__":
    main()
