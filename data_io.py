"""
data_io.py
----------
Recreates HW1 Task 1 (documents as distributions) plus the stopword-removal
rule stated in HW1's background section:
    - twitter: keep stop words (documents are ~10 words long)
    - amazon / classic / reuters (all datasets used from HW3 onward):
      remove stop words as usual

Also recreates the "load once, keep only vectors that occur in the
dataset, cache the small subset, L2-normalize" trick from HW1 Setup.

ASSUMPTIONS ABOUT THE .mat FORMAT
----------------------------------
The datasets come from github.com/joisino/reeval-wmd (Sato et al.'s
re-evaluation of Kusner's WMD benchmark), after running
`python duplication.py` to deduplicate. I do not have a copy of these
files in this environment, so the exact field names below are my best
reconstruction from the standard WMD-benchmark .mat layout (the one
Kusner et al.'s original repo and its derivatives use):

    'words' : (1, N) object array; words[0, i] is itself a (1, n_i)
              object array of token strings for document i
    'BOW_X' : (1, N) object array; BOW_X[0, i] is a (1, n_i) array of
              raw word counts, aligned index-for-index with words[0, i]
    'Y'     : (N,) or (1, N) array of integer class labels (1-indexed)
    'TR'    : (n_splits, n_train) 1-indexed train-document indices
    'TE'    : (n_splits, n_test)  1-indexed test-document indices
              (twitter/amazon/classic have multiple predefined splits;
              reuters ships with a single official split, i.e. TR/TE
              have one row -- matches HW4 §0.2's "reuters has a single
              official split")

FIRST THING TO DO on the real files: run `inspect_mat(path)` below and
diff its printed keys/shapes against the assumptions above. If they
differ, only `_load_raw_mat` needs to change -- everything downstream
(build_documents, kNN, FSW, ...) works off the (X, W, y, splits)
representation returned by `load_dataset`, not off .mat internals.
"""
import os
import json
import numpy as np
import scipy.io as sio

# A minimal, standard English stop-word list (NLTK's list, hard-coded so
# this file has no extra dependency). Used for amazon/classic/reuters.
STOP_WORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll
he's her here here's hers herself him himself his how how's i i'd i'll
i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our
ours ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them
themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll
we're we've were weren't what what's when when's where where's which
while who who's whom why why's with won't would wouldn't you you'd
you'll you're you've your yours yourself yourselves
""".split())


def inspect_mat(path):
    """Print top-level keys and shapes -- run this first on a real file."""
    m = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    for k, v in m.items():
        if k.startswith("__"):
            continue
        shape = getattr(v, "shape", None)
        print(f"{k}: type={type(v)} shape={shape}")
    return m


def _load_raw_mat(path):
    """Returns per-document (tokens, counts, vectors), labels, and split
    indices, straight from the .mat file. See module docstring for the
    assumed schema.

    The word vectors are taken directly from the file's per-document `X`
    cells (each `(d, n_i)`, already L2-normalized), which index-for-index
    align with `BOW_X` counts. `words` are kept only for reference.
    """
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "words" in m:
        words_cell = np.atleast_1d(m["words"])
        bow_cell = np.atleast_1d(m["BOW_X"])
        x_cell = np.atleast_1d(m["X"])
        y = np.asarray(m["Y"]).astype(int).reshape(-1)
        tr = _rows_to_indices(m["TR"])
        te = _rows_to_indices(m["TE"])
    elif "words_tr" in m:
        words_tr = np.atleast_1d(m["words_tr"])
        words_te = np.atleast_1d(m["words_te"])
        words_cell = np.concatenate([words_tr, words_te])

        bow_tr = np.atleast_1d(m["BOW_xtr"])
        bow_te = np.atleast_1d(m["BOW_xte"])
        bow_cell = np.concatenate([bow_tr, bow_te])

        x_tr = np.atleast_1d(m["xtr"])
        x_te = np.atleast_1d(m["xte"])
        x_cell = np.concatenate([x_tr, x_te])

        y_tr = np.asarray(m["ytr"]).astype(int).reshape(-1)
        y_te = np.asarray(m["yte"]).astype(int).reshape(-1)
        y = np.concatenate([y_tr, y_te])

        n_tr = len(words_tr)
        n_te = len(words_te)
        tr = [np.arange(n_tr)]
        te = [np.arange(n_tr, n_tr + n_te)]
    else:
        raise KeyError(
            f"Unknown .mat structure in {path}. "
            f"Keys found: {[k for k in m.keys() if not k.startswith('__')]}"
        )

    n_docs = len(words_cell)
    docs_tokens = []
    docs_counts = []
    X_list = []
    for i in range(n_docs):
        toks = [str(w) for w in np.atleast_1d(words_cell[i]).reshape(-1)]
        cnts = np.atleast_1d(bow_cell[i]).astype(float).reshape(-1)
        xmat = np.asarray(x_cell[i], dtype=np.float64)
        docs_tokens.append(toks)
        docs_counts.append(cnts)
        X_list.append(xmat)

    return docs_tokens, docs_counts, X_list, y, tr, te


def _rows_to_indices(cell):
    """Normalizes a TR/TE cell into a list of 0-indexed int arrays, one
    per outer split. Handles both rectangular (e.g. (5, n)) matrices and
    ragged object arrays (rows of different length, as produced by
    make_subsets.py)."""
    arr = np.asarray(cell)
    if arr.ndim == 2 and arr.dtype.kind != "O":
        return [row.astype(int) - 1 for row in arr]
    # ragged: cell is an object array whose elements are rows (possibly
    # themselves arrays); or a single 1-D row.
    if arr.ndim == 1 and arr.dtype.kind == "O":
        rows = arr
    elif arr.ndim == 2 and arr.dtype.kind == "O":
        rows = arr[0]
    else:
        rows = [arr]
    out = []
    for r in rows:
        out.append(np.asarray(r).reshape(-1).astype(int) - 1)
    return out


def _as_rows(idx_rows):
    """Yields each outer-split row of an index container, whether it's a
    2-D np array, a ragged object array, or a list of 1-D arrays."""
    if isinstance(idx_rows, (list, tuple)):
        return list(idx_rows)
    arr = np.asarray(idx_rows)
    if arr.ndim == 2 and arr.dtype.kind != "O":
        return list(arr)
    if arr.ndim == 2 and arr.dtype.kind == "O":
        return list(arr[0])
    if arr.ndim == 1 and arr.dtype.kind == "O":
        return list(arr)
    return [arr]


def build_vocab_subset(docs_tokens, full_embedding_loader, cache_path):
    """HW1 Setup trick: load the full pretrained embedding ONCE, keep
    only vectors for words that appear in this dataset, L2-normalize,
    and cache the small subset so future runs load instantly.

    full_embedding_loader: a zero-arg callable that returns a dict
    {word: raw_vector} for the FULL pretrained embedding (e.g. a
    gensim KeyedVectors wrapper). Only called on a cache miss.
    """
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True).item()
        return data["vocab"], data["vectors"]

    needed = set()
    for toks in docs_tokens:
        needed.update(toks)

    full = full_embedding_loader()
    vocab, vecs = [], []
    for w in needed:
        v = full.get(w)
        if v is None:
            continue
        v = np.asarray(v, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        vocab.append(w)
        vecs.append(v)
    vectors = np.stack(vecs).astype(np.float32)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, {"vocab": vocab, "vectors": vectors}, allow_pickle=True)
    return vocab, vectors


def build_documents(docs_tokens, docs_counts, vocab, vectors,
                     remove_stopwords):
    """HW1 Task 1: for each document, look up word vectors, discard
    words with no vector (and, from HW3 onward, stop words for
    amazon/classic/reuters), compute nBOW weights d_i = c_i / sum(c_j),
    discard documents left with <= 2 words.

    Returns
    -------
    X_list : list of (n_i, d) float32 arrays -- word vectors per doc
    W_list : list of (n_i,) float32 arrays -- nBOW weights per doc, sum to 1
    keep_mask : bool array over the original doc indices (False = discarded)
    n_discarded : int
    """
    word2idx = {w: i for i, w in enumerate(vocab)}
    X_list, W_list = [], []
    keep_mask = np.zeros(len(docs_tokens), dtype=bool)
    n_discarded = 0

    for doc_i, (toks, cnts) in enumerate(zip(docs_tokens, docs_counts)):
        xs, cs = [], []
        for tok, c in zip(toks, cnts):
            tok_l = tok.lower()
            if remove_stopwords and tok_l in STOP_WORDS:
                continue
            idx = word2idx.get(tok) or word2idx.get(tok_l)
            if idx is None:
                continue
            xs.append(vectors[idx])
            cs.append(c)

        if len(xs) <= 2:
            n_discarded += 1
            continue

        X = np.stack(xs).astype(np.float32)
        c = np.asarray(cs, dtype=np.float64)
        W = (c / c.sum()).astype(np.float32)

        X_list.append(X)
        W_list.append(W)
        keep_mask[doc_i] = True

    return X_list, W_list, keep_mask, n_discarded


def load_dataset(mat_path, embedding_name, full_embedding_loader,
                  cache_dir, dataset_name, remove_stopwords):
    """End-to-end loader for one dataset, using word vectors that are
    already shipped inside the .mat file (the file's per-doc `X` cells,
    already L2-normalized) instead of re-looking them up from a full
    pretrained embedding. This removes the fastText dependency so the
    pipeline can run on these cached benchmark files directly.

    Returns a dict with X (list of per-doc word-vector arrays), W (list
    of per-doc nBOW weight arrays), y (labels, filtered to kept docs),
    and TR/TE split indices remapped into the filtered (kept-only) index
    space, ready for cv_knn.py.

    Note: because the file carries no token->vector mapping, stopword
    removal can't be applied here; the nBOW weights come straight from
    the file's `BOW_X` counts (which already align with `X`).
    """
    docs_tokens, docs_counts, X_raw, y, tr, te = _load_raw_mat(mat_path)

    X_list, W_list = [], []
    keep_mask = np.zeros(len(X_raw), dtype=bool)
    n_discarded = 0
    for i, (xmat, counts) in enumerate(zip(X_raw, docs_counts)):
        xmat = np.asarray(xmat, dtype=np.float64)
        if xmat.ndim == 1:      # single-token doc stored as flat (d,)
            xmat = xmat[:, None]
        Xi = xmat.T.astype(np.float32)  # (d, n) -> (n, d)
        if Xi.shape[0] <= 2:  # same <=2-word discard rule as build_documents
            n_discarded += 1
            continue
        c = np.asarray(counts, dtype=np.float64)
        W = (c / c.sum()).astype(np.float32)
        X_list.append(Xi)
        W_list.append(W)
        keep_mask[i] = True

    # remap original -> kept indices; -1 marks a discarded (dropped) doc
    remap = -np.ones(len(X_raw), dtype=int)
    remap[keep_mask] = np.arange(keep_mask.sum())

    def remap_split(idx_rows):
        out = []
        for row in _as_rows(idx_rows):
            new_row = remap[row]
            out.append(new_row[new_row >= 0])
        return out

    result = {
        "X": X_list,
        "W": W_list,
        "y": y[keep_mask],
        "TR": remap_split(tr),
        "TE": remap_split(te),
        "n_discarded": int(n_discarded),
        "n_kept": int(keep_mask.sum()),
    }
    print(f"[{dataset_name}] kept {result['n_kept']} docs, "
          f"discarded {n_discarded} (<=2 words after filtering)")
    return result
