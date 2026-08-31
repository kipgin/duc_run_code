# HW4 code

Recreates the parts of HW1 and HW3 that HW4 depends on (data loading,
exact WMD, FSW embedding, the cache layout) since the original code was
lost, then runs the HW4 comparison itself.

## Files

| file | recreates | does |
|---|---|---|
| `cache_utils.py` | HW3 §0.7 | cache folder layout, `config.json`, resumable `timing.json` |
| `data_io.py` | HW1 Task 1 | load `.mat`, look up word vectors, build nBOW `(X, W)` per doc, stopword removal, discard ≤2-word docs |
| `wmd_distances.py` | HW1 Task 2 / "Addition to HW1" | exact W1/W2 via POT, resumable + checkpointed + progress bar |
| `fsw_embed.py` | HW3 §0.2 | FSW module creation (seed 0, fixed args), corpus embedding, caching vectors, instant Euclidean distance matrices |
| `cv_knn.py` | HW1 Task 3, updated per HW4 §0.1/§0.2 | outer splits, 5-fold inner CV, k∈{1..30}, smallest-k tie rule, vote ties broken by decreasing k |
| `timing_utils.py` | HW4 §0.3 deliverable 4 | per-query WMD timing, FSW pre-embedding + per-query timing, pooled mean±SD, warm-up |
| `run_hw4.py` | HW4 | orchestrates everything into `results/` CSVs/JSON and `figures/` PNGs |
| `make_report.py` | HW4 §0.3 deliverables | assembles a markdown report from `results/` + `figures/` (numbers only — you write the discussion) |

## Before running — things I could not verify without your data

I don't have access to the `.mat` files, the pretrained fastText vectors,
or your server, so three things in this code are my best reconstruction
and need a sanity check against your real files:

1. **`.mat` field names** in `data_io.py` (`words`, `BOW_X`, `Y`, `TR`,
   `TE`). Run `data_io.inspect_mat(path)` on one real file first; if the
   keys differ, only `_load_raw_mat` needs editing.
2. **`fasttext_full_loader()`** in `run_hw4.py` is a stub — point it at
   your actual fastText `crawl-300d-2M.vec` loader from HW1.
3. **`DATASET_MAT_PATHS`** in `run_hw4.py` — fill in real paths.
4. **`CACHE_ROOT`** — set the `WMD_CACHE_ROOT` env var, or edit
   `cache_utils.py`, to point at your existing HW3 cache (so the W2
   matrices and FSW-500/1000/2000 vectors you already have are found
   and reused rather than recomputed).

## Install

```bash
pip install torch fswlib pot scikit-learn scipy numpy matplotlib tqdm
```

## Run order

```bash
export WMD_CACHE_ROOT=/path/to/your/existing/cache

# one dataset at a time (recommended — lets you watch progress bars
# and re-run just one if something looks off):
python run_hw4.py --dataset amazon
python run_hw4.py --dataset classic
python run_hw4.py --dataset reuters

# or all three:
python run_hw4.py --dataset all
```

Expected runtime: dominated by (a) any missing exact-W1 matrix — this
is resumable, so it's safe to kill and restart; check the tqdm ETA
after a few minutes and message if it looks unreasonable, per the HW3
instructions — and (b) embedding FSW-250, which HW4 §0.2 says should
only take minutes since 500/1000/2000 are already cached. Everything
else (kNN+CV, plotting) is fast, on the order of seconds to low minutes
per dataset.

`--skip-timing` skips the (slower, since it doesn't use any cache —
it's deliberately timing raw compute) timing measurements if you want
the tables/figures first.

## Output

```
results/
  main_table_{dataset}.csv       # deliverable 1
  full_results_{dataset}.json    # per-split detail behind the CSV + figures
  time_table_{dataset}.csv       # deliverable 4
figures/
  cv_curves_{dataset}.png        # deliverable 2
  error_vs_m_{dataset}.png       # deliverable 3
```

Then run `make_report.py` to assemble these into a single markdown
report; add your written discussion (deliverable 5) into the generated
`report.md` before converting to PDF (e.g. with pandoc, or the doc
tooling of your choice).
