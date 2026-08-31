"""
cache_utils.py
--------------
Implements the caching discipline from HW3 §0.7:

distances/{dataset}/{embedding}_{weights}_{metric}/
    w.npy         the matrix
    config.json   dataset, embedding, weights, metric, date, code version
    timing.json   accumulated compute cost (never overwritten, only added to)

embeddings/{dataset}/{embedding}_{weights}_fsw{m}.npy
    the FSW vectors themselves (not their distances - those are computed
    on the fly, they're instant from cached vectors)

Rule: a new configuration (dataset, embedding, weights, metric) always gets
a new folder. We never overwrite an existing w.npy silently.
"""
import os
import json
import time
import socket
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np

# ---- Root locations -------------------------------------------------
# Point these at wherever your actual cache lives. Kept as module-level
# so every script in code/ agrees on the same cache root.
CACHE_ROOT = os.environ.get("WMD_CACHE_ROOT", os.path.expanduser("~/wmd_cache"))
DISTANCES_ROOT = os.path.join(CACHE_ROOT, "distances")
EMBEDDINGS_ROOT = os.path.join(CACHE_ROOT, "embeddings")


def _code_version():
    """Best-effort git commit hash of the code/ folder, else 'unknown'."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.check_output(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def machine_name():
    """hostname + a short device string, used for the HW4 time table's
    'machine' column."""
    host = socket.gethostname()
    device = "CPU"
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
        else:
            device = platform.processor() or "CPU"
    except ImportError:
        device = platform.processor() or "CPU"
    return f"{host} ({device})"


def dist_dir(dataset, embedding, weights, metric):
    """distances/{dataset}/{embedding}_{weights}_{metric}/"""
    folder = f"{embedding}_{weights}_{metric}"
    return os.path.join(DISTANCES_ROOT, dataset, folder)


def emb_path(dataset, embedding, weights, m):
    """embeddings/{dataset}/{embedding}_{weights}_fsw{m}.npy"""
    folder = os.path.join(EMBEDDINGS_ROOT, dataset)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{embedding}_{weights}_fsw{m}.npy")


def write_config(folder, dataset, embedding, weights, metric, extra=None):
    os.makedirs(folder, exist_ok=True)
    cfg_path = os.path.join(folder, "config.json")
    if os.path.exists(cfg_path):
        return  # never overwrite an existing config
    cfg = {
        "dataset": dataset,
        "embedding": embedding,
        "weights": weights,
        "metric": metric,
        "date_created": datetime.now(timezone.utc).isoformat(),
        "code_version": _code_version(),
    }
    if extra:
        cfg.update(extra)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


def load_timing(folder):
    path = os.path.join(folder, "timing.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "total_compute_seconds": 0.0,
        "pairs_computed": 0,
        "num_workers": 1,
        "machine": machine_name(),
        "runs": [],
    }


def save_timing_increment(folder, seconds, n_pairs, num_workers=1):
    """Accumulate (never overwrite) compute cost into timing.json, per the
    HW3 rule: 'every long computation records its cost in timing.json:
    total compute seconds (accumulated across resumed runs -- add, never
    overwrite), pairs computed, seconds per pair, number of workers, and
    machine name.'"""
    os.makedirs(folder, exist_ok=True)
    timing = load_timing(folder)
    timing["total_compute_seconds"] += seconds
    timing["pairs_computed"] += n_pairs
    timing["num_workers"] = num_workers
    timing["machine"] = machine_name()
    timing["seconds_per_pair"] = (
        timing["total_compute_seconds"] / max(timing["pairs_computed"], 1)
    )
    timing["runs"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seconds": seconds,
        "pairs": n_pairs,
    })
    path = os.path.join(folder, "timing.json")
    with open(path, "w") as f:
        json.dump(timing, f, indent=2)
    return timing


class Stopwatch:
    """Small helper: `with Stopwatch() as sw: ...` then sw.elapsed."""
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
