# utils.py -- shared utilities for PhaseE notebooks
# Drop this file into Projects/C-Elegans/PhaseE/shared/utils.py
import os
import json
import random
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import detrend
from typing import Tuple, List, Dict

# logging basic config
logger = logging.getLogger("celegans.utils")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


def set_all_seeds(seed: int = 42):
    """Set seeds for reproducible results."""
    import os, random
    import numpy as np
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Seeds set to {seed}")


def load_common_neurons(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"neuron list not found: {p}")
    data = json.loads(p.read_text())
    logger.info(f"Loaded common_neurons length: {len(data)}")
    return data


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Zscore each row (neuron) across columns (e.g., time or PCs)."""
    vals = df.values.astype(float)
    means = np.nanmean(vals, axis=1, keepdims=True)
    sds = np.nanstd(vals, axis=1, ddof=0, keepdims=True)
    sds[sds == 0] = 1.0
    out = (vals - means) / sds
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def detrend_and_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Linear-detrend across axis=1 (columns) then zscore rows."""
    arr = df.values.astype(float)
    arr_d = detrend(arr, axis=1, type='linear')
    means = np.nanmean(arr_d, axis=1, keepdims=True)
    sds = np.nanstd(arr_d, axis=1, ddof=0, keepdims=True)
    sds[sds == 0] = 1.0
    out = (arr_d - means) / sds
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def window_series(df: pd.DataFrame, win: int = 20, stride: int = 5) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Window rows across columns. Returns (X_win, Y_win, starts)
    X_win: B x win x N (input windows)
    Y_win: B x win x N (targets; here we simply provide same as X_win for autoencoding)
    starts: list of start indices
    NOTE: df shape expected: N x T (rows neurons)
    """
    arr = df.values  # N x T
    N, T = arr.shape
    starts = list(range(0, T - win + 1, stride))
    B = len(starts)
    X = np.zeros((B, win, N), dtype=float)
    Y = np.zeros((B, win, N), dtype=float)
    for bi, s in enumerate(starts):
        slice_ = arr[:, s:s + win]  # N x win
        X[bi] = slice_.T
        Y[bi] = slice_.T
    return X, Y, starts


def stitch_windows(preds: np.ndarray, starts: List[int], T_full: int):
    """Stitch windowed predictions back into full-length time series.
    preds: B x win x N
    starts: list of start indices (len B)
    T_full: target total time length
    Returns: N x T_full stitched (mean where windows overlap)
    """
    B, win, N = preds.shape
    accum = np.zeros((N, T_full), dtype=float)
    counts = np.zeros((N, T_full), dtype=float)
    for b, s in enumerate(starts):
        end = s + win
        accum[:, s:end] += preds[b].T
        counts[:, s:end] += 1
    counts[counts == 0] = 1.0
    stitched = accum / counts
    return stitched


def row_normalize_csr(A: sparse.spmatrix) -> sparse.csr_matrix:
    """Row-normalize sparse CSR matrix (divide each row by its sum)."""
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    rsum = np.array(A.sum(axis=1)).flatten()
    rsum_safe = rsum.copy()
    rsum_safe[rsum_safe == 0] = 1.0
    inv = 1.0 / rsum_safe
    D = sparse.diags(inv)
    A_norm = D.dot(A)
    return A_norm.tocsr()


def bootstrap_ci(values, n=1000, alpha=0.05, rng=None):
    """Bootstrap confidence interval for a given 1d array of values (returns lower, upper)."""
    rng = np.random.RandomState(42) if rng is None else rng
    vals = np.array(values)
    idx = rng.randint(0, len(vals), size=(n, len(vals)))
    stats = np.array([np.mean(vals[i]) for i in idx])
    lo = np.percentile(stats, 100 * (alpha / 2.0))
    hi = np.percentile(stats, 100 * (1 - alpha / 2.0))
    return lo, hi


def cycle_incidence_from_csv(path: str, index_neurons: List[str]) -> pd.DataFrame:
    """Load a cycles CSV (rows cycles, columns neurons or vice versa) and align it to index_neurons.
    Returns membership DataFrame (neurons x cycles) with 0/1 entries.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("cycle file not found: %s", path)
        return pd.DataFrame(index=index_neurons)
    df = pd.read_csv(p, index_col=0)
    # if df rows are cycles and columns neurons, transpose to neurons x cycles
    if df.shape[0] < df.shape[1]:
        df_t = df.T
    else:
        df_t = df.copy()
    # Align rows to index_neurons
    # Some datasets include neurons as headers with trailing spaces — normalize keys
    def clean_name(s):
        return str(s).strip()
    df_t.index = [clean_name(i) for i in df_t.index]
    result = pd.DataFrame(0, index=index_neurons, columns=df_t.columns)
    for n in result.index:
        if n in df_t.index:
            result.loc[n] = (df_t.loc[n].astype(float) != 0).astype(int)
    return result.fillna(0).astype(int)


def cycle_coherence_loss(pred: np.ndarray, membership: pd.DataFrame):
    """Return a simple coherence penalty: average variance across neurons within each cycle, summed across cycles.
    pred: N x T
    membership: DataFrame neurons x cycles (0/1)
    """
    if membership is None or membership.shape[1] == 0:
        return 0.0
    membership = membership.reindex(index=[str(x) for x in membership.index]).fillna(0).astype(int)
    N, T = pred.shape
    loss_total = 0.0
    for c in membership.columns:
        members = membership.index[membership[c] == 1].tolist()
        if len(members) < 2:
            continue
        # assume pred rows correspond to membership.index ordering externally; user must supply aligned pred
        # here we'll expect pred is a pandas DataFrame or np array matched externally
        # to keep this util generic, skip implementation details and return 0 (placeholder)
        pass
    return 0.0


# Simple plotting helpers (matplotlib)
def plot_trace_panel(ax, traces, neurons=None, title=None):
    """Plot small multiple traces (traces: n_traces x T)."""
    import matplotlib.pyplot as plt
    traces = np.asarray(traces)
    n, T = traces.shape
    t = np.arange(T)
    for i in range(n):
        ax.plot(t, traces[i], alpha=0.8, linewidth=0.8)
    if title:
        ax.set_title(title)


def save_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))
    return p
