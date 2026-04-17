"""Motif computation on directed C. elegans connectome.

Design decisions:

    (M1) Motifs are computed on the BINARY adjacency (edge present / absent),
         not on the weighted adjacency. Synapse count is valuable elsewhere;
         here we count topological patterns.

    (M2) Both chemical and gap-junction layers contribute to the underlying
         "structural" graph by default (`type_set="both"`). Callers can pick
         `"chemical"` or `"gap"` to run one-layer analyses.

    (M3) Per-node motif features are **participation counts**, with an
         explicit degree-residualized variant for downstream statistics.
         Motif counts are strongly driven by degree; naively correlating
         gene expression with motif count conflates "is this neuron a
         hub?" with "does this gene predict motif participation?".

    (M4) All motif counts are integers built from matrix powers of A
         (faster and less error-prone than iterating triples).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


TypeSet = Literal["chemical", "gap", "both"]


@dataclass(frozen=True)
class MotifFeatures:
    """Per-neuron motif counts + derived features."""
    neurons: np.ndarray
    features: pd.DataFrame  # columns: in_deg, out_deg, total_deg, ffl, cycle3, recip, two_in, two_out, clust

    def degree_residualize(self) -> pd.DataFrame:
        """Return features with motif columns residualized against total_deg.

        Rationale: ffl, cycle3, two_in, two_out, recip, clust all scale with
        degree. To find gene -> motif associations that aren't just gene -> degree,
        residualize each motif feature against total_deg via OLS and return
        the residuals.
        """
        from sklearn.linear_model import LinearRegression
        deg = self.features[["in_deg", "out_deg", "total_deg"]].values
        motif_cols = ["ffl", "cycle3", "recip", "two_in", "two_out", "clust"]
        out = self.features.copy()
        for col in motif_cols:
            y = self.features[col].values.astype(float)
            if np.std(y) < 1e-10:
                out[col + "_resid"] = 0.0
                continue
            reg = LinearRegression().fit(deg, y)
            yhat = reg.predict(deg)
            out[col + "_resid"] = y - yhat
        return out


def build_binary_adjacency(
    chem_adj: np.ndarray,
    gap_adj: np.ndarray,
    type_set: TypeSet = "both",
) -> np.ndarray:
    """Binary adjacency combining chemical and gap layers per (M2)."""
    if type_set == "chemical":
        A = chem_adj > 0
    elif type_set == "gap":
        A = gap_adj > 0
    elif type_set == "both":
        A = (chem_adj > 0) | (gap_adj > 0)
    else:
        raise ValueError(f"type_set={type_set!r}")
    np.fill_diagonal(A, 0)
    return A.astype(np.int32)


def compute_motifs(
    neurons: np.ndarray,
    chem_adj: np.ndarray,
    gap_adj: np.ndarray,
    type_set: TypeSet = "both",
) -> MotifFeatures:
    """Compute per-neuron motif features from a directed adjacency.

    Features (per neuron i):
      - in_deg, out_deg, total_deg  : basic degree
      - ffl          : number of feed-forward triangles including i anywhere
      - cycle3       : number of directed 3-cycles through i (diag of A^3)
      - recip        : number of reciprocal partners of i
      - two_in       : number of length-2 directed paths ending at i
      - two_out      : number of length-2 directed paths starting at i
      - clust        : (ffl + cycle3) / (out_deg * in_deg) - degree-normalized
                       clustering-like ratio (NaN -> 0)
    """
    A = build_binary_adjacency(chem_adj, gap_adj, type_set)
    N = A.shape[0]

    out_deg = A.sum(axis=1)
    in_deg = A.sum(axis=0)
    total_deg = out_deg + in_deg

    A2 = A @ A
    A3 = A2 @ A

    # FFL: count of triangles (i,j,k) with i->j, j->k, i->k, summed over all j,k for each i
    ffl_anchor_i = np.sum(A * A2, axis=1)  # for each i: sum_k A[i,k]*A2[i,k] = # of j where i->j->k and i->k
    # cycle3 through i = (A^3)_ii / 1 (each 3-cycle i->j->k->i contributes once to the diag)
    cycle3 = np.diag(A3)
    # reciprocal partners
    recip = np.diag(A @ A.T) * 0  # placeholder
    recip = np.sum(A * A.T, axis=1)
    # two-step paths
    two_out = A2.sum(axis=1)
    two_in = A2.sum(axis=0)

    # Clustering-like ratio (normalize by opportunity)
    denom = (out_deg * in_deg).astype(float)
    clust = np.where(denom > 0, (ffl_anchor_i + cycle3) / denom, 0.0)

    feats = pd.DataFrame({
        "in_deg": in_deg,
        "out_deg": out_deg,
        "total_deg": total_deg,
        "ffl": ffl_anchor_i,
        "cycle3": cycle3,
        "recip": recip,
        "two_in": two_in,
        "two_out": two_out,
        "clust": clust,
    }, index=neurons)

    return MotifFeatures(neurons=np.asarray(neurons), features=feats)
