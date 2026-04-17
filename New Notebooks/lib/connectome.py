"""Clean C. elegans connectome loader (Witvliet 2020).

Design decisions (all explicit, all overridable):

    (D1) L/R-distinct neurons are KEPT distinct. Reason: CeNGEN and Witvliet both
         use L/R-resolved names; ASEL vs ASER have distinct functions, so
         collapsing L/R loses real biology. Use `collapse_lr=True` only as an
         explicit sensitivity analysis, never as the default.

    (D2) Chemical and gap-junction (electrical) synapses are stored as separate
         adjacency matrices. They have different biology and different dynamics.
         Downstream analyses should opt-in to combining them.

    (D3) Edge weight is the raw synapse count from Witvliet. Not binarized
         unless the caller explicitly asks. No zero-weight "edges" stored
         (contrast with the dense-edgelist canonical CSV in the old pipeline,
         which serialized every ordered pair and introduced the degree=299
         artifact).

    (D4) Self-loops are reported, then dropped by default. Override with
         `keep_self_loops=True`.

    (D5) The neuron set is defined by union of pre/post names present in the
         source file; no external master-neuron-list is imposed at load time
         (that filtering belongs to downstream alignment with CeNGEN etc.).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from .paths import WITVLIET_FILES

ChemKind = Literal["chemical", "electrical", "both"]


@dataclass(frozen=True)
class Connectome:
    """Canonical connectome representation at one developmental stage."""
    stage: str
    neurons: np.ndarray  # sorted unique neuron names (L/R distinct by default)
    chem_adj: np.ndarray  # (N, N) int32, chemical synapse counts
    gap_adj: np.ndarray   # (N, N) int32, electrical (gap) synapse counts  (symmetric)
    edges_raw: pd.DataFrame  # tidy source rows, after policy decisions applied

    @property
    def n(self) -> int:
        return len(self.neurons)

    def idx(self, name: str) -> int:
        where = np.where(self.neurons == name)[0]
        if len(where) == 0:
            raise KeyError(name)
        return int(where[0])

    def summary(self) -> Dict[str, float]:
        """Biological summary statistics. Used by notebooks for preregistered checks."""
        chem_total = int(self.chem_adj.sum())
        gap_total = int(self.gap_adj.sum())
        chem_edges = int((self.chem_adj > 0).sum())  # distinct ordered pairs
        gap_edges = int((self.gap_adj > 0).sum())    # counts both (i,j) and (j,i) since we mirror
        return {
            "stage": self.stage,
            "n_neurons": self.n,
            "chem_total_synapses": chem_total,
            "gap_total_synapses": gap_total,  # sum over matrix; undirected so each synapse counted twice
            "chem_directed_edges": chem_edges,
            "gap_unique_undirected_edges": gap_edges // 2,
            "chem_mean_syn_per_edge": chem_total / chem_edges if chem_edges else float("nan"),
            "density_chem": chem_edges / (self.n * (self.n - 1)) if self.n > 1 else float("nan"),
        }


def _canonical_lr(name: str) -> str:
    """Collapse L/R suffix for sensitivity analyses only. Not the default path."""
    if not isinstance(name, str):
        return name
    if len(name) > 2 and name.endswith(("L", "R")) and name[-2].isalpha():
        return name[:-1]
    return name


def load_witvliet(
    stage: str = "adult",
    *,
    collapse_lr: bool = False,      # D1 — sensitivity only
    keep_self_loops: bool = False,  # D4
) -> Connectome:
    """Load one developmental stage from Witvliet et al. 2020.

    Parameters
    ----------
    stage : {"L1_1","L1_2","L1_3","L1_4","L2","L3","adult"}
        Developmental stage / replicate key.
    collapse_lr : bool
        If True, fold *L and *R names together (sensitivity analysis).
    keep_self_loops : bool
        If True, retain rows with pre == post.
    """
    if stage not in WITVLIET_FILES:
        raise ValueError(f"Unknown stage {stage!r}. Available: {list(WITVLIET_FILES)}")
    path = WITVLIET_FILES[stage]
    raw = pd.read_excel(path)

    expected = {"pre", "post", "type", "synapses"}
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(f"Witvliet file {path.name} missing columns: {missing}")

    df = raw.copy()
    df["pre"] = df["pre"].astype(str).str.strip()
    df["post"] = df["post"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df["synapses"] = pd.to_numeric(df["synapses"], errors="coerce").fillna(0).astype(int)

    # D1 — L/R policy
    if collapse_lr:
        df["pre"] = df["pre"].map(_canonical_lr)
        df["post"] = df["post"].map(_canonical_lr)
        # After collapse, sum synapses over merged (pre,post,type) triplets.
        df = (
            df.groupby(["pre", "post", "type"], as_index=False)["synapses"].sum()
        )

    # D4 — self-loops
    self_loops = int((df["pre"] == df["post"]).sum())
    if not keep_self_loops:
        df = df[df["pre"] != df["post"]].copy()

    # D2 — validate type vocabulary
    ok_types = {"chemical", "electrical"}
    unknown_types = set(df["type"]) - ok_types
    if unknown_types:
        # Witvliet always uses these two; anything else is a data surprise — surface it.
        raise ValueError(
            f"Unexpected synapse type(s) in {path.name}: {unknown_types}. "
            f"Expected {ok_types}."
        )

    neurons = np.array(sorted(set(df["pre"]).union(df["post"])))
    n2i = {n: i for i, n in enumerate(neurons)}

    N = len(neurons)
    chem = np.zeros((N, N), dtype=np.int32)
    gap = np.zeros((N, N), dtype=np.int32)

    for r in df.itertuples(index=False):
        i, j = n2i[r.pre], n2i[r.post]
        if r.type == "chemical":
            chem[i, j] += int(r.synapses)
        else:  # electrical / gap junction — treat as undirected by mirroring
            gap[i, j] += int(r.synapses)
            gap[j, i] += int(r.synapses)

    # Record the self-loop count in metadata (stashed in a dataframe attr).
    df.attrs["self_loops_dropped"] = self_loops if not keep_self_loops else 0
    df.attrs["collapse_lr"] = collapse_lr
    df.attrs["source_file"] = str(path)

    return Connectome(
        stage=stage,
        neurons=neurons,
        chem_adj=chem,
        gap_adj=gap,
        edges_raw=df.reset_index(drop=True),
    )


def save_connectome(conn: Connectome, out_dir: Path) -> Dict[str, Path]:
    """Persist a Connectome as a directory of reproducible artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    np.savez_compressed(
        out_dir / f"connectome_{conn.stage}.npz",
        neurons=conn.neurons,
        chem_adj=conn.chem_adj,
        gap_adj=conn.gap_adj,
    )
    paths["npz"] = out_dir / f"connectome_{conn.stage}.npz"
    conn.edges_raw.to_csv(out_dir / f"connectome_{conn.stage}_edges.csv", index=False)
    paths["edges_csv"] = out_dir / f"connectome_{conn.stage}_edges.csv"
    summary_df = pd.DataFrame([conn.summary()])
    summary_df.to_csv(out_dir / f"connectome_{conn.stage}_summary.csv", index=False)
    paths["summary_csv"] = out_dir / f"connectome_{conn.stage}_summary.csv"
    return paths


def load_all_stages(**kwargs) -> Dict[str, Connectome]:
    """Convenience: load every available Witvliet stage with identical policies."""
    return {k: load_witvliet(k, **kwargs) for k in WITVLIET_FILES}
