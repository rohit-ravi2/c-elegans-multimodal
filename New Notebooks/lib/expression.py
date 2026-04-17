"""CeNGEN bulk expression loader + Witvliet neuron alignment.

Design decisions:

    (E1) Primary expression source is Barrett et al. 2022 CeNGEN bulk-sorted
         RNA-seq (`Bulk_Sorted_TPM.csv`). Covers 41 neuron classes; this is
         the biological reality — bulk-sorting pools multiple cells per class,
         so single-cell granularity requires a different (single-cell) source.

    (E2) Witvliet neuron → CeNGEN class mapping is a cascade of rules, applied
         in order. Each rule is explicit and testable:
             1. exact match (e.g. ASEL → ASEL, AVA → AVA)
             2. strip trailing L/R (ADLL → ADL, RIAL → RIA)
             3. strip trailing digits (DA1 → DA, VB7 → VB)
             4. combination: strip L/R then digits (SMDDL → SMD)
         Neurons with no match after all rules get NaN expression vectors.

    (E3) Gene identifier primary key is WBGene ID. Symbol resolution uses the
         `Gene_Name` column of the TPM file; neither the TPM matrix nor the
         returned object silently coerces between symbols and IDs.

    (E4) Special cases preserved:
         - ASEL / ASER kept distinct (both have CeNGEN columns — they are
           functionally asymmetric neurons).
         - AWC_ON / AWC_OFF collapsed to AWC in Witvliet naming (Witvliet uses
           AWCL/AWCR), so Witvliet AWCL / AWCR → CeNGEN AWC. We do not have
           L/R or ON/OFF resolution for AWC in bulk.

    (E5) Alignment direction: we build a matrix `(neuron, gene)` indexed by
         Witvliet neuron names, with expression pulled from the mapped CeNGEN
         class. Multiple Witvliet neurons can share the same underlying CeNGEN
         vector (e.g. ADLL and ADLR both show ADL's expression).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .paths import CENGEN_BULK, CENGEN


BULK_TPM_FILE = CENGEN_BULK / "Bulk_Sorted_TPM.csv"

SC_THRESHOLDED_FILES = {
    "liberal":      CENGEN / "thresholded" / "021821_liberal_threshold1.csv",
    "medium":       CENGEN / "thresholded" / "021821_medium_threshold2.csv",      # CeNGEN default
    "conservative": CENGEN / "thresholded" / "021821_conservative_threshold3.csv",
    "stringent":    CENGEN / "thresholded" / "021821_stringent_threshold4.csv",
}


@dataclass(frozen=True)
class Expression:
    """Aligned expression matrix: one row per Witvliet neuron, one column per gene."""
    neurons: np.ndarray           # (N,) — Witvliet neuron names, may have NaN rows
    genes_wbg: np.ndarray         # (G,) — WBGene IDs, primary key
    genes_symbol: np.ndarray      # (G,) — symbols from CeNGEN Gene_Name column
    genes_sequence: np.ndarray    # (G,) — sequence names from CeNGEN
    tpm: np.ndarray               # (N, G) — TPM values; NaN for unmapped neurons
    mapping: pd.DataFrame         # per-neuron: [witvliet_name, cengen_class, rule_used]

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def n_genes(self) -> int:
        return len(self.genes_wbg)

    def symbol(self, wbg: str) -> Optional[str]:
        where = np.where(self.genes_wbg == wbg)[0]
        if len(where) == 0:
            return None
        s = self.genes_symbol[where[0]]
        return None if pd.isna(s) else str(s)

    def wbg(self, symbol: str) -> Optional[str]:
        where = np.where(self.genes_symbol == symbol)[0]
        if len(where) == 0:
            return None
        return str(self.genes_wbg[where[0]])

    def expression_for(self, neuron: str, gene: str) -> float:
        """Pull a (neuron, gene) value. `gene` can be WBGene or symbol."""
        ni = np.where(self.neurons == neuron)[0]
        if len(ni) == 0:
            raise KeyError(f"neuron {neuron!r} not in expression matrix")
        wbg = gene if gene.startswith("WBGene") else self.wbg(gene)
        if wbg is None:
            raise KeyError(f"gene {gene!r} not resolvable")
        gi = np.where(self.genes_wbg == wbg)[0]
        return float(self.tpm[ni[0], gi[0]])

    def neurons_with_expression(self) -> np.ndarray:
        """Neurons that got a real expression vector (mapped to a CeNGEN class)."""
        has_data = ~np.all(np.isnan(self.tpm), axis=1)
        return self.neurons[has_data]


def _strip_lr(name: str) -> str:
    """Strip trailing L/R. Accepts names ending in L/R regardless of whether the
    prior char is alpha or digit (IL1L -> IL1, VB01R -> VB01, ADAL -> ADA)."""
    if len(name) > 2 and name.endswith(("L", "R")):
        return name[:-1]
    return name


def _strip_trailing_digits(name: str) -> str:
    i = len(name)
    while i > 0 and name[i-1].isdigit():
        i -= 1
    return name[:i] if i > 0 else name


def map_witvliet_to_cengen(
    witvliet_neurons: List[str],
    cengen_classes: List[str],
) -> pd.DataFrame:
    """Map each Witvliet neuron to a CeNGEN class via rule cascade.

    Rules applied in order:
      1 exact              (ASEL -> ASEL)
      2 strip_lr           (AVAL -> AVA)
      3 strip_digits       (DA1 -> DA)
      4 strip_lr_then_digits (VB01L -> VB)
      5 cengen_split_dv    (IL2DL -> IL2_DV; single-cell CeNGEN splits
                            dorsal/ventral vs lateral/right on some classes)
      6 cengen_split_lr    (IL2L -> IL2_LR)
    Rules 5-6 only fire when the single-cell CeNGEN file has `<base>_DV` or
    `<base>_LR` columns (these aren't present in bulk CeNGEN).
    """
    cengen_set = set(cengen_classes)
    rows = []
    for neuron in witvliet_neurons:
        matched, rule = None, None

        if neuron in cengen_set:
            matched, rule = neuron, "exact"
        else:
            stripped_lr = _strip_lr(neuron)
            if stripped_lr in cengen_set:
                matched, rule = stripped_lr, "strip_lr"
            else:
                stripped_digits = _strip_trailing_digits(neuron)
                if stripped_digits in cengen_set:
                    matched, rule = stripped_digits, "strip_digits"
                else:
                    stripped_both = _strip_trailing_digits(_strip_lr(neuron))
                    if stripped_both in cengen_set:
                        matched, rule = stripped_both, "strip_lr_then_digits"

        if matched is None:
            # Rule 5/6/7 — CeNGEN single-cell DV/LR split + plain-base fallback.
            # For positional suffixes (DL, DR, VL, VR, D, V), try these targets
            # in order: <base>_DV, <base>_LR, plain <base>. First hit wins.
            n_nd = _strip_trailing_digits(neuron)
            positional = [
                ("DL", ["DV", "LR"]),
                ("DR", ["DV", "LR"]),
                ("VL", ["DV", "LR"]),
                ("VR", ["DV", "LR"]),
                ("D",  ["DV"]),
                ("V",  ["DV"]),
                ("L",  ["LR"]),
                ("R",  ["LR"]),
            ]
            for suffix, split_groups in positional:
                if n_nd.endswith(suffix) and len(n_nd) > len(suffix):
                    base = n_nd[:-len(suffix)]
                    # Try split variants first
                    for g in split_groups:
                        cand = f"{base}_{g}"
                        if cand in cengen_set:
                            matched, rule = cand, f"cengen_split_{g.lower()}"
                            break
                    if matched is not None:
                        break
                    # Fall back to plain base
                    if base in cengen_set:
                        matched, rule = base, "strip_positional"
                        break

        rows.append({"witvliet_name": neuron, "cengen_class": matched, "rule_used": rule})
    return pd.DataFrame(rows)


def load_expression(witvliet_neurons: List[str]) -> Expression:
    """Load CeNGEN bulk TPM and align it to the given Witvliet neuron list."""
    tpm_df = pd.read_csv(BULK_TPM_FILE)

    # Basic schema checks
    for col in ("Gene_Name", "Sequence_Name", "Wormbase_ID"):
        if col not in tpm_df.columns:
            raise ValueError(f"Expected column {col!r} missing from {BULK_TPM_FILE.name}")

    cengen_class_cols = [c for c in tpm_df.columns
                         if c not in ("Gene_Name", "Sequence_Name", "Wormbase_ID")]

    # Build the mapping
    mapping = map_witvliet_to_cengen(list(witvliet_neurons), cengen_class_cols)

    # Build (neurons × genes) matrix
    N = len(witvliet_neurons)
    G = len(tpm_df)
    tpm = np.full((N, G), np.nan, dtype=np.float32)

    class_to_vec = {c: tpm_df[c].to_numpy(dtype=np.float32) for c in cengen_class_cols}
    for i, row in mapping.iterrows():
        cc = row["cengen_class"]
        if isinstance(cc, str) and cc in class_to_vec:
            tpm[i] = class_to_vec[cc]

    return Expression(
        neurons=np.asarray(witvliet_neurons),
        genes_wbg=tpm_df["Wormbase_ID"].astype(str).to_numpy(),
        genes_symbol=tpm_df["Gene_Name"].astype(str).to_numpy(),
        genes_sequence=tpm_df["Sequence_Name"].astype(str).to_numpy(),
        tpm=tpm,
        mapping=mapping,
    )


def load_expression_singlecell(
    witvliet_neurons: List[str],
    threshold: str = "medium",
) -> Expression:
    """Load single-cell CeNGEN thresholded expression and align to Witvliet neurons.

    Compared to `load_expression` (bulk Barrett 2022):
      - 128 neuron classes covered instead of 41
      - Per-class values come from single-cell pseudobulking with a CeNGEN
        threshold applied (4 threshold levels; default 'medium' = level 2)
      - Classes like IL2_DV, IL2_LR, RMD_DV, RMD_LR, RME_DV, RME_LR,
        AWC_ON, AWC_OFF, VD_DD are resolved via cengen_split rules.
    """
    if threshold not in SC_THRESHOLDED_FILES:
        raise ValueError(
            f"threshold={threshold!r}; expected one of {list(SC_THRESHOLDED_FILES)}"
        )
    path = SC_THRESHOLDED_FILES[threshold]
    tpm_df = pd.read_csv(path)

    # Schema: first three cols are metadata, the rest are neuron-class columns.
    meta_cols = {"Unnamed: 0", "gene_name", "Wormbase_ID"}
    actual_meta = [c for c in tpm_df.columns if c in meta_cols]
    for required in ("gene_name", "Wormbase_ID"):
        if required not in tpm_df.columns:
            raise ValueError(
                f"Expected column {required!r} missing from {path.name}"
            )

    cengen_class_cols = [c for c in tpm_df.columns if c not in meta_cols]
    mapping = map_witvliet_to_cengen(list(witvliet_neurons), cengen_class_cols)

    N = len(witvliet_neurons)
    G = len(tpm_df)
    tpm = np.full((N, G), np.nan, dtype=np.float32)
    class_to_vec = {c: tpm_df[c].to_numpy(dtype=np.float32) for c in cengen_class_cols}
    for i, row in mapping.iterrows():
        cc = row["cengen_class"]
        if isinstance(cc, str) and cc in class_to_vec:
            tpm[i] = class_to_vec[cc]

    return Expression(
        neurons=np.asarray(witvliet_neurons),
        genes_wbg=tpm_df["Wormbase_ID"].astype(str).to_numpy(),
        genes_symbol=tpm_df["gene_name"].astype(str).to_numpy(),
        genes_sequence=tpm_df["gene_name"].astype(str).to_numpy(),  # no sequence col here; duplicate
        tpm=tpm,
        mapping=mapping,
    )


def save_expression(expr: Expression, out_dir: Path) -> Dict[str, Path]:
    """Persist an Expression object as npz + CSV mapping + gene metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    np.savez_compressed(
        out_dir / "expression_tpm.npz",
        neurons=expr.neurons,
        genes_wbg=expr.genes_wbg,
        tpm=expr.tpm,
    )
    paths["npz"] = out_dir / "expression_tpm.npz"

    pd.DataFrame({
        "wbgene": expr.genes_wbg,
        "symbol": expr.genes_symbol,
        "sequence": expr.genes_sequence,
    }).to_csv(out_dir / "expression_genes.csv", index=False)
    paths["genes_csv"] = out_dir / "expression_genes.csv"

    expr.mapping.to_csv(out_dir / "expression_neuron_mapping.csv", index=False)
    paths["mapping_csv"] = out_dir / "expression_neuron_mapping.csv"

    return paths
