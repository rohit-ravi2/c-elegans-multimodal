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

from .paths import CENGEN_BULK


BULK_TPM_FILE = CENGEN_BULK / "Bulk_Sorted_TPM.csv"


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
    if len(name) > 2 and name.endswith(("L", "R")) and name[-2].isalpha():
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
    """Map each Witvliet neuron to a CeNGEN class via the E2 rule cascade."""
    cengen_set = set(cengen_classes)
    rows = []
    for neuron in witvliet_neurons:
        matched, rule = None, None
        # Rule 1 — exact
        if neuron in cengen_set:
            matched, rule = neuron, "exact"
        else:
            # Rule 2 — strip L/R
            stripped_lr = _strip_lr(neuron)
            if stripped_lr in cengen_set:
                matched, rule = stripped_lr, "strip_lr"
            else:
                # Rule 3 — strip digits
                stripped_digits = _strip_trailing_digits(neuron)
                if stripped_digits in cengen_set:
                    matched, rule = stripped_digits, "strip_digits"
                else:
                    # Rule 4 — strip L/R then digits
                    stripped_both = _strip_trailing_digits(_strip_lr(neuron))
                    if stripped_both in cengen_set:
                        matched, rule = stripped_both, "strip_lr_then_digits"
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
