"""Reference biological classifications.

Loer & Rand 2022 (WormAtlas-associated) is the authoritative NT mapping for
all 302 hermaphrodite C. elegans neurons. Used as ground truth for:
  - Sanity-checking expression alignment
  - Filtering neurons to NT-class subsets in downstream analyses
  - Providing ground-truth labels for classification tasks

Design decisions:

    (R1) Row-level NT assignments are per-neuron (e.g. ADAL, ADAR), not per-class.
         Class-level aggregation is derived from this (handled by caller).

    (R2) NT1 (primary transmitter) is the canonical assignment; NT2 (co-released)
         is also preserved for neurons with dual transmitter capacity.

    (R3) "Unknown" NT neurons are kept in the table but excluded from NT-group
         sanity checks; they would contaminate class-level comparisons.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .paths import DATA

NT_TABLE = DATA / "expression/neurotransmitter/Ce_NTtables_Loer&Rand2022.xlsx"


@dataclass(frozen=True)
class NTReference:
    """Per-neuron neurotransmitter assignments from Loer & Rand 2022."""
    table: pd.DataFrame   # columns: neuron_class, neuron, nt1, nt2, location, notes

    def neurons_by_nt(self, nt_contains: str) -> List[str]:
        """All neurons whose NT1 contains the given token (e.g. 'Acetylcholine', 'GABA')."""
        mask = self.table["nt1"].astype(str).str.contains(nt_contains, case=False, na=False)
        return sorted(self.table.loc[mask, "neuron"].astype(str).unique().tolist())

    def cholinergic(self) -> List[str]:
        return self.neurons_by_nt("Acetylcholine")

    def gabaergic(self) -> List[str]:
        return self.neurons_by_nt("GABA")

    def glutamatergic(self) -> List[str]:
        return self.neurons_by_nt("Glutamate")

    def dopaminergic(self) -> List[str]:
        return self.neurons_by_nt("Dopamine")

    def serotonergic(self) -> List[str]:
        return self.neurons_by_nt("Serotonin")

    def nt_of(self, neuron: str) -> Optional[str]:
        row = self.table[self.table["neuron"] == neuron]
        if len(row) == 0:
            return None
        v = row["nt1"].iloc[0]
        return None if pd.isna(v) else str(v)

    def class_of(self, neuron: str) -> Optional[str]:
        """The neuron-class name (as Loer & Rand use it), if resolvable."""
        row = self.table[self.table["neuron"] == neuron]
        if len(row) == 0:
            return None
        v = row["neuron_class"].iloc[0]
        return None if pd.isna(v) else str(v)

    def counts(self) -> pd.Series:
        return self.table["nt1"].value_counts(dropna=False)


def load_nt_reference() -> NTReference:
    """Load Loer & Rand 2022 hermaphrodite NT table, cleaned."""
    import warnings
    with warnings.catch_warnings():
        # openpyxl emits a benign "Unknown extension" warning on this file
        warnings.simplefilter("ignore")
        df = pd.read_excel(NT_TABLE, sheet_name="Hermaphrodite, sorted by neuron", header=3)

    # The first column is an internal row count; rename by position.
    df.columns = ["_h0", "Nrow", "neuron_class", "neuron", "sex", "nt1", "nt2", "location", "notes"][: len(df.columns)]

    # Drop header artifacts and rows without a neuron name
    df = df[df["neuron"].notna() & (df["neuron"] != "Neuron")].copy()

    # Forward-fill neuron_class (the xlsx only fills the class on the first member row)
    df["neuron_class"] = df["neuron_class"].ffill()

    df = df.reset_index(drop=True)[["neuron_class", "neuron", "nt1", "nt2", "location", "notes"]]
    return NTReference(table=df)
