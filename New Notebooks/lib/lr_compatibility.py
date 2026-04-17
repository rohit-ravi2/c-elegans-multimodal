"""Ligand-receptor compatibility atlas for C. elegans edge prediction.

Design decisions:

    (L1) Per-neuron ligand annotations from Bentley et al. 2016 (curated, evidence-
         backed neuron -> {neurotransmitter, neuropeptide} mappings). Collapsed
         to class level when downstream needs class-level features.

    (L2) Per-neuron receptor annotations from Bentley (explicit expression-
         evidence for 30 receptors across ~291 neurons).

    (L3) Canonical ligand-receptor pairing matrix derived from C. elegans
         literature. Asymmetric and incomplete — it covers the major classical
         transmitters, the four monoamines (dopamine, serotonin, octopamine,
         tyramine), and the best-characterized peptide pairs. Gaps are explicit
         (treated as "unknown", not "doesn't bind").

    (L4) Compatibility score at class-pair level:
             score(P -> Q) = sum over (ligand L, receptor R) of
                             1[P expresses L] * 1[Q expresses R] * M[L, R]
         where M[L,R] is 1 if the pair is a known binding, 0 otherwise.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from .paths import DATA


BENTLEY_FILE = DATA / "expression/bentley2016/Bentley_et_al_2016_expression.csv"


# ----- L3: canonical ligand-receptor pairs -----
# Ligand name -> set of receptor symbols (in Bentley-style capitalization).
# Sources: WormBase receptor annotations; canonical C. elegans neuropharmacology
# (e.g. classical review: Bargmann 1998 / Hobert annotations).
CANONICAL_LR_PAIRS: Dict[str, Set[str]] = {
    # Monoamines
    "Dopamine":   {"DOP-1", "DOP-2", "DOP-3", "DOP-4", "DOP-5"},
    "Serotonin":  {"SER-1", "SER-2", "SER-3", "SER-4", "SER-5", "SER-6", "SER-7", "MOD-1"},
    "Octopamine": {"SER-3", "SER-6", "OCTR-1"},
    "Tyramine":   {"TYRA-2", "TYRA-3", "SER-2", "LGC-55"},

    # Peptides — from Bentley 2016, Beets 2023, well-characterized pairs
    "PDF-1":      {"PDFR-1"},
    "PDF-2":      {"PDFR-1"},
    "FLP-1":      {"NPR-4", "NPR-11"},
    "FLP-2":      {"NPR-1", "FRPR-19"},
    "FLP-3":      {"NPR-10", "NPR-5"},
    "FLP-6":      {"FRPR-6", "FRPR-19"},
    "FLP-7":      {"NPR-22"},
    "FLP-8":      {"NPR-22"},
    "FLP-11":     {"NPR-1", "NPR-4", "NPR-5", "NPR-22", "FRPR-3"},
    "FLP-13":     {"NPR-1", "DMSR-1", "FRPR-19"},
    "FLP-14":     {"FRPR-19", "NPR-11"},
    "FLP-15":     {"NPR-3", "NPR-5", "NPR-11"},
    "FLP-17":     {"EGL-6"},
    "FLP-18":     {"NPR-1", "NPR-4", "NPR-5", "NPR-10", "NPR-11"},
    "FLP-19":     {"NPR-1"},
    "FLP-21":     {"NPR-1"},
    "FLP-22":     {"NPR-1", "NPR-22"},
    "NLP-1":      {"NPR-11"},
    "NLP-3":      {"CKR-2", "NPR-17"},
    "NLP-8":      {"NPR-17"},
    "NLP-11":     {"NPR-17"},
    "NLP-12":     {"CKR-1", "CKR-2"},
    "NLP-14":     {"NPR-11", "NPR-17"},
    "NLP-15":     {"NMUR-4"},
    "NLP-17":     {"NPR-17"},
    "NLP-18":     {"CKR-1", "CKR-2", "NMUR-2"},
    "NLP-21":     {"NPR-17"},
    "NLP-40":     {"AEX-2"},
    "NTC-1":      {"NTR-1", "NTR-2"},
}


@dataclass(frozen=True)
class LRCompatibility:
    """Ligand-receptor compatibility atlas."""
    # neuron -> set of ligands it expresses (from Bentley NT + Neuropeptide)
    neuron_to_ligands: Dict[str, Set[str]]
    # neuron -> set of receptors it expresses (from Bentley Receptor rows)
    neuron_to_receptors: Dict[str, Set[str]]
    # canonical L->R pair dict
    canonical_pairs: Dict[str, Set[str]]

    def class_ligands(self, class_members: List[str]) -> Set[str]:
        """Union of ligands expressed across all neurons in a class."""
        out = set()
        for n in class_members:
            out |= self.neuron_to_ligands.get(n, set())
        return out

    def class_receptors(self, class_members: List[str]) -> Set[str]:
        """Union of receptors expressed across all neurons in a class."""
        out = set()
        for n in class_members:
            out |= self.neuron_to_receptors.get(n, set())
        return out

    def compatibility_score(
        self,
        pre_class_members: List[str],
        post_class_members: List[str],
    ) -> Tuple[int, List[Tuple[str, str]]]:
        """Return (count of matched L-R pairs, list of matched pairs).

        A pair (L, R) contributes if:
          - L is a canonical ligand name in CANONICAL_LR_PAIRS,
          - some neuron in the pre-class expresses L,
          - some neuron in the post-class expresses R,
          - R is in CANONICAL_LR_PAIRS[L].
        """
        pre_ligands = self.class_ligands(pre_class_members)
        post_receptors = self.class_receptors(post_class_members)
        matches: List[Tuple[str, str]] = []
        for L in pre_ligands:
            if L in self.canonical_pairs:
                for R in self.canonical_pairs[L] & post_receptors:
                    matches.append((L, R))
        return len(matches), matches


def load_lr_atlas() -> LRCompatibility:
    """Build the per-neuron ligand and receptor sets from Bentley 2016."""
    df = pd.read_csv(BENTLEY_FILE)

    ligand_rels = df[df["Relationship"].isin(["Neuropeptide", "Neurotransmitter"])]
    receptor_rels = df[df["Relationship"] == "Receptor"]

    neuron_to_ligands: Dict[str, Set[str]] = {}
    for _, row in ligand_rels.iterrows():
        neuron = str(row["Entity1"]).strip()
        ligand = str(row["Entity2"]).strip()
        neuron_to_ligands.setdefault(neuron, set()).add(ligand)

    neuron_to_receptors: Dict[str, Set[str]] = {}
    for _, row in receptor_rels.iterrows():
        neuron = str(row["Entity1"]).strip()
        recv = str(row["Entity2"]).strip()
        neuron_to_receptors.setdefault(neuron, set()).add(recv)

    return LRCompatibility(
        neuron_to_ligands=neuron_to_ligands,
        neuron_to_receptors=neuron_to_receptors,
        canonical_pairs=CANONICAL_LR_PAIRS,
    )
