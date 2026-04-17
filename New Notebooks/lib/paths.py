"""Canonical paths used across the New Notebooks stack.

All paths resolve relative to the C-Elegans project root, discovered by walking
up from this file's location. No hardcoded absolute paths.
"""
from __future__ import annotations
from pathlib import Path

_LIB_DIR = Path(__file__).resolve().parent
_NEW_NB_DIR = _LIB_DIR.parent
PROJECT_ROOT = _NEW_NB_DIR.parent
assert (PROJECT_ROOT / "data").is_dir(), f"Unexpected layout: {PROJECT_ROOT}"

DATA = PROJECT_ROOT / "data"
CONNECTOME = DATA / "connectome"
WITVLIET = CONNECTOME / "witvliet2020"
EXPRESSION = DATA / "expression"
CENGEN = EXPRESSION / "cengen"
CENGEN_BULK = CENGEN / "bulk_barrett2022"
CENGEN_AGGR = CENGEN / "aggregated"
CENGEN_L4 = CENGEN / "single_cell_L4"
LINEAGE = DATA / "lineage"
WORMBASE = DATA / "wormbase_release_WS297"

DERIVED = _NEW_NB_DIR / "data_derived"
DERIVED.mkdir(parents=True, exist_ok=True)

WITVLIET_FILES = {
    "L1_1": WITVLIET / "witvliet_2020_1 L1.xlsx",
    "L1_2": WITVLIET / "witvliet_2020_2 L1.xlsx",
    "L1_3": WITVLIET / "witvliet_2020_3 L1.xlsx",
    "L1_4": WITVLIET / "witvliet_2020_4 L1.xlsx",
    "L2":   WITVLIET / "witvliet_2020_5 L2.xlsx",
    "L3":   WITVLIET / "witvliet_2020_6 L3.xlsx",
    "adult": WITVLIET / "witvliet_2020_7 adult.xlsx",
}
