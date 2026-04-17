"""Canonical path resolver for C-Elegans project.
Import this from any notebook: `from utils.paths import *`
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / 'data'
CONNECTOME = DATA / 'connectome'
EXPRESSION = DATA / 'expression'
CENGEN = EXPRESSION / 'cengen'
GENOMES = DATA / 'genomes'
WORMBASE = DATA / 'wormbase_release_WS297'
LINEAGE = DATA / 'lineage'
ATTRIBUTES = DATA / 'attributes'
PAPER_SUPPS = DATA / 'paper_supplements'
THIRD_PARTY = DATA / 'third_party_releases'

ANALYSIS = ROOT / 'analysis'
MODELS = ROOT / 'models'
SIMULATION = ROOT / 'simulation'
MANIFESTS = ROOT / 'manifests'
NOTEBOOKS = ROOT / 'notebooks'
RESULTS = ROOT / 'RESULTS'

# Common sub-locations
CENGEN_DERIVED = CENGEN / 'derived'
CENGEN_BULK = CENGEN / 'bulk_barrett2022'
CENGEN_SINGLE_L4 = CENGEN / 'single_cell_L4'
CENGEN_AGGR = CENGEN / 'aggregated'
CENGEN_CONCORD = CENGEN / 'concordance'
CENGEN_GOLD = CENGEN / 'gold'
CENGEN_THRESH = CENGEN / 'thresholded'

COOK2019 = CONNECTOME / 'cook2019'
WHITE = CONNECTOME / 'white_MoW'
WITVLIET = CONNECTOME / 'witvliet2020'
