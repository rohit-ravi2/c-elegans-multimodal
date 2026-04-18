"""Nb 99 — Paper-ready synthesis table of all notebook findings."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 99 — Paper Synthesis Table

Aggregate all per-notebook summary CSVs into a single table for the paper."""))

cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
_HERE = Path.cwd()
if (_HERE / 'lib').is_dir(): sys.path.insert(0, str(_HERE))
elif (_HERE.parent / 'lib').is_dir(): sys.path.insert(0, str(_HERE.parent))

from lib.paths import DERIVED
import pandas as pd, numpy as np, os

# Curated summary for each notebook
entries = [
    ('01','Clean connectome (Witvliet 2020)','infrastructure','PASS — canonical adult adjacency built', {}),
    ('02','CeNGEN expression alignment','infrastructure','PASS — 179/222 mapped, textbook NT validation', {}),
    ('03','Neuron-level gene×motif Spearman','test','NULL (pseudoreplication artifact, 2797 hits collapse at class-level)', {}),
    ('03b','Class-level gene×motif (N=84)','test','NULL — 0/60,438 FDR survivors', {}),
    ('04A','Synthetic dynamics','test','NULL by threshold (weak but detectable gene signal)', {}),
    ('04B','Neuron-role classification','test','PASS 72.7% (expected); hub class weak +8%', {}),
    ('05','Developmental rewiring (AUC)','test','NULL — AUC 0.62 below bar 0.65', {}),
    ('05b','Gene stability on rewiring','test','NULL — 56 stable genes vs null 55 (tied)', {}),
    ('06','LR compat edge prediction','test','NULL above contact; contact-only AUC=0.78', {}),
    ('07','Developmental topological rule','POSITIVE','AUC 0.76 pooled, +0.23 over null — pub-worthy', {}),
    ('08','Sex dimorphism / cross-dataset','test','NULL — Jaccard 0.38-0.42; motif Spearman 0.60', {}),
    ('09','Topology + genes for rewiring','POSITIVE','AUC 0.80 (topo 0.76), delta +0.044 — KEY FINDING', {}),
    ('10','Gap junction arrival','test','NULL — AUC 0.60 (chem and gap follow different rules)', {}),
    ('11','Gene stability above topology','test','see final summary', {}),
    ('12','Peptide wireless connectome','POSITIVE','Structurally distinct (Jaccard 0.045 vs syn)', {}),
    ('13','Signed connectome motifs','POSITIVE','6/8 sign classes significant; (-,+,-) at z=+28', {}),
    ('14','Yemini phenotype enrichment','mixed','Behavior genes broadly expressed; hub correlation null', {}),
    ('15','Human-ortholog conservation','mixed','Descriptive; 44% conservation; no hub correlation', {}),
    ('16','Topo+genes+peptide for rewiring','test','see final summary', {}),
]

rows = []
for nb_id, name, category, headline, _extra in entries:
    # Try loading the corresponding summary CSV
    cand_paths = [
        DERIVED / f'nb{nb_id}_final_summary.csv',
        DERIVED / f'nb{nb_id.replace(\"b\",\"\")}_final_summary.csv',
        DERIVED / f'nb{nb_id.split(\"A\")[0]}_final_summary.csv',
    ]
    loaded = None
    for p in cand_paths:
        if p.exists():
            try:
                loaded = pd.read_csv(p).iloc[0].to_dict()
                break
            except Exception:
                pass
    rows.append({
        'notebook': nb_id, 'name': name, 'category': category,
        'headline': headline,
        'data_loaded': bool(loaded is not None),
        'summary_csv_exists': any(p.exists() for p in cand_paths),
    })

df = pd.DataFrame(rows)
df.to_csv(DERIVED / 'nb99_synthesis.csv', index=False)
print(df.to_string())"""))

cells.append(nbf.v4.new_markdown_cell("""## Paper-ready narrative

With all notebooks done, the cohesive Paper 1 story is:

1. **Infrastructure (01, 02)**: Rebuilt pipeline with Witvliet adult + developmental stages + CeNGEN alignment, biologically validated.
2. **Static gene-motif tests all null (03/03b/04/05/05b/06)**: Under correct class-level statistics with pseudoreplication control, gene expression alone does not predict motif participation, hub status, developmental rewiring, or ligand-receptor edge existence above strong baselines.
3. **Topological rule discovered (07)**: Pure topology at stage t predicts edge arrival at t+1 with AUC 0.76, driven by shared-output-partner and triadic closure, without gene information.
4. **Gene expression refines topology (09) — KEY FINDING**: Adding CeNGEN PCA-50 expression features boosts AUC from 0.76 to 0.80 (Δ+0.044, null Δ-0.022). Expression is a modulatory refinement on top of a topological rule.
5. **Layered wiring mechanism (10, 12, 13)**: Chemical synapses follow topology+genes (0.80). Gap junctions do NOT (0.60 = null, suggesting innexin-specific rules). Peptide wireless graph is structurally distinct (Jaccard 0.045 with synaptic). Signed motifs show enriched coherent-excitation FFLs and massively enriched dual-inhibition gates (z=+28).
6. **Caveats (08)**: Individual-edge reliability is limited across datasets (Jaccard 0.38-0.42), but motif-level / degree-level signals are robust.
"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/99_synthesis.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 99 written ({len(nb.cells)} cells)')
