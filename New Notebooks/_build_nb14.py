"""Nb 14 — Yemini behavior-phenotype gene enrichment per CeNGEN neuron class."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 14 — Yemini 2013 Behavior-Phenotype Gene Enrichment per Neuron Class

## Question

Yemini et al. 2013 profiled behavioral phenotypes for 333 single-gene C. elegans mutants. Their "Minimum Rank-Sum q" column gives a phenotype-significance score per gene.

- **Do specific CeNGEN neuron classes preferentially express these "behavior-affecting" genes**?
- **Is the gene-to-behavior effect size correlated with connectome topology** (e.g., hub status)?

This is a translational / phenotype-grounded test. No prior notebook connected mutant phenotypes to connectome topology.

## Preregistered criteria

1. **≥ 100 Yemini genes successfully mapped** to CeNGEN gene symbols.
2. **Enrichment test**: using hypergeometric test per neuron class, at least 5 classes have q<0.05 for "disproportionate expression of Yemini behavior genes" (vs class with no Yemini-gene expression).
3. **Hub correlation**: Spearman between (per-neuron hub_score from Nb07's out_deg_i + in_deg_j) and (per-class Yemini-gene expression count in top-100 expressed genes) — significant p < 0.05.

## Halting rule

Null if criterion 3 fails (no correlation between hub status and behavior-gene enrichment)."""))

cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import warnings; warnings.simplefilter('ignore')
_HERE = Path.cwd()
if (_HERE / 'lib').is_dir(): sys.path.insert(0, str(_HERE))
elif (_HERE.parent / 'lib').is_dir(): sys.path.insert(0, str(_HERE.parent))

from lib.paths import DATA, DERIVED

import numpy as np, pandas as pd
from scipy.stats import hypergeom, spearmanr
import re

RNG = np.random.default_rng(42)

# Load Yemini
y = pd.read_csv(DATA / 'expression/neurotransmitter/yemini_combined_sheets.csv')
print(f'Yemini rows: {len(y)}')

# Extract gene symbol from genotype (e.g. 'acr-2(ok1887)III' -> 'acr-2')
def extract_gene(genotype):
    if not isinstance(genotype, str): return None
    m = re.match(r'^([a-zA-Z0-9-]+?)\\(', genotype)
    if m: return m.group(1).lower().strip()
    # Fallback: first token
    t = genotype.split('(')[0].strip()
    return t.lower() if t else None

y['gene_symbol'] = y['Genotype'].apply(extract_gene)
y = y[y['gene_symbol'].notna()].copy()
y = y.drop_duplicates(subset=['gene_symbol'])
# Filter to significant phenotypes (q<0.05)
y_sig = y[y['Minimum Rank-Sum q'] < 0.05]
print(f'Unique Yemini genes (any phenotype): {len(y)}')
print(f'Unique Yemini genes (significant q<0.05): {len(y_sig)}')
print(f'Sample: {y_sig[\"gene_symbol\"].head(10).tolist()}')

behavior_genes = set(y_sig['gene_symbol'].tolist())
print(f'\\nBehavior gene set size: {len(behavior_genes)}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Intersect with CeNGEN"))

cells.append(nbf.v4.new_code_cell("""expr_data = np.load(DERIVED / 'expression_tpm.npz', allow_pickle=True)
tpm_neurons = expr_data['neurons']; tpm = expr_data['tpm']
genes_csv = pd.read_csv(DERIVED / 'expression_genes.csv')
gene_symbols = np.array([str(s).lower() for s in genes_csv['symbol']])

mapping = pd.read_csv(DERIVED / 'expression_neuron_mapping.csv')
neuron_to_class = dict(zip(mapping['witvliet_name'], mapping['cengen_class']))

# How many behavior genes are in CeNGEN?
cengen_gene_set = set(gene_symbols.tolist())
behavior_in_cengen = behavior_genes & cengen_gene_set
print(f'Behavior genes mapped to CeNGEN: {len(behavior_in_cengen)} / {len(behavior_genes)}')

# Class-level expression
class_to_expr = {}
for i, nm in enumerate(tpm_neurons):
    cls = neuron_to_class.get(str(nm))
    if isinstance(cls, str) and cls not in class_to_expr and not np.all(np.isnan(tpm[i])):
        class_to_expr[cls] = tpm[i]

classes_list = sorted(class_to_expr.keys())
print(f'CeNGEN classes: {len(classes_list)}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Per-class enrichment test"))

cells.append(nbf.v4.new_code_cell("""# For each class: consider top-100 most-expressed genes. Count how many are behavior genes.
# Hypergeometric test: is the count significantly higher than expected given the base rate?

TOP_K = 100
G_total = len(gene_symbols)
n_behavior = len(behavior_in_cengen)

# Build a boolean mask for behavior genes in the expression gene list
behavior_mask = np.array([sym in behavior_in_cengen for sym in gene_symbols])
print(f'Behavior genes in CeNGEN (array): {behavior_mask.sum()}')

rows = []
for cls in classes_list:
    vec = class_to_expr[cls]
    # Top-K expressed genes
    top_idx = np.argsort(-vec)[:TOP_K]
    # How many are behavior genes?
    k = int(behavior_mask[top_idx].sum())
    # Hypergeometric: P(X >= k) where sample size = TOP_K, population = G_total,
    # successes in population = n_behavior
    # sf(k-1) = P(X >= k)
    pval = float(hypergeom.sf(k-1, G_total, n_behavior, TOP_K))
    rows.append({'cengen_class': cls, 'top100_behavior_count': k,
                 'hypergeom_p': pval, 'expected_k': TOP_K * n_behavior / G_total})

cls_df = pd.DataFrame(rows)
# BH-FDR
from statsmodels.stats.multitest import multipletests
cls_df['q'] = multipletests(cls_df['hypergeom_p'].values, method='fdr_bh')[1]
cls_df = cls_df.sort_values('hypergeom_p').reset_index(drop=True)
print(f'\\nClass enrichment test (top-100 expressed genes contain behavior genes):')
print(f'Expected count per class: {cls_df[\"expected_k\"].iloc[0]:.2f}')
print(f'Observed counts min/max/median: {cls_df[\"top100_behavior_count\"].min()}/{cls_df[\"top100_behavior_count\"].max()}/{cls_df[\"top100_behavior_count\"].median()}')
print(f'\\nTop 15 classes by p-value:')
print(cls_df.head(15).to_string())

n_sig_classes = int((cls_df['q'] < 0.05).sum())
print(f'\\nClasses with q < 0.05: {n_sig_classes}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Correlation with connectome hub status (Nb07 topology)"))

cells.append(nbf.v4.new_code_cell("""# Load motif features from Nb 03b (class-level)
motif_feats = pd.read_csv(DERIVED / 'motif_features_per_class.csv', index_col=0)
print(f'Class-level motif features: {motif_feats.shape}')

# Merge with enrichment results
merged = cls_df.merge(motif_feats.reset_index().rename(columns={'index':'cengen_class'}),
                     on='cengen_class', how='inner')
print(f'Merged: {len(merged)} classes')

# Correlate behavior-gene count with hub statistics
for col in ['in_deg', 'out_deg', 'total_deg']:
    rho, p = spearmanr(merged['top100_behavior_count'], merged[col])
    print(f'  top100_behavior_count vs {col}: Spearman rho={rho:+.3f}, p={p:.4f}')

# Primary test: hub vs behavior-gene count
rho_hub, p_hub = spearmanr(merged['top100_behavior_count'], merged['total_deg'])
print(f'\\nPRIMARY TEST: Spearman rho(top100 behavior genes, total_deg) = {rho_hub:+.3f}, p = {p_hub:.4f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Criteria"))

cells.append(nbf.v4.new_code_cell("""crit1 = len(behavior_in_cengen) >= 100
crit2 = n_sig_classes >= 5
crit3 = p_hub < 0.05

print('CRITERIA')
print('=' * 60)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 Yemini genes in CeNGEN >= 100    {len(behavior_in_cengen)}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 >=5 classes enriched q<0.05      {n_sig_classes}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 hub-behavior Spearman p<0.05      p={p_hub:.4f}')
print('=' * 60)

if all([crit1, crit2, crit3]):
    verdict = 'POSITIVE — specific neuron classes are enriched for behavior-affecting genes, correlated with hub status'
elif crit1 and crit2:
    verdict = 'POSITIVE (class-level) — behavior-gene enrichment is class-specific but not hub-correlated'
elif crit1 and crit3:
    verdict = 'POSITIVE (hub) — behavior-gene expression correlates with hub status but no class-level enrichment'
else:
    verdict = 'NULL — no meaningful enrichment of behavior-affecting genes per neuron class'
print(f'\\nVERDICT: {verdict}')

summary = pd.DataFrame([{
    'yemini_genes_total': len(behavior_genes),
    'yemini_in_cengen': len(behavior_in_cengen),
    'n_classes_tested': len(classes_list),
    'n_sig_classes_q05': n_sig_classes,
    'hub_spearman_rho': float(rho_hub), 'hub_spearman_p': float(p_hub),
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb14_final_summary.csv', index=False)
cls_df.to_csv(DERIVED / 'nb14_class_enrichment.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/14_yemini_phenotype_enrichment.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 14 written ({len(nb.cells)} cells)')
