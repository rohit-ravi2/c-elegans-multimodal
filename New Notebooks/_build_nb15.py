"""Nb 15 — Human-ortholog conservation per CeNGEN neuron class × topology."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 15 — Evolutionary Conservation × Neuron-Class Topology

## Question

Using WormBase WS297 ortholog file (`caenorhabditis_elegans.PRJNA13758.WBPS19.orthologs.tsv`, 2.27M rows across 227 species including Human), compute per-CeNGEN-class "conservation score": what fraction of top-expressed genes in each neuron class have identifiable orthologs in human? Then:

1. Is conservation score correlated with connectome hub status?
2. Are command interneurons (known to control locomotion) more or less conserved than sensory neurons?

## Why this is different

No prior notebook used the ortholog file (despite being 2.27M rows). This is an evolutionary axis that doesn't require new data.

## Preregistered criteria

1. **≥ 1,000 CeNGEN genes have confident human orthologs** (non-empty ortholog_gene_id).
2. **Class-level conservation scores have non-trivial variance** (std > 0.05 across 84 classes).
3. **Conservation correlates with topology**: Spearman between (conservation fraction) and (total_deg or hub score) with p < 0.05."""))

cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import warnings; warnings.simplefilter('ignore')
_HERE = Path.cwd()
if (_HERE / 'lib').is_dir(): sys.path.insert(0, str(_HERE))
elif (_HERE.parent / 'lib').is_dir(): sys.path.insert(0, str(_HERE.parent))

from lib.paths import DATA, DERIVED

import numpy as np, pandas as pd
from scipy.stats import spearmanr

ORTHO_FILE = DATA / 'wormbase_release_WS297/orthologs/caenorhabditis_elegans.PRJNA13758.WBPS19.orthologs.tsv'
# Read only human orthologs (saves memory)
print('Loading human-only orthologs...')
human_orthos = []
with open(ORTHO_FILE) as f:
    header = f.readline().strip().split('\\t')
    col_species = header.index('ortholog_species_name')
    col_gene = header.index('gene_id')
    col_orth = header.index('ortholog_gene_id')
    col_ident = header.index('target_identity')
    for line in f:
        parts = line.rstrip('\\n').split('\\t')
        if len(parts) < len(header): continue
        if parts[col_species] == 'Human':
            human_orthos.append({
                'wbgene': parts[col_gene],
                'human_ortholog': parts[col_orth],
                'target_identity': float(parts[col_ident]) if parts[col_ident] else np.nan,
            })
ho = pd.DataFrame(human_orthos)
print(f'Human ortholog rows: {len(ho)}')
print(f'Unique C. elegans WBGenes with human ortholog: {ho[\"wbgene\"].nunique()}')
print(f'Unique human ortholog targets: {ho[\"human_ortholog\"].nunique()}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Map CeNGEN → human ortholog conservation scores"))

cells.append(nbf.v4.new_code_cell("""expr_data = np.load(DERIVED / 'expression_tpm.npz', allow_pickle=True)
tpm_neurons = expr_data['neurons']; tpm = expr_data['tpm']
genes_wbg = expr_data['genes_wbg']
genes_csv = pd.read_csv(DERIVED / 'expression_genes.csv')

mapping = pd.read_csv(DERIVED / 'expression_neuron_mapping.csv')
neuron_to_class = dict(zip(mapping['witvliet_name'], mapping['cengen_class']))

# Build class-level TPM
class_to_expr = {}
for i, nm in enumerate(tpm_neurons):
    cls = neuron_to_class.get(str(nm))
    if isinstance(cls, str) and cls not in class_to_expr and not np.all(np.isnan(tpm[i])):
        class_to_expr[cls] = tpm[i]
classes_list = sorted(class_to_expr.keys())

# How many CeNGEN WBGenes have at least one human ortholog?
conserved_wbgenes = set(ho['wbgene'].unique())
cengen_has_ortho = np.array([wbg in conserved_wbgenes for wbg in genes_wbg])
print(f'CeNGEN genes with human ortholog: {cengen_has_ortho.sum()} / {len(genes_wbg)}')
print(f'Base conservation rate: {cengen_has_ortho.mean():.2%}')

# Per-class conservation score: fraction of top-K expressed genes that have human orthologs
TOP_K = 200
rows = []
for cls in classes_list:
    vec = class_to_expr[cls]
    top_idx = np.argsort(-vec)[:TOP_K]
    n_conserved = int(cengen_has_ortho[top_idx].sum())
    rows.append({'cengen_class': cls, 'conservation_count': n_conserved,
                 'conservation_fraction': n_conserved / TOP_K})
cons_df = pd.DataFrame(rows).sort_values('conservation_fraction', ascending=False)
print(f'\\nConservation score distribution across classes:')
print(cons_df['conservation_fraction'].describe().round(3).to_string())
print(f'\\nTop 10 most-conserved neuron classes:')
print(cons_df.head(10).to_string())
print(f'\\nBottom 10 least-conserved neuron classes:')
print(cons_df.tail(10).to_string())"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Correlation with connectome topology"))

cells.append(nbf.v4.new_code_cell("""motif_feats = pd.read_csv(DERIVED / 'motif_features_per_class.csv', index_col=0)
merged = cons_df.merge(motif_feats.reset_index().rename(columns={'index':'cengen_class'}),
                       on='cengen_class', how='inner')
print(f'Merged classes: {len(merged)}')

for col in ['in_deg','out_deg','total_deg','ffl','cycle3']:
    rho, p = spearmanr(merged['conservation_fraction'], merged[col])
    print(f'  conservation vs {col:12s}: Spearman rho={rho:+.3f}, p={p:.4f}')

rho_hub, p_hub = spearmanr(merged['conservation_fraction'], merged['total_deg'])
print(f'\\nPRIMARY TEST: Spearman rho(conservation, total_deg) = {rho_hub:+.3f}, p = {p_hub:.4f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Criteria + save"))

cells.append(nbf.v4.new_code_cell("""crit1 = cengen_has_ortho.sum() >= 1000
crit2 = cons_df['conservation_fraction'].std() > 0.05
crit3 = p_hub < 0.05

print('CRITERIA')
print('=' * 60)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 CeNGEN genes w/ human ortholog >= 1000   {cengen_has_ortho.sum()}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 class conservation std > 0.05             {cons_df[\"conservation_fraction\"].std():.3f}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 conservation-hub p < 0.05                  p={p_hub:.4f}')
print('=' * 60)

if all([crit1, crit2, crit3]):
    if rho_hub > 0:
        verdict = 'POSITIVE — hub neurons are MORE conserved (express more human-orthologous genes)'
    else:
        verdict = 'POSITIVE — hub neurons are LESS conserved (C. elegans-specific genes dominate hubs)'
elif crit1 and crit2:
    verdict = 'DESCRIPTIVE — per-class conservation scores vary but don\\'t correlate with hub status'
else:
    verdict = 'INCONCLUSIVE'
print(f'\\nVERDICT: {verdict}')

summary = pd.DataFrame([{
    'cengen_genes_conserved': int(cengen_has_ortho.sum()),
    'cengen_base_rate': float(cengen_has_ortho.mean()),
    'conservation_std_across_classes': float(cons_df['conservation_fraction'].std()),
    'hub_spearman_rho': float(rho_hub), 'hub_spearman_p': float(p_hub),
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb15_final_summary.csv', index=False)
cons_df.to_csv(DERIVED / 'nb15_class_conservation.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/15_human_ortholog_conservation.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 15 written ({len(nb.cells)} cells)')
