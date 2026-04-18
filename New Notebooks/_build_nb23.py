"""Nb 23 — Multiplex hubs: neurons in top 10% of BOTH synaptic and peptide graphs (descriptive)."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 23 — Multiplex Hubs (Descriptive)

## Question

Nb12 showed the peptide wireless connectome is structurally distinct from the synaptic graph (Jaccard 0.045). But some neurons may be hubs in BOTH. These "multiplex hubs" would be bridge neurons translating between fast-reflex wiring and slow-neuromodulatory state.

Identify the multiplex hubs and do a light enrichment analysis.

## Scope

This is **descriptive** — not a formal hypothesis test. N ≈ 20 multiplex hubs is too small for FDR on 13k genes. Instead we report:
- List of multiplex hubs
- Their known NT identities and functional classes
- Whether they cluster in any annotated category

## Preregistered criteria (descriptive)

1. **≥ 5 multiplex hubs identified** (neurons top-10% in both graphs).
2. **Report the list and cross-check with Loer & Rand NT classification**."""))

cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import warnings; warnings.simplefilter('ignore')
_HERE = Path.cwd()
if (_HERE / 'lib').is_dir(): sys.path.insert(0, str(_HERE))
elif (_HERE.parent / 'lib').is_dir(): sys.path.insert(0, str(_HERE.parent))

from lib.paths import DERIVED
from lib.reference import load_nt_reference
import numpy as np, pandas as pd

adult = np.load(DERIVED / 'connectome_adult.npz', allow_pickle=True)
w_neurons = np.array([str(n) for n in adult['neurons']])
w_chem = adult['chem_adj']
w_gap  = adult['gap_adj']

pep = np.load(DERIVED / 'nb12_peptide_adjacency.npz', allow_pickle=True)
A_pep = pep['A_peptide'].astype(np.int32)
pep_neurons = np.array([str(n) for n in pep['neurons']])

nt = load_nt_reference()

# Alignment
assert len(w_neurons) == len(pep_neurons)
assert (w_neurons == pep_neurons).all(), 'Witvliet and peptide neuron order differ'

print(f'N={len(w_neurons)} neurons, synaptic edges={int((w_chem>0).sum())}, peptide edges={A_pep.sum()}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Compute multiplex hub membership"))

cells.append(nbf.v4.new_code_cell("""# Synaptic: directed in+out degree
syn_total = (w_chem > 0).astype(int).sum(axis=1) + (w_chem > 0).astype(int).sum(axis=0)
# Peptide: directed in+out degree
pep_total = A_pep.sum(axis=1) + A_pep.sum(axis=0)

thresh_syn = np.percentile(syn_total, 90)
thresh_pep = np.percentile(pep_total, 90)

syn_hub = (syn_total >= thresh_syn)
pep_hub = (pep_total >= thresh_pep)
multi_hub = syn_hub & pep_hub
syn_only = syn_hub & ~pep_hub
pep_only = pep_hub & ~syn_hub

print(f'Synaptic hubs (top 10%):  {syn_hub.sum()} (threshold >={thresh_syn:.0f})')
print(f'Peptide hubs (top 10%):   {pep_hub.sum()} (threshold >={thresh_pep:.0f})')
print(f'MULTIPLEX HUBS (both):    {multi_hub.sum()}')
print(f'Synaptic-only hubs:       {syn_only.sum()}')
print(f'Peptide-only hubs:        {pep_only.sum()}')

multi_hub_neurons = w_neurons[multi_hub]
syn_only_neurons = w_neurons[syn_only]
pep_only_neurons = w_neurons[pep_only]"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Multiplex hubs with NT classification"))

cells.append(nbf.v4.new_code_cell("""def nt_of(n):
    v = nt.nt_of(n)
    if v is None: return 'Unknown'
    s = v.lower()
    if 'acetylcholine' in s: return 'ACh'
    if 'gaba' in s: return 'GABA'
    if 'glutamate' in s: return 'Glu'
    if 'dopamine' in s: return 'Dopamine'
    if 'serotonin' in s: return 'Serotonin'
    if 'octopamine' in s: return 'Octopamine'
    if 'tyramine' in s: return 'Tyramine'
    return str(v)

multi_df = pd.DataFrame({
    'neuron': multi_hub_neurons,
    'syn_total_degree': syn_total[multi_hub],
    'pep_total_degree': pep_total[multi_hub],
    'NT': [nt_of(n) for n in multi_hub_neurons],
}).sort_values('syn_total_degree', ascending=False)

print('MULTIPLEX HUBS (top 10% in both synaptic AND peptide graphs):')
print(multi_df.to_string(index=False))
print(f'\\nNT breakdown:')
print(multi_df['NT'].value_counts().to_string())

# Compare to syn-only hubs
syn_only_df = pd.DataFrame({
    'neuron': syn_only_neurons,
    'NT': [nt_of(n) for n in syn_only_neurons],
})
print(f'\\nSyn-only hub NT breakdown:')
print(syn_only_df['NT'].value_counts().to_string())

pep_only_df = pd.DataFrame({
    'neuron': pep_only_neurons,
    'NT': [nt_of(n) for n in pep_only_neurons],
})
print(f'\\nPep-only hub NT breakdown:')
print(pep_only_df['NT'].value_counts().to_string())"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — NT enrichment test (Fisher exact)"))

cells.append(nbf.v4.new_code_cell("""from scipy.stats import fisher_exact

# Is any NT class over-represented among multiplex hubs?
for nt_class in ['ACh','Glu','GABA','Dopamine','Serotonin']:
    # 2x2 table: (multiplex hub, other) x (NT=class, NT!=class)
    all_nts = np.array([nt_of(n) for n in w_neurons])
    a = int(((all_nts == nt_class) & multi_hub).sum())  # this-NT AND multiplex
    b = int(((all_nts == nt_class) & ~multi_hub).sum()) # this-NT AND not-multiplex
    c = int(((all_nts != nt_class) & multi_hub).sum())
    d = int(((all_nts != nt_class) & ~multi_hub).sum())
    try:
        odds, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
    except Exception:
        odds, pval = 0, 1
    total_nt = a + b
    expected = multi_hub.sum() * total_nt / len(w_neurons) if len(w_neurons) else 0
    print(f'  {nt_class:12s}: multiplex={a}/{multi_hub.sum()} vs all-neurons {total_nt}/{len(w_neurons)}  expected≈{expected:.1f}  odds={odds:.2f}  p={pval:.3e}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Criteria + save"))

cells.append(nbf.v4.new_code_cell("""crit1 = multi_hub.sum() >= 5

print('CRITERIA')
print('=' * 60)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] >= 5 multiplex hubs      {multi_hub.sum()}')
print('=' * 60)

if crit1:
    verdict = f'DESCRIPTIVE — {multi_hub.sum()} multiplex hubs identified; see list for cross-system bridges'
else:
    verdict = 'INCONCLUSIVE — too few multiplex hubs for analysis'
print(f'VERDICT: {verdict}')

summary = pd.DataFrame([{
    'n_synaptic_hubs': int(syn_hub.sum()),
    'n_peptide_hubs': int(pep_hub.sum()),
    'n_multiplex_hubs': int(multi_hub.sum()),
    'syn_threshold': float(thresh_syn),
    'pep_threshold': float(thresh_pep),
    'multiplex_hubs_list': ';'.join(multi_hub_neurons.tolist()),
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb23_final_summary.csv', index=False)
multi_df.to_csv(DERIVED / 'nb23_multiplex_hubs.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/23_multiplex_hubs.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 23 written ({len(nb.cells)} cells)')
