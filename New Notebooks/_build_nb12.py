"""Nb 12 — Peptide 'wireless' connectome: build A_peptide, structurally characterize vs A_synaptic."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 12 — Peptide 'Wireless' Connectome Structural Characterization

## Question

C. elegans has a parallel peptidergic signaling layer — neurons secrete neuropeptides that can diffuse and activate receptors on distant cells, forming a 'wireless' connectome. This graph has been characterized separately from the synaptic connectome (Ripoll-Sanchez 2023, Bentley 2016).

Using Bentley 2016's curated neuron→peptide and neuron→receptor data + canonical peptide-receptor pairs (same as Nb06), build:

**A_peptide[i,j] = 1 if (neuron i secretes ligand L) AND (neuron j expresses receptor R) AND (L,R) is a known pair**

Then characterize:
1. How does A_peptide compare structurally to A_synaptic (Witvliet adult chemical)?
2. Are the peptide-graph hubs the same as synaptic hubs?
3. Do rich-club / small-world properties differ?
4. What fraction of synaptic edges are ALSO peptide-connected?

## Why this is different from 06

Nb06 used peptide compatibility as a FEATURE for predicting synaptic edges. Null result (compat didn't add above contact).

This notebook is NOT about predicting synapses. It's about characterizing the peptide graph as its own object and comparing its structure to the synaptic graph. This is a descriptive/network-science finding, not a predictive model.

## Preregistered criteria

1. **A_peptide is non-trivial**: density 0.005–0.30, not zero, not fully connected.
2. **Peptide-synaptic edge overlap non-trivial**: Jaccard ≥ 0.05 AND ≤ 0.70 (some overlap but not redundant).
3. **Peptide and synaptic degree sequences differ**: Spearman correlation between per-neuron peptide-degree and synaptic-degree < 0.90 (they are different hubs, not the same).
4. **Peptide graph has distinct clustering properties**: mean clustering coefficient different from synaptic by ≥ 0.05.

Any/all passing = "peptide connectome is a structurally distinct overlay on the synaptic connectome." All failing = peptide signaling simply mirrors synaptic structure."""))

cells.append(nbf.v4.new_code_cell("""import sys, time
from pathlib import Path
import warnings; warnings.simplefilter('ignore')
_HERE = Path.cwd()
if (_HERE / 'lib').is_dir(): sys.path.insert(0, str(_HERE))
elif (_HERE.parent / 'lib').is_dir(): sys.path.insert(0, str(_HERE.parent))

from lib.paths import DERIVED
from lib.lr_compatibility import load_lr_atlas, CANONICAL_LR_PAIRS

import numpy as np, pandas as pd
from scipy.stats import spearmanr

adult = np.load(DERIVED / 'connectome_adult.npz', allow_pickle=True)
w_neurons = np.array([str(n) for n in adult['neurons']])
w_chem = adult['chem_adj']
print(f'Witvliet: {len(w_neurons)} neurons, {int((w_chem > 0).sum())} chem edges')

lr = load_lr_atlas()
print(f'Bentley LR atlas loaded')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Build A_peptide per-neuron"))

cells.append(nbf.v4.new_code_cell("""# A_peptide[i, j] = 1 if some peptide-receptor pair (L, R) is covered:
#   i secretes L, j expresses R, (L, R) in CANONICAL_LR_PAIRS

N = len(w_neurons)
A_pep = np.zeros((N, N), dtype=np.int32)
pep_neurons = [n for n in w_neurons if n in lr.neuron_to_ligands and n in lr.neuron_to_receptors]
print(f'Neurons with both ligand AND receptor info: {len(pep_neurons)}')

# Consider ALL neurons (i as ligand source, j as receptor target), even if missing some info
for i, nrn_i in enumerate(w_neurons):
    ligands_i = lr.neuron_to_ligands.get(nrn_i, set())
    if not ligands_i: continue
    for j, nrn_j in enumerate(w_neurons):
        if i == j: continue
        receptors_j = lr.neuron_to_receptors.get(nrn_j, set())
        if not receptors_j: continue
        # Does any (L in ligands_i) have a receptor match in receptors_j?
        hit = False
        for L in ligands_i:
            if L in CANONICAL_LR_PAIRS:
                if CANONICAL_LR_PAIRS[L] & receptors_j:
                    hit = True
                    break
        if hit:
            A_pep[i, j] = 1

print(f'A_peptide: {A_pep.sum()} directed edges')
density_pep = A_pep.sum() / (N * (N - 1))
print(f'Density: {density_pep:.4f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Compare structure to A_synaptic"))

cells.append(nbf.v4.new_code_cell("""# Synaptic binary
A_syn = (w_chem > 0).astype(np.int32)
np.fill_diagonal(A_syn, 0)
density_syn = A_syn.sum() / (N * (N - 1))

# Edge sets
syn_edges = set((i, j) for i in range(N) for j in range(N) if i != j and A_syn[i, j])
pep_edges = set((i, j) for i in range(N) for j in range(N) if i != j and A_pep[i, j])
inter = syn_edges & pep_edges
union = syn_edges | pep_edges
jaccard = len(inter) / len(union) if union else 0

print(f'Synaptic edges:      {len(syn_edges)}')
print(f'Peptide edges:       {len(pep_edges)}')
print(f'Shared:              {len(inter)}')
print(f'Jaccard (syn, pep):  {jaccard:.4f}')
print(f'Fraction of syn that is also pep: {len(inter)/max(len(syn_edges),1):.3f}')
print(f'Fraction of pep that is also syn: {len(inter)/max(len(pep_edges),1):.3f}')

# Degree correlations
syn_in = A_syn.sum(axis=0); syn_out = A_syn.sum(axis=1)
pep_in = A_pep.sum(axis=0); pep_out = A_pep.sum(axis=1)
rho_in, p_in = spearmanr(syn_in, pep_in)
rho_out, p_out = spearmanr(syn_out, pep_out)
print(f'\\nIn-degree Spearman (syn vs pep):  {rho_in:+.3f} (p={p_in:.2e})')
print(f'Out-degree Spearman (syn vs pep): {rho_out:+.3f} (p={p_out:.2e})')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Clustering coefficients"))

cells.append(nbf.v4.new_code_cell("""def local_clustering_directed(A):
    # For directed graph: clustering coefficient per node
    # Using the formula: C(i) = (A+A^T)^3[i,i] / (d_tot[i]*(d_tot[i]-1) - 2 * sum_k A[i,k]*A[k,i])
    # Easier approximation: take undirected version
    A_und = ((A + A.T) > 0).astype(np.int32)
    deg = A_und.sum(axis=1)
    N = A.shape[0]
    cc = np.zeros(N)
    A2 = A_und @ A_und
    triangles = np.diag(A_und @ A2) / 2
    for i in range(N):
        if deg[i] < 2:
            cc[i] = 0
        else:
            cc[i] = 2 * triangles[i] / (deg[i] * (deg[i] - 1))
    return cc

cc_syn = local_clustering_directed(A_syn)
cc_pep = local_clustering_directed(A_pep)
print(f'Mean clustering: syn={cc_syn.mean():.3f}, pep={cc_pep.mean():.3f}, diff={abs(cc_syn.mean()-cc_pep.mean()):.3f}')
print(f'Mean degree: syn={A_syn.sum(axis=1).mean():.1f}, pep={A_pep.sum(axis=1).mean():.1f}')

# Correlation between per-neuron clustering across graphs
rho_cc, p_cc = spearmanr(cc_syn, cc_pep)
print(f'Clustering Spearman (syn vs pep): {rho_cc:+.3f} (p={p_cc:.2e})')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Preregistered criteria"))

cells.append(nbf.v4.new_code_cell("""crit1 = 0.005 <= density_pep <= 0.30
crit2 = 0.05 <= jaccard <= 0.70
crit3 = min(abs(rho_in), abs(rho_out)) < 0.90
crit4 = abs(cc_syn.mean() - cc_pep.mean()) >= 0.05

print('CRITERIA')
print('=' * 60)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 A_pep density in [0.005, 0.30]    {density_pep:.4f}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 Jaccard in [0.05, 0.70]            {jaccard:.4f}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 min degree Spearman < 0.90         min={min(abs(rho_in), abs(rho_out)):.3f}')
print(f'  [{\"PASS\" if crit4 else \"FAIL\"}] 4 mean CC diff >= 0.05               {abs(cc_syn.mean() - cc_pep.mean()):.3f}')
print('=' * 60)
n_pass = sum([crit1, crit2, crit3, crit4])
print(f'{n_pass}/4 criteria pass')

if n_pass >= 3:
    verdict = 'STRUCTURALLY DISTINCT — peptide connectome is a distinct graph overlay'
elif n_pass >= 2:
    verdict = 'PARTIAL — peptide graph shares some properties with synaptic but differs in others'
else:
    verdict = 'REDUNDANT — peptide graph essentially mirrors synaptic structure'
print(f'\\nVERDICT: {verdict}')

summary = pd.DataFrame([{
    'A_pep_edges': int(A_pep.sum()),
    'A_pep_density': density_pep,
    'A_syn_edges': int(A_syn.sum()),
    'A_syn_density': density_syn,
    'jaccard': jaccard,
    'deg_spearman_in': float(rho_in),
    'deg_spearman_out': float(rho_out),
    'mean_cc_syn': float(cc_syn.mean()),
    'mean_cc_pep': float(cc_pep.mean()),
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb12_final_summary.csv', index=False)
np.savez_compressed(DERIVED / 'nb12_peptide_adjacency.npz', A_peptide=A_pep, neurons=w_neurons)
print(summary.T.to_string())"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/12_peptide_wireless_connectome.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 12 written ({len(nb.cells)} cells)')
