"""Nb 16 — Does peptide-graph connectivity add to topology+genes edge arrival model?"""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 16 — Topology + Genes + Peptide-Graph Edge Arrival

## Question

Nb09 showed topology + CeNGEN gene PCs predict edge arrival at AUC 0.80. Nb12 showed the peptide-signaling graph is structurally distinct from the synaptic graph (Jaccard 0.045).

**Does adding peptide-graph connectivity features** (is there a pre-post peptide signaling edge? Does the post neuron receive peptide signals from the pre neuron's peptide-neighborhood?) **improve on Nb09's AUC 0.80?**

If yes: peptide-wireless graph adds predictive power beyond synaptic topology + gene expression.
If no: peptide graph is redundant with topology + genes for this task.

## Preregistered criteria

1. **Topology+genes AUC reproduces Nb09**: AUC ∈ [0.78, 0.82].
2. **Topology+genes+peptide AUC ≥ topology+genes AUC** (monotone improvement).
3. **Permutation null** (shuffle peptide features, keep topology+genes): shuffled delta < 0.
4. **Delta ≥ 0.01**: meaningful incremental improvement."""))

cells.append(nbf.v4.new_code_cell("""import sys, time
from pathlib import Path
import warnings; warnings.simplefilter('ignore')
_HERE = Path.cwd()
if (_HERE / 'lib').is_dir(): sys.path.insert(0, str(_HERE))
elif (_HERE.parent / 'lib').is_dir(): sys.path.insert(0, str(_HERE.parent))

from lib.paths import DERIVED
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(42)

# Load Nb07 candidates, expression, peptide graph, mapping
cand = pd.read_csv(DERIVED / 'nb07_pooled_candidates.csv')
print(f'Nb07 candidates: {len(cand)}')

pep = np.load(DERIVED / 'nb12_peptide_adjacency.npz', allow_pickle=True)
A_pep = pep['A_peptide'].astype(np.int32)
pep_neurons = np.array([str(n) for n in pep['neurons']])
print(f'A_peptide: {A_pep.shape}, edges={A_pep.sum()}')

expr_data = np.load(DERIVED / 'expression_tpm.npz', allow_pickle=True)
tpm_neurons = expr_data['neurons']; tpm = expr_data['tpm']
mapping = pd.read_csv(DERIVED / 'expression_neuron_mapping.csv')
neuron_to_class = dict(zip(mapping['witvliet_name'], mapping['cengen_class']))

# Common-185 list (from Nb07)
STAGES = ['L1_1','L1_2','L1_3','L1_4','L2','L3','adult']
s_neurons = {}
for s in STAGES:
    d = np.load(DERIVED / 'developmental' / f'connectome_{s}.npz', allow_pickle=True)
    s_neurons[s] = set(str(n) for n in d['neurons'])
common = sorted(set.intersection(*s_neurons.values()))
print(f'Common-185: {len(common)}')

# Build peptide-graph index map (Witvliet neurons -> peptide-graph index)
pep_idx = {n: i for i, n in enumerate(pep_neurons)}"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Build peptide features per candidate"))

cells.append(nbf.v4.new_code_cell("""# For each candidate (i, j) from Nb07:
#   pep_direct: does peptide edge exist from i to j?
#   pep_reverse: from j to i?
#   pep_shared_targets: # of k such that i->k and j->k in peptide graph
#   pep_shared_sources: # of k such that k->i and k->j in peptide graph

# Map Nb07 i, j indices (into common-185) to peptide-graph indices
def nb07_idx_to_pep_idx(idx):
    n = common[int(idx)]
    return pep_idx.get(n)

# Precompute peptide-graph matrix on common-185 ordering
# Instead, just work in peptide-graph space directly
A_p = A_pep

pep_direct = np.zeros(len(cand), dtype=np.int32)
pep_reverse = np.zeros(len(cand), dtype=np.int32)
pep_shared_targets = np.zeros(len(cand), dtype=np.int32)
pep_shared_sources = np.zeros(len(cand), dtype=np.int32)

for idx, row in cand.iterrows():
    pi = nb07_idx_to_pep_idx(row['i'])
    pj = nb07_idx_to_pep_idx(row['j'])
    if pi is None or pj is None: continue
    pep_direct[idx] = A_p[pi, pj]
    pep_reverse[idx] = A_p[pj, pi]
    pep_shared_targets[idx] = int((A_p[pi, :] & A_p[pj, :]).sum())
    pep_shared_sources[idx] = int((A_p[:, pi] & A_p[:, pj]).sum())

print(f'pep_direct > 0: {(pep_direct > 0).sum()}')
print(f'pep_shared_targets mean: {pep_shared_targets.mean():.2f}')
print(f'pep_shared_sources mean: {pep_shared_sources.mean():.2f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Gene PCs (reproduce Nb09 features)"))

cells.append(nbf.v4.new_code_cell("""class_to_expr = {}
for i, nm in enumerate(tpm_neurons):
    cls = neuron_to_class.get(str(nm))
    if isinstance(cls, str) and cls not in class_to_expr and not np.all(np.isnan(tpm[i])):
        class_to_expr[cls] = tpm[i]
classes_list = sorted(class_to_expr.keys())
X_cls = np.stack([np.log1p(class_to_expr[c]) for c in classes_list])
pca_model = PCA(n_components=50, random_state=42).fit(X_cls)
X_pca = pca_model.transform(X_cls)
class_to_pc = {c: X_pca[i] for i, c in enumerate(classes_list)}
ZERO = np.zeros(50)

def get_pc(n):
    cls = neuron_to_class.get(n)
    if isinstance(cls, str) and cls in class_to_pc:
        return class_to_pc[cls]
    return ZERO

gene_pre = np.stack([get_pc(common[int(r['i'])]) for _, r in cand.iterrows()])
gene_post = np.stack([get_pc(common[int(r['j'])]) for _, r in cand.iterrows()])"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Three nested models"))

cells.append(nbf.v4.new_code_cell("""TOP_COVS = ['out_deg_i','in_deg_j','shared_out','shared_in','triangle_closure','reverse_2step']
X_topo = cand[TOP_COVS].values.astype(float)
X_topo_genes = np.concatenate([X_topo, gene_pre, gene_post], axis=1)

pep_feats = np.stack([pep_direct, pep_reverse, pep_shared_targets, pep_shared_sources], axis=1).astype(float)
X_full = np.concatenate([X_topo_genes, pep_feats], axis=1)

y = cand['arrived'].values.astype(int)

def cv_auc(X, y, n_splits=5, random_state=42, C=1.0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    for tr, te in skf.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C))
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.array(aucs)

aucs_topo = cv_auc(X_topo, y)
aucs_tg = cv_auc(X_topo_genes, y)
aucs_full = cv_auc(X_full, y)
print(f'Topology-only:            AUC {aucs_topo.mean():.3f} ± {aucs_topo.std():.3f}')
print(f'Topology + genes:         AUC {aucs_tg.mean():.3f} ± {aucs_tg.std():.3f}  (Nb09 replicate)')
print(f'Topology + genes + peptide: AUC {aucs_full.mean():.3f} ± {aucs_full.std():.3f}')
delta_gtop = aucs_tg.mean() - aucs_topo.mean()
delta_pepadd = aucs_full.mean() - aucs_tg.mean()
print(f'\\nDelta (genes over topology): {delta_gtop:+.4f}')
print(f'Delta (peptide over topo+genes): {delta_pepadd:+.4f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Permutation null on peptide features"))

cells.append(nbf.v4.new_code_cell("""N_PERM = 50
null_deltas = []
t0 = time.time()
for i in range(N_PERM):
    perm = RNG.permutation(X_full.shape[0])
    X_shuf = X_full.copy()
    X_shuf[:, -4:] = X_full[perm, -4:]
    aucs_s = cv_auc(X_shuf, y, random_state=42+i)
    null_deltas.append(aucs_s.mean() - aucs_tg.mean())
    if (i+1) % 10 == 0:
        print(f'  perm {i+1}/{N_PERM} ({time.time()-t0:.1f}s)  last delta: {null_deltas[-1]:+.4f}')
null_deltas = np.array(null_deltas)
print(f'\\nNull deltas (shuffle peptide feat): mean={null_deltas.mean():+.4f}, 95pct={np.percentile(null_deltas, 95):+.4f}')
print(f'Observed peptide-add delta: {delta_pepadd:+.4f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 5 — Criteria"))

cells.append(nbf.v4.new_code_cell("""crit1 = 0.78 <= aucs_tg.mean() <= 0.82
crit2 = delta_pepadd >= 0  # monotone
crit3 = np.percentile(null_deltas, 95) < delta_pepadd
crit4 = delta_pepadd >= 0.01

print('CRITERIA')
print('=' * 60)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 Nb09 reproduces in [0.78, 0.82]     {aucs_tg.mean():.3f}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 Peptide-add delta >= 0              {delta_pepadd:+.4f}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 Peptide-add > null 95pct             {delta_pepadd:+.4f} > {np.percentile(null_deltas, 95):+.4f}')
print(f'  [{\"PASS\" if crit4 else \"FAIL\"}] 4 Peptide-add delta >= 0.01            {delta_pepadd:+.4f}')
print('=' * 60)

if crit1 and crit4 and crit3:
    verdict = 'POSITIVE — peptide-graph features add meaningfully above topology+genes'
elif crit1 and crit3:
    verdict = 'WEAK POSITIVE — peptide beats null but adds < 0.01 AUC'
elif crit1 and crit2:
    verdict = 'MONOTONE — peptide features don\\'t hurt but don\\'t clearly add either'
else:
    verdict = 'NULL — peptide features redundant with topology+genes'
print(f'\\nVERDICT: {verdict}')

summary = pd.DataFrame([{
    'auc_topology': float(aucs_topo.mean()),
    'auc_topo_genes': float(aucs_tg.mean()),
    'auc_full_with_peptide': float(aucs_full.mean()),
    'delta_genes_over_topo': float(delta_gtop),
    'delta_peptide_over_topogenes': float(delta_pepadd),
    'null_95pct_pepdelta': float(np.percentile(null_deltas, 95)),
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb16_final_summary.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/16_topology_genes_peptide.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 16 written ({len(nb.cells)} cells)')
