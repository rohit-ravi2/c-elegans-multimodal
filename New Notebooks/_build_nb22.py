"""Nb 22 — Three-layer integrated model: contact + topology + genes for edge arrival (with bootstrap CIs)."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 22 — Three-Layer Integrated Model + Uncertainty

## The new paper headline

Nb20 changed the picture. Contact is the strongest single predictor (AUC 0.836), topology adds specificity (+0.038), genes add further refinement (+0.044 in Nb09). This notebook builds the **full three-layer model** and reports bootstrap CIs on every layer's contribution.

## Preregistered criteria

1. **Contact-only AUC reproduces Nb20**: ∈ [0.82, 0.85].
2. **Contact + topology AUC reproduces Nb20**: ∈ [0.86, 0.89].
3. **Contact + topology + genes AUC > 0.88**.
4. **Each layer adds ≥ 0.02 AUC**: contact alone → add topology → add genes, each step gives ≥ 0.02.
5. **Bootstrap CIs on all deltas exclude zero** (1,000 resamples).

## Halting rule

If criterion 4 fails for any layer: that layer is redundant with the lower layers. Report honestly.

If all 5 pass: we have the cleanest possible mechanistic decomposition of C. elegans chemical synapse formation: **contact → topology → genes**, with quantified contributions per layer."""))

cells.append(nbf.v4.new_code_cell("""import sys, time
from pathlib import Path
import warnings; warnings.simplefilter('ignore')
_HERE = Path.cwd()
if (_HERE / 'lib').is_dir(): sys.path.insert(0, str(_HERE))
elif (_HERE.parent / 'lib').is_dir(): sys.path.insert(0, str(_HERE.parent))

from lib.paths import DATA, DERIVED
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(42)

cand = pd.read_csv(DERIVED / 'nb07_pooled_candidates.csv')
print(f'Pooled candidates: {len(cand)}')

# Common-185
STAGES = ['L1_1','L1_2','L1_3','L1_4','L2','L3','adult']
s_neurons = {}
for s in STAGES:
    d = np.load(DERIVED / 'developmental' / f'connectome_{s}.npz', allow_pickle=True)
    s_neurons[s] = set(str(n) for n in d['neurons'])
common = sorted(set.intersection(*s_neurons.values()))

# Contact matrix (from Nb20 logic)
contact_xlsx = DATA / 'connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx'
contact_df = pd.read_excel(contact_xlsx, sheet_name='adult nerve ring neighbors', header=0, index_col=0)
contact_df = contact_df.loc[:, ~contact_df.columns.str.startswith('Unnamed')]
contact_df = contact_df[contact_df.index.map(lambda x: isinstance(x, str))]
contact_df = contact_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

name_to_idx = {n: i for i, n in enumerate(common)}
contact_in_common = np.zeros((len(common), len(common)), dtype=np.float32)
for row_name in contact_df.index:
    if row_name not in name_to_idx: continue
    ri = name_to_idx[row_name]
    for col_name in contact_df.columns:
        if col_name not in name_to_idx: continue
        ci = name_to_idx[col_name]
        val = contact_df.loc[row_name, col_name]
        if pd.notna(val):
            contact_in_common[ri, ci] = float(val)
contact_in_common = (contact_in_common + contact_in_common.T) / 2

contact_area = np.array([contact_in_common[int(r['i']), int(r['j'])] for _, r in cand.iterrows()])
log_contact = np.log1p(contact_area)
contact_present = (contact_area > 0).astype(int)

# Gene features (PCA-50)
expr_data = np.load(DERIVED / 'expression_tpm.npz', allow_pickle=True)
tpm_neurons = expr_data['neurons']; tpm = expr_data['tpm']
mapping = pd.read_csv(DERIVED / 'expression_neuron_mapping.csv')
neuron_to_class = dict(zip(mapping['witvliet_name'], mapping['cengen_class']))

class_to_expr = {}
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
def get_pc(n): return class_to_pc.get(neuron_to_class.get(n, ''), ZERO)

gene_pre = np.stack([get_pc(common[int(r['i'])]) for _, r in cand.iterrows()])
gene_post = np.stack([get_pc(common[int(r['j'])]) for _, r in cand.iterrows()])

print(f'Features: topology, contact, genes loaded')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Build nested feature sets"))

cells.append(nbf.v4.new_code_cell("""TOP_COVS = ['out_deg_i','in_deg_j','shared_out','shared_in','triangle_closure','reverse_2step']
X_topo = cand[TOP_COVS].values.astype(float)
X_contact = np.stack([log_contact, contact_present], axis=1)

# Models (nested):
#   M1 = contact only
#   M2 = contact + topology
#   M3 = contact + topology + genes
#   Also: genes only, topology only for context

X_M1 = X_contact
X_M2 = np.concatenate([X_contact, X_topo], axis=1)
X_M3 = np.concatenate([X_contact, X_topo, gene_pre, gene_post], axis=1)
X_topo_only = X_topo
X_genes_only = np.concatenate([gene_pre, gene_post], axis=1)

y = cand['arrived'].values.astype(int)
print(f'N={len(y)}, arrivals={y.sum()} ({y.mean():.3f})')
print(f'M1 (contact): {X_M1.shape}')
print(f'M2 (contact+topo): {X_M2.shape}')
print(f'M3 (contact+topo+genes): {X_M3.shape}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Paired bootstrap AUCs (1,000 resamples)"))

cells.append(nbf.v4.new_code_cell("""def get_oof_preds(X, y, n_splits=5, random_state=42, C=1.0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros_like(y, dtype=float)
    for tr, te in skf.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof

t0 = time.time()
oof_M1 = get_oof_preds(X_M1, y)
print(f'M1 done ({time.time()-t0:.1f}s)')
oof_M2 = get_oof_preds(X_M2, y)
print(f'M2 done ({time.time()-t0:.1f}s)')
oof_M3 = get_oof_preds(X_M3, y)
print(f'M3 done ({time.time()-t0:.1f}s)')
oof_topo = get_oof_preds(X_topo_only, y)
print(f'topo-only done ({time.time()-t0:.1f}s)')
oof_genes = get_oof_preds(X_genes_only, y)
print(f'genes-only done ({time.time()-t0:.1f}s)')

def auc_from(oof, y):
    return roc_auc_score(y, oof)

print(f'\\nAUC (5-fold OOF):')
print(f'  M1 contact-only:              {auc_from(oof_M1, y):.4f}')
print(f'  M2 contact+topology:          {auc_from(oof_M2, y):.4f}')
print(f'  M3 contact+topology+genes:    {auc_from(oof_M3, y):.4f}')
print(f'  topology-only:                {auc_from(oof_topo, y):.4f}')
print(f'  genes-only:                   {auc_from(oof_genes, y):.4f}')"""))

cells.append(nbf.v4.new_code_cell("""# Paired bootstrap for CIs on each AUC AND on layer-by-layer deltas
n_boot = 1000
N = len(y)

bs_M1 = np.zeros(n_boot); bs_M2 = np.zeros(n_boot); bs_M3 = np.zeros(n_boot)
bs_topo = np.zeros(n_boot); bs_genes = np.zeros(n_boot)
bs_delta_topo_over_contact = np.zeros(n_boot)     # M2 - M1
bs_delta_genes_over_topo_contact = np.zeros(n_boot) # M3 - M2
bs_delta_topo_over_nothing = np.zeros(n_boot)

for b in range(n_boot):
    idx = RNG.choice(N, N, replace=True)
    bs_M1[b] = roc_auc_score(y[idx], oof_M1[idx])
    bs_M2[b] = roc_auc_score(y[idx], oof_M2[idx])
    bs_M3[b] = roc_auc_score(y[idx], oof_M3[idx])
    bs_topo[b] = roc_auc_score(y[idx], oof_topo[idx])
    bs_genes[b] = roc_auc_score(y[idx], oof_genes[idx])
    bs_delta_topo_over_contact[b] = bs_M2[b] - bs_M1[b]
    bs_delta_genes_over_topo_contact[b] = bs_M3[b] - bs_M2[b]

def ci(a): return f'{np.mean(a):.4f} [{np.percentile(a, 2.5):.4f}, {np.percentile(a, 97.5):.4f}]'

print('BOOTSTRAP 95% CIs (1000 resamples):')
print(f'  Contact only (M1):                   {ci(bs_M1)}')
print(f'  Topology only:                       {ci(bs_topo)}')
print(f'  Genes only:                          {ci(bs_genes)}')
print(f'  Contact + Topology (M2):             {ci(bs_M2)}')
print(f'  Contact + Topology + Genes (M3):     {ci(bs_M3)}')
print(f'\\nDELTAS:')
print(f'  Δ Topology above Contact:            {ci(bs_delta_topo_over_contact)}')
print(f'  Δ Genes above Contact+Topology:      {ci(bs_delta_genes_over_topo_contact)}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Preregistered criteria"))

cells.append(nbf.v4.new_code_cell("""crit1 = 0.82 <= np.mean(bs_M1) <= 0.85
crit2 = 0.86 <= np.mean(bs_M2) <= 0.89
crit3 = np.mean(bs_M3) > 0.88
crit4a = np.mean(bs_delta_topo_over_contact) >= 0.02
crit4b = np.mean(bs_delta_genes_over_topo_contact) >= 0.02
crit5a = np.percentile(bs_delta_topo_over_contact, 2.5) > 0
crit5b = np.percentile(bs_delta_genes_over_topo_contact, 2.5) > 0

print('PREREGISTERED CRITERIA')
print('=' * 70)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 Contact AUC in [0.82, 0.85]                 {np.mean(bs_M1):.4f}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 Contact+Topology AUC in [0.86, 0.89]         {np.mean(bs_M2):.4f}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 Full AUC > 0.88                               {np.mean(bs_M3):.4f}')
print(f'  [{\"PASS\" if crit4a else \"FAIL\"}] 4a Topology adds >= 0.02 above contact        {np.mean(bs_delta_topo_over_contact):+.4f}')
print(f'  [{\"PASS\" if crit4b else \"FAIL\"}] 4b Genes adds >= 0.02 above contact+topology  {np.mean(bs_delta_genes_over_topo_contact):+.4f}')
print(f'  [{\"PASS\" if crit5a else \"FAIL\"}] 5a Topology delta CI excludes 0               CI low={np.percentile(bs_delta_topo_over_contact, 2.5):+.4f}')
print(f'  [{\"PASS\" if crit5b else \"FAIL\"}] 5b Genes delta CI excludes 0                  CI low={np.percentile(bs_delta_genes_over_topo_contact, 2.5):+.4f}')
print('=' * 70)

all_pass = all([crit1, crit2, crit3, crit4a, crit4b, crit5a, crit5b])
n_pass = sum([crit1, crit2, crit3, crit4a, crit4b, crit5a, crit5b])

if all_pass:
    verdict = 'STRONG POSITIVE — all 7 criteria pass; three-layer model is paper-ready'
elif n_pass >= 5:
    verdict = 'POSITIVE — key results hold; minor criteria failures (e.g., window mismatch)'
elif crit3 and (crit4a or crit4b):
    verdict = 'PARTIAL — full AUC high and at least one layer adds meaningfully'
else:
    verdict = 'INCOMPLETE — three-layer story is more complex than preregistered'

print(f'\\n{n_pass}/7 criteria pass')
print(f'VERDICT: {verdict}')

summary = pd.DataFrame([{
    'auc_contact_only':       float(np.mean(bs_M1)),
    'auc_contact_topo':       float(np.mean(bs_M2)),
    'auc_contact_topo_genes': float(np.mean(bs_M3)),
    'auc_contact_ci_low':     float(np.percentile(bs_M1, 2.5)),
    'auc_contact_ci_high':    float(np.percentile(bs_M1, 97.5)),
    'auc_full_ci_low':        float(np.percentile(bs_M3, 2.5)),
    'auc_full_ci_high':       float(np.percentile(bs_M3, 97.5)),
    'delta_topology':         float(np.mean(bs_delta_topo_over_contact)),
    'delta_topology_ci_low':  float(np.percentile(bs_delta_topo_over_contact, 2.5)),
    'delta_topology_ci_high': float(np.percentile(bs_delta_topo_over_contact, 97.5)),
    'delta_genes':            float(np.mean(bs_delta_genes_over_topo_contact)),
    'delta_genes_ci_low':     float(np.percentile(bs_delta_genes_over_topo_contact, 2.5)),
    'delta_genes_ci_high':    float(np.percentile(bs_delta_genes_over_topo_contact, 97.5)),
    'verdict':                verdict,
}])
summary.to_csv(DERIVED / 'nb22_final_summary.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/22_three_layer_integrated.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 22 written ({len(nb.cells)} cells)')
