"""Nb 24 — Contact-stratified three-layer model: is the AUC 0.89 real or inflated by no-contact tautology?"""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 24 — Contact-Stratified Three-Layer Test (Removing the Tautology)

## Why this exists

Nb 20 and Nb 22 reported AUC 0.836 (contact) and 0.890 (full model). These numbers include candidate pairs with **contact = 0** — pairs that physically *cannot* form a synapse. The classifier gets easy wins on these (no contact → no arrival is tautologically true), inflating the AUC.

**The non-trivial question:** Among pairs that DO physically touch, can we predict which ones form chemical synapses across L1→adult development? This strips away the "you need to touch to form a synapse" tautology and asks the actual mechanistic question.

## Preregistered criteria

1. **Contact-stratified N ≥ 20,000**: enough data to test.
2. **Contact-only AUC in [0.55, 0.75]**: if contact area alone is very high (>0.85) on the stratified subset, the tautology isn't fully removed; if below 0.55, contact carries no signal and the earlier 0.836 was pure tautology.
3. **Contact + topology AUC ≥ 0.70** on the stratified subset.
4. **Full (contact + topology + genes) AUC ≥ 0.72** on the stratified subset.
5. **Each layer adds ≥ 0.02 AUC** (same as Nb22).
6. **Bootstrap 95% CIs on all deltas exclude zero.**

## Halting rule

- If contact-stratified full AUC is **< 0.68**: the AUC 0.890 from Nb22 was largely tautological, and we don't actually have a strong predictive signal for developmental wiring. Honest null.
- If contact-stratified full AUC is **in [0.68, 0.80]**: there is a real signal but it's much smaller than Nb22 suggested. Paper reframes with realistic numbers.
- If contact-stratified full AUC is **≥ 0.80**: the Nb22 headline survives; the tautology concern is minor."""))

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
print(f'Nb07 pooled candidates: {len(cand)}')

STAGES = ['L1_1','L1_2','L1_3','L1_4','L2','L3','adult']
s_neurons = {}
for s in STAGES:
    d = np.load(DERIVED / 'developmental' / f'connectome_{s}.npz', allow_pickle=True)
    s_neurons[s] = set(str(n) for n in d['neurons'])
common = sorted(set.intersection(*s_neurons.values()))

# Contact matrix
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
        if pd.notna(val): contact_in_common[ri, ci] = float(val)
contact_in_common = (contact_in_common + contact_in_common.T) / 2

contact_area = np.array([contact_in_common[int(r['i']), int(r['j'])] for _, r in cand.iterrows()])

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

print(f'contact_area stats: 0s={int((contact_area == 0).sum())}, >0={int((contact_area > 0).sum())}, median(>0)={np.median(contact_area[contact_area>0]):.1f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Apply contact stratification"))

cells.append(nbf.v4.new_code_cell("""# Keep only candidates where contact > 0
contact_mask = contact_area > 0
cand_s = cand.loc[contact_mask].reset_index(drop=True)
contact_area_s = contact_area[contact_mask]
log_contact_s = np.log1p(contact_area_s)
gene_pre_s = gene_pre[contact_mask]
gene_post_s = gene_post[contact_mask]
y_s = cand_s['arrived'].values.astype(int)

print(f'Full pooled set:           {len(cand)}  arrivals={int(cand[\"arrived\"].sum())}')
print(f'Contact-stratified subset: {len(cand_s)}  arrivals={int(y_s.sum())}  rate={y_s.mean():.4f}')
print(f'\\nBreakdown by transition:')
print(cand_s['transition'].value_counts().to_string())
print(f'\\nContact area distribution (in strat subset):')
print(pd.Series(contact_area_s).describe().round(1).to_string())"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Nested models on contact-stratified subset"))

cells.append(nbf.v4.new_code_cell("""TOP_COVS = ['out_deg_i','in_deg_j','shared_out','shared_in','triangle_closure','reverse_2step']
X_topo_s    = cand_s[TOP_COVS].values.astype(float)
X_contact_s = log_contact_s.reshape(-1, 1)  # continuous contact area only (all > 0 now)
X_M1_s = X_contact_s                                             # contact only (continuous)
X_M2_s = np.concatenate([X_contact_s, X_topo_s], axis=1)         # + topology
X_M3_s = np.concatenate([X_contact_s, X_topo_s, gene_pre_s, gene_post_s], axis=1)  # + genes

def get_oof(X, y, n_splits=5, random_state=42, C=1.0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros_like(y, dtype=float)
    for tr, te in skf.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof

t0 = time.time()
oof_M1 = get_oof(X_M1_s, y_s); print(f'M1 done ({time.time()-t0:.1f}s)')
oof_M2 = get_oof(X_M2_s, y_s); print(f'M2 done ({time.time()-t0:.1f}s)')
oof_M3 = get_oof(X_M3_s, y_s); print(f'M3 done ({time.time()-t0:.1f}s)')
oof_topo_only = get_oof(X_topo_s, y_s); print(f'topo-only done ({time.time()-t0:.1f}s)')

print(f'\\nAUC on contact-stratified subset (N={len(y_s)}):')
print(f'  Contact-only (continuous):   {roc_auc_score(y_s, oof_M1):.4f}')
print(f'  Topology-only:                {roc_auc_score(y_s, oof_topo_only):.4f}')
print(f'  Contact + Topology:           {roc_auc_score(y_s, oof_M2):.4f}')
print(f'  Contact + Topology + Genes:   {roc_auc_score(y_s, oof_M3):.4f}')

print(f'\\n--- COMPARISON TO NB22 (with non-contacting pairs included) ---')
print(f'  Nb22 contact-only:          0.833')
print(f'  Nb22 full 3-layer:          0.890')
print(f'  Nb24 contact-only (strat):  {roc_auc_score(y_s, oof_M1):.4f}')
print(f'  Nb24 full 3-layer (strat):  {roc_auc_score(y_s, oof_M3):.4f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Paired bootstrap CIs on every layer"))

cells.append(nbf.v4.new_code_cell("""N = len(y_s); n_boot = 1000
bs_M1 = np.zeros(n_boot); bs_M2 = np.zeros(n_boot); bs_M3 = np.zeros(n_boot)
bs_topo_only = np.zeros(n_boot)
bs_d_topo = np.zeros(n_boot); bs_d_genes = np.zeros(n_boot)
for b in range(n_boot):
    idx = RNG.choice(N, N, replace=True)
    bs_M1[b] = roc_auc_score(y_s[idx], oof_M1[idx])
    bs_M2[b] = roc_auc_score(y_s[idx], oof_M2[idx])
    bs_M3[b] = roc_auc_score(y_s[idx], oof_M3[idx])
    bs_topo_only[b] = roc_auc_score(y_s[idx], oof_topo_only[idx])
    bs_d_topo[b] = bs_M2[b] - bs_M1[b]
    bs_d_genes[b] = bs_M3[b] - bs_M2[b]

def ci(a): return f'{np.mean(a):.4f} [{np.percentile(a, 2.5):.4f}, {np.percentile(a, 97.5):.4f}]'

print('CONTACT-STRATIFIED BOOTSTRAP (1000 resamples):')
print(f'  Contact only:             {ci(bs_M1)}')
print(f'  Topology only:            {ci(bs_topo_only)}')
print(f'  Contact + Topology:       {ci(bs_M2)}')
print(f'  Full (C+T+G):             {ci(bs_M3)}')
print(f'  Δ Topology above contact: {ci(bs_d_topo)}')
print(f'  Δ Genes above C+T:        {ci(bs_d_genes)}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Criteria and reinterpretation"))

cells.append(nbf.v4.new_code_cell("""auc_c     = float(np.mean(bs_M1))
auc_ct    = float(np.mean(bs_M2))
auc_full  = float(np.mean(bs_M3))
auc_topo  = float(np.mean(bs_topo_only))
d_topo    = float(np.mean(bs_d_topo))
d_genes   = float(np.mean(bs_d_genes))

crit1 = len(y_s) >= 20000
crit2 = 0.55 <= auc_c <= 0.75
crit3 = auc_ct >= 0.70
crit4 = auc_full >= 0.72
crit5a = d_topo >= 0.02
crit5b = d_genes >= 0.02
crit6a = np.percentile(bs_d_topo, 2.5) > 0
crit6b = np.percentile(bs_d_genes, 2.5) > 0

print('PREREGISTERED CRITERIA (contact-stratified)')
print('=' * 70)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 N_stratified >= 20,000                    {len(y_s)}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 Contact-only AUC in [0.55, 0.75]          {auc_c:.3f}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 Contact+Topology AUC >= 0.70               {auc_ct:.3f}')
print(f'  [{\"PASS\" if crit4 else \"FAIL\"}] 4 Full AUC >= 0.72                            {auc_full:.3f}')
print(f'  [{\"PASS\" if crit5a else \"FAIL\"}] 5a Topology delta >= 0.02                   {d_topo:+.4f}')
print(f'  [{\"PASS\" if crit5b else \"FAIL\"}] 5b Genes delta >= 0.02                      {d_genes:+.4f}')
print(f'  [{\"PASS\" if crit6a else \"FAIL\"}] 6a Topology delta CI excludes 0             low={np.percentile(bs_d_topo, 2.5):+.4f}')
print(f'  [{\"PASS\" if crit6b else \"FAIL\"}] 6b Genes delta CI excludes 0                low={np.percentile(bs_d_genes, 2.5):+.4f}')
print('=' * 70)

n_pass = sum([crit1, crit2, crit3, crit4, crit5a, crit5b, crit6a, crit6b])
print(f'{n_pass}/8 criteria pass')

if auc_full < 0.68:
    verdict = 'HONEST NULL — Nb22 AUC 0.89 was largely tautological; real developmental signal is marginal'
elif 0.68 <= auc_full < 0.80:
    verdict = 'REAL BUT SMALLER — signal exists above tautology but much smaller than Nb22 suggested'
elif auc_full >= 0.80:
    verdict = 'NB22 HEADLINE SURVIVES — tautology concern is minor; three-layer model is real'
print(f'\\nVERDICT: {verdict}')

summary = pd.DataFrame([{
    'N_stratified': int(len(y_s)),
    'arrivals': int(y_s.sum()),
    'base_rate': float(y_s.mean()),
    'auc_contact_only':    auc_c,
    'auc_topology_only':   auc_topo,
    'auc_contact_topo':    auc_ct,
    'auc_full':            auc_full,
    'auc_contact_ci_low':  float(np.percentile(bs_M1, 2.5)),
    'auc_contact_ci_high': float(np.percentile(bs_M1, 97.5)),
    'auc_full_ci_low':     float(np.percentile(bs_M3, 2.5)),
    'auc_full_ci_high':    float(np.percentile(bs_M3, 97.5)),
    'delta_topo':          d_topo,
    'delta_topo_ci_low':   float(np.percentile(bs_d_topo, 2.5)),
    'delta_topo_ci_high':  float(np.percentile(bs_d_topo, 97.5)),
    'delta_genes':         d_genes,
    'delta_genes_ci_low':  float(np.percentile(bs_d_genes, 2.5)),
    'delta_genes_ci_high': float(np.percentile(bs_d_genes, 97.5)),
    'verdict':             verdict,
}])
summary.to_csv(DERIVED / 'nb24_final_summary.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/24_contact_stratified_three_layer.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 24 written ({len(nb.cells)} cells)')
