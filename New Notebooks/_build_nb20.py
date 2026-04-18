"""Nb 20 — Physical contact null: is Nb07's topological rule just a proxy for 3D contact?"""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 20 — Physical Contact Null Check

## The robustness question

Nb06 found that the Brittin/Zhen nerve-ring contact matrix predicts **static adult synaptic edges** at AUC 0.78. Nb07 found that chemical topology predicts **edge arrival** at AUC 0.76. These numbers are suspiciously close. **Could Nb07's topology signal be nothing more than a proxy for whether two neurons physically touch?**

If contact alone predicts arrival at AUC ≥ 0.74, then Nb07's "topological rule" claim is substantially weakened — topology would be a downstream consequence of 3D anatomy, not an independent rule.

## What we're doing

For each developmental transition, ask three questions:
1. **Contact-only AUC**: Using only the adult nerve-ring contact matrix, predict edge arrival at each transition.
2. **Topology vs contact head-to-head**: Which predicts better?
3. **Does topology ADD above contact?** This is the load-bearing test.

## Preregistered criteria

1. **Contact matrix is loadable** for ≥ 170 of the 185 common neurons.
2. **Contact-only AUC is in [0.55, 0.85]**. (If below 0.55, contact isn't a predictor at all; if above 0.85, it's doing all the work.)
3. **Topology-only reproduces Nb07**: AUC ∈ [0.74, 0.78].
4. **KEY TEST**: Topology + contact AUC ≥ contact-only AUC + 0.02. (Topology adds beyond 3D proximity.)
5. **Permutation null** (shuffle topology features, keep contact): shuffled delta < 0.

## Halting rule

**If criterion 4 fails** (topology doesn't add above contact): Nb07's topological rule claim weakens dramatically. Paper must reframe as "contact-mediated wiring" rather than "topological rule."

**If criterion 4 passes**: Nb07/09 are robust to the obvious confound. Topology captures something beyond physical proximity.

## Important limitation

We only have adult + L4 contact matrices (Brittin nerve-ring data). We use **adult contact as a proxy for all transitions**. This is conservative: if contact at adult stage already captures edge arrival at earlier stages, that would be the strongest possible confound claim. If even using adult contact as a proxy, topology still adds, the topology signal is robust."""))

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
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(42)

# Load Nb07 pooled candidates (has all topology features + transition labels)
cand = pd.read_csv(DERIVED / 'nb07_pooled_candidates.csv')
print(f'Nb07 candidates: {len(cand)}')
print(cand['transition'].value_counts().to_string())

# Common-185 neuron list
STAGES = ['L1_1','L1_2','L1_3','L1_4','L2','L3','adult']
s_neurons = {}
for s in STAGES:
    d = np.load(DERIVED / 'developmental' / f'connectome_{s}.npz', allow_pickle=True)
    s_neurons[s] = set(str(n) for n in d['neurons'])
common = sorted(set.intersection(*s_neurons.values()))
print(f'Common-185: {len(common)}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Load Brittin adult nerve-ring contact matrix"))

cells.append(nbf.v4.new_code_cell("""contact_xlsx = DATA / 'connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx'
contact_df = pd.read_excel(contact_xlsx, sheet_name='adult nerve ring neighbors', header=0, index_col=0)
contact_df = contact_df.loc[:, ~contact_df.columns.str.startswith('Unnamed')]
contact_df = contact_df[contact_df.index.map(lambda x: isinstance(x, str))]
contact_df = contact_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

contact_neurons = set(contact_df.index)
common_with_contact = [n for n in common if n in contact_neurons]
print(f'Common-185 neurons in contact matrix: {len(common_with_contact)}/{len(common)}')

# Restrict contact matrix to common-185 and check dimensions
cm_idx = [i for i, n in enumerate(common) if n in contact_neurons]
contact_in_common = np.zeros((len(common), len(common)), dtype=np.float32)
name_to_idx = {n: i for i, n in enumerate(common)}
for row_name in contact_df.index:
    if row_name not in name_to_idx: continue
    ri = name_to_idx[row_name]
    for col_name in contact_df.columns:
        if col_name not in name_to_idx: continue
        ci = name_to_idx[col_name]
        val = contact_df.loc[row_name, col_name]
        if pd.notna(val):
            contact_in_common[ri, ci] = float(val)

# Contact is symmetric (physical adjacency); ensure
contact_in_common = (contact_in_common + contact_in_common.T) / 2
print(f'Contact matrix on common-185: {contact_in_common.shape}')
print(f'Nonzero contact entries: {int((contact_in_common > 0).sum())}')
print(f'Contact value range: [{contact_in_common.min():.0f}, {contact_in_common.max():.0f}]')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Build contact features per candidate"))

cells.append(nbf.v4.new_code_cell("""# For each candidate (i, j) in Nb07 pool:
#   contact_area     : raw contact area value between i and j in nerve-ring EM
#   log_contact      : log1p of contact area (handles orders of magnitude)
#   contact_present  : binary — is contact > 0?

contact_area = np.array([contact_in_common[int(r['i']), int(r['j'])] for _, r in cand.iterrows()])
log_contact = np.log1p(contact_area)
contact_present = (contact_area > 0).astype(int)

print(f'Candidates with any contact: {int((contact_area > 0).sum())} / {len(cand)}')
print(f'Log-contact range: [{log_contact.min():.2f}, {log_contact.max():.2f}]')

# Sanity: among candidates with contact, what's the arrival rate vs without?
y = cand['arrived'].values.astype(int)
p_arrive_contact = (y[contact_present == 1].mean()) if (contact_present == 1).sum() else 0
p_arrive_nocontact = (y[contact_present == 0].mean()) if (contact_present == 0).sum() else 0
print(f'\\nP(arrive | has contact):   {p_arrive_contact:.4f}  (n={int((contact_present == 1).sum())})')
print(f'P(arrive | no contact):    {p_arrive_nocontact:.4f}  (n={int((contact_present == 0).sum())})')
print(f'Enrichment ratio: {p_arrive_contact / max(p_arrive_nocontact, 1e-6):.2f}x')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Nested AUC comparison"))

cells.append(nbf.v4.new_code_cell("""TOP_COVS = ['out_deg_i','in_deg_j','shared_out','shared_in','triangle_closure','reverse_2step']
X_topo = cand[TOP_COVS].values.astype(float)
X_contact = np.stack([log_contact, contact_present], axis=1)
X_both = np.concatenate([X_topo, X_contact], axis=1)

def cv_auc(X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    for tr, te in skf.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.array(aucs)

aucs_topo = cv_auc(X_topo, y)
aucs_contact = cv_auc(X_contact, y)
aucs_both = cv_auc(X_both, y)

delta_topo_above_contact = aucs_both.mean() - aucs_contact.mean()
delta_contact_above_topo = aucs_both.mean() - aucs_topo.mean()

print(f'Topology-only AUC:      {aucs_topo.mean():.3f} ± {aucs_topo.std():.3f}')
print(f'Contact-only AUC:       {aucs_contact.mean():.3f} ± {aucs_contact.std():.3f}')
print(f'Topology + contact AUC: {aucs_both.mean():.3f} ± {aucs_both.std():.3f}')
print(f'\\nDelta (topology+contact - contact-only): {delta_topo_above_contact:+.4f}')
print(f'Delta (topology+contact - topology-only): {delta_contact_above_topo:+.4f}')
print(f'\\n>> Interpretation:')
print(f'   If topology AUC > contact AUC: topology captures something contact does not')
print(f'   If they are similar but topology+contact >> each alone: they capture complementary info')
print(f'   If contact alone matches topology: Nb07 signal is mostly 3D-proximity')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Permutation null"))

cells.append(nbf.v4.new_code_cell("""N_PERM = 100
null_delta_topo = []
t0 = time.time()
for i in range(N_PERM):
    perm = RNG.permutation(X_both.shape[0])
    X_shuf = X_both.copy()
    X_shuf[:, :len(TOP_COVS)] = X_both[perm, :len(TOP_COVS)]  # shuffle topology features; keep contact
    aucs_s = cv_auc(X_shuf, y, random_state=42+i).mean()
    null_delta_topo.append(aucs_s - aucs_contact.mean())
    if (i+1) % 25 == 0:
        print(f'  {i+1}/{N_PERM} ({time.time()-t0:.1f}s)')
null_delta_topo = np.array(null_delta_topo)
print(f'\\nNull delta (topology-shuffle, keep contact): mean={null_delta_topo.mean():+.4f}, 95pct={np.percentile(null_delta_topo, 95):+.4f}')
print(f'Observed topology-above-contact delta: {delta_topo_above_contact:+.4f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 5 — Preregistered criteria"))

cells.append(nbf.v4.new_code_cell("""crit1 = len(common_with_contact) >= 170
crit2 = 0.55 <= aucs_contact.mean() <= 0.85
crit3 = 0.74 <= aucs_topo.mean() <= 0.78
crit4 = delta_topo_above_contact >= 0.02
crit5 = delta_topo_above_contact > np.percentile(null_delta_topo, 95)

print('PREREGISTERED CRITERIA')
print('=' * 70)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 contact matrix covers >= 170 neurons        {len(common_with_contact)}/185')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 Contact-only AUC in [0.55, 0.85]             {aucs_contact.mean():.3f}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 Topology-only AUC in [0.74, 0.78]             {aucs_topo.mean():.3f}')
print(f'  [{\"PASS\" if crit4 else \"FAIL\"}] 4 Topology adds >= 0.02 above contact           {delta_topo_above_contact:+.4f}')
print(f'  [{\"PASS\" if crit5 else \"FAIL\"}] 5 Delta > null 95pct                            {delta_topo_above_contact:+.4f} vs {np.percentile(null_delta_topo, 95):+.4f}')
print('=' * 70)

# CRITICAL INTERPRETATION
if crit4 and crit5:
    verdict = 'POSITIVE — topology contains genuine signal BEYOND physical contact'
    implication = 'Nb07/09 topological rule is robust to the contact confound'
elif crit4 or crit5:
    verdict = 'PARTIAL — topology adds some signal above contact but weakly'
    implication = 'Topology is PARTIALLY a proxy for 3D contact; paper should acknowledge this'
else:
    verdict = 'NULL — topology does NOT add above contact'
    implication = 'CRITICAL: Nb07/09 signal is substantially a proxy for physical adjacency. Reframe paper.'

print(f'\\nVERDICT: {verdict}')
print(f'IMPLICATION: {implication}')

summary = pd.DataFrame([{
    'n_common_with_contact': len(common_with_contact),
    'auc_topology_only': float(aucs_topo.mean()),
    'auc_contact_only': float(aucs_contact.mean()),
    'auc_topology_plus_contact': float(aucs_both.mean()),
    'delta_topology_above_contact': float(delta_topo_above_contact),
    'delta_contact_above_topology': float(delta_contact_above_topo),
    'null_95pct_topology_delta': float(np.percentile(null_delta_topo, 95)),
    'verdict': verdict,
    'paper_implication': implication,
}])
summary.to_csv(DERIVED / 'nb20_final_summary.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/20_contact_null_check.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 20 written ({len(nb.cells)} cells)')
