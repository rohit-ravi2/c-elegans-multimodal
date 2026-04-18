"""Nb 09 — Does adding gene expression improve Nb07's topology-only AUC 0.76?
This is THE critical follow-up test for the Nb07 positive finding."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 09 — Does Gene Expression Add to Nb07's Topology-Only Edge Arrival Model?

## Why this is the critical paper-level test

Nb07 found that purely topological features (triangle closure, shared outputs, in/out degree) predict developmental edge arrival at **AUC 0.76** — a strong positive signal without any gene information.

The scientific claim hinges on what happens when gene expression features are added:
- **If gene features add substantially (AUC ≥ 0.80)**: wiring is topological + gene-modulated. Paper reframes as "topological rule plus gene refinement."
- **If gene features add marginally (0.76 → 0.78)**: wiring is primarily topological; genes are weak modulators at best.
- **If gene features add nothing (AUC stays 0.76)**: wiring is topological only; gene-expression based models of the connectome may be fundamentally mis-framed.

## Why this is genuinely different from prior notebooks

Every prior notebook tested gene-based models without a strong topological baseline.
- Nb 03/03b: genes → motifs, no topological baseline, null.
- Nb 05: genes → developmental rewiring, no topological control, weak AUC 0.62.
- Nb 06: L-R compat → edges, contact-stratified, null above contact.

Now we have a topology-only baseline at AUC 0.76. Adding genes on top is the **proper test** — it shows whether genes add *anything* to what topology already captures.

## Preregistered criteria

1. **Delta (topology + genes) − (topology only) ≥ 0.02**: genes add at least modestly.
2. **Permutation null** (shuffle gene features, keep topology): shuffled delta ≤ 0.
3. **If delta < 0.02**: wiring is topological-only. This is *consistent* with Nb03/03b/05/06 nulls — and that consistency is itself a paper-level finding.

## Halting rule

If delta ≥ 0.02 AND beats null: paper becomes "topology + gene refinement" positive story.
If delta < 0.02: paper becomes "wiring is topological, prior gene-failures confirmed" story — both publishable, different angles."""))

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

# Load Nb07 pooled candidates
cand = pd.read_csv(DERIVED / 'nb07_pooled_candidates.csv')
print(f'Pooled candidates from Nb07: {len(cand)}')
print(f'Columns: {list(cand.columns)}')

# Load CeNGEN expression
expr_data = np.load(DERIVED / 'expression_tpm.npz', allow_pickle=True)
tpm_neurons = expr_data['neurons']; tpm = expr_data['tpm']
mapping = pd.read_csv(DERIVED / 'expression_neuron_mapping.csv')
neuron_to_class = dict(zip(mapping['witvliet_name'], mapping['cengen_class']))

# Witvliet neurons for Nb07 were the common 185 across stages
adult = np.load(DERIVED / 'connectome_adult.npz', allow_pickle=True)
w_neurons = np.array([str(n) for n in adult['neurons']])

# Build class-level PCA-50 expression
class_to_expr = {}
for i, nm in enumerate(tpm_neurons):
    cls = neuron_to_class.get(str(nm))
    if isinstance(cls, str) and cls not in class_to_expr and not np.all(np.isnan(tpm[i])):
        class_to_expr[cls] = tpm[i]
print(f'CeNGEN classes with expression: {len(class_to_expr)}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Map Nb07 candidates (neuron-level indices) back to CeNGEN class-level gene features"))

cells.append(nbf.v4.new_code_cell("""# Nb07 candidates use 'i' and 'j' — integer indices into the common-185 neuron list
# Need to recover the neuron name, then map to CeNGEN class, then to expression vector.
STAGES = ['L1_1','L1_2','L1_3','L1_4','L2','L3','adult']
stage0 = np.load(DERIVED / 'developmental' / 'connectome_L1_1.npz', allow_pickle=True)
stage1 = np.load(DERIVED / 'developmental' / 'connectome_adult.npz', allow_pickle=True)
# The common-185 list is the intersection; reconstruct it identically
s_neurons = {}
for s in STAGES:
    d = np.load(DERIVED / 'developmental' / f'connectome_{s}.npz', allow_pickle=True)
    s_neurons[s] = set(str(n) for n in d['neurons'])
common = sorted(set.intersection(*s_neurons.values()))
print(f'Common-185 list: {len(common)} neurons')

# Build PCA-50 expression per CeNGEN class
classes_list = sorted(class_to_expr.keys())
X_cls = np.stack([np.log1p(class_to_expr[c]) for c in classes_list])
pca_model = PCA(n_components=50, random_state=42).fit(X_cls)
X_pca = pca_model.transform(X_cls)
class_to_pc = {c: X_pca[i] for i, c in enumerate(classes_list)}
# Fallback: zero vector for classes without expression
ZERO_PC = np.zeros(50)

def neuron_to_pc(neuron):
    cls = neuron_to_class.get(neuron)
    if isinstance(cls, str) and cls in class_to_pc:
        return class_to_pc[cls]
    return ZERO_PC

# Build per-candidate gene features
gene_feats_pre = np.stack([neuron_to_pc(common[int(r['i'])]) for _, r in cand.iterrows()])
gene_feats_post = np.stack([neuron_to_pc(common[int(r['j'])]) for _, r in cand.iterrows()])
# Also: indicator for whether pre and post both have expression data
both_have_expr = (~(np.all(gene_feats_pre==0, axis=1) | np.all(gene_feats_post==0, axis=1))).astype(int)

print(f'Candidates with BOTH pre & post expression: {both_have_expr.sum()} / {len(cand)}')
print(f'  this is the subset on which gene features can add anything')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Head-to-head: topology only vs topology + gene PCs"))

cells.append(nbf.v4.new_code_cell("""TOPOLOGY_COVS = ['out_deg_i', 'in_deg_j', 'shared_out', 'shared_in', 'triangle_closure', 'reverse_2step']

X_topo = cand[TOPOLOGY_COVS].values.astype(float)
X_full = np.concatenate([X_topo, gene_feats_pre, gene_feats_post], axis=1)
y = cand['arrived'].values.astype(int)

def cv_auc_robust(X, y, n_splits=5, random_state=42, C=1.0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    for tr, te in skf.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C))
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.array(aucs)

aucs_topo = cv_auc_robust(X_topo, y)
aucs_full = cv_auc_robust(X_full, y)
delta = aucs_full.mean() - aucs_topo.mean()

print(f'Topology-only AUC:       {aucs_topo.mean():.3f} ± {aucs_topo.std():.3f}')
print(f'Topology + genes AUC:    {aucs_full.mean():.3f} ± {aucs_full.std():.3f}')
print(f'Delta (full - topology): {delta:+.4f}')

# Also test: genes ALONE (as a sanity check)
X_genes = np.concatenate([gene_feats_pre, gene_feats_post], axis=1)
aucs_genes = cv_auc_robust(X_genes, y)
print(f'\\nGenes-only AUC (sanity): {aucs_genes.mean():.3f} ± {aucs_genes.std():.3f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Permutation null: shuffle gene features, keep topology"))

cells.append(nbf.v4.new_code_cell("""N_PERM = 50
null_deltas = []
t0 = time.time()
for i in range(N_PERM):
    perm = RNG.permutation(X_full.shape[0])
    X_shuf = X_full.copy()
    X_shuf[:, len(TOPOLOGY_COVS):] = X_full[perm, len(TOPOLOGY_COVS):]
    aucs_s = cv_auc_robust(X_shuf, y, random_state=42+i)
    null_deltas.append(aucs_s.mean() - aucs_topo.mean())
    if (i+1) % 10 == 0:
        print(f'  {i+1}/{N_PERM} ({time.time()-t0:.1f}s)  recent null deltas: {[f\"{d:+.4f}\" for d in null_deltas[-3:]]}')
null_deltas = np.array(null_deltas)

print(f'\\nPermutation null deltas: mean={null_deltas.mean():+.4f}, 95pct={np.percentile(null_deltas, 95):+.4f}, max={null_deltas.max():+.4f}')
print(f'Observed delta:          {delta:+.4f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Preregistered criteria"))

cells.append(nbf.v4.new_code_cell("""crit1 = delta >= 0.02
crit2 = delta > np.percentile(null_deltas, 95)

print('PREREGISTERED CRITERIA')
print('=' * 70)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 Delta (topo+genes - topo) >= 0.02    delta={delta:+.4f}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 Delta > null 95pct                    null95={np.percentile(null_deltas, 95):+.4f}')
print('=' * 70)

if crit1 and crit2:
    verdict = 'POSITIVE — gene expression adds meaningfully above topology for edge arrival'
    paper_angle = 'Paper: topology is the primary rule, gene expression is a modest refinement layer'
elif crit2:  # not meaningful effect size but above null
    verdict = 'WEAK POSITIVE — tiny gene contribution above null but <0.02 delta'
    paper_angle = 'Paper: topology dominates; gene contribution exists but is too small to be biologically meaningful'
else:
    verdict = 'NULL — gene expression does NOT add above topology for edge arrival'
    paper_angle = 'Paper: wiring is topological, gene expression is not predictive over topology'

print(f'\\nVERDICT: {verdict}')
print(f'PAPER ANGLE: {paper_angle}')

summary = pd.DataFrame([{
    'N_candidates': int(len(cand)),
    'topology_only_auc': float(aucs_topo.mean()),
    'topology_plus_genes_auc': float(aucs_full.mean()),
    'delta': float(delta),
    'null_95pct_delta': float(np.percentile(null_deltas, 95)),
    'null_mean_delta': float(null_deltas.mean()),
    'genes_only_auc': float(aucs_genes.mean()),
    'verdict': verdict,
    'paper_angle': paper_angle,
}])
summary.to_csv(DERIVED / 'nb09_final_summary.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells

with open('/home/rohit/Desktop/C-Elegans/New Notebooks/09_topology_vs_topology_plus_genes.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 09 written ({len(nb.cells)} cells)')
