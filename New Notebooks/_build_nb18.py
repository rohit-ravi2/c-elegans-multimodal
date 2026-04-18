"""Nb 18 — Bootstrap confidence intervals on Nb07/09/17's AUCs."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 18 — Bootstrap Confidence Intervals on Paper's Key AUCs

## Why

The paper reports AUC values (0.76 topology, 0.80 topology+genes, delta +0.044). A reviewer will demand confidence intervals. This notebook runs 1,000-bootstrap percentile CIs on each metric, providing the final rigor level.

## Preregistered criteria

1. **Topology-only AUC 95% CI excludes 0.70**.
2. **Topology+genes AUC 95% CI excludes topology-only upper bound** — i.e., CI for (full AUC - topology AUC) excludes 0.
3. Report CIs for: Nb07 topology-only, Nb09 topology+genes, delta, and per-transition AUCs."""))

cells.append(nbf.v4.new_code_cell("""import sys
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
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(42)

cand = pd.read_csv(DERIVED / 'nb07_pooled_candidates.csv')
expr_data = np.load(DERIVED / 'expression_tpm.npz', allow_pickle=True)
tpm_neurons = expr_data['neurons']; tpm = expr_data['tpm']
mapping = pd.read_csv(DERIVED / 'expression_neuron_mapping.csv')
neuron_to_class = dict(zip(mapping['witvliet_name'], mapping['cengen_class']))

STAGES = ['L1_1','L1_2','L1_3','L1_4','L2','L3','adult']
s_neurons = {}
for s in STAGES:
    d = np.load(DERIVED / 'developmental' / f'connectome_{s}.npz', allow_pickle=True)
    s_neurons[s] = set(str(n) for n in d['neurons'])
common = sorted(set.intersection(*s_neurons.values()))

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
def get_pc(idx):
    n = common[int(idx)]
    cls = neuron_to_class.get(n)
    return class_to_pc.get(cls, ZERO)

gene_pre = np.stack([get_pc(r['i']) for _, r in cand.iterrows()])
gene_post = np.stack([get_pc(r['j']) for _, r in cand.iterrows()])

TOP_COVS = ['out_deg_i','in_deg_j','shared_out','shared_in','triangle_closure','reverse_2step']
X_topo = cand[TOP_COVS].values.astype(float)
X_full = np.concatenate([X_topo, gene_pre, gene_post], axis=1)
y = cand['arrived'].values.astype(int)
print(f'Dataset: N={len(y)}, arrivals={y.sum()}')"""))

cells.append(nbf.v4.new_code_cell("""# 5-fold CV AUC with 1000 bootstrap resamples on the test predictions
from sklearn.model_selection import StratifiedKFold

def bootstrap_cv_aucs(X, y, n_boot=1000, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # Collect per-fold OOF probabilities
    oof = np.zeros_like(y, dtype=float)
    for tr, te in skf.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    # Now bootstrap CI via resampling indices (keeping y and oof paired)
    N = len(y)
    bs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = RNG.choice(N, N, replace=True)
        bs[b] = roc_auc_score(y[idx], oof[idx])
    return bs

print('Bootstrap topology-only AUC (1000 boots)...')
bs_topo = bootstrap_cv_aucs(X_topo, y, n_boot=1000)
print('Bootstrap topology+genes AUC (1000 boots)...')
bs_full = bootstrap_cv_aucs(X_full, y, n_boot=1000)

# Delta bootstrap: use the SAME resampling indices to get paired bootstrap of delta
# (Not easily shareable with existing function — do a fresh paired bootstrap below)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_topo = np.zeros_like(y, dtype=float)
oof_full = np.zeros_like(y, dtype=float)
for tr, te in skf.split(X_topo, y):
    m_topo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    m_topo.fit(X_topo[tr], y[tr])
    oof_topo[te] = m_topo.predict_proba(X_topo[te])[:, 1]
    m_full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    m_full.fit(X_full[tr], y[tr])
    oof_full[te] = m_full.predict_proba(X_full[te])[:, 1]

N = len(y)
n_boot = 1000
delta_bs = np.zeros(n_boot)
for b in range(n_boot):
    idx = RNG.choice(N, N, replace=True)
    delta_bs[b] = roc_auc_score(y[idx], oof_full[idx]) - roc_auc_score(y[idx], oof_topo[idx])

print(f'\\nAUC topology-only:        {np.mean(bs_topo):.3f}  95% CI [{np.percentile(bs_topo, 2.5):.3f}, {np.percentile(bs_topo, 97.5):.3f}]')
print(f'AUC topology+genes:       {np.mean(bs_full):.3f}  95% CI [{np.percentile(bs_full, 2.5):.3f}, {np.percentile(bs_full, 97.5):.3f}]')
print(f'Delta (full - topology):  {np.mean(delta_bs):+.4f}  95% CI [{np.percentile(delta_bs, 2.5):+.4f}, {np.percentile(delta_bs, 97.5):+.4f}]')"""))

cells.append(nbf.v4.new_markdown_cell("## Criteria"))

cells.append(nbf.v4.new_code_cell("""topo_ci_lower = np.percentile(bs_topo, 2.5)
full_ci_lower = np.percentile(bs_full, 2.5)
delta_ci_lower = np.percentile(delta_bs, 2.5)
delta_ci_upper = np.percentile(delta_bs, 97.5)

crit1 = topo_ci_lower > 0.70
crit2 = delta_ci_lower > 0

print('CRITERIA')
print('=' * 60)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 Topology AUC 95% CI lower bound > 0.70    {topo_ci_lower:.3f}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 Delta 95% CI lower bound > 0               {delta_ci_lower:+.4f}')
print('=' * 60)

if crit1 and crit2:
    verdict = 'BOTH PASS — Nb07/Nb09 findings survive rigorous uncertainty quantification'
elif crit1:
    verdict = 'PARTIAL — topology robust but gene contribution marginal'
elif crit2:
    verdict = 'PARTIAL — gene delta robust but topology AUC CI wider'
else:
    verdict = 'NULL — neither pass uncertainty quantification'
print(f'\\nVERDICT: {verdict}')

summary = pd.DataFrame([{
    'topology_auc_mean': float(np.mean(bs_topo)),
    'topology_auc_ci_low': float(np.percentile(bs_topo, 2.5)),
    'topology_auc_ci_high': float(np.percentile(bs_topo, 97.5)),
    'full_auc_mean': float(np.mean(bs_full)),
    'full_auc_ci_low': float(np.percentile(bs_full, 2.5)),
    'full_auc_ci_high': float(np.percentile(bs_full, 97.5)),
    'delta_mean': float(np.mean(delta_bs)),
    'delta_ci_low': float(delta_ci_lower),
    'delta_ci_high': float(delta_ci_upper),
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb18_final_summary.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/18_bootstrap_cis.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 18 written ({len(nb.cells)} cells)')
