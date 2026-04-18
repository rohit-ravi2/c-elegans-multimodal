"""Nb 17 — Cross-stage generalization: does Nb07/09 rule learned on one transition predict another?"""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 17 — Cross-Stage Generalization of the Topology+Genes Rule

## Question

Nb07 and Nb09 pool across 3 developmental transitions (L1_cons→L2, L2→L3, L3→adult). Does the rule learned on ONE transition GENERALIZE to another? This tests whether the wiring rule is truly developmental (stage-invariant) or whether a single transition drove the pooled signal.

Two cross-stage tests:
1. **Train on L1_cons→L2, test on L3→adult** (furthest-apart transitions)
2. **Train on L2→L3, test on L1_cons→L2 AND L3→adult**

If AUC degrades dramatically (< 0.60) on held-out transitions: the rule is transition-specific, weakening the paper. If it holds (≥ 0.70): the rule is a genuine developmental invariant.

## Preregistered criteria

1. **Test-on-different-transition AUC ≥ 0.70** in at least 2 of 3 train/test splits.
2. **Degradation ≤ 0.10** from within-transition AUC (Nb07 individual transitions were 0.72, 0.82, 0.81)."""))

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

# Load Nb07 candidates (has transition label)
cand = pd.read_csv(DERIVED / 'nb07_pooled_candidates.csv')
print(f'Nb07 candidates: {len(cand)}')
print(cand['transition'].value_counts().to_string())

# Gene features (reproduce Nb09)
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
    if isinstance(cls, str) and cls in class_to_pc:
        return class_to_pc[cls]
    return ZERO

gene_pre = np.stack([get_pc(r['i']) for _, r in cand.iterrows()])
gene_post = np.stack([get_pc(r['j']) for _, r in cand.iterrows()])"""))

cells.append(nbf.v4.new_markdown_cell("## Cross-stage train/test splits"))

cells.append(nbf.v4.new_code_cell("""TOP_COVS = ['out_deg_i','in_deg_j','shared_out','shared_in','triangle_closure','reverse_2step']
X_topo = cand[TOP_COVS].values.astype(float)
X_full = np.concatenate([X_topo, gene_pre, gene_post], axis=1)
y = cand['arrived'].values.astype(int)

transitions = sorted(cand['transition'].unique())
print(f'Transitions: {transitions}')

def train_test_auc(train_mask, test_mask, X, y):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    model.fit(X[train_mask], y[train_mask])
    pred = model.predict_proba(X[test_mask])[:, 1]
    return roc_auc_score(y[test_mask], pred)

results = []
for train_tr in transitions:
    train_mask = (cand['transition'] == train_tr).values
    for test_tr in transitions:
        if test_tr == train_tr: continue
        test_mask = (cand['transition'] == test_tr).values
        auc_topo = train_test_auc(train_mask, test_mask, X_topo, y)
        auc_full = train_test_auc(train_mask, test_mask, X_full, y)
        results.append({
            'train_on': train_tr, 'test_on': test_tr,
            'auc_topology': auc_topo, 'auc_topo_plus_genes': auc_full,
            'delta': auc_full - auc_topo,
        })
results_df = pd.DataFrame(results)
print(results_df.to_string())"""))

cells.append(nbf.v4.new_markdown_cell("## Criteria"))

cells.append(nbf.v4.new_code_cell("""# How many of the 6 cross-stage tests have full AUC >= 0.70?
n_strong = int((results_df['auc_topo_plus_genes'] >= 0.70).sum())

# Compare to within-transition 5-fold CV baseline (Nb07 per-transition: 0.72, 0.82, 0.81)
within_aucs = {'L1_consensus->L2': 0.720, 'L2->L3': 0.819, 'L3->adult': 0.809}

degradations = []
for _, row in results_df.iterrows():
    within = within_aucs[row['test_on']]
    deg = within - row['auc_topo_plus_genes']
    degradations.append(deg)
results_df['degradation_from_within'] = degradations

n_small_degradation = int((results_df['degradation_from_within'] <= 0.10).sum())

print(f'\\nN cross-stage tests with AUC >= 0.70: {n_strong}/6')
print(f'N with degradation <= 0.10 from within-transition baseline: {n_small_degradation}/6')
print(f'\\nMean cross-stage AUC (topo+genes): {results_df[\"auc_topo_plus_genes\"].mean():.3f}')

crit1 = n_strong >= 4  # ≥2/3 "transitions-as-train" have ≥ test-AUC 0.70
crit2 = n_small_degradation >= 4

print('\\nCRITERIA')
print('=' * 60)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 cross-stage AUC >= 0.70 in >=4/6    {n_strong}/6')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 degradation <= 0.10 in >=4/6        {n_small_degradation}/6')
print('=' * 60)

if crit1 and crit2:
    verdict = 'STRONG GENERALIZATION — topology+genes rule holds across developmental transitions'
elif crit1 or crit2:
    verdict = 'PARTIAL GENERALIZATION — rule mostly holds but some transitions differ'
else:
    verdict = 'POOR GENERALIZATION — rule is transition-specific'
print(f'\\nVERDICT: {verdict}')

summary = pd.DataFrame([{
    'n_cross_stage_high_auc': n_strong,
    'n_low_degradation': n_small_degradation,
    'mean_cross_stage_auc': float(results_df['auc_topo_plus_genes'].mean()),
    'mean_cross_stage_degradation': float(results_df['degradation_from_within'].mean()),
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb17_final_summary.csv', index=False)
results_df.to_csv(DERIVED / 'nb17_cross_stage_results.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/17_cross_stage_generalization.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 17 written ({len(nb.cells)} cells)')
