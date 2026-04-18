"""Nb 10 — Gap junction edge-arrival parallels Nb07, on electrical synapses."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 10 — Gap Junction Edge Arrival (Nb07 complement on electrical synapses)

## Question

Nb07 found that topology at stage t predicts chemical synapse arrival at t+1 (AUC 0.76). Gap junctions (electrical synapses) are a biologically distinct wiring layer with different formation rules (innexin proteins, bilateral coupling). Does the same topological rule hold for gap junctions?

## Biological expectations

Gap junctions:
- **Are undirected** (treat symmetrically)
- **Tend to form within cell classes** (homotypic, L/R pairs, similar neurons)
- **Depend on innexin protein expression** (INX-1 through INX-22)
- **Are sparser** than chemical synapses

Prior expectation: gap junction arrival may be LESS predictable from topology (more biology-driven) but shared-neighbor / transitive-closure logic could still hold.

## Preregistered criteria

1. **N transitions ≥ 3** (same as Nb07).
2. **Per-transition N ≥ 100 candidates**.
3. **Pooled AUC ≥ 0.60** (gap junctions are noisier; lower bar than Nb07's 0.76).
4. **Permutation null 95pct < observed**.

## Halting rule

If AUC < 0.55: gap junction arrival is not predictable from topology — may require innexin expression. If 0.55 ≤ AUC < 0.70: weak topological signal. If AUC ≥ 0.70: parallel to Nb07's finding."""))

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
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(42)

DEV = DERIVED / 'developmental'
STAGES = ['L1_1','L1_2','L1_3','L1_4','L2','L3','adult']
stage_data = {}
for s in STAGES:
    d = np.load(DEV / f'connectome_{s}.npz', allow_pickle=True)
    neurons = np.array([str(n) for n in d['neurons']])
    stage_data[s] = {'neurons': neurons, 'gap': d['gap_adj'],
                     'n2i': {n: i for i, n in enumerate(neurons)}}
    print(f'{s}: {len(neurons)} neurons, {int((d[\"gap_adj\"]>0).sum())//2} undirected gap edges')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Common neurons + L1 consensus"))

cells.append(nbf.v4.new_code_cell("""common = sorted(set(stage_data['L1_1']['neurons']).intersection(
    *[set(stage_data[s]['neurons']) for s in STAGES[1:]]
))
print(f'Common: {len(common)}')

def gap_on_common(s):
    idx = [stage_data[s]['n2i'][n] for n in common]
    G = stage_data[s]['gap'][np.ix_(idx, idx)] > 0
    np.fill_diagonal(G, 0)
    # Gap junctions are undirected; ensure symmetric
    G = (G | G.T).astype(np.int32)
    return G

G_by_stage = {s: gap_on_common(s) for s in STAGES}
for s in STAGES:
    edges = int(G_by_stage[s].sum() // 2)
    print(f'  {s}: {edges} undirected gap edges among {len(common)} common neurons')

G_L1 = np.stack([G_by_stage[s] for s in ['L1_1','L1_2','L1_3','L1_4']]).sum(axis=0)
G_L1_cons = (G_L1 >= 3).astype(np.int32)
G_L1_cons = ((G_L1_cons + G_L1_cons.T) > 0).astype(np.int32)  # ensure sym
print(f'\\nL1 consensus (>=3/4 replicates): {G_L1_cons.sum()//2} undirected edges')

TRANSITIONS = [
    ('L1_consensus','L2', G_L1_cons, G_by_stage['L2']),
    ('L2','L3', G_by_stage['L2'], G_by_stage['L3']),
    ('L3','adult', G_by_stage['L3'], G_by_stage['adult']),
]"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Topological features (symmetric, reflecting undirected gap junctions)"))

cells.append(nbf.v4.new_code_cell("""def build_gap_candidates(G_prev, G_next):
    N = G_prev.shape[0]
    deg = G_prev.sum(axis=1)
    shared_neighbors = G_prev @ G_prev  # (i,j) = # of k such that k is gap-neighbor of both i,j
    # 2-hop reachability via gap graph
    rows = []
    for i in range(N):
        for j in range(i+1, N):  # undirected — each unordered pair once
            if G_prev[i, j] == 1: continue  # already has edge
            rows.append({
                'i': i, 'j': j,
                'arrived': int(G_next[i, j]),
                'deg_i': deg[i], 'deg_j': deg[j],
                'deg_product': float(deg[i]) * float(deg[j]),
                'deg_sum': deg[i] + deg[j],
                'shared_neighbors': shared_neighbors[i, j],
            })
    return pd.DataFrame(rows)

per_t = {}
for prev_name, next_name, G_prev, G_next in TRANSITIONS:
    c = build_gap_candidates(G_prev, G_next)
    per_t[(prev_name, next_name)] = c
    print(f'{prev_name} -> {next_name}: {len(c)} candidates, {c[\"arrived\"].sum()} arrivals (rate {c[\"arrived\"].mean():.3f})')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Fit logistic per transition + pooled"))

cells.append(nbf.v4.new_code_cell("""COVS = ['deg_i','deg_j','deg_product','deg_sum','shared_neighbors']

def cv_auc(X, y, n_splits=5, random_state=42):
    if y.sum() < 5 or (y==0).sum() < 5:
        return np.array([np.nan]*n_splits), np.zeros(X.shape[1])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []; coefs = []
    for tr, te in skf.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        coefs.append(m.named_steps['logisticregression'].coef_[0])
    return np.array(aucs), np.mean(coefs, axis=0)

results_per_t = {}
for (pn, nn), c in per_t.items():
    X = c[COVS].values.astype(float); y = c['arrived'].values.astype(int)
    aucs, coefs = cv_auc(X, y)
    results_per_t[(pn, nn)] = {'aucs': aucs, 'coefs': coefs, 'N': len(c), 'n_pos': int(y.sum())}
    print(f'{pn} -> {nn}: AUC {aucs.mean():.3f} ± {aucs.std():.3f}  N={len(c)} arrivals={y.sum()}')
    print(f'  coefs: {dict(zip(COVS, [round(v, 3) for v in coefs]))}')

pooled = pd.concat([c.assign(transition=f'{pn}->{nn}') for (pn,nn), c in per_t.items()], ignore_index=True)
X = pooled[COVS].values.astype(float); y = pooled['arrived'].values.astype(int)
aucs_p, coefs_p = cv_auc(X, y)
print(f'\\nPooled AUC: {aucs_p.mean():.3f} ± {aucs_p.std():.3f}')
print('Pooled coefs:')
for c, v in zip(COVS, coefs_p):
    print(f'  {c:20s} {v:+.4f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Permutation null + criteria"))

cells.append(nbf.v4.new_code_cell("""N_PERM = 100
null_aucs = []
for i in range(N_PERM):
    y_p = y.copy()
    for tr in pooled['transition'].unique():
        mask = (pooled['transition']==tr).values
        y_p[mask] = RNG.permutation(y[mask])
    a, _ = cv_auc(X, y_p, random_state=42+i)
    null_aucs.append(a.mean())
null_aucs = np.array(null_aucs)
print(f'Null: mean={null_aucs.mean():.3f}, 95pct={np.percentile(null_aucs, 95):.3f}, max={null_aucs.max():.3f}')

crit1 = len(TRANSITIONS) >= 3
crit2 = all(r['N'] >= 100 for r in results_per_t.values())
crit3 = aucs_p.mean() >= 0.60
crit4 = aucs_p.mean() > np.percentile(null_aucs, 95)

print('\\nCRITERIA')
print('=' * 60)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 >=3 transitions                {len(TRANSITIONS)}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 per-trans N >= 100              min={min(r[\"N\"] for r in results_per_t.values())}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 pooled AUC >= 0.60              {aucs_p.mean():.3f}')
print(f'  [{\"PASS\" if crit4 else \"FAIL\"}] 4 AUC > null 95pct                {aucs_p.mean():.3f} vs {np.percentile(null_aucs, 95):.3f}')
print('=' * 60)
all_pass = all([crit1, crit2, crit3, crit4])

if all_pass and aucs_p.mean() >= 0.70:
    verdict = 'POSITIVE — strong topological rule for gap junction arrival'
elif all_pass:
    verdict = 'WEAK POSITIVE — topological signal exists for gap junctions but weaker than chemical'
else:
    verdict = 'NULL — gap junction arrival not predictable from topology alone'
print(f'\\nVERDICT: {verdict}')

summary = pd.DataFrame([{
    'n_transitions': len(TRANSITIONS),
    'pooled_N': len(pooled),
    'pooled_arrivals': int(y.sum()),
    'pooled_auc': float(aucs_p.mean()),
    'null_95pct': float(np.percentile(null_aucs, 95)),
    'shared_neighbors_coef': float(coefs_p[COVS.index('shared_neighbors')]),
    'deg_product_coef': float(coefs_p[COVS.index('deg_product')]),
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb10_final_summary.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/10_gap_junction_arrival.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 10 written ({len(nb.cells)} cells)')
