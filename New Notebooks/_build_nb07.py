"""Build Notebook 07 — developmental edge-arrival hazard model."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 07 — Developmental Edge-Arrival as a Topological Hazard

## Question

When a chemical synapse appears between some developmental stage S_t and S_{t+1}, is its arrival predicted by the *local topology at stage S_t* (shared partners, triangle-closure potential, 2-hop reachability)?

This asks whether C. elegans wiring unfolds as a **topological rule** independent of gene expression. It's a complement to 05/06: 05 asked if gene expression predicts rewiring (weak positive → null under stability), 06 asked if L-R compatibility predicts edges (null above contact). 07 asks if the connectome at stage t predicts itself forward — i.e., "preferential attachment" or "triadic closure" style dynamics.

## Why it's different from 03/04/05/06

All prior notebooks either:
- Tested gene-feature → topology-feature correlation at one stage (static)
- Treated edge labels as static classes (stable / added / never)

This notebook treats edge arrival as an **event** whose hazard depends on *topology at the preceding stage*. Genes aren't used at all — this is a pure structural dynamics question.

## Preregistered criteria

1. **N transitions ≥ 3** (we have 6 ordered L1→L1→L2→L3→adult transitions if we chain L1 replicates as independent; at minimum L1_mean → L2 → L3 → adult gives 3 transitions).
2. **Per-transition N ≥ 200 candidate "potential edges"** (pairs that could form an edge next stage but currently don't).
3. **Logistic model with topological covariates beats chance AUC ≥ 0.60**.
4. **Triangle-closure covariate (common neighbors) has positive significant coefficient** (p < 0.01 via permutation).
5. **Permutation null (shuffle outcome labels within transitions) 95pct < observed AUC**.

## Halting rule

If AUC < 0.55 across all transitions: declare null — "edge arrival in C. elegans is not predictable from local topology at preceding stage, at this resolution."

If AUC ≥ 0.60 with significant triangle-closure coefficient: real finding — a topological rule for wiring dynamics."""))

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
    stage_data[s] = {'neurons': neurons, 'chem': d['chem_adj'],
                     'n2i': {n: i for i, n in enumerate(neurons)}}
    print(f'{s}: {len(neurons)} neurons, {int((d[\"chem_adj\"]>0).sum())} chem edges')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Define ordered transitions + L1-consensus starting point"))

cells.append(nbf.v4.new_code_cell("""# Common neuron set across ALL stages (so we can compare like-for-like)
common = sorted(set(stage_data['L1_1']['neurons']).intersection(
    *[set(stage_data[s]['neurons']) for s in STAGES[1:]]
))
print(f'Common neurons (all 7 stages): {len(common)}')

# Build binary chem adjacency on common set, per stage
def chem_on_common(s, common):
    idx = [stage_data[s]['n2i'][n] for n in common]
    A = stage_data[s]['chem'][np.ix_(idx, idx)] > 0
    np.fill_diagonal(A, 0)
    return A.astype(np.int32)

A_by_stage = {s: chem_on_common(s, common) for s in STAGES}
for s in STAGES:
    print(f'  {s}: {A_by_stage[s].sum()} directed edges among {len(common)} common neurons')

# L1 consensus: edge present if in >= 3 of 4 L1 replicates
A_L1 = np.stack([A_by_stage[s] for s in ['L1_1','L1_2','L1_3','L1_4']]).sum(axis=0)
A_L1_cons = (A_L1 >= 3).astype(np.int32)
print(f'\\nL1 consensus (>=3/4 replicates): {A_L1_cons.sum()} edges')

TRANSITIONS = [
    ('L1_consensus', 'L2', A_L1_cons, A_by_stage['L2']),
    ('L2', 'L3', A_by_stage['L2'], A_by_stage['L3']),
    ('L3', 'adult', A_by_stage['L3'], A_by_stage['adult']),
]
print(f'\\nTransitions: {[(t[0], t[1]) for t in TRANSITIONS]}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Compute topological covariates per transition"))

cells.append(nbf.v4.new_code_cell("""def topological_features(A_prev):
    N = A_prev.shape[0]
    out_deg = A_prev.sum(axis=1)
    in_deg = A_prev.sum(axis=0)
    A2 = A_prev @ A_prev
    # For each ordered pair (i, j):
    #   shared_out: # of k such that i->k and j->k  =>  (A_prev @ A_prev.T)[i, j]
    #   shared_in:  # of k such that k->i and k->j  =>  (A_prev.T @ A_prev)[i, j]
    #   triangle_closure: # of k such that i->k and k->j  =>  A2[i, j]
    #   reverse_2step:     # of k such that j->k and k->i =>  A2[j, i]
    shared_out = A_prev @ A_prev.T
    shared_in  = A_prev.T @ A_prev
    return {
        'out_deg_i': out_deg,
        'in_deg_j':  in_deg,
        'shared_out': shared_out,
        'shared_in':  shared_in,
        'triangle_closure': A2,
        'reverse_2step': A2.T,
    }

def build_candidate_table(A_prev, A_next):
    N = A_prev.shape[0]
    feats = topological_features(A_prev)
    rows = []
    for i in range(N):
        for j in range(N):
            if i == j: continue
            prev = int(A_prev[i, j])
            nxt  = int(A_next[i, j])
            # Candidate = edge did NOT exist at prev (so can potentially arrive)
            if prev == 1: continue
            rows.append({
                'i': i, 'j': j,
                'arrived': nxt,  # 1 if arrived at next stage
                'out_deg_i': feats['out_deg_i'][i],
                'in_deg_j':  feats['in_deg_j'][j],
                'shared_out': feats['shared_out'][i, j],
                'shared_in':  feats['shared_in'][i, j],
                'triangle_closure': feats['triangle_closure'][i, j],
                'reverse_2step':    feats['reverse_2step'][i, j],
            })
    return pd.DataFrame(rows)

per_transition = {}
for prev_name, next_name, A_prev, A_next in TRANSITIONS:
    cand = build_candidate_table(A_prev, A_next)
    arrival_rate = cand['arrived'].mean()
    print(f'{prev_name} -> {next_name}: {len(cand)} candidates, {cand[\"arrived\"].sum()} arrivals (rate {arrival_rate:.3f})')
    per_transition[(prev_name, next_name)] = cand"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Fit logistic hazard per transition + joint model"))

cells.append(nbf.v4.new_code_cell("""COVARIATES = ['out_deg_i', 'in_deg_j', 'shared_out', 'shared_in', 'triangle_closure', 'reverse_2step']

def cv_logistic(X, y, n_splits=5, random_state=42):
    if y.sum() < 5 or (y==0).sum() < 5:
        return np.array([np.nan]*n_splits), np.zeros(X.shape[1])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    coefs = []
    for tr, te in skf.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        coefs.append(m.named_steps['logisticregression'].coef_[0])
    return np.array(aucs), np.mean(coefs, axis=0)

per_trans_results = {}
for (prev_name, next_name), cand in per_transition.items():
    X = cand[COVARIATES].values.astype(float)
    y = cand['arrived'].values.astype(int)
    aucs, coefs = cv_logistic(X, y)
    per_trans_results[(prev_name, next_name)] = {'aucs': aucs, 'coefs': coefs, 'N': len(cand), 'n_pos': int(y.sum())}
    print(f'{prev_name} -> {next_name}: AUC {aucs.mean():.3f} ± {aucs.std():.3f}  N={len(cand)}  arrivals={y.sum()}')
    print(f'  coefs: {dict(zip(COVARIATES, [round(c,3) for c in coefs]))}')"""))

cells.append(nbf.v4.new_code_cell("""# Pooled analysis: concatenate all transitions with a "transition" indicator
pooled_rows = []
for (prev_name, next_name), cand in per_transition.items():
    c2 = cand.copy()
    c2['transition'] = f'{prev_name}->{next_name}'
    pooled_rows.append(c2)
pooled = pd.concat(pooled_rows, ignore_index=True)
print(f'Pooled candidates: {len(pooled)}')
print(f'Pooled arrivals:   {pooled[\"arrived\"].sum()}  (rate {pooled[\"arrived\"].mean():.3f})')

X = pooled[COVARIATES].values.astype(float)
y = pooled['arrived'].values.astype(int)
aucs_pooled, coefs_pooled = cv_logistic(X, y)
print(f'\\nPooled AUC: {aucs_pooled.mean():.3f} ± {aucs_pooled.std():.3f}')
print(f'Pooled coefs:')
for c, v in zip(COVARIATES, coefs_pooled):
    print(f'  {c:20s} {v:+.4f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Permutation null (shuffle outcome within each transition)"))

cells.append(nbf.v4.new_code_cell("""N_PERM = 100
null_aucs = []
t0 = time.time()
for i in range(N_PERM):
    y_perm = y.copy()
    for tr in pooled['transition'].unique():
        mask = (pooled['transition'] == tr).values
        y_perm[mask] = RNG.permutation(y[mask])
    a, _ = cv_logistic(X, y_perm, random_state=42+i)
    null_aucs.append(a.mean())
    if (i+1) % 25 == 0:
        print(f'  {i+1}/{N_PERM} perms ({time.time()-t0:.1f}s)')
null_aucs = np.array(null_aucs)
print(f'\\nPermutation null: mean={null_aucs.mean():.3f}, 95pct={np.percentile(null_aucs, 95):.3f}, max={null_aucs.max():.3f}')
print(f'Observed pooled AUC: {aucs_pooled.mean():.3f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 5 — Preregistered criteria"))

cells.append(nbf.v4.new_code_cell("""triangle_idx = COVARIATES.index('triangle_closure')
shared_out_idx = COVARIATES.index('shared_out')
tc_coef = coefs_pooled[triangle_idx]
so_coef = coefs_pooled[shared_out_idx]

# Permutation p-value for triangle-closure coefficient
null_tc = []
for i in range(100):
    y_perm = y.copy()
    for tr in pooled['transition'].unique():
        mask = (pooled['transition'] == tr).values
        y_perm[mask] = RNG.permutation(y[mask])
    _, coefs = cv_logistic(X, y_perm, random_state=42+i)
    null_tc.append(coefs[triangle_idx])
null_tc = np.array(null_tc)
p_tc = float((np.abs(null_tc) >= abs(tc_coef)).mean())
print(f'triangle_closure observed coef: {tc_coef:+.4f}')
print(f'triangle_closure null 95% range: [{np.percentile(null_tc, 2.5):+.4f}, {np.percentile(null_tc, 97.5):+.4f}]')
print(f'triangle_closure two-tailed permutation p: {p_tc:.4f}')

crit1 = len(TRANSITIONS) >= 3
crit2 = all(r['N'] >= 200 for r in per_trans_results.values())
crit3 = aucs_pooled.mean() >= 0.60
crit4 = (tc_coef > 0) and (p_tc < 0.01)
crit5 = aucs_pooled.mean() > np.percentile(null_aucs, 95)

print('\\nPREREGISTERED CRITERIA')
print('=' * 70)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 N transitions >= 3                        {len(TRANSITIONS)}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 Per-transition N >= 200 candidates         min={min(r[\"N\"] for r in per_trans_results.values())}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 Pooled AUC >= 0.60                          {aucs_pooled.mean():.3f}')
print(f'  [{\"PASS\" if crit4 else \"FAIL\"}] 4 Triangle-closure positive w/ p<0.01          coef={tc_coef:+.4f}, p={p_tc:.4f}')
print(f'  [{\"PASS\" if crit5 else \"FAIL\"}] 5 AUC > null 95pct                             {aucs_pooled.mean():.3f} vs {np.percentile(null_aucs, 95):.3f}')
print('=' * 70)
all_pass = all([crit1, crit2, crit3, crit4, crit5])

if all_pass:
    verdict = 'POSITIVE — topological rule governs edge arrival during C. elegans development'
elif crit3 and crit5:
    verdict = 'POSITIVE (AUC) — signal real above null but specific covariate interpretation weak'
elif crit3:
    verdict = 'WEAK POSITIVE (AUC above bar but not above null)'
else:
    verdict = 'NULL — edge arrival not predictable from topology at preceding stage'
print(f'\\nVERDICT: {verdict}')

summary = pd.DataFrame([{
    'n_transitions': len(TRANSITIONS),
    'pooled_N': len(pooled),
    'pooled_arrivals': int(y.sum()),
    'pooled_auc': float(aucs_pooled.mean()),
    'null_95pct': float(np.percentile(null_aucs, 95)),
    'triangle_closure_coef': float(tc_coef),
    'triangle_closure_perm_p': p_tc,
    'shared_out_coef': float(so_coef),
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb07_final_summary.csv', index=False)
print(summary.T.to_string())
pooled.to_csv(DERIVED / 'nb07_pooled_candidates.csv', index=False)"""))

nb.cells = cells

with open('/home/rohit/Desktop/C-Elegans/New Notebooks/07_developmental_edge_hazard.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 07 written ({len(nb.cells)} cells)')
