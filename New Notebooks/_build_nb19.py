"""Nb 19 — Electrical Scaffold Hypothesis: does the gap-junction graph at stage t predict chemical-edge arrival at t+1?"""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 19 — Electrical Scaffold Hypothesis

## The hypothesis (biologically motivated)

In mammalian cortical development, transient gap junctions form temporary electrical networks that synchronize firing; this synchronous activity guides later chemical synapse formation, after which the gap junctions disappear. Does the same logic hold in *C. elegans*?

**If the electrical-scaffold hypothesis is correct**: neurons connected by a gap junction at stage t should be disproportionately likely to form a chemical synapse by stage t+1.

## What we're doing

For each developmental transition (L1_consensus → L2, L2 → L3, L3 → adult), ask: given the gap-junction connectome at stage t, does a direct or near-direct gap connection predict chemical-edge arrival at t+1? We test this **above** Nb07's topological rule (shared chemical partners, triadic closure) — so the question is specifically whether electrical-graph proximity adds signal beyond chemical-graph topology.

## Preregistered criteria

1. **Baseline recapitulates Nb07**: topology-only AUC ≥ 0.75 on the pooled candidate set. (Sanity.)
2. **Electrical features alone are predictive**: AUC ≥ 0.60 when using only gap-junction features (direct gap edge, gap-graph neighbors, gap-graph 2-hop reachability).
3. **Electrical adds above chemical topology**: delta (topology + gap features) − (topology alone) ≥ 0.02 AUC.
4. **Delta is not random**: permutation null (shuffle gap features across candidate pairs, keep topology aligned) gives 95pct delta < observed.
5. **Effect is larger at earlier transitions**: AUC improvement from gap features is ≥ 0.02 greater for L1→L2 than for L3→adult. (The scaffold hypothesis predicts gap junctions matter MOST early.)

## Halting rules

- If criterion 3 fails: gap junctions at stage t DO NOT predict chemical arrival at t+1 beyond topology. Null finding, report honestly.
- If 3 passes but 5 fails: gap junctions add signal but not in a developmental-scaffold-specific way (could be a static correlation).
- If 3 AND 5 pass: strong support for the electrical-scaffold hypothesis."""))

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

# Load all 7 Witvliet stages
STAGES = ['L1_1','L1_2','L1_3','L1_4','L2','L3','adult']
stage_data = {}
for s in STAGES:
    d = np.load(DERIVED / 'developmental' / f'connectome_{s}.npz', allow_pickle=True)
    n = np.array([str(x) for x in d['neurons']])
    stage_data[s] = {
        'neurons': n,
        'chem': d['chem_adj'],
        'gap':  d['gap_adj'],
        'n2i': {x: i for i, x in enumerate(n)},
    }
    print(f'{s}: {len(n)} neurons, chem={int((d[\"chem_adj\"]>0).sum())} gap={int((d[\"gap_adj\"]>0).sum())//2}')

# Common set across all stages
common = sorted(set(stage_data['L1_1']['neurons']).intersection(
    *[set(stage_data[s]['neurons']) for s in STAGES[1:]]
))
N_common = len(common)
print(f'\\nCommon neurons across all 7 stages: {N_common}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Build per-transition candidate tables with BOTH chemical-topology AND gap-junction features"))

cells.append(nbf.v4.new_code_cell("""def chem_on_common(s):
    idx = [stage_data[s]['n2i'][n] for n in common]
    A = (stage_data[s]['chem'][np.ix_(idx, idx)] > 0).astype(np.int32)
    np.fill_diagonal(A, 0)
    return A

def gap_on_common(s):
    idx = [stage_data[s]['n2i'][n] for n in common]
    G = stage_data[s]['gap'][np.ix_(idx, idx)] > 0
    np.fill_diagonal(G, 0)
    G = ((G | G.T)).astype(np.int32)  # ensure symmetric
    return G

# L1 consensus on chem + gap
L1s = ['L1_1','L1_2','L1_3','L1_4']
A_chem_L1cons = (np.stack([chem_on_common(s) for s in L1s]).sum(axis=0) >= 3).astype(np.int32)
A_gap_L1cons  = (np.stack([gap_on_common(s)  for s in L1s]).sum(axis=0) >= 3).astype(np.int32)
A_gap_L1cons  = ((A_gap_L1cons + A_gap_L1cons.T) > 0).astype(np.int32)

TRANSITIONS = [
    ('L1_cons', 'L2', A_chem_L1cons, chem_on_common('L2'), A_gap_L1cons, gap_on_common('L2')),
    ('L2', 'L3',     chem_on_common('L2'),    chem_on_common('L3'),    gap_on_common('L2'),    gap_on_common('L3')),
    ('L3', 'adult',  chem_on_common('L3'),    chem_on_common('adult'), gap_on_common('L3'),    gap_on_common('adult')),
]
print(f'Transitions to analyze: {[(t[0], t[1]) for t in TRANSITIONS]}')"""))

cells.append(nbf.v4.new_code_cell("""def topological_features(A_chem):
    N = A_chem.shape[0]
    out_deg = A_chem.sum(axis=1)
    in_deg  = A_chem.sum(axis=0)
    A2 = A_chem @ A_chem
    shared_out = A_chem @ A_chem.T
    shared_in  = A_chem.T @ A_chem
    return {
        'out_deg_i': out_deg,
        'in_deg_j':  in_deg,
        'shared_out': shared_out,
        'shared_in':  shared_in,
        'triangle_closure': A2,
        'reverse_2step': A2.T,
    }

def gap_features(A_gap):
    N = A_gap.shape[0]
    gap_deg = A_gap.sum(axis=1)
    gap_2hop = A_gap @ A_gap
    gap_shared = A_gap @ A_gap.T  # shared gap-neighbors (symmetric since gap is undirected)
    return {
        'gap_direct': A_gap,       # 1 if i-j gap junction exists
        'gap_deg_i': gap_deg,
        'gap_deg_j': gap_deg,
        'gap_2hop':  gap_2hop,
        'gap_shared': gap_shared,
    }

def build_candidates(A_chem_prev, A_chem_next, A_gap_prev):
    N = A_chem_prev.shape[0]
    topo = topological_features(A_chem_prev)
    gap  = gap_features(A_gap_prev)
    rows = []
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if A_chem_prev[i, j] == 1: continue  # edge already present
            rows.append({
                'i': i, 'j': j,
                'arrived': int(A_chem_next[i, j]),
                # topology
                'out_deg_i': topo['out_deg_i'][i],
                'in_deg_j':  topo['in_deg_j'][j],
                'shared_out': topo['shared_out'][i, j],
                'shared_in':  topo['shared_in'][i, j],
                'triangle_closure': topo['triangle_closure'][i, j],
                'reverse_2step':    topo['reverse_2step'][i, j],
                # gap-junction features
                'gap_direct':      gap['gap_direct'][i, j],
                'gap_deg_i':       gap['gap_deg_i'][i],
                'gap_deg_j':       gap['gap_deg_j'][j],
                'gap_2hop':        gap['gap_2hop'][i, j],
                'gap_shared':      gap['gap_shared'][i, j],
            })
    return pd.DataFrame(rows)

per_t = {}
for prev_name, next_name, A_chem_p, A_chem_n, A_gap_p, A_gap_n in TRANSITIONS:
    c = build_candidates(A_chem_p, A_chem_n, A_gap_p)
    per_t[(prev_name, next_name)] = c
    n_arr = int(c['arrived'].sum())
    n_direct_gap = int((c['gap_direct'] == 1).sum())
    n_both = int(((c['gap_direct'] == 1) & (c['arrived'] == 1)).sum())
    print(f'{prev_name}->{next_name}: N={len(c)}, arrivals={n_arr}, candidates w/ direct gap={n_direct_gap}, arrivals WITH gap={n_both}')
    # Enrichment: P(arrive | gap) vs P(arrive | no gap)
    if n_direct_gap > 0 and (len(c) - n_direct_gap) > 0:
        p_w_gap = n_both / n_direct_gap
        p_wo_gap = (n_arr - n_both) / max(1, (len(c) - n_direct_gap))
        print(f'    P(arrive|gap) = {p_w_gap:.3f}  P(arrive|no gap) = {p_wo_gap:.3f}  ratio = {p_w_gap/max(p_wo_gap, 1e-6):.2f}x')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Nested AUC comparison per transition + pooled"))

cells.append(nbf.v4.new_code_cell("""TOP_COVS = ['out_deg_i','in_deg_j','shared_out','shared_in','triangle_closure','reverse_2step']
GAP_COVS = ['gap_direct','gap_deg_i','gap_deg_j','gap_2hop','gap_shared']

def cv_auc(X, y, n_splits=5, random_state=42, C=1.0):
    if y.sum() < 5 or (y==0).sum() < 5:
        return np.array([np.nan])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    for tr, te in skf.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C))
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.array(aucs)

per_trans_results = {}
for (pn, nn), c in per_t.items():
    X_topo = c[TOP_COVS].values.astype(float)
    X_gap  = c[GAP_COVS].values.astype(float)
    X_full = np.concatenate([X_topo, X_gap], axis=1)
    y = c['arrived'].values.astype(int)
    auc_topo = cv_auc(X_topo, y).mean()
    auc_gap  = cv_auc(X_gap, y).mean()
    auc_full = cv_auc(X_full, y).mean()
    delta = auc_full - auc_topo
    per_trans_results[(pn, nn)] = {
        'auc_topo': auc_topo, 'auc_gap': auc_gap, 'auc_full': auc_full, 'delta': delta,
        'N': len(c), 'n_pos': int(y.sum()),
    }
    print(f'{pn}->{nn}: topo={auc_topo:.3f}, gap-only={auc_gap:.3f}, full={auc_full:.3f}, delta={delta:+.4f}  (N={len(c)}, pos={y.sum()})')

# Pooled
pooled = pd.concat([c.assign(transition=f'{pn}->{nn}') for (pn,nn), c in per_t.items()], ignore_index=True)
X_topo = pooled[TOP_COVS].values.astype(float)
X_gap  = pooled[GAP_COVS].values.astype(float)
X_full = np.concatenate([X_topo, X_gap], axis=1)
y = pooled['arrived'].values.astype(int)

auc_pooled_topo = cv_auc(X_topo, y).mean()
auc_pooled_gap  = cv_auc(X_gap, y).mean()
auc_pooled_full = cv_auc(X_full, y).mean()
delta_pooled = auc_pooled_full - auc_pooled_topo

print(f'\\nPOOLED:')
print(f'  Topology-only AUC: {auc_pooled_topo:.3f}')
print(f'  Gap-only AUC:      {auc_pooled_gap:.3f}')
print(f'  Full AUC:          {auc_pooled_full:.3f}')
print(f'  Delta (gap adds):  {delta_pooled:+.4f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Permutation null on gap features"))

cells.append(nbf.v4.new_code_cell("""N_PERM = 100
null_deltas = []
t0 = time.time()
for i in range(N_PERM):
    perm = RNG.permutation(X_full.shape[0])
    X_shuf = X_full.copy()
    X_shuf[:, len(TOP_COVS):] = X_full[perm, len(TOP_COVS):]  # shuffle gap features, keep topology
    aucs_s = cv_auc(X_shuf, y, random_state=42+i).mean()
    null_deltas.append(aucs_s - auc_pooled_topo)
    if (i+1) % 25 == 0:
        print(f'  {i+1}/{N_PERM}  ({time.time()-t0:.1f}s)  last delta: {null_deltas[-1]:+.4f}')
null_deltas = np.array(null_deltas)
print(f'\\nNull deltas (shuffle gap features): mean={null_deltas.mean():+.4f}, 95pct={np.percentile(null_deltas, 95):+.4f}')
print(f'Observed gap-add delta: {delta_pooled:+.4f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Preregistered criteria"))

cells.append(nbf.v4.new_code_cell("""# Earliest transition: L1_cons->L2.  Latest: L3->adult.
earliest_delta = per_trans_results[('L1_cons','L2')]['delta']
latest_delta   = per_trans_results[('L3','adult')]['delta']

crit1 = auc_pooled_topo >= 0.75
crit2 = auc_pooled_gap  >= 0.60
crit3 = delta_pooled >= 0.02
crit4 = delta_pooled > np.percentile(null_deltas, 95)
crit5 = (earliest_delta - latest_delta) >= 0.02

print('PREREGISTERED CRITERIA')
print('=' * 70)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 Topology AUC >= 0.75              {auc_pooled_topo:.3f}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 Gap-only AUC >= 0.60                 {auc_pooled_gap:.3f}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 Delta (gap add) >= 0.02              {delta_pooled:+.4f}')
print(f'  [{\"PASS\" if crit4 else \"FAIL\"}] 4 Delta > null 95pct                   {delta_pooled:+.4f} vs {np.percentile(null_deltas, 95):+.4f}')
print(f'  [{\"PASS\" if crit5 else \"FAIL\"}] 5 Earliest > Latest delta (scaffold)   L1cons->L2 {earliest_delta:+.4f} vs L3->adult {latest_delta:+.4f}')
print('=' * 70)

critical_pass = crit3 and crit4  # delta real and above null
scaffold_pass = crit5

if critical_pass and scaffold_pass:
    verdict = 'STRONG POSITIVE — gap junctions scaffold later chemical synapse formation (developmental gradient present)'
elif critical_pass:
    verdict = 'POSITIVE — gap junctions add predictive power, but no developmental-scaffold gradient'
elif delta_pooled > 0:
    verdict = 'WEAK POSITIVE — gap features help but below threshold'
else:
    verdict = 'NULL — gap junctions do NOT scaffold chemical synapse formation above topology'

print(f'\\nVERDICT: {verdict}')

summary = pd.DataFrame([{
    'transitions': len(TRANSITIONS),
    'pooled_N': len(pooled),
    'pooled_arrivals': int(y.sum()),
    'auc_topology': auc_pooled_topo,
    'auc_gap_only': auc_pooled_gap,
    'auc_full': auc_pooled_full,
    'delta_gap_adds': delta_pooled,
    'null_95pct_delta': float(np.percentile(null_deltas, 95)),
    'earliest_delta': earliest_delta,
    'latest_delta': latest_delta,
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb19_final_summary.csv', index=False)
print(summary.T.to_string())

per_trans_df = pd.DataFrame([
    {'transition': f'{pn}->{nn}', **{k: v for k, v in r.items() if k not in ['aucs', 'coefs']}}
    for (pn, nn), r in per_trans_results.items()
])
per_trans_df.to_csv(DERIVED / 'nb19_per_transition.csv', index=False)
print('\\nPer-transition table:')
print(per_trans_df.to_string())"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/19_electrical_scaffold.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 19 written ({len(nb.cells)} cells)')
