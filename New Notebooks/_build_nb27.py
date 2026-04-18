"""Nb 27 — Channel selection 4-way classification with NT-tautology ablation."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 27 — Channel-Selection 4-Way Classification with NT-Tautology Ablation

## The moonshot question

Among the 10,060 contacting neuron pairs in Nb25, each pair falls into one of four channels:
- **neither**: contact but no edge of either kind
- **synapse-only**: contact + chemical synapse, no peptide edge
- **peptide-only**: contact + peptide edge, no chemical synapse
- **both**: contact + chemical synapse + peptide edge

Can we predict *which channel* a pair ends up in from features? And — critically — is the predictive signal real, or is it just recovering the way we built the peptide graph in the first place?

## The tautology trap (why this needs careful design)

`A_peptide` was constructed in Nb12 from Bentley 2016 annotations of (pre-neuron ligand) × (post-neuron receptor) × canonical L-R pairs. So pre-neuron NT-class is a **definitional** input to peptide-edge presence. Nb25 already showed monoamine pre-neurons are 3–10× enriched in peptide-only pairs (p = 10⁻¹⁵ to 10⁻²⁹) — that finding is real but partially tautological.

Any classifier using pre-NT as a feature will trivially nail peptide-only classification. The non-trivial question is whether **topology and contact alone** can predict channel choice — i.e., does the position of a neuron pair in the connectome independently encode which signaling mode the pair uses?

## Three nested models (preregistered hierarchy)

- **Model C (topology-only)**: contact area + 6 topology features. No cell-identity information.
- **Model B (+ gene PCA)**: Model C + 50 gene PCs each for pre and post. Genes partially encode NT but not deterministically.
- **Model A (+ NT labels)**: Model B + explicit one-hot NT identity. Expected ceiling; partially tautological.

The load-bearing comparison is **Model C vs chance**. If topology-only can distinguish channel classes at above-chance AUC, that means network position carries information about signaling mode that is NOT just cell-type identity. That's the moonshot.

## Preregistered criteria

1. **Class balance**: each of the 4 classes has ≥ 100 samples.
2. **Model C macro-AUC ≥ 0.60**: topology-only beats chance non-trivially.
3. **Model C per-class AUC ≥ 0.55** for at least 3 of 4 classes.
4. **Delta (Model B − Model C) ≥ 0.02** on macro-AUC: genes add above topology.
5. **Delta (Model A − Model B) ≥ 0.05** on peptide-only class specifically: confirms the tautology reservoir (this is the *expected* inflation from NT identity).
6. **Bootstrap 95% CIs on all Model C per-class AUCs exclude 0.50** (chance).

## Halting rule

- **If Model C macro-AUC < 0.55**: topology has no independent signal for channel choice. The tautology is doing all the work. Null result — monoaminergic-vs-synaptic choice is a cell-type property, not a network property.
- **If Model C macro-AUC ∈ [0.55, 0.65]**: weak topology signal. Report honestly.
- **If Model C macro-AUC ≥ 0.65**: topology and contact carry real channel-selection information beyond cell identity. Genuine new finding."""))

cells.append(nbf.v4.new_code_cell("""import sys, time
from pathlib import Path
import warnings; warnings.simplefilter('ignore')
_HERE = Path.cwd()
if (_HERE / 'lib').is_dir(): sys.path.insert(0, str(_HERE))
elif (_HERE.parent / 'lib').is_dir(): sys.path.insert(0, str(_HERE.parent))

from lib.paths import DATA, DERIVED
from lib.reference import load_nt_reference

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(42)

# Load core data
adult = np.load(DERIVED / 'connectome_adult.npz', allow_pickle=True)
w_neurons = np.array([str(n) for n in adult['neurons']])
A_syn = (adult['chem_adj'] > 0).astype(np.int32)
np.fill_diagonal(A_syn, 0)

pep = np.load(DERIVED / 'nb12_peptide_adjacency.npz', allow_pickle=True)
A_pep = pep['A_peptide'].astype(np.int32)
np.fill_diagonal(A_pep, 0)

# Contact matrix
contact_xlsx = DATA / 'connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx'
contact_df = pd.read_excel(contact_xlsx, sheet_name='adult nerve ring neighbors', header=0, index_col=0)
contact_df = contact_df.loc[:, ~contact_df.columns.str.startswith('Unnamed')]
contact_df = contact_df[contact_df.index.map(lambda x: isinstance(x, str))]
contact_df = contact_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
name_to_idx = {n: i for i, n in enumerate(w_neurons)}
C_mat = np.zeros((len(w_neurons), len(w_neurons)), dtype=np.float32)
for rn in contact_df.index:
    if rn not in name_to_idx: continue
    ri = name_to_idx[rn]
    for cn in contact_df.columns:
        if cn not in name_to_idx: continue
        ci = name_to_idx[cn]
        v = contact_df.loc[rn, cn]
        if pd.notna(v): C_mat[ri, ci] = float(v)
C_mat = (C_mat + C_mat.T) / 2

nt = load_nt_reference()

mapping = pd.read_csv(DERIVED / 'expression_neuron_mapping.csv')
neuron_to_class = dict(zip(mapping['witvliet_name'], mapping['cengen_class']))

expr_data = np.load(DERIVED / 'expression_tpm.npz', allow_pickle=True)
tpm_neurons = expr_data['neurons']; tpm = expr_data['tpm']
class_to_expr = {}
for i, nm in enumerate(tpm_neurons):
    cls = neuron_to_class.get(str(nm))
    if isinstance(cls, str) and cls not in class_to_expr and not np.all(np.isnan(tpm[i])):
        class_to_expr[cls] = tpm[i]

print(f'Neurons: {len(w_neurons)}, Synapse edges: {int(A_syn.sum())}, Peptide edges: {int(A_pep.sum())}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Build the 4-class channel labels on contacting pairs"))

cells.append(nbf.v4.new_code_cell("""# 4-class labels
N = len(w_neurons)
rows = []
for i in range(N):
    for j in range(N):
        if i == j: continue
        c = float(C_mat[i, j])
        if c <= 0: continue
        s_edge = int(A_syn[i, j])
        p_edge = int(A_pep[i, j])
        if s_edge == 0 and p_edge == 0: label = 0  # neither
        elif s_edge == 1 and p_edge == 0: label = 1  # synapse-only
        elif s_edge == 0 and p_edge == 1: label = 2  # peptide-only
        else: label = 3  # both
        rows.append({'i': i, 'j': j, 'pre': w_neurons[i], 'post': w_neurons[j],
                     'contact_area': c, 'label': label})

pairs = pd.DataFrame(rows)
print(f'Contacting pairs: {len(pairs)}')
class_counts = pairs['label'].value_counts().sort_index()
class_names = {0: 'neither', 1: 'syn-only', 2: 'pep-only', 3: 'both'}
for cls, cnt in class_counts.items():
    print(f'  Class {cls} ({class_names[cls]}): {cnt}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Build feature sets (Models C, B, A)"))

cells.append(nbf.v4.new_code_cell("""# --- Topology features ---
A_bin = A_syn.copy()  # use chemical graph for topology
out_deg = A_bin.sum(axis=1)
in_deg = A_bin.sum(axis=0)
A2 = A_bin @ A_bin
shared_out = A_bin @ A_bin.T
shared_in = A_bin.T @ A_bin
triangle_closure = A2
reverse_2step = A2.T

log_contact = np.log1p(pairs['contact_area'].values)
topo_cols = {
    'log_contact': log_contact,
    'out_deg_i': np.array([out_deg[r['i']] for _, r in pairs.iterrows()]),
    'in_deg_j':  np.array([in_deg[r['j']] for _, r in pairs.iterrows()]),
    'shared_out': np.array([shared_out[r['i'], r['j']] for _, r in pairs.iterrows()]),
    'shared_in':  np.array([shared_in[r['i'], r['j']]  for _, r in pairs.iterrows()]),
    'triangle_closure': np.array([triangle_closure[r['i'], r['j']] for _, r in pairs.iterrows()]),
    'reverse_2step':    np.array([reverse_2step[r['i'], r['j']] for _, r in pairs.iterrows()]),
}
X_C = np.column_stack([topo_cols[k] for k in ['log_contact','out_deg_i','in_deg_j','shared_out','shared_in','triangle_closure','reverse_2step']])
print(f'Model C features: {X_C.shape}')

# --- Gene PCA features ---
classes_list = sorted(class_to_expr.keys())
X_cls = np.stack([np.log1p(class_to_expr[c]) for c in classes_list])
pca_model = PCA(n_components=50, random_state=42).fit(X_cls)
X_pca = pca_model.transform(X_cls)
class_to_pc = {c: X_pca[idx] for idx, c in enumerate(classes_list)}
ZERO = np.zeros(50)
def get_pc(n): return class_to_pc.get(neuron_to_class.get(n, ''), ZERO)

gene_pre = np.stack([get_pc(r['pre']) for _, r in pairs.iterrows()])
gene_post = np.stack([get_pc(r['post']) for _, r in pairs.iterrows()])
X_B = np.concatenate([X_C, gene_pre, gene_post], axis=1)
print(f'Model B features: {X_B.shape}')

# --- NT identity features (one-hot) ---
def get_nt_simple(n):
    v = nt.nt_of(n)
    if v is None: return 'Unknown'
    s = v.lower()
    if 'acetylcholine' in s: return 'ACh'
    if 'gaba' in s: return 'GABA'
    if 'glutamate' in s: return 'Glu'
    if 'dopamine' in s: return 'DA'
    if 'serotonin' in s: return '5HT'
    if 'octopamine' in s: return 'OA'
    if 'tyramine' in s: return 'TA'
    return 'Other'

pre_nt = np.array([get_nt_simple(r['pre']) for _, r in pairs.iterrows()])
post_nt = np.array([get_nt_simple(r['post']) for _, r in pairs.iterrows()])
NT_CATS = ['ACh','GABA','Glu','DA','5HT','OA','TA','Unknown','Other']
ohe = OneHotEncoder(categories=[NT_CATS, NT_CATS], sparse_output=False, handle_unknown='ignore')
nt_oh = ohe.fit_transform(np.column_stack([pre_nt, post_nt]))
X_A = np.concatenate([X_B, nt_oh], axis=1)
print(f'Model A features: {X_A.shape}')

y4 = pairs['label'].values"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — 4-class CV: per-class one-vs-rest AUC + macro"))

cells.append(nbf.v4.new_code_cell("""def oof_proba(X, y, n_splits=5, random_state=42, C=1.0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros((len(y), 4))
    for tr, te in skf.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, multi_class='multinomial', C=C))
        m.fit(X[tr], y[tr])
        # Multinomial gives proba for all classes
        proba = m.predict_proba(X[te])
        # Align columns to ALL 4 classes
        present_classes = m.classes_
        for i_out, c in enumerate(present_classes):
            oof[te, c] = proba[:, i_out]
    return oof

def per_class_auc(y_true, y_proba):
    aucs = {}
    for c in [0, 1, 2, 3]:
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() < 10:
            aucs[c] = np.nan
        else:
            aucs[c] = roc_auc_score(y_bin, y_proba[:, c])
    return aucs

t0 = time.time()
oof_C = oof_proba(X_C, y4); print(f'Model C done ({time.time()-t0:.1f}s)')
oof_B = oof_proba(X_B, y4); print(f'Model B done ({time.time()-t0:.1f}s)')
oof_A = oof_proba(X_A, y4); print(f'Model A done ({time.time()-t0:.1f}s)')

for model_name, oof in [('Model C (topo+contact only)', oof_C),
                        ('Model B (+ gene PCA)', oof_B),
                        ('Model A (+ NT identity)', oof_A)]:
    aucs = per_class_auc(y4, oof)
    macro = np.nanmean(list(aucs.values()))
    print(f'\\n{model_name}:')
    for c in sorted(aucs):
        print(f'  Class {c} ({class_names[c]:<10s}): AUC {aucs[c]:.4f}')
    print(f'  Macro-AUC: {macro:.4f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Bootstrap 95% CIs on per-class AUCs"))

cells.append(nbf.v4.new_code_cell("""N_PAIRS = len(y4)
n_boot = 500

def bootstrap_per_class(oof, y, n_boot=n_boot):
    bs = {c: np.zeros(n_boot) for c in [0, 1, 2, 3]}
    for b in range(n_boot):
        idx = RNG.choice(N_PAIRS, N_PAIRS, replace=True)
        for c in [0, 1, 2, 3]:
            y_bin = (y[idx] == c).astype(int)
            if y_bin.sum() < 5 or (y_bin == 0).sum() < 5:
                bs[c][b] = np.nan
            else:
                bs[c][b] = roc_auc_score(y_bin, oof[idx, c])
    return bs

print('Running bootstraps on each model...')
bs_C = bootstrap_per_class(oof_C, y4)
bs_B = bootstrap_per_class(oof_B, y4)
bs_A = bootstrap_per_class(oof_A, y4)

def ci(arr):
    arr = arr[~np.isnan(arr)]
    return f'{np.mean(arr):.3f} [{np.percentile(arr, 2.5):.3f}, {np.percentile(arr, 97.5):.3f}]'

print('\\nModel C (topology + contact only) per-class AUC with 95% CIs:')
for c in [0, 1, 2, 3]:
    print(f'  Class {c} ({class_names[c]:<10s}): {ci(bs_C[c])}')
print('\\nModel B (+ gene PCA):')
for c in [0, 1, 2, 3]:
    print(f'  Class {c} ({class_names[c]:<10s}): {ci(bs_B[c])}')
print('\\nModel A (+ NT one-hot):')
for c in [0, 1, 2, 3]:
    print(f'  Class {c} ({class_names[c]:<10s}): {ci(bs_A[c])}')

# Deltas
print('\\n--- Deltas (Model B − Model C, per-class) ---')
for c in [0, 1, 2, 3]:
    d = bs_B[c] - bs_C[c]
    d = d[~np.isnan(d)]
    print(f'  Class {c}: {np.mean(d):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]')
print('\\n--- Deltas (Model A − Model B, per-class; tautology reservoir) ---')
for c in [0, 1, 2, 3]:
    d = bs_A[c] - bs_B[c]
    d = d[~np.isnan(d)]
    print(f'  Class {c}: {np.mean(d):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 5 — Preregistered criteria check"))

cells.append(nbf.v4.new_code_cell("""# Point estimates
aucs_C = per_class_auc(y4, oof_C)
aucs_B = per_class_auc(y4, oof_B)
aucs_A = per_class_auc(y4, oof_A)

macro_C = np.nanmean(list(aucs_C.values()))
macro_B = np.nanmean(list(aucs_B.values()))
macro_A = np.nanmean(list(aucs_A.values()))
delta_B_C = macro_B - macro_C
delta_A_B = macro_A - macro_B
delta_A_B_peptide = aucs_A[2] - aucs_B[2]  # for peptide-only class

# Class balance
min_class = int(pairs['label'].value_counts().min())

# Per-class C AUC >= 0.55 for at least 3 of 4
per_class_c_pass = sum(1 for c in [0, 1, 2, 3] if aucs_C[c] >= 0.55)

# CIs exclude 0.5 for each Model C per-class
ci_exclude_chance = sum(1 for c in [0, 1, 2, 3] if np.nanpercentile(bs_C[c], 2.5) > 0.50)

crit1 = min_class >= 100
crit2 = macro_C >= 0.60
crit3 = per_class_c_pass >= 3
crit4 = delta_B_C >= 0.02
crit5 = delta_A_B_peptide >= 0.05
crit6 = ci_exclude_chance == 4

print('PREREGISTERED CRITERIA')
print('=' * 72)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 min class count >= 100                     min={min_class}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 Model C macro-AUC >= 0.60                    {macro_C:.4f}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 Per-class C AUC >= 0.55 for >=3/4 classes    {per_class_c_pass}/4')
print(f'  [{\"PASS\" if crit4 else \"FAIL\"}] 4 Delta B-C macro >= 0.02 (genes add)          {delta_B_C:+.4f}')
print(f'  [{\"PASS\" if crit5 else \"FAIL\"}] 5 Delta A-B peptide-class >= 0.05              {delta_A_B_peptide:+.4f}')
print(f'  [{\"PASS\" if crit6 else \"FAIL\"}] 6 All Model C class CIs exclude 0.50 chance    {ci_exclude_chance}/4')
print('=' * 72)
n_pass = sum([crit1, crit2, crit3, crit4, crit5, crit6])

# The LOAD-BEARING interpretation
if macro_C < 0.55:
    verdict = 'NULL — topology has no channel-selection information. Monoaminergic-vs-synaptic is purely a cell-type property.'
elif 0.55 <= macro_C < 0.65:
    verdict = 'WEAK POSITIVE — topology weakly encodes channel choice; main signal is NT identity'
else:
    verdict = 'POSITIVE — topology and contact independently predict channel choice above chance'

print(f'\\n{n_pass}/6 criteria pass')
print(f'VERDICT: {verdict}')
print(f'\\nKEY NUMBERS FOR PAPER:')
print(f'  Model C (topology+contact only) macro-AUC:  {macro_C:.4f}')
print(f'  Model B (+ gene PCA) macro-AUC:             {macro_B:.4f}')
print(f'  Model A (+ NT labels) macro-AUC:            {macro_A:.4f}')
print(f'  Tautology reservoir on peptide class (A-B): {delta_A_B_peptide:+.4f}')

summary = pd.DataFrame([{
    'n_pairs': len(y4),
    'n_neither': int((y4==0).sum()), 'n_syn_only': int((y4==1).sum()),
    'n_pep_only': int((y4==2).sum()), 'n_both': int((y4==3).sum()),
    'modelC_macro_auc': macro_C,
    'modelB_macro_auc': macro_B,
    'modelA_macro_auc': macro_A,
    'delta_B_over_C': delta_B_C,
    'delta_A_over_B_peptide_class': delta_A_B_peptide,
    'modelC_neither': aucs_C[0], 'modelC_syn': aucs_C[1], 'modelC_pep': aucs_C[2], 'modelC_both': aucs_C[3],
    'modelB_neither': aucs_B[0], 'modelB_syn': aucs_B[1], 'modelB_pep': aucs_B[2], 'modelB_both': aucs_B[3],
    'modelA_neither': aucs_A[0], 'modelA_syn': aucs_A[1], 'modelA_pep': aucs_A[2], 'modelA_both': aucs_A[3],
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb27_final_summary.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/27_channel_selection_ablation.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 27 written ({len(nb.cells)} cells)')
