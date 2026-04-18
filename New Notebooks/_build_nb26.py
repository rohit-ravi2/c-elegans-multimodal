"""Nb 26 — Distilled Developmental Rulebook from Nb24's black-box model."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 26 — Distilled Developmental Rulebook

## Goal

Nb 24 achieves AUC 0.823 on contact-stratified edge arrival with a black-box logistic regression over ~110 features (contact, topology, gene PCA). This notebook asks: **can we distill that prediction into a small, interpretable rulebook that a biologist could read and hypothesize from?**

## Concrete approach

Two parallel distillations, compared to the Nb24 black-box baseline:

1. **Decision tree** (max 10 leaves): gives `if-then` rules, one per leaf path.
2. **L1-logistic** over a curated interpretable feature set (transformed topology, contact bins, top gene PCs): gives sparse weighted sum.

We report whichever achieves higher AUC at the "≤10 terms" constraint. Both get bootstrap CIs.

## Preregistered criteria

1. **Distilled AUC ≥ 0.78** (loses ≤ 0.045 from black-box 0.823).
2. **≤ 10 rules/terms total**.
3. **Bootstrap 95% CI on distilled AUC excludes 0.70** (clearly above a reasonable null).
4. **Rule stability**: refit on 5 cross-validation splits; the top-3 rules (by contribution) appear in ≥ 3 of 5 fits.
5. **Each rule has a biological interpretation** (manually annotated post-hoc).

## Halting rule

- If distilled AUC < 0.74: we cannot get interpretability AND accuracy simultaneously. Honest null on rulebook attempt; paper uses black-box numbers only.
- If distilled AUC ∈ [0.74, 0.78]: marginal; report cautiously as "low-accuracy interpretable" version.
- If distilled AUC ≥ 0.78: the rulebook is the deliverable — 10 rules, biologically annotated."""))

cells.append(nbf.v4.new_code_cell("""import sys, time
from pathlib import Path
import warnings; warnings.simplefilter('ignore')
_HERE = Path.cwd()
if (_HERE / 'lib').is_dir(): sys.path.insert(0, str(_HERE))
elif (_HERE.parent / 'lib').is_dir(): sys.path.insert(0, str(_HERE.parent))

from lib.paths import DATA, DERIVED
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(42)

# Rebuild Nb24's feature set (contact-stratified)
cand = pd.read_csv(DERIVED / 'nb07_pooled_candidates.csv')

STAGES = ['L1_1','L1_2','L1_3','L1_4','L2','L3','adult']
s_neurons = {}
for s in STAGES:
    d = np.load(DERIVED / 'developmental' / f'connectome_{s}.npz', allow_pickle=True)
    s_neurons[s] = set(str(n) for n in d['neurons'])
common = sorted(set.intersection(*s_neurons.values()))

# Contact
contact_xlsx = DATA / 'connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx'
contact_df = pd.read_excel(contact_xlsx, sheet_name='adult nerve ring neighbors', header=0, index_col=0)
contact_df = contact_df.loc[:, ~contact_df.columns.str.startswith('Unnamed')]
contact_df = contact_df[contact_df.index.map(lambda x: isinstance(x, str))]
contact_df = contact_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
name_to_idx = {n: i for i, n in enumerate(common)}
contact_mat = np.zeros((len(common), len(common)), dtype=np.float32)
for rn in contact_df.index:
    if rn not in name_to_idx: continue
    ri = name_to_idx[rn]
    for cn in contact_df.columns:
        if cn not in name_to_idx: continue
        ci = name_to_idx[cn]
        v = contact_df.loc[rn, cn]
        if pd.notna(v): contact_mat[ri, ci] = float(v)
contact_mat = (contact_mat + contact_mat.T) / 2
contact_area = np.array([contact_mat[int(r['i']), int(r['j'])] for _, r in cand.iterrows()])
log_contact = np.log1p(contact_area)

# Restrict to contact-stratified
contact_mask = contact_area > 0
cand_s = cand.loc[contact_mask].reset_index(drop=True)
log_contact_s = log_contact[contact_mask]
y_s = cand_s['arrived'].values.astype(int)
print(f'Contact-stratified candidates: {len(cand_s)} (arrivals={y_s.sum()})')

# Gene PCA (from Nb24)
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

gene_pre = np.stack([get_pc(common[int(r['i'])]) for _, r in cand_s.iterrows()])
gene_post = np.stack([get_pc(common[int(r['j'])]) for _, r in cand_s.iterrows()])

TOP_COVS = ['out_deg_i','in_deg_j','shared_out','shared_in','triangle_closure','reverse_2step']
X_topo = cand_s[TOP_COVS].values.astype(float)

# For the interpretable model, use only TOP-5 gene PCs (not all 50) — the rest are noisier
GENE_K = 5
X_full = np.concatenate([
    log_contact_s.reshape(-1, 1),
    X_topo,
    gene_pre[:, :GENE_K],
    gene_post[:, :GENE_K],
], axis=1)
feat_names = ['log_contact'] + TOP_COVS + [f'pre_PC{i+1}' for i in range(GENE_K)] + [f'post_PC{i+1}' for i in range(GENE_K)]
print(f'Distillation feature set: {len(feat_names)} features')
print(f'Features: {feat_names}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Black-box baseline (reproduce Nb24)"))

cells.append(nbf.v4.new_code_cell("""# Full Nb24 feature set for baseline
X_full_blackbox = np.concatenate([
    log_contact_s.reshape(-1, 1),
    X_topo,
    gene_pre,
    gene_post,
], axis=1)

def oof_preds(X, y, model_fn, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros_like(y, dtype=float)
    for tr, te in skf.split(X, y):
        m = model_fn()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof

def lr_default():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))

oof_baseline = oof_preds(X_full_blackbox, y_s, lr_default)
auc_baseline = roc_auc_score(y_s, oof_baseline)
print(f'Black-box baseline AUC (Nb24 reproduction): {auc_baseline:.4f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Decision tree distillation (≤ 10 leaves)"))

cells.append(nbf.v4.new_code_cell("""# Try several depths; pick the one that gives best AUC at <= 10 leaves
results = []
for max_leaves in [4, 6, 8, 10]:
    def tree_fn(ml=max_leaves):
        return DecisionTreeClassifier(max_leaf_nodes=ml, min_samples_leaf=200, random_state=42)
    oof_tree = oof_preds(X_full, y_s, tree_fn)
    auc_tree = roc_auc_score(y_s, oof_tree)
    results.append({'max_leaves': max_leaves, 'auc': auc_tree, 'oof': oof_tree})
    print(f'  Tree with max_leaves={max_leaves}: AUC={auc_tree:.4f}')

# Pick tree with the best AUC
best_tree = max(results, key=lambda r: r['auc'])
print(f'\\nBest tree: max_leaves={best_tree[\"max_leaves\"]}, AUC={best_tree[\"auc\"]:.4f}')

# Fit on FULL dataset to extract the rules
tree_full = DecisionTreeClassifier(max_leaf_nodes=best_tree['max_leaves'], min_samples_leaf=200, random_state=42)
tree_full.fit(X_full, y_s)
print(f'Tree n_leaves: {tree_full.get_n_leaves()}')
print(f'\\n--- Tree rules ---')
print(export_text(tree_full, feature_names=feat_names, max_depth=20))

oof_tree_best = best_tree['oof']"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — L1 logistic distillation (sparse interpretable model)"))

cells.append(nbf.v4.new_code_cell("""# Use L1-logistic with varying C to target <= 10 non-zero features
def l1_fn(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=5000))

l1_results = []
for C in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
    oof_l1 = oof_preds(X_full, y_s, lambda C=C: l1_fn(C))
    auc_l1 = roc_auc_score(y_s, oof_l1)
    # Also count non-zero features on a final fit
    m_final = l1_fn(C)
    m_final.fit(X_full, y_s)
    coefs = m_final.named_steps['logisticregression'].coef_[0]
    n_nonzero = int((np.abs(coefs) > 1e-8).sum())
    l1_results.append({'C': C, 'auc': auc_l1, 'n_nonzero': n_nonzero, 'oof': oof_l1, 'coefs': coefs})
    print(f'  C={C}: AUC={auc_l1:.4f}, non-zero features={n_nonzero}')

# Pick the L1 that has AUC close to best AND <= 10 non-zero features
l1_eligible = [r for r in l1_results if r['n_nonzero'] <= 10]
if l1_eligible:
    best_l1 = max(l1_eligible, key=lambda r: r['auc'])
    print(f'\\nBest L1 with <=10 features: C={best_l1[\"C\"]}, AUC={best_l1[\"auc\"]:.4f}, n_nonzero={best_l1[\"n_nonzero\"]}')
    # Print the non-zero features with coefficients
    scaler_mean = np.mean(X_full, axis=0)
    scaler_std = np.std(X_full, axis=0) + 1e-12
    print(f'\\n--- L1 non-zero features (standardized coefficients) ---')
    idx_sorted = np.argsort(-np.abs(best_l1['coefs']))
    for i in idx_sorted:
        if abs(best_l1['coefs'][i]) > 1e-8:
            print(f'  {feat_names[i]:24s}  {best_l1[\"coefs\"][i]:+.4f}')
    oof_l1_best = best_l1['oof']
else:
    print('No L1 model hit <=10 features cleanly; using sparsest one')
    best_l1 = min(l1_results, key=lambda r: r['n_nonzero'])
    oof_l1_best = best_l1['oof']"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Bootstrap CIs on both distilled models"))

cells.append(nbf.v4.new_code_cell("""n_boot = 1000
N = len(y_s)

def bootstrap_auc(oof, y, n_boot=n_boot):
    bs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = RNG.choice(N, N, replace=True)
        bs[b] = roc_auc_score(y[idx], oof[idx])
    return bs

bs_blackbox = bootstrap_auc(oof_baseline, y_s)
bs_tree = bootstrap_auc(oof_tree_best, y_s)
bs_l1 = bootstrap_auc(oof_l1_best, y_s)

def ci(a): return f'{np.mean(a):.4f} [{np.percentile(a, 2.5):.4f}, {np.percentile(a, 97.5):.4f}]'

print('Model comparison with 95% bootstrap CIs:')
print(f'  Black-box (Nb24 full):        {ci(bs_blackbox)}')
print(f'  Tree (max_leaves={best_tree[\"max_leaves\"]}): {ci(bs_tree)}')
print(f'  L1-logistic (C={best_l1[\"C\"]}, {best_l1[\"n_nonzero\"]} features): {ci(bs_l1)}')

# Pick best distilled
if np.mean(bs_tree) >= np.mean(bs_l1):
    best_distilled_name = 'tree'
    best_distilled_auc = np.mean(bs_tree)
    best_distilled_ci = bs_tree
    best_n_rules = int(tree_full.get_n_leaves())
else:
    best_distilled_name = 'l1_logistic'
    best_distilled_auc = np.mean(bs_l1)
    best_distilled_ci = bs_l1
    best_n_rules = best_l1['n_nonzero']

print(f'\\nBest distilled model: {best_distilled_name}')
print(f'AUC: {best_distilled_auc:.4f} [{np.percentile(best_distilled_ci, 2.5):.4f}, {np.percentile(best_distilled_ci, 97.5):.4f}]')
print(f'Number of rules/terms: {best_n_rules}')
print(f'Loss from black-box: {np.mean(bs_blackbox) - best_distilled_auc:.4f} AUC points')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 5 — Rule stability via 5-fold CV"))

cells.append(nbf.v4.new_code_cell("""# For each CV fold, refit the tree, extract rules, check which features appear at which split
skf_stab = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_rules = []  # list of sets of (feature_name, threshold_sign) pairs per fold
for fold_i, (tr, te) in enumerate(skf_stab.split(X_full, y_s)):
    t = DecisionTreeClassifier(max_leaf_nodes=best_tree['max_leaves'], min_samples_leaf=200, random_state=42)
    t.fit(X_full[tr], y_s[tr])
    # Extract features used in splits
    tree_struct = t.tree_
    features_used = set()
    for node_id in range(tree_struct.node_count):
        if tree_struct.feature[node_id] >= 0:
            fi = tree_struct.feature[node_id]
            features_used.add(feat_names[fi])
    fold_rules.append(features_used)
    print(f'Fold {fold_i+1}: features={sorted(features_used)}')

# Which features appear in >=3 of 5 folds?
from collections import Counter
feature_appearance = Counter()
for fold_set in fold_rules:
    for f in fold_set:
        feature_appearance[f] += 1
stable_features = sorted([f for f, c in feature_appearance.items() if c >= 3])
print(f'\\nFeatures stable across >=3/5 folds: {stable_features}')
print(f'Stability count: {dict(feature_appearance)}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 6 — Extract the final rulebook"))

cells.append(nbf.v4.new_code_cell("""# Walk the full-data tree and extract each leaf as a rule
def extract_rules(tree_model, feat_names):
    tree = tree_model.tree_
    rules = []
    def recurse(node_id, conditions):
        if tree.feature[node_id] >= 0:  # internal
            f_name = feat_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]
            # Left child: feature <= threshold
            recurse(tree.children_left[node_id], conditions + [(f_name, '<=', threshold)])
            recurse(tree.children_right[node_id], conditions + [(f_name, '>',  threshold)])
        else:
            # Leaf
            counts = tree.value[node_id][0]
            n_total = int(counts.sum())
            n_positive = int(counts[1])
            prob = n_positive / n_total if n_total else 0
            rules.append({'conditions': conditions, 'n_samples': n_total, 'n_positive': n_positive, 'probability': prob})
    recurse(0, [])
    return rules

rules = extract_rules(tree_full, feat_names)
print(f'{len(rules)} leaf rules')
print()
# Sort by arrival probability (most interesting ones first)
rules_sorted = sorted(rules, key=lambda r: -r['probability'])

print('=' * 90)
print(f'{\"Rule\":<4s} {\"Arrival Prob\":>12s} {\"N samples\":>10s} {\"N positives\":>12s}  Conditions')
print('=' * 90)
for i, r in enumerate(rules_sorted):
    conds = ' AND '.join([f'{f} {op} {thr:.3f}' for f, op, thr in r['conditions']])
    print(f'R{i+1:<3d} {r[\"probability\"]:>12.3f} {r[\"n_samples\"]:>10d} {r[\"n_positive\"]:>12d}  {conds}')

rulebook_df = pd.DataFrame([{
    'rule_id': f'R{i+1}',
    'arrival_probability': r['probability'],
    'n_samples': r['n_samples'],
    'n_arrivals': r['n_positive'],
    'conditions': ' AND '.join([f'{f} {op} {thr:.3f}' for f, op, thr in r['conditions']]),
} for i, r in enumerate(rules_sorted)])
rulebook_df.to_csv(DERIVED / 'nb26_rulebook.csv', index=False)"""))

cells.append(nbf.v4.new_markdown_cell("## Step 7 — Preregistered criteria"))

cells.append(nbf.v4.new_code_cell("""crit1 = best_distilled_auc >= 0.78
crit2 = best_n_rules <= 10
crit3 = np.percentile(best_distilled_ci, 2.5) > 0.70
# Stability: at least 3 features appear in >=3/5 folds AND these are among the top tree features
top_tree_features = set()
for r in rules_sorted[:5]:  # top 5 rules by probability
    for f, _, _ in r['conditions']:
        top_tree_features.add(f)
stable_in_top = len(top_tree_features & set(stable_features))
crit4 = stable_in_top >= 3

print('PREREGISTERED CRITERIA')
print('=' * 70)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 Distilled AUC >= 0.78            {best_distilled_auc:.4f}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 <= 10 rules/terms                   {best_n_rules}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 AUC CI lower bound > 0.70          {np.percentile(best_distilled_ci, 2.5):.4f}')
print(f'  [{\"PASS\" if crit4 else \"FAIL\"}] 4 >=3 top-rule features stable (3/5 folds)  {stable_in_top}')
print('=' * 70)
n_pass = sum([crit1, crit2, crit3, crit4])
print(f'{n_pass}/4 criteria pass')

if n_pass == 4:
    verdict = 'POSITIVE — rulebook successfully distills the black-box model'
elif n_pass == 3:
    verdict = 'WEAK POSITIVE — rulebook works but with caveats'
elif best_distilled_auc >= 0.74:
    verdict = 'MARGINAL — distilled model is interpretable but loses meaningful accuracy'
else:
    verdict = 'NULL — cannot distill the black-box model into ≤10 rules without losing accuracy'
print(f'\\nVERDICT: {verdict}')

summary = pd.DataFrame([{
    'black_box_auc':      float(np.mean(bs_blackbox)),
    'distilled_model':    best_distilled_name,
    'distilled_auc':      float(best_distilled_auc),
    'distilled_ci_low':   float(np.percentile(best_distilled_ci, 2.5)),
    'distilled_ci_high':  float(np.percentile(best_distilled_ci, 97.5)),
    'n_rules':            int(best_n_rules),
    'accuracy_loss':      float(np.mean(bs_blackbox) - best_distilled_auc),
    'stable_features':    ','.join(stable_features),
    'verdict':            verdict,
}])
summary.to_csv(DERIVED / 'nb26_final_summary.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/26_distilled_rulebook.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 26 written ({len(nb.cells)} cells)')
