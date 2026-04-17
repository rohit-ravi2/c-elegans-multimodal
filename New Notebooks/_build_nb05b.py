"""Generate 05b notebook with nbformat (avoids JSON-escape problems)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 05b — Gene-Level Stability Selection on Developmental Rewiring Classifier

## Why this notebook exists

Notebook 05 produced a weak positive signal on developmental rewiring (AUC ~0.62, above null 0.55 but below the preregistered 0.65 bar). The classifier used PCA-50 features, which hides gene-level interpretation.

This notebook asks: given the weak AUC signal exists, which specific genes contribute to it, and is the top-gene ranking robust across bootstrap samples?

## Important caveats up front

1. **CeNGEN is L4 larval, not L1.** Claims of the form "L1 expression predicts adult wiring" are unsupported. The defensible claim: "L4 expression signatures discriminate class-pairs that already connect at L1 from those that only form connections by adulthood."
2. **At AUC 0.62, gene-level hits are inherently noisy.** Stability selection (bootstrap frequency) separates robust hits from noise.
3. **No pre-specified gene list.** Unbiased selection runs first; saved output on disk; candidate genes (flp-21, ceh-31, blos-2) are only looked up AFTER ranking is fixed.

## Preregistered criteria

1. Stability selection produces >=10 genes with freq_max >= 0.5.
2. On 20 shuffled-label permutations, the real count exceeds the permutation null 95th percentile.
3. Of the top-50 stability-ranked genes, at least 5 fall into a canonical functional category (TF / neuropeptide / ion channel / adhesion / synaptic).

## Halting rule

If criterion 1 fails: null, end. If 1 passes but 2 fails: weak positive. If all pass: report top genes as descriptive finding grounded in AUC~0.62 signal."""))

cells.append(nbf.v4.new_code_cell("""import sys, time
from pathlib import Path
_HERE = Path.cwd()
if (_HERE / 'lib').is_dir(): sys.path.insert(0, str(_HERE))
elif (_HERE.parent / 'lib').is_dir(): sys.path.insert(0, str(_HERE.parent))

from lib.paths import DERIVED

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

RNG = np.random.default_rng(42)
C_LR = 1.0

labels = pd.read_csv(DERIVED / 'nb05_pair_labels.csv')
print(f'Nb05 class-pair labels: {len(labels)} pairs')
print(labels['label'].value_counts().to_string())"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Build raw-gene feature matrix"))

cells.append(nbf.v4.new_code_cell("""mapping_df = pd.read_csv(DERIVED / 'expression_neuron_mapping.csv')
neuron_to_class = dict(zip(mapping_df['witvliet_name'], mapping_df['cengen_class']))

expr_data = np.load(DERIVED / 'expression_tpm.npz', allow_pickle=True)
neurons_e = expr_data['neurons']
tpm = expr_data['tpm']
genes_wbg = expr_data['genes_wbg']
genes_csv = pd.read_csv(DERIVED / 'expression_genes.csv')
gene_symbols = genes_csv['symbol'].values

class_to_expr = {}
for i, nm in enumerate(neurons_e):
    cls = neuron_to_class.get(str(nm))
    if isinstance(cls, str) and cls not in class_to_expr and not np.all(np.isnan(tpm[i])):
        class_to_expr[cls] = tpm[i]

classes_list = sorted(class_to_expr.keys())
X_cls = np.stack([class_to_expr[c] for c in classes_list])
X_cls_log = np.log1p(X_cls)

n_expressed = (X_cls > 0).sum(axis=0)
means = X_cls.mean(axis=0)
stds = X_cls.std(axis=0)
cv = np.where(means > 0, stds / (means + 1e-9), 0.0)
keep = (n_expressed >= 5) & (cv >= 1.0)
G_kept = int(keep.sum())
print(f'Genes passing filter: {G_kept}')

X_cls_filt = X_cls_log[:, keep]
wbg_filt = genes_wbg[keep]
sym_filt = gene_symbols[keep]

class_idx = {c: i for i, c in enumerate(classes_list)}

def pair_features_raw(pre, post):
    return np.concatenate([X_cls_filt[class_idx[pre]], X_cls_filt[class_idx[post]]])

X_pairs = np.stack([pair_features_raw(r['pre_class'], r['post_class']) for _, r in labels.iterrows()])
y_pairs = (labels['label'] == 'added').astype(int).values
print(f'Feature matrix: {X_pairs.shape}  (N pairs x 2*{G_kept})')
print(f'y: stable={int((y_pairs==0).sum())}, added={int((y_pairs==1).sum())}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Baseline: raw-gene L1-logistic AUC"))

cells.append(nbf.v4.new_code_cell("""def cv_auc(X, y, n_splits=5, random_state=42, C=C_LR):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    for tr, te in skf.split(X, y):
        model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=2000)
        )
        model.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
    return np.array(aucs)

t0 = time.time()
aucs_raw = cv_auc(X_pairs, y_pairs, C=C_LR)
print(f'Raw-gene Lasso AUC (5 folds): {[f\"{a:.3f}\" for a in aucs_raw]}')
print(f'  mean={aucs_raw.mean():.3f} std={aucs_raw.std():.3f}  (Nb05 PCA baseline: 0.616)')
print(f'  fit time: {time.time()-t0:.1f}s')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Bootstrap stability selection"))

cells.append(nbf.v4.new_code_cell("""N_BOOT = 100
SUBSAMPLE_FRAC = 0.7

P = X_pairs.shape[1]
freq = np.zeros(P, dtype=float)
coef_sum = np.zeros(P, dtype=float)

scaler = StandardScaler(with_mean=False).fit(X_pairs)
X_scaled = scaler.transform(X_pairs)

pos = np.where(y_pairs == 1)[0]
neg = np.where(y_pairs == 0)[0]
n_pos = int(SUBSAMPLE_FRAC * len(pos))
n_neg = int(SUBSAMPLE_FRAC * len(neg))

print(f'Running {N_BOOT} bootstrap Lasso fits...')
t0 = time.time()
for b in range(N_BOOT):
    idx = np.concatenate([RNG.choice(pos, n_pos, replace=False),
                          RNG.choice(neg, n_neg, replace=False)])
    try:
        model = LogisticRegression(penalty='l1', solver='liblinear', C=C_LR, max_iter=2000)
        model.fit(X_scaled[idx], y_pairs[idx])
        coef = model.coef_[0]
        freq += (coef != 0).astype(float)
        coef_sum += coef
    except Exception:
        pass
    if (b + 1) % 20 == 0:
        print(f'  {b+1}/{N_BOOT} ({time.time()-t0:.1f}s)')
freq /= N_BOOT
mean_coef = coef_sum / N_BOOT
print(f'Done in {time.time()-t0:.1f}s')"""))

cells.append(nbf.v4.new_code_cell("""freq_pre  = freq[:G_kept]
freq_post = freq[G_kept:]
coef_pre  = mean_coef[:G_kept]
coef_post = mean_coef[G_kept:]

per_gene = pd.DataFrame({
    'wbgene': wbg_filt,
    'symbol': sym_filt,
    'freq_pre': freq_pre,
    'freq_post': freq_post,
    'freq_max': np.maximum(freq_pre, freq_post),
    'coef_pre_mean': coef_pre,
    'coef_post_mean': coef_post,
})
per_gene = per_gene.sort_values('freq_max', ascending=False).reset_index(drop=True)
per_gene.to_csv(DERIVED / 'nb05b_gene_stability.csv', index=False)
print(f'Saved gene stability ranking: {DERIVED / \"nb05b_gene_stability.csv\"}')
print(f'\\nTop 30 genes by freq_max:')
print(per_gene.head(30).to_string())"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Permutation null"))

cells.append(nbf.v4.new_code_cell("""N_PERM = 20
N_BOOT_PERM = 50

n_real_top = int((per_gene['freq_max'] >= 0.5).sum())
print(f'Real data: {n_real_top} genes at freq_max >= 0.5')

def stability_run(X, y, n_boot, rng):
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    n_pos = int(SUBSAMPLE_FRAC * len(pos))
    n_neg = int(SUBSAMPLE_FRAC * len(neg))
    freq_local = np.zeros(X.shape[1])
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                              rng.choice(neg, n_neg, replace=False)])
        try:
            m = LogisticRegression(penalty='l1', solver='liblinear', C=C_LR, max_iter=2000)
            m.fit(X[idx], y[idx])
            freq_local += (m.coef_[0] != 0).astype(float)
        except Exception:
            pass
    return freq_local / n_boot

print(f'Running {N_PERM} permutation stability runs ({N_BOOT_PERM} bootstraps each)...')
t0 = time.time()
perm_top_counts = []
for p in range(N_PERM):
    y_perm = RNG.permutation(y_pairs)
    freq_p = stability_run(X_scaled, y_perm, N_BOOT_PERM, RNG)
    freq_max_p = np.maximum(freq_p[:G_kept], freq_p[G_kept:])
    perm_top_counts.append(int((freq_max_p >= 0.5).sum()))
    if (p + 1) % 5 == 0:
        print(f'  {p+1}/{N_PERM} done ({time.time()-t0:.1f}s), recent counts: {perm_top_counts[-5:]}')
perm_top_counts = np.array(perm_top_counts)
print(f'\\nDone in {time.time()-t0:.1f}s')
print(f'Permutation null count distribution (# genes at freq_max>=0.5):')
print(f'  mean: {perm_top_counts.mean():.1f}')
print(f'  95pct: {np.percentile(perm_top_counts, 95):.1f}')
print(f'  max:  {perm_top_counts.max()}')
print(f'\\nReal: {n_real_top} vs null 95pct: {np.percentile(perm_top_counts, 95):.1f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 5 — Preregistered criteria + top-50 inspection"))

cells.append(nbf.v4.new_code_cell("""import re

CATEGORIES = {
    'TF':            re.compile(r'^(ceh|lim|egl-|vab-|unc-3$|unc-4$|unc-30|unc-42|unc-86|unc-55|ttx-3|mec-3|hbl-1|pag-3|zag-1|alr-1|cog-1|ast-1|ztf-|mbr-1|fkh-|nhr-|hlh-|lin-11|lin-32|tlp-1)', re.I),
    'neuropeptide':  re.compile(r'^(flp-|nlp-|pdf-|npr-|ins-|nmur)', re.I),
    'ion_channel':   re.compile(r'^(unc-7$|unc-9$|twk-|slo-|exp-2|egl-23|egl-36|egl-2$|cca-|shk-|shl-|kvs-|kcnl-|acd-|asic-|unc-103|unc-36|eat-16|itr-|nca-)', re.I),
    'adhesion_guide':re.compile(r'^(sax-|unc-5$|unc-6$|unc-40$|ptp-|nlg-|nrx-|cdh-|cam-|lad-|syg-|zig-|rig-|ina-1|unc-44|unc-73|slt-|sma-|plx-|mig-|igcm-|dma-1|madd-)', re.I),
    'synaptic':      re.compile(r'^(unc-13$|unc-17$|unc-18$|unc-64$|snb-|snt-|syn-|syx-|sbt-|unc-31$|ric-4|exp-3|unc-46)', re.I),
}

def categorize(sym):
    if not isinstance(sym, str): return None
    for cat, pat in CATEGORIES.items():
        if pat.match(sym):
            return cat
    return None

top50 = per_gene.head(50).copy()
top50['category'] = top50['symbol'].apply(categorize)
cat_counts = top50['category'].value_counts(dropna=False)
print('Top-50 functional category breakdown:')
print(cat_counts.to_string())
n_categorized = int(top50['category'].notna().sum())

crit1 = n_real_top >= 10
crit2 = n_real_top > np.percentile(perm_top_counts, 95)
crit3 = n_categorized >= 5

print('\\nPREREGISTERED CRITERIA')
print('=' * 70)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 N_stable_genes (freq>=0.5) >= 10        n={n_real_top}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 Real count > null 95pct                  real={n_real_top}, null={np.percentile(perm_top_counts, 95):.1f}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 Top-50 functional-category hits >= 5     n_cat={n_categorized}')
print('=' * 70)

top50.to_csv(DERIVED / 'nb05b_top50_annotated.csv', index=False)
print(f'\\nTop 50 with categories:')
print(top50[['symbol','wbgene','freq_max','coef_pre_mean','coef_post_mean','category']].to_string())"""))

cells.append(nbf.v4.new_markdown_cell("## Step 6 — Post-analysis lookup: candidate genes (flp-21, ceh-31, blos-2)"))

cells.append(nbf.v4.new_code_cell("""CANDIDATES = ['flp-21', 'ceh-31', 'blos-2']
print('Candidate gene lookup (post-analysis, descriptive only):')
for sym in CANDIDATES:
    row = per_gene[per_gene['symbol'] == sym]
    if len(row) == 0:
        print(f'  {sym:10s}  NOT in variance-filtered gene set')
    else:
        rank = int(per_gene.index[per_gene['symbol'] == sym].tolist()[0]) + 1
        r = row.iloc[0]
        print(f'  {sym:10s}  rank={rank}/{len(per_gene)}  freq_max={r[\"freq_max\"]:.3f}  freq_pre={r[\"freq_pre\"]:.3f}  freq_post={r[\"freq_post\"]:.3f}  coef_pre={r[\"coef_pre_mean\"]:+.4f}  coef_post={r[\"coef_post_mean\"]:+.4f}')

print('\\nTop-3 gene in each functional category (unbiased alternative):')
pgc = per_gene.copy(); pgc['category'] = pgc['symbol'].apply(categorize)
for cat in CATEGORIES.keys():
    sub = pgc[pgc['category'] == cat].head(3)
    if len(sub):
        print(f'\\n  {cat}:')
        for _, rr in sub.iterrows():
            rank = int(per_gene.index[per_gene['symbol'] == rr['symbol']].tolist()[0]) + 1
            print(f'    {rr[\"symbol\"]:12s} rank={rank}  freq_max={rr[\"freq_max\"]:.3f}  coef_pre={rr[\"coef_pre_mean\"]:+.4f}  coef_post={rr[\"coef_post_mean\"]:+.4f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 7 — Final verdict"))

cells.append(nbf.v4.new_code_cell("""print('=' * 70)
print('NOTEBOOK 05b — GENE-LEVEL STABILITY VERDICT')
print('=' * 70)
print(f'Baseline AUC (raw-gene Lasso): {aucs_raw.mean():.3f}')
print(f'Genes at freq_max >= 0.5:       {n_real_top}')
print(f'Permutation null 95pct:         {np.percentile(perm_top_counts, 95):.1f}')
print(f'Top-50 with known categories:   {n_categorized}')

if crit1 and crit2 and crit3:
    verdict = 'POSITIVE — robust gene-level signal with functional enrichment'
elif crit1 and crit2:
    verdict = 'POSITIVE (robust stability) but weak functional enrichment'
elif crit1:
    verdict = 'WEAK POSITIVE — stability exists but not robust above permutation null'
else:
    verdict = 'NULL — no robust gene-level identification possible'
print(f'\\nVerdict: {verdict}')

summary = pd.DataFrame([{
    'raw_gene_auc': float(aucs_raw.mean()),
    'n_stable_genes_freq_ge_0_5': n_real_top,
    'null_95pct_count': float(np.percentile(perm_top_counts, 95)),
    'n_top50_categorized': n_categorized,
    'crit1_pass': bool(crit1), 'crit2_pass': bool(crit2), 'crit3_pass': bool(crit3),
    'verdict': verdict,
    'flp21_in_set': bool((per_gene['symbol'] == 'flp-21').any()),
    'ceh31_in_set': bool((per_gene['symbol'] == 'ceh-31').any()),
    'blos2_in_set': bool((per_gene['symbol'] == 'blos-2').any()),
}])
summary.to_csv(DERIVED / 'nb05b_final_summary.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells

with open('/home/rohit/Desktop/C-Elegans/New Notebooks/05b_rewiring_gene_stability.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 05b written ({len(nb.cells)} cells)')
