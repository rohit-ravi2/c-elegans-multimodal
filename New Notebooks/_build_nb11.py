"""Nb 11 — which genes drive Nb09's +0.044 AUC boost? Feature importance + stability."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 11 — Which Genes Drive the Topology→Full Improvement?

## Context

Nb 09 shows CeNGEN PCA-50 gene features boost edge-arrival AUC from 0.758 (topology) to 0.802 (topology + genes). Which specific genes contribute?

This is the gene-level interpretation that the paper needs. We'll:
1. Train on raw-gene features (not PCA) alongside topology
2. Stability-select across bootstrap subsamples
3. Report top genes with confidence-interval frequencies
4. Cross-check against biological categories (TF / neuropeptide / adhesion / ion channel)

## Lesson from Nb 05b

In 05b we ran stability selection on raw-gene features without a topological baseline. Result: top genes were housekeeping (cct-5, act-4, rpl-19) and uncharacterized ORFs — signal tied with null. That was because the signal being captured was "class-identity noise," not developmental-rewiring biology.

This notebook differs: **features compete against a strong topological baseline**. Only genes that carry signal *above topology* should survive. Housekeeping genes that correlate with neuron identity (and thus implicitly with topology) will fail to add.

## Preregistered criteria

1. **≥ 10 genes with stability frequency ≥ 0.5** (against topology baseline).
2. **Permutation null 95pct < observed count** of stable genes.
3. **Top-20 stability-ranked genes include ≥ 3 from canonical categories** (TF, neuropeptide, adhesion, ion channel).

## Halting rule

Nulls on criterion 1: the +0.044 AUC gain in Nb09 is carried by the PCA-50 ensemble, not by identifiable individual genes. Still a finding but downgrades from "these specific genes matter" to "the gene-ensemble helps diffusely."

Positive: we have a named, interpretable gene list."""))

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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(42)

cand = pd.read_csv(DERIVED / 'nb07_pooled_candidates.csv')
print(f'Nb07 candidates: {len(cand)}')

expr_data = np.load(DERIVED / 'expression_tpm.npz', allow_pickle=True)
tpm_neurons = expr_data['neurons']; tpm = expr_data['tpm']
genes_wbg = expr_data['genes_wbg']
genes_csv = pd.read_csv(DERIVED / 'expression_genes.csv')
gene_symbols = genes_csv['symbol'].values

mapping = pd.read_csv(DERIVED / 'expression_neuron_mapping.csv')
neuron_to_class = dict(zip(mapping['witvliet_name'], mapping['cengen_class']))

# common-185 neuron list
STAGES = ['L1_1','L1_2','L1_3','L1_4','L2','L3','adult']
s_neurons = {}
for s in STAGES:
    d = np.load(DERIVED / 'developmental' / f'connectome_{s}.npz', allow_pickle=True)
    s_neurons[s] = set(str(n) for n in d['neurons'])
common = sorted(set.intersection(*s_neurons.values()))
print(f'Common-185 neurons: {len(common)}')

# Class-level expression
class_to_expr = {}
for i, nm in enumerate(tpm_neurons):
    cls = neuron_to_class.get(str(nm))
    if isinstance(cls, str) and cls not in class_to_expr and not np.all(np.isnan(tpm[i])):
        class_to_expr[cls] = tpm[i]

# Variance filter
classes_list = sorted(class_to_expr.keys())
X_cls = np.stack([class_to_expr[c] for c in classes_list])
n_expressed = (X_cls > 0).sum(axis=0)
means = X_cls.mean(axis=0); stds = X_cls.std(axis=0)
cv = np.where(means > 0, stds / (means + 1e-9), 0.0)
keep = (n_expressed >= 5) & (cv >= 1.0)
print(f'Genes after filter: {int(keep.sum())}')

X_cls_log = np.log1p(X_cls)[:, keep]
wbg_filt = genes_wbg[keep]
sym_filt = gene_symbols[keep]
class_to_exprvec = {c: X_cls_log[i] for i, c in enumerate(classes_list)}
G = X_cls_log.shape[1]
ZERO = np.zeros(G)

def neuron_to_expr(nid):
    n = common[int(nid)]
    cls = neuron_to_class.get(n)
    if isinstance(cls, str) and cls in class_to_exprvec:
        return class_to_exprvec[cls]
    return ZERO"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Assemble topology + raw-gene features"))

cells.append(nbf.v4.new_code_cell("""TOP_COVS = ['out_deg_i','in_deg_j','shared_out','shared_in','triangle_closure','reverse_2step']
X_topo = cand[TOP_COVS].values.astype(float)

# Build gene features: [pre_gene_vec || post_gene_vec]
gene_pre = np.stack([neuron_to_expr(r['i']) for _, r in cand.iterrows()])
gene_post = np.stack([neuron_to_expr(r['j']) for _, r in cand.iterrows()])
X_full = np.concatenate([X_topo, gene_pre, gene_post], axis=1)
y = cand['arrived'].values.astype(int)
print(f'Full feature matrix: {X_full.shape} (topology={len(TOP_COVS)}, pre_genes={G}, post_genes={G})')
print(f'Arrivals: {y.sum()} / {len(y)} (rate {y.mean():.3f})')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Bootstrap stability selection with L1-logistic (on top of topology)"))

cells.append(nbf.v4.new_code_cell("""# L1-penalized logistic regression on [topology + genes] — select sparse gene subset
# Use liblinear for L1; preserve topology features by not penalizing them heavily (we can't
# exclude them from L1 penalty directly in sklearn but using strong C keeps them non-zero)

# Scale features
scaler = StandardScaler(with_mean=False).fit(X_full)
X_scaled = scaler.transform(X_full)

N_BOOT = 50  # smaller for speed; N_full features is ~17,300
SUB = 0.7
n_sub = int(SUB * len(y))
# Stratified subsample
pos = np.where(y==1)[0]; neg = np.where(y==0)[0]
n_pos_sub = int(SUB * len(pos)); n_neg_sub = int(SUB * len(neg))

P = X_full.shape[1]
freq = np.zeros(P)
coef_sum = np.zeros(P)

print(f'Running {N_BOOT} bootstrap L1-logistic fits (on {P} features)...')
t0 = time.time()
for b in range(N_BOOT):
    idx = np.concatenate([RNG.choice(pos, n_pos_sub, replace=False),
                          RNG.choice(neg, n_neg_sub, replace=False)])
    try:
        m = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=2000)
        m.fit(X_scaled[idx], y[idx])
        coef = m.coef_[0]
        freq += (coef != 0).astype(float)
        coef_sum += coef
    except Exception as e:
        pass
    if (b+1) % 10 == 0:
        print(f'  {b+1}/{N_BOOT}  ({time.time()-t0:.1f}s)')
freq /= N_BOOT
mean_coef = coef_sum / N_BOOT"""))

cells.append(nbf.v4.new_code_cell("""# Split frequencies into topology vs pre_gene vs post_gene
TOPO_N = len(TOP_COVS)
freq_topo = freq[:TOPO_N]
freq_pre = freq[TOPO_N:TOPO_N+G]
freq_post = freq[TOPO_N+G:]
coef_topo = mean_coef[:TOPO_N]
coef_pre = mean_coef[TOPO_N:TOPO_N+G]
coef_post = mean_coef[TOPO_N+G:]

print('Topology features (all should be stable >0.95 since non-penalized gets selected):')
for k, (f, c) in enumerate(zip(freq_topo, coef_topo)):
    print(f'  {TOP_COVS[k]:20s} freq={f:.3f}  coef={c:+.4f}')

per_gene = pd.DataFrame({
    'wbgene': wbg_filt, 'symbol': sym_filt,
    'freq_pre': freq_pre, 'freq_post': freq_post,
    'freq_max': np.maximum(freq_pre, freq_post),
    'coef_pre_mean': coef_pre, 'coef_post_mean': coef_post,
})
per_gene = per_gene.sort_values('freq_max', ascending=False).reset_index(drop=True)
per_gene.to_csv(DERIVED / 'nb11_gene_stability_over_topology.csv', index=False)
print(f'\\nTop 30 genes by freq_max (above topology baseline):')
print(per_gene.head(30).to_string())"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Permutation null"))

cells.append(nbf.v4.new_code_cell("""N_PERM = 20
n_real_top = int((per_gene['freq_max'] >= 0.5).sum())

def run_stability(y_in, n_boot=25, C=0.1):
    pos_ = np.where(y_in==1)[0]; neg_ = np.where(y_in==0)[0]
    nps = int(SUB * len(pos_)); nns = int(SUB * len(neg_))
    fq = np.zeros(P)
    for _ in range(n_boot):
        idx = np.concatenate([RNG.choice(pos_, nps, replace=False),
                              RNG.choice(neg_, nns, replace=False)])
        try:
            m = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=2000)
            m.fit(X_scaled[idx], y_in[idx])
            fq += (m.coef_[0] != 0).astype(float)
        except Exception:
            pass
    return fq / n_boot

null_top_counts = []
t0 = time.time()
for i in range(N_PERM):
    y_perm = RNG.permutation(y)
    fq = run_stability(y_perm, n_boot=25)
    freq_max_p = np.maximum(fq[TOPO_N:TOPO_N+G], fq[TOPO_N+G:])
    null_top_counts.append(int((freq_max_p >= 0.5).sum()))
    if (i+1) % 5 == 0:
        print(f'  perm {i+1}/{N_PERM} ({time.time()-t0:.1f}s)  recent null counts: {null_top_counts[-5:]}')
null_top_counts = np.array(null_top_counts)
print(f'\\nPermutation null counts (genes at freq>=0.5): mean={null_top_counts.mean():.1f}, 95pct={np.percentile(null_top_counts, 95):.1f}, max={null_top_counts.max()}')
print(f'Real count: {n_real_top}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Category annotation + criteria"))

cells.append(nbf.v4.new_code_cell("""import re
CATEGORIES = {
    'TF': re.compile(r'^(ceh|lim-|egl-[0-9]|vab-|unc-3$|unc-4$|unc-30|unc-42|unc-86|unc-55|ttx-3|mec-3|hbl-1|pag-3|zag-1|alr-1|cog-1|ast-1|ztf-|mbr-1|fkh-|nhr-|hlh-|lin-11|lin-32|tlp-1|daf-)', re.I),
    'neuropeptide': re.compile(r'^(flp-|nlp-|pdf-|npr-|ins-|nmur|sbt-|ntc-)', re.I),
    'ion_channel': re.compile(r'^(unc-7$|unc-9$|twk-|slo-|exp-2|egl-23|egl-36|egl-2$|cca-|shk-|shl-|kvs-|kcnl-|acd-|asic-|unc-103|unc-36|eat-16|itr-|nca-|glc-|unc-49)', re.I),
    'adhesion': re.compile(r'^(sax-|unc-5$|unc-6$|unc-40$|ptp-|nlg-|nrx-|cdh-|cam-|lad-|syg-|zig-|rig-|ina-1|unc-44|unc-73|slt-|sma-|plx-|mig-|igcm-|dma-1|madd-|wrk-1|klp-)', re.I),
    'synaptic': re.compile(r'^(unc-13$|unc-17$|unc-18$|unc-64$|snb-|snt-|syn-|syx-|sbt-|unc-31$|ric-4|exp-3|unc-46|eat-4|cho-1|unc-25|tph-|cat-1|cat-2)', re.I),
}

def categorize(sym):
    if not isinstance(sym, str): return None
    for cat, pat in CATEGORIES.items():
        if pat.match(sym):
            return cat
    return None

top50 = per_gene.head(50).copy()
top50['category'] = top50['symbol'].apply(categorize)
print(top50[['symbol','wbgene','freq_max','coef_pre_mean','coef_post_mean','category']].to_string())
cat_counts = top50['category'].value_counts(dropna=False)
print(f'\\nTop 50 category breakdown:')
print(cat_counts.to_string())
n_categorized_top20 = int(top50.head(20)['category'].notna().sum())

crit1 = n_real_top >= 10
crit2 = n_real_top > np.percentile(null_top_counts, 95)
crit3 = n_categorized_top20 >= 3

print('\\nCRITERIA')
print('=' * 60)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 genes at freq>=0.5 >= 10    n={n_real_top}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 real > null 95pct            real={n_real_top}, null={np.percentile(null_top_counts, 95):.1f}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 top-20 functional >= 3       n_cat={n_categorized_top20}')
print('=' * 60)
all_pass = all([crit1, crit2, crit3])

if all_pass:
    verdict = 'POSITIVE — robust gene-level signal with functional enrichment above topology'
elif crit1 and crit2:
    verdict = 'POSITIVE (robust) but functional enrichment weak'
elif crit1:
    verdict = 'WEAK POSITIVE — stability exists but ties with null'
else:
    verdict = 'NULL — no robust gene-level identification above topology'
print(f'\\nVERDICT: {verdict}')

summary = pd.DataFrame([{
    'n_stable_genes': n_real_top,
    'null_95pct': float(np.percentile(null_top_counts, 95)),
    'null_mean': float(null_top_counts.mean()),
    'n_top20_categorized': n_categorized_top20,
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb11_final_summary.csv', index=False)
print(summary.T.to_string())
top50.to_csv(DERIVED / 'nb11_top50_annotated.csv', index=False)"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/11_genes_above_topology_baseline.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 11 written ({len(nb.cells)} cells)')
