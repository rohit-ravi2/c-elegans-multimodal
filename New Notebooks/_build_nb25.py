"""Nb 25 — Peptide Compensation Hypothesis: do peptide edges fill in where synaptic edges don't go?"""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 25 — Peptide Compensation Hypothesis

## The question

Nb 12 showed the peptide-signaling graph and synaptic connectome are structurally distinct: Jaccard of only 0.045. Do these two graphs COMPLEMENT each other (peptide fills where synapse doesn't) or are they INDEPENDENT (no relationship between their absence/presence)?

This is a direct test of the common idea that peptide signaling "extends" the connectome into a parallel control plane. Prior findings relevant to this question:
- Nb 12: node-degree Spearman between synaptic and peptide graphs is +0.11 to +0.28 (positive, weak) — at the hub level there's some redundancy, not compensation.
- Nb 23: 5 multiplex hubs (RIM, RMG, RIG) are in top-10% of BOTH graphs — these are co-hubs, not complementary.

But node-level patterns don't answer the edge-level question. Here we test three specific compensation claims at the edge level.

## Three preregistered tests

### Test 1 — Edge-level complementarity
Among pairs in physical contact, is the presence of a peptide edge INVERSELY associated with the presence of a synaptic edge, after controlling for contact area?

- Model: `logit(pep_edge) ~ log_contact + syn_edge`
- Compensatory: `coef(syn_edge) < 0` (pairs with synapse less likely to have peptide)
- Redundant: `coef(syn_edge) > 0` (pairs with synapse also have peptide — reinforcement)
- Independent: `coef(syn_edge) ≈ 0`
- **Criterion 1 pass**: `|coef(syn_edge)|` has p < 0.01 with 1000-bootstrap CI

### Test 2 — Triangle closure by modality
For triples (i, j, k) where the synaptic graph has i→j AND j→k (a path of length 2), is the direct edge i→k MORE likely to be in the peptide graph when it's NOT in the synaptic graph?

- Compute P(pep i→k | syn i→j, syn j→k, NO syn i→k) vs P(pep i→k | syn i→j, syn j→k, AND syn i→k)
- Compensation: former > latter by ≥ 20% relative
- **Criterion 2 pass**: relative difference ≥ 20% with 1000-bootstrap CI excluding zero

### Test 3 — NT-class biology of "peptide-only" pairs
For pairs with: contact > 0 AND peptide edge AND NO synaptic edge — are specific pre-neuron NT classes (Dopamine, Serotonin, Octopamine, Tyramine — the classical neuromodulators) over-represented?

- Fisher's exact test per NT class
- **Criterion 3 pass**: at least one monoamine NT class shows p < 0.05 enrichment

## Halting rule

If all 3 fail: peptide and synaptic graphs are INDEPENDENT — no mechanistic complementarity. Report null, narrative becomes "peptide signaling forms a parallel but unrelated network."

If ≥ 1 passes: the specific form of compensation is the finding. Focus the paper on whichever test gave signal."""))

cells.append(nbf.v4.new_code_cell("""import sys, time
from pathlib import Path
import warnings; warnings.simplefilter('ignore')
_HERE = Path.cwd()
if (_HERE / 'lib').is_dir(): sys.path.insert(0, str(_HERE))
elif (_HERE.parent / 'lib').is_dir(): sys.path.insert(0, str(_HERE.parent))

from lib.paths import DATA, DERIVED
from lib.reference import load_nt_reference

import numpy as np, pandas as pd
from scipy.stats import fisher_exact
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RNG = np.random.default_rng(42)

# Load all three graphs on the SAME neuron list
adult = np.load(DERIVED / 'connectome_adult.npz', allow_pickle=True)
w_neurons = np.array([str(n) for n in adult['neurons']])
A_syn = (adult['chem_adj'] > 0).astype(np.int32)
np.fill_diagonal(A_syn, 0)

pep = np.load(DERIVED / 'nb12_peptide_adjacency.npz', allow_pickle=True)
A_pep = pep['A_peptide'].astype(np.int32)
pep_neurons = np.array([str(n) for n in pep['neurons']])
assert (w_neurons == pep_neurons).all(), 'Neuron order differs between Witvliet and peptide graphs'
np.fill_diagonal(A_pep, 0)

# Contact matrix on same ordering
contact_xlsx = DATA / 'connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx'
contact_df = pd.read_excel(contact_xlsx, sheet_name='adult nerve ring neighbors', header=0, index_col=0)
contact_df = contact_df.loc[:, ~contact_df.columns.str.startswith('Unnamed')]
contact_df = contact_df[contact_df.index.map(lambda x: isinstance(x, str))]
contact_df = contact_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

name_to_idx = {n: i for i, n in enumerate(w_neurons)}
N = len(w_neurons)
C_mat = np.zeros((N, N), dtype=np.float32)
for row_name in contact_df.index:
    if row_name not in name_to_idx: continue
    ri = name_to_idx[row_name]
    for col_name in contact_df.columns:
        if col_name not in name_to_idx: continue
        ci = name_to_idx[col_name]
        val = contact_df.loc[row_name, col_name]
        if pd.notna(val): C_mat[ri, ci] = float(val)
C_mat = (C_mat + C_mat.T) / 2

nt = load_nt_reference()

print(f'Neurons: {N}')
print(f'Synaptic edges: {int(A_syn.sum())}')
print(f'Peptide edges:  {int(A_pep.sum())}')
print(f'Contact > 0 entries (directed): {int((C_mat > 0).sum())}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Build edge-level table (contact-stratified, directed pairs)"))

cells.append(nbf.v4.new_code_cell("""rows = []
for i in range(N):
    for j in range(N):
        if i == j: continue
        c = float(C_mat[i, j])
        if c <= 0: continue  # contact-stratified
        rows.append({
            'i': i, 'j': j,
            'pre': w_neurons[i], 'post': w_neurons[j],
            'contact_area': c,
            'syn_edge': int(A_syn[i, j]),
            'pep_edge': int(A_pep[i, j]),
        })
pairs = pd.DataFrame(rows)
print(f'Contacting ordered pairs: {len(pairs)}')

# Breakdown
breakdown = pairs.groupby(['syn_edge','pep_edge']).size().reset_index(name='count')
print(f'\\nBreakdown:')
print(breakdown.to_string(index=False))

# Expected if independent
p_syn = pairs['syn_edge'].mean()
p_pep = pairs['pep_edge'].mean()
print(f'\\nP(syn_edge)=  {p_syn:.4f}')
print(f'P(pep_edge)=  {p_pep:.4f}')
print(f'If independent, P(syn AND pep) = {p_syn * p_pep:.4f}')
print(f'Observed      P(syn AND pep) = {((pairs[\"syn_edge\"]==1) & (pairs[\"pep_edge\"]==1)).mean():.4f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Test 1 — Edge-level complementarity via logistic regression"))

cells.append(nbf.v4.new_code_cell("""# Model: pep_edge ~ log_contact + syn_edge
X1 = pairs[['contact_area', 'syn_edge']].copy()
X1['log_contact'] = np.log1p(X1['contact_area'])
X_design = sm.add_constant(X1[['log_contact', 'syn_edge']].values.astype(float))
y1 = pairs['pep_edge'].values.astype(int)

model = sm.Logit(y1, X_design).fit(disp=0)
print(model.summary())

coef_syn = float(model.params[2])
pval_syn = float(model.pvalues[2])
# Bootstrap 95% CI on the coefficient
n_boot = 1000
bs_coefs = np.zeros(n_boot)
for b in range(n_boot):
    idx = RNG.choice(len(pairs), len(pairs), replace=True)
    try:
        m_b = sm.Logit(y1[idx], X_design[idx]).fit(disp=0)
        bs_coefs[b] = m_b.params[2]
    except Exception:
        bs_coefs[b] = np.nan
bs_coefs = bs_coefs[~np.isnan(bs_coefs)]
ci_low, ci_high = np.percentile(bs_coefs, 2.5), np.percentile(bs_coefs, 97.5)

print(f'\\nTest 1 — coefficient on syn_edge (controlling for log_contact):')
print(f'  Point estimate:  {coef_syn:+.4f}')
print(f'  p-value:         {pval_syn:.4e}')
print(f'  95% bootstrap CI: [{ci_low:+.4f}, {ci_high:+.4f}]')

if coef_syn < 0:
    direction = 'NEGATIVE (compensatory — pairs with synapse have FEWER peptide edges)'
elif coef_syn > 0:
    direction = 'POSITIVE (redundant — pairs with synapse ALSO have peptide edges)'
else:
    direction = 'NULL (independent)'
print(f'  Direction: {direction}')
crit1_pass = pval_syn < 0.01
print(f'  Criterion 1 (p < 0.01): {\"PASS\" if crit1_pass else \"FAIL\"}')"""))

cells.append(nbf.v4.new_markdown_cell("## Test 2 — Triangle closure by modality"))

cells.append(nbf.v4.new_code_cell("""# For each ordered (i, k) pair where there exists at least one j with syn i->j and syn j->k:
# - count peptide-edge i->k when syn i->k IS present
# - count peptide-edge i->k when syn i->k is ABSENT
# Compute rates.

syn_mat = A_syn
pep_mat = A_pep

# syn_2step[i, k] = number of j such that syn i->j and syn j->k
syn_2step = syn_mat @ syn_mat  # int matrix

# Pairs with syn_2step > 0 (a syn path of length 2 exists from i to k)
has_2path = (syn_2step > 0) & (np.eye(N, dtype=bool) == False)
print(f'Pairs (i,k) with syn 2-path: {int(has_2path.sum())}')

# Among these, split by whether direct syn i->k exists
direct_syn_exists = has_2path & (syn_mat > 0)
direct_syn_absent = has_2path & (syn_mat == 0)

# Peptide rate in each subset
n_direct_exist = int(direct_syn_exists.sum())
n_direct_absent = int(direct_syn_absent.sum())
pep_rate_direct_exist = float((pep_mat[direct_syn_exists] > 0).mean()) if n_direct_exist else np.nan
pep_rate_direct_absent = float((pep_mat[direct_syn_absent] > 0).mean()) if n_direct_absent else np.nan

print(f'\\nP(pep i->k | syn 2-path AND direct syn exists):    {pep_rate_direct_exist:.4f}  (n={n_direct_exist})')
print(f'P(pep i->k | syn 2-path AND direct syn absent):    {pep_rate_direct_absent:.4f}  (n={n_direct_absent})')

# Compensation predicts: higher pep rate when syn is ABSENT
if pep_rate_direct_absent > 0 and pep_rate_direct_exist > 0:
    rel_diff = (pep_rate_direct_absent - pep_rate_direct_exist) / pep_rate_direct_exist
    print(f'\\nRelative difference (absent - exist)/exist: {rel_diff:+.3f} ({rel_diff*100:+.1f}%)')
else:
    rel_diff = np.nan

# Bootstrap: resample (i, k) pairs within each stratum and recompute rates
idx_exist = np.where(direct_syn_exists.flatten())[0]
idx_absent = np.where(direct_syn_absent.flatten())[0]
pep_flat = (pep_mat > 0).flatten().astype(int)

bs_rel_diff = np.zeros(n_boot)
for b in range(n_boot):
    s_exist = RNG.choice(idx_exist, len(idx_exist), replace=True)
    s_absent = RNG.choice(idx_absent, len(idx_absent), replace=True)
    r_exist = pep_flat[s_exist].mean()
    r_absent = pep_flat[s_absent].mean()
    bs_rel_diff[b] = (r_absent - r_exist) / max(r_exist, 1e-9)
ci_low_r, ci_high_r = np.percentile(bs_rel_diff, 2.5), np.percentile(bs_rel_diff, 97.5)

print(f'95% bootstrap CI on relative diff: [{ci_low_r:+.3f}, {ci_high_r:+.3f}]')
crit2_pass = (ci_low_r > 0.20)  # compensation requires >= +20%
print(f'Criterion 2 (relative diff >= +20% with CI excluding 0.20): {\"PASS\" if crit2_pass else \"FAIL\"}')"""))

cells.append(nbf.v4.new_markdown_cell("## Test 3 — NT-class biology of 'peptide-only' pairs"))

cells.append(nbf.v4.new_code_cell("""# Pairs with contact AND peptide AND NO synapse
peptide_only = pairs[(pairs['pep_edge'] == 1) & (pairs['syn_edge'] == 0)].copy()
syn_only = pairs[(pairs['pep_edge'] == 0) & (pairs['syn_edge'] == 1)].copy()
both = pairs[(pairs['pep_edge'] == 1) & (pairs['syn_edge'] == 1)].copy()
neither = pairs[(pairs['pep_edge'] == 0) & (pairs['syn_edge'] == 0)].copy()

print(f'Peptide-only:    {len(peptide_only)}')
print(f'Syn-only:        {len(syn_only)}')
print(f'Both:            {len(both)}')
print(f'Neither:         {len(neither)}')

def nt_of(n):
    v = nt.nt_of(n)
    if v is None: return None
    s = v.lower()
    if 'acetylcholine' in s: return 'ACh'
    if 'gaba' in s: return 'GABA'
    if 'glutamate' in s: return 'Glu'
    if 'dopamine' in s: return 'Dopamine'
    if 'serotonin' in s: return 'Serotonin'
    if 'octopamine' in s: return 'Octopamine'
    if 'tyramine' in s: return 'Tyramine'
    return 'Other'

for g in [peptide_only, syn_only, both, neither]:
    g['pre_NT'] = g['pre'].map(nt_of)

# For each NT class, Fisher exact test: is this NT enriched in peptide-only vs background?
# Background: all contacting pairs with contact + !syn (so peptide-only + neither)
print('\\nNT-class enrichment in peptide-only pairs:')
print(f'{\"NT\":<12s} {\"in_pep_only\":>12s} {\"in_background\":>15s} {\"odds\":>8s} {\"p (greater)\":>12s}')

bkg = pd.concat([peptide_only, neither])
results = []
for nt_class in ['ACh','GABA','Glu','Dopamine','Serotonin','Octopamine','Tyramine']:
    a = int((peptide_only['pre_NT'] == nt_class).sum())
    b = int(len(peptide_only) - a)
    c = int((neither['pre_NT'] == nt_class).sum())
    d = int(len(neither) - c)
    try:
        odds, p = fisher_exact([[a, b], [c, d]], alternative='greater')
    except Exception:
        odds, p = 0, 1
    results.append({'NT': nt_class, 'in_pep_only': a, 'pep_only_total': len(peptide_only),
                    'in_neither': c, 'neither_total': len(neither),
                    'odds_ratio': odds, 'p_enrichment': p})
    print(f'{nt_class:<12s} {a:>4d} / {len(peptide_only):<5d} {c:>6d} / {len(neither):<6d} {odds:>8.2f} {p:>12.3e}')

nt_df = pd.DataFrame(results)
monoamine_sig = nt_df[nt_df['NT'].isin(['Dopamine','Serotonin','Octopamine','Tyramine'])]
crit3_pass = bool((monoamine_sig['p_enrichment'] < 0.05).any())
print(f'\\nCriterion 3 (>=1 monoamine class with p<0.05 enrichment): {\"PASS\" if crit3_pass else \"FAIL\"}')"""))

cells.append(nbf.v4.new_markdown_cell("## Summary"))

cells.append(nbf.v4.new_code_cell("""print('NOTEBOOK 25 — PEPTIDE COMPENSATION HYPOTHESIS')
print('=' * 70)

print('\\nTest 1 (edge-level complementarity):')
print(f'  syn_edge coefficient in logit(pep_edge ~ log_contact + syn_edge):')
print(f'    {coef_syn:+.4f}  p={pval_syn:.4e}  CI=[{ci_low:+.4f}, {ci_high:+.4f}]')
print(f'  {\"PASS\" if crit1_pass else \"FAIL\"} (p<0.01)')

print('\\nTest 2 (triangle-closure compensation):')
print(f'  P(pep | 2-path, no direct syn): {pep_rate_direct_absent:.4f}')
print(f'  P(pep | 2-path, direct syn):    {pep_rate_direct_exist:.4f}')
print(f'  Relative diff: {rel_diff:+.3f}  CI=[{ci_low_r:+.3f}, {ci_high_r:+.3f}]')
print(f'  {\"PASS\" if crit2_pass else \"FAIL\"} (relative >= +20% with CI excluding 0.20)')

print('\\nTest 3 (NT-class enrichment in peptide-only pairs):')
sig_nts = nt_df[nt_df['p_enrichment'] < 0.05]
if len(sig_nts):
    for _, r in sig_nts.iterrows():
        print(f'  {r[\"NT\"]}: odds={r[\"odds_ratio\"]:.2f}, p={r[\"p_enrichment\"]:.3e}')
else:
    print(f'  No NT class significantly enriched')
print(f'  {\"PASS\" if crit3_pass else \"FAIL\"} (>=1 monoamine with p<0.05)')

n_pass = sum([crit1_pass, crit2_pass, crit3_pass])
print(f'\\n{n_pass} of 3 criteria pass')

if n_pass == 3:
    verdict = 'STRONG — peptide compensates for synapse gaps in multiple ways'
elif n_pass == 2:
    verdict = 'PARTIAL — peptide compensation detectable in 2 of 3 tests'
elif n_pass == 1:
    verdict = 'WEAK — peptide compensation only visible in one test (specify which)'
else:
    verdict = 'NULL — peptide and synaptic graphs are independent at edge level (no compensation)'
print(f'\\nVERDICT: {verdict}')

summary = pd.DataFrame([{
    'n_contacting_pairs': len(pairs),
    'n_peptide_only': len(peptide_only),
    'n_syn_only': len(syn_only),
    'n_both': len(both),
    'n_neither': len(neither),
    'test1_coef_syn_edge': coef_syn,
    'test1_p': pval_syn,
    'test1_ci_low': ci_low,
    'test1_ci_high': ci_high,
    'test1_pass': crit1_pass,
    'test2_rel_diff': rel_diff,
    'test2_ci_low': ci_low_r,
    'test2_ci_high': ci_high_r,
    'test2_pass': crit2_pass,
    'test3_any_monoamine_sig': crit3_pass,
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb25_final_summary.csv', index=False)
nt_df.to_csv(DERIVED / 'nb25_nt_enrichment.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/25_peptide_compensation.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 25 written ({len(nb.cells)} cells)')
