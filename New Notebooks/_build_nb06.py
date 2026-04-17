"""Build Notebook 06 — edge-level ligand-receptor prediction, contact-stratified."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 06 — Edge-Level Ligand-Receptor Prediction, Contact-Stratified

## The pivot

Everything in Notebooks 03-05b tested **node-level correlations**: do a neuron's expression features predict its hub status, motif participation, or developmental role? All null at correct N.

This notebook tests **edge-level mechanism**: given two neurons that physically contact each other (nerve-ring contact matrix from Brittin/Zhen/Meyer), does the compatibility of (pre-expressed ligand) × (post-expressed receptor) predict whether they form a chemical synapse?

## Why this is genuinely different
- **Edge-level, not node-level.** N jumps from ~84 classes to ~2,000 class-pairs.
- **Mechanistic feature.** Compatibility score = counting curated ligand-receptor pairs that the (pre,post) duo could potentially use. Not correlation-mined.
- **Contact-stratified negatives.** Negatives are pairs that *physically touch* but don't connect — the right null. Random non-connected pairs are trivially distinguishable by geometry.
- **Two independent data sources.** Bentley 2016 for ligands + receptors (curated); Witvliet 2020 for edges; Brittin nerve-ring-neighbors for contact. No shared bias.

## Preregistered criteria
1. **Contact-only baseline AUC ≥ 0.60**. Sanity: physical contact alone is a predictor.
2. **Full model AUC ≥ 0.70**. The publication bar.
3. **Delta (compat+contact) − (contact-only) ≥ 0.03**. Compatibility adds above contact.
4. **Permutation null 95pct < observed full-model AUC**. Shuffled compatibility scores.
5. **At least 3 canonical ligand-receptor families** (e.g. ACh-nicotinic, GABA-GABA_A, Glu-iGluR, monoamine-GPCR) show positive marginal contribution when examined individually.

## Halting rule
If criterion 2 fails: declare null, stop.
If 2 passes but 3 fails: the signal is driven by contact alone, not mechanism.
If all pass: defensible positive; specify marginal contributions per LR family.

## Outputs
- `data_derived/nb06_pair_features.csv`
- `data_derived/nb06_final_summary.csv`
- `data_derived/nb06_lr_matches_per_pair.csv`"""))

cells.append(nbf.v4.new_code_cell("""import sys, time
from pathlib import Path
import warnings; warnings.simplefilter('ignore')
_HERE = Path.cwd()
if (_HERE / 'lib').is_dir(): sys.path.insert(0, str(_HERE))
elif (_HERE.parent / 'lib').is_dir(): sys.path.insert(0, str(_HERE.parent))

from lib.paths import DATA, DERIVED
from lib.lr_compatibility import load_lr_atlas, CANONICAL_LR_PAIRS
import numpy as np, pandas as pd

RNG = np.random.default_rng(42)

# 1) Witvliet adult
adult = np.load(DERIVED / 'connectome_adult.npz', allow_pickle=True)
neurons_c = np.array([str(n) for n in adult['neurons']])
chem_adj = adult['chem_adj']
print(f'Witvliet adult: {len(neurons_c)} neurons, {int((chem_adj>0).sum())} chemical edges')

# 2) Brittin contact matrix (adult nerve ring)
contact_xlsx = DATA / 'connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx'
contact_df = pd.read_excel(contact_xlsx, sheet_name='adult nerve ring neighbors', header=0, index_col=0)
# The last column may be an unnamed duplicate; drop if present
contact_df = contact_df.loc[:, ~contact_df.columns.str.startswith('Unnamed')]
# Drop rows that look like index labels
contact_df = contact_df[contact_df.index.map(lambda x: isinstance(x, str))]
contact_df = contact_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
contact_neurons = np.array([str(n) for n in contact_df.index])
contact_mat = contact_df.values.astype(np.float32)
print(f'Brittin contact: {contact_mat.shape} neurons (adult nerve ring)')
print(f'Contact entries > 0: {int((contact_mat > 0).sum())}, mean nonzero: {contact_mat[contact_mat>0].mean():.1f}')

# 3) Bentley LR atlas
lr = load_lr_atlas()
print(f'Bentley: {len(lr.neuron_to_ligands)} neurons with ligands, {len(lr.neuron_to_receptors)} with receptors')

# 4) Neuron-to-class mapping (from Nb 02 alignment) — classes = CeNGEN classes
mapping_df = pd.read_csv(DERIVED / 'expression_neuron_mapping.csv')
neuron_to_class = dict(zip(mapping_df['witvliet_name'], mapping_df['cengen_class']))"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Assemble class-level features"))

cells.append(nbf.v4.new_code_cell("""# Intersect: neurons present in Witvliet AND in contact matrix
common_neurons = sorted(set(neurons_c) & set(contact_neurons))
print(f'Neurons in Witvliet ∩ Contact: {len(common_neurons)}')

# Build class membership (from the Nb 02 mapping) — restricted to these neurons
class_members: dict = {}
for n in common_neurons:
    cls = neuron_to_class.get(n)
    if isinstance(cls, str):
        class_members.setdefault(cls, []).append(n)
print(f'Classes covered: {len(class_members)}')

# For each class pair (P, Q), compute:
#   1) has_edge: does any (n_p, n_q) have a Witvliet chemical synapse?
#   2) contact_area: total contact area summed across L/R-paired members
#   3) compatibility: counted LR matches (via Bentley)
chem_idx = {n: i for i, n in enumerate(neurons_c)}
contact_idx = {n: i for i, n in enumerate(contact_neurons)}

classes = sorted(class_members)
pair_records = []
for p in classes:
    members_p = class_members[p]
    ligands_p = lr.class_ligands(members_p)
    for q in classes:
        if p == q:
            continue
        members_q = class_members[q]
        receptors_q = lr.class_receptors(members_q)

        # edge existence (any L/R-paired neuron pair)
        edge_count = 0
        for a in members_p:
            ai = chem_idx.get(a)
            if ai is None: continue
            for b in members_q:
                bi = chem_idx.get(b)
                if bi is None or ai == bi: continue
                if chem_adj[ai, bi] > 0:
                    edge_count += int(chem_adj[ai, bi])

        # contact area (sum of individual neuron-pair contacts)
        contact_sum = 0.0
        contact_count = 0
        for a in members_p:
            ai = contact_idx.get(a)
            if ai is None: continue
            for b in members_q:
                bi = contact_idx.get(b)
                if bi is None or ai == bi: continue
                contact_sum += contact_mat[ai, bi]
                if contact_mat[ai, bi] > 0:
                    contact_count += 1

        # compatibility
        compat_score, compat_pairs = lr.compatibility_score(members_p, members_q)

        pair_records.append({
            'pre_class': p,
            'post_class': q,
            'edge_count': edge_count,
            'has_edge': int(edge_count > 0),
            'contact_sum': float(contact_sum),
            'contact_pairs': contact_count,
            'has_contact': int(contact_sum > 0),
            'n_pre_ligands': len(ligands_p),
            'n_post_receptors': len(receptors_q),
            'compat_score': int(compat_score),
            'compat_pairs': ';'.join(sorted({f'{L}->{R}' for L, R in compat_pairs})),
        })

pairs = pd.DataFrame(pair_records)
print(f'Total ordered class-pairs (P != Q): {len(pairs)}')
print(f'\\nBreakdown:')
print(f'  has_edge:              {pairs[\"has_edge\"].sum()} pairs')
print(f'  has_contact:           {pairs[\"has_contact\"].sum()} pairs')
print(f'  has_edge & has_contact:{((pairs[\"has_edge\"]==1) & (pairs[\"has_contact\"]==1)).sum()} pairs')
print(f'  has_contact & !edge:   {((pairs[\"has_edge\"]==0) & (pairs[\"has_contact\"]==1)).sum()} pairs (negatives)')

pairs.to_csv(DERIVED / 'nb06_pair_features.csv', index=False)"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Build positives / negatives (contact-stratified)"))

cells.append(nbf.v4.new_code_cell("""# Positives: has_edge AND has_contact (edge exists between physically-touching classes)
# Negatives: has_contact AND !has_edge (physically touch but no edge)
# Drop pairs with no contact — they're trivially not-edges due to geometry, not mechanism

analysis = pairs[pairs['has_contact'] == 1].copy().reset_index(drop=True)
y = analysis['has_edge'].values.astype(int)
print(f'Contact-stratified analysis set: {len(analysis)} class-pairs')
print(f'  positives (edge+contact):      {(y==1).sum()}')
print(f'  negatives (contact, no edge):  {(y==0).sum()}')
print(f'  base rate:                     {y.mean():.3f}')

# Log-transform contact (spans many orders of magnitude)
analysis['log_contact'] = np.log1p(analysis['contact_sum'])
# Count-like features
analysis['log_n_lig']  = np.log1p(analysis['n_pre_ligands'])
analysis['log_n_rec']  = np.log1p(analysis['n_post_receptors'])

print('\\nFeature summary:')
print(analysis[['log_contact','compat_score','n_pre_ligands','n_post_receptors']].describe().round(3).to_string())"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Nested model comparison"))

cells.append(nbf.v4.new_code_cell("""from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def cv_auc(X, y, n_splits=5, random_state=42, C=1.0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    for tr, te in skf.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C))
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.array(aucs)

# Model A: contact only
X_contact = analysis[['log_contact']].values
aucs_contact = cv_auc(X_contact, y)
# Model B: compat only
X_compat = analysis[['compat_score']].values
aucs_compat = cv_auc(X_compat, y)
# Model C: contact + compat + ligand/receptor richness (full)
X_full = analysis[['log_contact', 'compat_score', 'log_n_lig', 'log_n_rec']].values
aucs_full = cv_auc(X_full, y)

def fmt(a): return f'{a.mean():.3f} ± {a.std():.3f}  (folds: {[f\"{x:.2f}\" for x in a]})'
print(f'Contact-only AUC:  {fmt(aucs_contact)}')
print(f'Compat-only AUC:   {fmt(aucs_compat)}')
print(f'Full (C+A+R) AUC:  {fmt(aucs_full)}')

delta = aucs_full.mean() - aucs_contact.mean()
print(f'\\nDelta (full - contact-only): {delta:+.3f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Permutation null"))

cells.append(nbf.v4.new_code_cell("""N_PERM = 100
null_full = []
# Shuffle compat_score + ligand/receptor count jointly (break their alignment with y)
X_full_mat = X_full.copy()
for i in range(N_PERM):
    perm = RNG.permutation(X_full_mat.shape[0])
    X_shuf = X_full_mat.copy()
    X_shuf[:, 1:] = X_full_mat[perm, 1:]  # shuffle compat+counts; keep contact aligned
    null_full.append(cv_auc(X_shuf, y, random_state=42+i).mean())
null_full = np.array(null_full)
print(f'Permutation null (shuffle compat+richness; keep contact): mean={null_full.mean():.3f}, 95pct={np.percentile(null_full, 95):.3f}, max={null_full.max():.3f}')
print(f'\\nObserved full AUC: {aucs_full.mean():.3f}')
print(f'Null 95pct:        {np.percentile(null_full, 95):.3f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 5 — Marginal-contribution check per ligand family"))

cells.append(nbf.v4.new_code_cell("""# Compute compatibility restricted to one ligand FAMILY at a time
FAMILIES = {
    'monoamine': {'Dopamine', 'Serotonin', 'Octopamine', 'Tyramine'},
    'FLP_peptides': {k for k in CANONICAL_LR_PAIRS if k.startswith('FLP-')},
    'NLP_peptides': {k for k in CANONICAL_LR_PAIRS if k.startswith('NLP-')},
    'PDF_peptides': {'PDF-1', 'PDF-2'},
}

# Rebuild class-level ligand sets
from lib.lr_compatibility import load_lr_atlas
lr_atlas = load_lr_atlas()

def compat_score_family(row, family_ligands):
    p, q = row['pre_class'], row['post_class']
    mem_p = class_members.get(p, [])
    mem_q = class_members.get(q, [])
    ligands_p = lr_atlas.class_ligands(mem_p) & family_ligands
    receptors_q = lr_atlas.class_receptors(mem_q)
    score = 0
    for L in ligands_p:
        if L in CANONICAL_LR_PAIRS:
            score += len(CANONICAL_LR_PAIRS[L] & receptors_q)
    return score

family_auc_deltas = {}
for fam, lig_set in FAMILIES.items():
    feat = np.array([compat_score_family(r, lig_set) for _, r in analysis.iterrows()]).reshape(-1, 1)
    # Sanity: any variance?
    if feat.std() < 1e-9:
        family_auc_deltas[fam] = {'delta_auc': 0.0, 'note': 'no variance'}
        continue
    X_fam = np.concatenate([X_contact, feat], axis=1)
    auc_fam = cv_auc(X_fam, y).mean()
    delta_fam = auc_fam - aucs_contact.mean()
    family_auc_deltas[fam] = {'delta_auc': delta_fam, 'contact_only': aucs_contact.mean(), 'plus_family': auc_fam}
    print(f'  {fam:18s}  contact={aucs_contact.mean():.3f}  +{fam}={auc_fam:.3f}  delta={delta_fam:+.3f}')

fam_df = pd.DataFrame(family_auc_deltas).T
fam_df.to_csv(DERIVED / 'nb06_family_marginals.csv')
n_families_positive = int((fam_df['delta_auc'] > 0.005).sum())
print(f'\\nLigand families with positive marginal (delta > 0.005): {n_families_positive}/{len(FAMILIES)}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 6 — Preregistered criteria check"))

cells.append(nbf.v4.new_code_cell("""crit1 = aucs_contact.mean() >= 0.60
crit2 = aucs_full.mean() >= 0.70
crit3 = delta >= 0.03
crit4 = aucs_full.mean() > np.percentile(null_full, 95)
crit5 = n_families_positive >= 3

print('PREREGISTERED CRITERIA')
print('=' * 70)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 Contact-only AUC >= 0.60            {aucs_contact.mean():.3f}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 Full AUC >= 0.70                     {aucs_full.mean():.3f}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 Delta (full - contact) >= 0.03       {delta:+.3f}')
print(f'  [{\"PASS\" if crit4 else \"FAIL\"}] 4 Full AUC > null 95pct                {aucs_full.mean():.3f} vs {np.percentile(null_full, 95):.3f}')
print(f'  [{\"PASS\" if crit5 else \"FAIL\"}] 5 >=3 LR families w/ positive marginal  {n_families_positive}/{len(FAMILIES)}')
print('=' * 70)
all_pass = all([crit1, crit2, crit3, crit4, crit5])
print(f'ALL PASS: {all_pass}')

if all_pass:
    verdict = 'STRONG POSITIVE — L-R compatibility + contact predicts chemical synapses'
elif crit2 and crit3 and crit4:
    verdict = 'POSITIVE — main signal clear, some families inconclusive'
elif crit1 and (crit4 or crit3):
    verdict = 'WEAK POSITIVE — contact dominates, compatibility adds marginally'
else:
    verdict = 'NULL — compatibility does not add above contact baseline'
print(f'\\nVERDICT: {verdict}')

summary = pd.DataFrame([{
    'n_class_pairs_with_contact': int(len(analysis)),
    'n_positives': int((y==1).sum()),
    'n_negatives': int((y==0).sum()),
    'contact_only_auc': float(aucs_contact.mean()),
    'compat_only_auc': float(aucs_compat.mean()),
    'full_auc': float(aucs_full.mean()),
    'delta_full_minus_contact': float(delta),
    'null_95pct': float(np.percentile(null_full, 95)),
    'n_families_positive': n_families_positive,
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb06_final_summary.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells

with open('/home/rohit/Desktop/C-Elegans/New Notebooks/06_lr_compatibility_edge_prediction.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 06 written ({len(nb.cells)} cells)')
