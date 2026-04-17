"""Build Notebook 08 — sex dimorphism via Cook 2019 hermaphrodite vs male."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 08 — Sex Dimorphism as Within-Species Control (Cook 2019)

## Question

Cook et al. 2019 published both hermaphrodite AND male C. elegans connectomes at class level. We have the file (`connectome/cook2019/SI 7 Cell class connectome adjacency matrices`) and haven't touched the male one. Three questions:

1. **How consistent are motif / edge patterns across sexes** for the ~100 shared neuron classes? This is a natural internal replication control — much stronger than permutation null.
2. **Do sex-specific edges** (present in male only, or hermaphrodite only) have distinctive features on classes that exist in both sexes?
3. **Combined with Witvliet**: are Cook-hermaphrodite and Witvliet-adult edges consistent? If so, the N doubles for cross-dataset tests.

## Why it's different from 03/04/05/06/07

- No prior notebook used Cook 2019's male connectome at all.
- Sex dimorphism is a clean biological contrast that isn't captured by permutation nulls.
- This is a **reliability/robustness** analysis rather than a new mechanistic hypothesis. If connectome motifs don't replicate across sexes (at shared classes), *all* prior gene-motif work is unreliable regardless of statistics.

## Preregistered criteria

1. **Sex-shared edge overlap ≥ 70%** on shared neuron classes. (Sanity: sex dimorphism should mostly preserve core wiring.)
2. **Motif-feature rank-correlation Spearman ρ ≥ 0.70** between sexes for shared classes. (If motif statistics aren't sex-invariant, motif-based analyses are fundamentally unreliable.)
3. **Sex-specific edge count > 20** (enough to have a meaningful "sex-specific" category for analysis).
4. **Cook-hermaphrodite vs Witvliet-adult**: edge overlap ≥ 60% on shared neuron class-pairs. (Cross-dataset replication.)

## Halting rule

If criteria 1 AND 2 both pass: motif-based analyses are trustworthy across sexes. Proceed to sex-specific edge analysis.
If 1 OR 2 fails: entire class-level motif approach is unreliable. Report as a methods-level null."""))

cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import warnings; warnings.simplefilter('ignore')
_HERE = Path.cwd()
if (_HERE / 'lib').is_dir(): sys.path.insert(0, str(_HERE))
elif (_HERE.parent / 'lib').is_dir(): sys.path.insert(0, str(_HERE.parent))

from lib.paths import DATA, DERIVED
from lib.motifs import compute_motifs
import numpy as np, pandas as pd
from scipy.stats import spearmanr

COOK_FILE = DATA / 'connectome/cook2019/SI 7 Cell class connectome adjacency matrices, corrected July 2020.xlsx'
xl = pd.ExcelFile(COOK_FILE)
print(f'Cook 2019 SI 7 sheets: {xl.sheet_names}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Load Cook 2019 herm + male adjacencies (class level)"))

cells.append(nbf.v4.new_code_cell("""def load_cook_sheet(sheet_name):
    # Cook 2019 format: a square matrix with row/column headers being cell class names.
    # The first few rows and columns may be merged cells / legend — use best-effort parsing.
    raw = pd.read_excel(COOK_FILE, sheet_name=sheet_name, header=None)
    # Find the row with the column headers (first row where all cells are strings / names)
    # and the column with row labels.
    # Strategy: scan for a row where cells are class-name-like strings.
    for hdr_row in range(20):
        row_vals = raw.iloc[hdr_row].dropna()
        if len(row_vals) > 30 and row_vals.astype(str).str.len().mean() < 8:
            # probably a row of short class names
            break
    header = raw.iloc[hdr_row]
    # Column containing row labels
    col_labels = raw.iloc[hdr_row+1:, 0].dropna()
    # First column that looks like numeric data starts at col 1 or later
    body_start_col = 1
    for c in range(1, raw.shape[1]):
        try:
            float(raw.iloc[hdr_row+1, c])
            body_start_col = c
            break
        except Exception:
            continue
    body = raw.iloc[hdr_row+1:, body_start_col:].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    col_names = header.iloc[body_start_col:].astype(str).tolist()
    row_names = raw.iloc[hdr_row+1:hdr_row+1+body.shape[0], 0].astype(str).tolist()
    # Clean row names (strip whitespace)
    row_names = [r.strip() for r in row_names]
    col_names = [c.strip() for c in col_names]
    # Build df
    df = pd.DataFrame(body.values, index=row_names, columns=col_names)
    # Restrict to square intersection (rows in cols and cols in rows)
    common = [x for x in df.index if x in df.columns]
    df = df.loc[common, common]
    return df

herm_chem = load_cook_sheet('herm chem grouped')
male_chem = load_cook_sheet('male chem grouped')
print(f'Herm chem: {herm_chem.shape}')
print(f'Male chem: {male_chem.shape}')
print(f'Herm chem range: [{herm_chem.values.min()}, {herm_chem.values.max()}]')
print(f'Male chem range: [{male_chem.values.min()}, {male_chem.values.max()}]')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Find shared neuron classes between sexes"))

cells.append(nbf.v4.new_code_cell("""# Share class intersection
shared_classes = sorted(set(herm_chem.index) & set(male_chem.index))
herm_only = sorted(set(herm_chem.index) - set(male_chem.index))
male_only = sorted(set(male_chem.index) - set(herm_chem.index))
print(f'Shared classes: {len(shared_classes)}')
print(f'Herm-only:      {len(herm_only)}   e.g. {herm_only[:10]}')
print(f'Male-only:      {len(male_only)}   e.g. {male_only[:10]}')

# Restrict adjacencies to shared classes
H = herm_chem.loc[shared_classes, shared_classes].values
M = male_chem.loc[shared_classes, shared_classes].values
H_bin = (H > 0).astype(int)
M_bin = (M > 0).astype(int)
print(f'\\nShared-class adjacency: {H_bin.shape}')
print(f'  herm edges: {H_bin.sum()}, male edges: {M_bin.sum()}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Edge overlap between sexes"))

cells.append(nbf.v4.new_code_cell("""H_set = set((i, j) for i in range(len(shared_classes)) for j in range(len(shared_classes)) if i != j and H_bin[i, j] > 0)
M_set = set((i, j) for i in range(len(shared_classes)) for j in range(len(shared_classes)) if i != j and M_bin[i, j] > 0)
union = H_set | M_set
intersection = H_set & M_set
overlap_rate = len(intersection) / len(union) if len(union) else 0
print(f'Herm-only edges:  {len(H_set - M_set)}')
print(f'Male-only edges:  {len(M_set - H_set)}')
print(f'Shared edges:     {len(intersection)}')
print(f'Jaccard overlap:  {overlap_rate:.3f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Motif feature correlation between sexes"))

cells.append(nbf.v4.new_code_cell("""# Compute per-class motif features on each sex
N = len(shared_classes)
H_names = np.array(shared_classes)

def quick_motifs(A_bin):
    A = A_bin.astype(np.int32)
    out_deg = A.sum(axis=1)
    in_deg = A.sum(axis=0)
    A2 = A @ A
    A3 = A2 @ A
    ffl = np.sum(A * A2, axis=1)
    cycle3 = np.diag(A3)
    recip = np.sum(A * A.T, axis=1)
    return pd.DataFrame({'in_deg': in_deg, 'out_deg': out_deg,
                         'ffl': ffl, 'cycle3': cycle3, 'recip': recip}, index=H_names)

mf_herm = quick_motifs(H_bin)
mf_male = quick_motifs(M_bin)

cross_sex_corr = {}
for col in ['in_deg','out_deg','ffl','cycle3','recip']:
    r, p = spearmanr(mf_herm[col], mf_male[col])
    cross_sex_corr[col] = {'rho': float(r), 'p': float(p)}
    print(f'  {col:10s} Spearman rho = {r:+.3f}, p = {p:.2e}')

# Average
avg_rho = np.mean([v['rho'] for v in cross_sex_corr.values()])
print(f'\\nAverage Spearman rho: {avg_rho:+.3f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 5 — Cross-dataset consistency: Cook herm vs Witvliet adult"))

cells.append(nbf.v4.new_code_cell("""# Load Witvliet adult
adult = np.load(DERIVED / 'connectome_adult.npz', allow_pickle=True)
w_neurons = np.array([str(n) for n in adult['neurons']])
w_chem = adult['chem_adj']

# Map to class labels via Nb02 mapping
mapping = pd.read_csv(DERIVED / 'expression_neuron_mapping.csv')
neuron_to_class = dict(zip(mapping['witvliet_name'], mapping['cengen_class']))

# Collapse Witvliet to class level: class-pair has edge if any constituent neuron pair has edge
witvliet_class_edges = {}
for i, ni in enumerate(w_neurons):
    ci = neuron_to_class.get(str(ni))
    if not isinstance(ci, str): continue
    for j, nj in enumerate(w_neurons):
        if i == j: continue
        cj = neuron_to_class.get(str(nj))
        if not isinstance(cj, str): continue
        if w_chem[i, j] > 0:
            witvliet_class_edges[(ci, cj)] = witvliet_class_edges.get((ci, cj), 0) + int(w_chem[i, j])

# Restrict to classes shared between Cook-herm and Witvliet (via our mapping)
cook_classes = set(shared_classes)
witvliet_classes = set(v for v in neuron_to_class.values() if isinstance(v, str))
common_cw = sorted(cook_classes & witvliet_classes)
print(f'Cook shared classes ∩ Witvliet-mapped classes: {len(common_cw)}')

H_cw_edges = set()
for i, p in enumerate(shared_classes):
    for j, q in enumerate(shared_classes):
        if i != j and H_bin[i, j] > 0 and p in common_cw and q in common_cw:
            H_cw_edges.add((p, q))

W_cw_edges = set(e for e in witvliet_class_edges if e[0] in common_cw and e[1] in common_cw)

cw_union = H_cw_edges | W_cw_edges
cw_intersection = H_cw_edges & W_cw_edges
cw_overlap = len(cw_intersection) / len(cw_union) if cw_union else 0
print(f'Cook-herm edges on common classes:     {len(H_cw_edges)}')
print(f'Witvliet edges on common classes:      {len(W_cw_edges)}')
print(f'Shared (both datasets):                {len(cw_intersection)}')
print(f'Jaccard Cook-herm vs Witvliet:         {cw_overlap:.3f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 6 — Preregistered criteria"))

cells.append(nbf.v4.new_code_cell("""crit1 = overlap_rate >= 0.70
crit2 = avg_rho >= 0.70
crit3 = min(len(H_set - M_set), len(M_set - H_set)) >= 20
crit4 = cw_overlap >= 0.60

print('PREREGISTERED CRITERIA')
print('=' * 70)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 Sex-shared edge Jaccard >= 0.70          {overlap_rate:.3f}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 Avg cross-sex motif Spearman >= 0.70      {avg_rho:.3f}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 Sex-specific edges (min class) >= 20       min={min(len(H_set - M_set), len(M_set - H_set))}')
print(f'  [{\"PASS\" if crit4 else \"FAIL\"}] 4 Cook-herm vs Witvliet Jaccard >= 0.60      {cw_overlap:.3f}')
print('=' * 70)
all_pass = all([crit1, crit2, crit3, crit4])

if all_pass:
    verdict = 'POSITIVE — connectome motifs are sex-invariant AND cross-dataset-reproducible'
elif crit1 and crit2:
    verdict = 'PARTIAL POSITIVE — sex-invariance holds but Cook-Witvliet cross-dataset drifts'
elif crit4:
    verdict = 'PARTIAL POSITIVE — Cook-Witvliet replicates but sex-differences are large'
else:
    verdict = 'NULL — motif statistics not reliable across sexes OR across independent datasets'
print(f'\\nVERDICT: {verdict}')

summary = pd.DataFrame([{
    'shared_classes': len(shared_classes),
    'herm_edges': len(H_set),
    'male_edges': len(M_set),
    'sex_edge_jaccard': overlap_rate,
    'avg_motif_spearman': avg_rho,
    'cook_herm_vs_witvliet_jaccard': cw_overlap,
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb08_final_summary.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells

with open('/home/rohit/Desktop/C-Elegans/New Notebooks/08_sex_dimorphism.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 08 written ({len(nb.cells)} cells)')
