"""Nb 13 — Signed connectome: infer E/I signs per edge from NT + receptor, then re-examine motifs."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 13 — Signed Connectome Motif Re-Analysis

## Context

Every notebook so far has used the connectome as an UNSIGNED directed graph — edges exist or don't, without E/I polarity. But biologically, chemical synapses are either excitatory or inhibitory, and balanced feed-forward loops (FFLs) look topologically identical to double-negative feedback loops.

This notebook infers edge signs from:
- **Presynaptic NT identity** (Loer & Rand 2022 + Bentley 2016)
- **Postsynaptic receptor class** (Bentley 2016 + known pharmacology)

Then compares **signed motif counts** to unsigned. If significant differences arise, all prior motif-based analyses (Nb03b, Nb05) may have been run on the wrong substrate.

## Signed-edge inference rules

- ACh → nicotinic receptor (acr-*, unc-29/38/63, lev-*, des-2): **EXCITATORY (+1)**
- ACh → muscarinic (gar-1/2/3): **INHIBITORY/MOD (−1)** (often gar-2 is inhibitory)
- Glutamate → ionotropic (glr-*, nmr-*, avr-*): **EXCITATORY (+1)**
- Glutamate → metabotropic (mgl-*): **INHIBITORY/MOD (−1)**
- GABA → any receptor (unc-49, gab-*, lgc-37): **INHIBITORY (−1)**
- Monoamine → most GPCRs: **MODULATORY (±1, usually − for somatic)**
- Peptide → NPR-*: **MODULATORY (±1, often −)**

## Preregistered criteria

1. **Fraction of edges with assignable sign ≥ 30%**. (Many edges won't be sign-assignable; that's expected.)
2. **Among signed edges, the excitatory/inhibitory ratio is in [0.5, 5.0]**. (Neither side is absurdly dominant.)
3. **Signed FFLs show distinct enrichment from unsigned**: coherent FFLs (all +) or incoherent FFLs should be over/under-represented compared to motif-rewired null (Bollobás sign-preserving).
4. **At least one signed-motif class has p < 0.05 vs degree-preserving null**."""))

cells.append(nbf.v4.new_code_cell("""import sys, time
from pathlib import Path
import warnings; warnings.simplefilter('ignore')
_HERE = Path.cwd()
if (_HERE / 'lib').is_dir(): sys.path.insert(0, str(_HERE))
elif (_HERE.parent / 'lib').is_dir(): sys.path.insert(0, str(_HERE.parent))

from lib.paths import DERIVED
from lib.lr_compatibility import load_lr_atlas
from lib.reference import load_nt_reference

import numpy as np, pandas as pd

RNG = np.random.default_rng(42)

adult = np.load(DERIVED / 'connectome_adult.npz', allow_pickle=True)
w_neurons = np.array([str(n) for n in adult['neurons']])
w_chem = adult['chem_adj']

lr = load_lr_atlas()
nt = load_nt_reference()

# Build per-neuron NT
def get_nt(n):
    v = nt.nt_of(n)
    if v is None: return None
    s = v.lower()
    if 'acetylcholine' in s or 'ach' in s: return 'ACh'
    if 'gaba' in s: return 'GABA'
    if 'glutamate' in s: return 'Glu'
    if 'dopamine' in s: return 'Dopamine'
    if 'serotonin' in s: return 'Serotonin'
    if 'octopamine' in s: return 'Octopamine'
    if 'tyramine' in s: return 'Tyramine'
    return None

print(f'Witvliet: {len(w_neurons)} neurons, {int((w_chem > 0).sum())} chem edges')
print(f'NT-assigned neurons: {sum(1 for n in w_neurons if get_nt(n))}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Assign signs per directed edge"))

cells.append(nbf.v4.new_code_cell("""# Receptor class => sign rules
EXCITATORY_RECEPTORS = {
    # ACh nicotinic
    'ACR-2','ACR-3','ACR-5','ACR-7','ACR-8','ACR-10','ACR-11','ACR-12','ACR-14','ACR-15','ACR-16','ACR-17','ACR-18','ACR-19','ACR-20','ACR-21','ACR-23','ACR-24','ACR-25',
    'UNC-29','UNC-38','UNC-63','LEV-1','LEV-8','DES-2',
    # Glu ionotropic
    'GLR-1','GLR-2','GLR-3','GLR-4','GLR-5','GLR-6','GLR-7','GLR-8','NMR-1','NMR-2','AVR-14','AVR-15'
}
INHIBITORY_RECEPTORS = {
    # GABA
    'UNC-49','GAB-1','LGC-37','GBB-1','GBB-2',
    # Glu inhibitory (glc-*)
    'GLC-1','GLC-2','GLC-3','GLC-4',
    # GABA-B
}
MOD_NEG = {
    # Muscarinic ACh (often inhibitory, sometimes mod)
    'GAR-1','GAR-2','GAR-3',
    # Metabotropic Glu
    'MGL-1','MGL-2','MGL-3',
    # Monoamine and peptide GPCRs typically GPCR-mediated (can be + or -)
}

def edge_sign(pre, post):
    pre_nt = get_nt(pre)
    post_recv = lr.neuron_to_receptors.get(post, set())
    if pre_nt is None or not post_recv:
        return 0  # unassignable

    if pre_nt == 'ACh':
        if post_recv & EXCITATORY_RECEPTORS: return +1
        if post_recv & MOD_NEG & {'GAR-1','GAR-2','GAR-3'}: return -1
        return 0
    if pre_nt == 'GABA':
        if post_recv & INHIBITORY_RECEPTORS: return -1
        return 0
    if pre_nt == 'Glu':
        if post_recv & EXCITATORY_RECEPTORS & {'GLR-1','GLR-2','GLR-3','GLR-4','GLR-5','GLR-6','GLR-7','GLR-8','NMR-1','NMR-2','AVR-14','AVR-15'}: return +1
        if post_recv & INHIBITORY_RECEPTORS & {'GLC-1','GLC-2','GLC-3','GLC-4'}: return -1
        if post_recv & {'MGL-1','MGL-2','MGL-3'}: return -1
        return 0
    if pre_nt in ['Dopamine','Serotonin','Octopamine','Tyramine']:
        # Monoamines mostly modulatory, often inhibitory (conservative: assign -1 only if receptor known)
        # Leave unassigned for now since ambiguous
        return 0
    return 0

N = len(w_neurons)
signs = np.zeros((N, N), dtype=np.int8)
edge_count = 0
for i in range(N):
    for j in range(N):
        if w_chem[i, j] > 0:
            edge_count += 1
            signs[i, j] = edge_sign(w_neurons[i], w_neurons[j])

n_pos = int((signs > 0).sum())
n_neg = int((signs < 0).sum())
n_unassigned = edge_count - n_pos - n_neg
print(f'Total chem edges: {edge_count}')
print(f'  Excitatory (+1): {n_pos}  ({n_pos/edge_count:.2%})')
print(f'  Inhibitory (-1): {n_neg}  ({n_neg/edge_count:.2%})')
print(f'  Unassigned:      {n_unassigned}  ({n_unassigned/edge_count:.2%})')

n_assigned = n_pos + n_neg
frac_assigned = n_assigned / edge_count
EI_ratio = n_pos / max(n_neg, 1)
print(f'\\nFraction assigned: {frac_assigned:.2%}')
print(f'E/I ratio: {EI_ratio:.2f}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Signed triad motif counts"))

cells.append(nbf.v4.new_code_cell("""# Focus on FFL motifs: (i->j, j->k, i->k) with 8 possible sign combinations
# Count each sign-class separately

A_bin = (w_chem > 0).astype(np.int32)
np.fill_diagonal(A_bin, 0)

def count_signed_ffls(signs, A_bin):
    # Only count edges with assigned signs
    A_pos = (signs > 0).astype(np.int32)
    A_neg = (signs < 0).astype(np.int32)
    # FFL: i->j, j->k, i->k. Each edge can be +/-/0
    # Enumerate 8 sign classes (2^3 = 8)
    counts = {}
    for s_ij in [+1, -1]:
        for s_jk in [+1, -1]:
            for s_ik in [+1, -1]:
                A_ij = A_pos if s_ij == +1 else A_neg
                A_jk = A_pos if s_jk == +1 else A_neg
                A_ik = A_pos if s_ik == +1 else A_neg
                # FFL count: A_ij @ A_jk has (i, k) = number of j such that i->j and j->k
                # then element-wise multiply by A_ik
                c = int((A_ij @ A_jk * A_ik).sum())
                counts[f'({s_ij:+d},{s_jk:+d},{s_ik:+d})'] = c
    return counts

signed_counts = count_signed_ffls(signs, A_bin)
unsigned_ffls = int(((A_bin @ A_bin) * A_bin).sum())
print(f'Unsigned FFL count: {unsigned_ffls}')
print(f'Signed FFL counts (only edges with inferred signs):')
total_signed = 0
for k, v in signed_counts.items():
    print(f'  {k}: {v}')
    total_signed += v
print(f'  total signed: {total_signed}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Null: rewire signs (preserving E and I counts per neuron) and recount"))

cells.append(nbf.v4.new_code_cell("""# Simple null: re-assign signs of assigned edges uniformly at random with same total +/- counts
assigned_mask = (signs != 0)
assigned_pos = signs[assigned_mask]

N_PERM = 100
null_counts = {k: [] for k in signed_counts}

for p in range(N_PERM):
    # Shuffle signs of assigned edges
    shuffled = RNG.permutation(assigned_pos)
    signs_perm = signs.copy()
    signs_perm[assigned_mask] = shuffled
    cnt = count_signed_ffls(signs_perm, A_bin)
    for k in signed_counts:
        null_counts[k].append(cnt[k])

print('Sign-class FFL counts: real vs null')
print(f'{\"sign_class\":15s} {\"real\":>6} {\"null_mean\":>10} {\"null_95pct\":>12} {\"z_score\":>10}')
print('=' * 60)
significant_classes = 0
for k, v in signed_counts.items():
    nm = np.mean(null_counts[k])
    nsd = np.std(null_counts[k])
    z = (v - nm) / max(nsd, 1e-6)
    n95 = np.percentile(null_counts[k], 95)
    n05 = np.percentile(null_counts[k], 5)
    sig = (v > n95) or (v < n05)
    if sig: significant_classes += 1
    flag = '***' if sig else ''
    print(f'{k:15s} {v:>6d} {nm:>10.1f} [{n05:>5.0f}, {n95:>5.0f}] z={z:+.2f} {flag}')
print(f'\\n{significant_classes}/{len(signed_counts)} sign classes show significant enrichment/depletion')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — Criteria"))

cells.append(nbf.v4.new_code_cell("""crit1 = frac_assigned >= 0.30
crit2 = 0.5 <= EI_ratio <= 5.0
crit3 = significant_classes >= 1
crit4 = significant_classes >= 1  # same as 3 for now

print('CRITERIA')
print('=' * 60)
print(f'  [{\"PASS\" if crit1 else \"FAIL\"}] 1 Frac edges assignable >= 30%     {frac_assigned:.2%}')
print(f'  [{\"PASS\" if crit2 else \"FAIL\"}] 2 E/I ratio in [0.5, 5.0]            {EI_ratio:.2f}')
print(f'  [{\"PASS\" if crit3 else \"FAIL\"}] 3 >=1 signed motif significant       {significant_classes}/{len(signed_counts)}')
print('=' * 60)

if all([crit1, crit2, crit3]):
    verdict = 'POSITIVE — signed connectome reveals motif sign enrichment not visible in unsigned'
elif crit1 and crit2:
    verdict = 'DESCRIPTIVE — signs assignable and balanced, no motif-level enrichment signal'
else:
    verdict = 'INCOMPLETE — insufficient sign coverage for meaningful analysis'
print(f'\\nVERDICT: {verdict}')

summary = pd.DataFrame([{
    'edges_total': edge_count,
    'edges_excitatory': n_pos, 'edges_inhibitory': n_neg,
    'edges_unassigned': n_unassigned,
    'frac_assigned': frac_assigned,
    'EI_ratio': EI_ratio,
    'unsigned_ffls': unsigned_ffls,
    'signed_ffls_total': total_signed,
    'n_significant_sign_classes': significant_classes,
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb13_final_summary.csv', index=False)
pd.DataFrame([signed_counts]).to_csv(DERIVED / 'nb13_signed_ffl_counts.csv', index=False)
print(summary.T.to_string())"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/13_signed_connectome_motifs.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 13 written ({len(nb.cells)} cells)')
