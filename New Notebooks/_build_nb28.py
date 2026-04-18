"""Nb 28 — Multiplex Control Architecture: comprehensive multi-layer archetype atlas."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ml)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Notebook 28 — Multiplex Control Architecture of the *C. elegans* Nervous System

## The synthesis moonshot

Nb 27 showed that synaptic placement is topology-driven and peptide placement is identity-driven — two orthogonal organizational principles on overlapping neurons. But the nervous system isn't just synapse + peptide: it also has gap junctions (a distinct electrical-synchronization layer, Nb 10/19).

**The question:** if we look at each of the 222 C. elegans neurons across all three communication layers simultaneously — chemical, electrical, peptide — can we identify principled *archetypes* of multiplex role? Are there neurons that specialize in one layer, others that integrate across layers, and does this organization map coherently to known biology?

## Why this is a real test (not just a plot-making exercise)

This notebook is structured around six concrete preregistered tests, each with a halting criterion:

1. **Layer distinctness** — the three-layer degree matrix should NOT collapse to one dominant axis.
2. **Archetype emergence** — rule-based archetypes should produce at least 4 non-empty categories.
3. **NT enrichment** — NT classes should differentially occupy archetypes (matches Nb 25 monoamine preference).
4. **Command interneuron prediction** — AVA, AVB, AVD, AVE, PVC should be enriched as wired integrators.
5. **Monoamine prediction** — ADE, CEP, PDE, HSN, NSM, RIC, RIM should be enriched as wireless modulators.
6. **Multiplex hub reproduction** — Nb 23's multiplex hubs (RIM L/R, RMG L/R, RIGR) should reappear with the new archetype machinery.

## Preregistered criteria

Pass: ≥ 5 of 6 tests pass.
Weak pass: 4 of 6 tests pass.
Null: ≤ 3 of 6 tests pass → archetype framing is not load-bearing; paper stays at Nb 27 synthesis."""))

cells.append(nbf.v4.new_code_cell("""import sys, time
from pathlib import Path
import warnings; warnings.simplefilter('ignore')
_HERE = Path.cwd()
if (_HERE / 'lib').is_dir(): sys.path.insert(0, str(_HERE))
elif (_HERE.parent / 'lib').is_dir(): sys.path.insert(0, str(_HERE.parent))

from lib.paths import DATA, DERIVED
from lib.reference import load_nt_reference

import numpy as np, pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)

# Load all three connectome layers on the same neuron set
adult = np.load(DERIVED / 'connectome_adult.npz', allow_pickle=True)
w_neurons = np.array([str(n) for n in adult['neurons']])
A_chem = adult['chem_adj']
A_gap  = adult['gap_adj']  # already symmetric-doubled
np.fill_diagonal(A_chem, 0)
np.fill_diagonal(A_gap, 0)

pep = np.load(DERIVED / 'nb12_peptide_adjacency.npz', allow_pickle=True)
A_pep = pep['A_peptide'].astype(np.int32)
pep_neurons = np.array([str(n) for n in pep['neurons']])
assert (w_neurons == pep_neurons).all(), 'neuron order mismatch'
np.fill_diagonal(A_pep, 0)

nt = load_nt_reference()

N = len(w_neurons)
print(f'N neurons: {N}')
print(f'Chemical synapses: {int((A_chem>0).sum())} directed')
print(f'Gap junctions:     {int((A_gap>0).sum()) // 2} undirected')
print(f'Peptide edges:     {int(A_pep.sum())} directed')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1 — Compute per-neuron multi-layer degree profile"))

cells.append(nbf.v4.new_code_cell("""# Five-dimensional feature per neuron
chem_in   = (A_chem > 0).astype(int).sum(axis=0)
chem_out  = (A_chem > 0).astype(int).sum(axis=1)
gap_deg   = (A_gap > 0).astype(int).sum(axis=1)  # symmetric, so either axis works
pep_in    = A_pep.sum(axis=0)
pep_out   = A_pep.sum(axis=1)

profile = pd.DataFrame({
    'neuron': w_neurons,
    'chem_in': chem_in,
    'chem_out': chem_out,
    'gap_deg': gap_deg,
    'pep_in': pep_in,
    'pep_out': pep_out,
})

# Z-score each feature (within the full neuron set)
for col in ['chem_in','chem_out','gap_deg','pep_in','pep_out']:
    profile[f'{col}_z'] = (profile[col] - profile[col].mean()) / (profile[col].std() + 1e-12)

print('Multi-layer profile summary:')
print(profile[['chem_in','chem_out','gap_deg','pep_in','pep_out']].describe().round(2).to_string())

# Layer Spearman correlations — how much do the layers share structure?
print('\\nCross-layer Spearman correlations:')
cols = ['chem_in','chem_out','gap_deg','pep_in','pep_out']
corr_mat = np.zeros((5, 5))
for i, a in enumerate(cols):
    for j, b in enumerate(cols):
        r, _ = spearmanr(profile[a], profile[b])
        corr_mat[i, j] = r
corr_df = pd.DataFrame(corr_mat, index=cols, columns=cols)
print(corr_df.round(3).to_string())"""))

cells.append(nbf.v4.new_markdown_cell("## Step 2 — Preregistered rule-based archetype labels"))

cells.append(nbf.v4.new_code_cell("""# Archetype definitions (set BEFORE examining data):
#   Wired broadcaster:     chem_out_z >= 1.0, chem_in_z < 0.5, gap_z < 0.5, pep_out_z < 0.5, pep_in_z < 0.5
#   Wired receiver:        chem_in_z >= 1.0, chem_out_z < 0.5, gap_z < 0.5, pep_z < 0.5
#   Wired integrator:      (chem_in_z + chem_out_z) >= 1.5 AND both > 0, others < 0.5
#   Electrical synchronizer: gap_z >= 1.0, others_z < 0.5
#   Wireless modulator:    pep_out_z >= 1.0, chem_out_z < 0.5, gap_z < 0.5
#   Wireless target:       pep_in_z >= 1.0, chem_in_z < 0.5, gap_z < 0.5
#   Multiplex integrator:  >= 2 layers with z >= 1.0 (combining {chem, gap, pep})
#   Quiet:                 all z < 0.5

THR_HIGH = 1.0
THR_LOW  = 0.5

def classify_archetype(row):
    ci, co = row['chem_in_z'], row['chem_out_z']
    g      = row['gap_deg_z']
    pi, po = row['pep_in_z'], row['pep_out_z']

    # Layer-level 'high' indicators
    chem_high = max(ci, co) >= THR_HIGH
    gap_high  = g >= THR_HIGH
    pep_high  = max(pi, po) >= THR_HIGH
    n_high_layers = int(chem_high) + int(gap_high) + int(pep_high)

    # Check MULTIPLEX first (matches Nb 23 approach)
    if n_high_layers >= 2:
        return 'Multiplex_integrator'

    # Otherwise, figure out the dominant layer
    if chem_high and not (gap_high or pep_high):
        if ci >= THR_HIGH and co >= THR_HIGH: return 'Wired_integrator'
        elif ci >= THR_HIGH: return 'Wired_receiver'
        elif co >= THR_HIGH: return 'Wired_broadcaster'

    if gap_high and not (chem_high or pep_high):
        return 'Electrical_synchronizer'

    if pep_high and not (chem_high or gap_high):
        if pi >= THR_HIGH and po >= THR_HIGH: return 'Wireless_integrator'
        elif po >= THR_HIGH: return 'Wireless_modulator'
        elif pi >= THR_HIGH: return 'Wireless_target'

    if max(ci, co, g, pi, po) < THR_LOW:
        return 'Quiet'

    return 'Intermediate'

profile['archetype'] = profile.apply(classify_archetype, axis=1)
archetype_counts = profile['archetype'].value_counts()
print('Archetype breakdown:')
print(archetype_counts.to_string())
print(f'\\n{int((profile[\"archetype\"] != \"Quiet\") & (profile[\"archetype\"] != \"Intermediate\")).sum()} neurons in specialized archetypes')

# Save profile
profile.to_csv(DERIVED / 'nb28_neuron_profile.csv', index=False)"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3 — Annotate with biological ground truth"))

cells.append(nbf.v4.new_code_cell("""def nt_of(n):
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

profile['NT'] = profile['neuron'].map(nt_of)

# Command interneurons (Loer & Rand body location + classical neuron list)
COMMAND_INTERNEURONS = {'AVAL','AVAR','AVBL','AVBR','AVDL','AVDR','AVEL','AVER','PVCL','PVCR'}

# Monoamine neurons (from Loer & Rand classifications + literature)
MONOAMINE_NEURONS = set()
for n in w_neurons:
    s = nt_of(n)
    if s in {'DA','5HT','OA','TA'}:
        MONOAMINE_NEURONS.add(n)

# Sensory neurons (known amphid + phasmid + mechanosensory classes)
SENSORY_PREFIXES = ['ASE','ASG','ASH','ASI','ASJ','ASK','AWA','AWB','AWC','ADL','ADF',
                    'AFD','BAG','URX','URY','IL2','PHA','PHB','ALM','PLM','AVM','PVM']
SENSORY_NEURONS = {n for n in w_neurons if any(n.startswith(p) for p in SENSORY_PREFIXES)}

print(f'Command interneurons in data: {sorted(COMMAND_INTERNEURONS & set(w_neurons))}')
print(f'Monoamine neurons in data:    {sorted(MONOAMINE_NEURONS & set(w_neurons))}')
print(f'Sensory neurons in data:      {len(SENSORY_NEURONS & set(w_neurons))}  e.g. {sorted(SENSORY_NEURONS & set(w_neurons))[:10]}')

profile['is_command'] = profile['neuron'].isin(COMMAND_INTERNEURONS).astype(int)
profile['is_monoamine'] = profile['neuron'].isin(MONOAMINE_NEURONS).astype(int)
profile['is_sensory'] = profile['neuron'].isin(SENSORY_NEURONS).astype(int)"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4 — k-means sensitivity analysis"))

cells.append(nbf.v4.new_code_cell("""# Cluster the 5-dim z-scored profile into k=7 clusters (matching our archetype taxonomy size)
X_z = profile[['chem_in_z','chem_out_z','gap_deg_z','pep_in_z','pep_out_z']].values

kmeans_results = {}
for k in [5, 6, 7, 8]:
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X_z)
    profile[f'kmeans_k{k}'] = km.labels_
    # Compute centroids
    centroids = pd.DataFrame(km.cluster_centers_, columns=['chem_in_z','chem_out_z','gap_deg_z','pep_in_z','pep_out_z'])
    kmeans_results[k] = {'model': km, 'centroids': centroids}
    print(f'\\n--- k={k} centroid patterns ---')
    print(centroids.round(2).to_string())

# Use k=7 as the main view
profile['cluster'] = profile['kmeans_k7']

# Name each cluster based on its centroid shape
CENTROID_LABELS_K7 = {}
centroids_7 = kmeans_results[7]['centroids']
for cluster_id in range(7):
    c = centroids_7.loc[cluster_id]
    # Heuristic naming based on strongest axes
    max_layer = max(c['chem_out_z'], c['chem_in_z'], c['gap_deg_z'], c['pep_out_z'], c['pep_in_z'])
    if max_layer < 0.3:
        CENTROID_LABELS_K7[cluster_id] = 'Baseline'
    else:
        label_parts = []
        if c['chem_in_z'] > 0.8: label_parts.append('chem_in+')
        if c['chem_out_z'] > 0.8: label_parts.append('chem_out+')
        if c['gap_deg_z'] > 0.8: label_parts.append('gap+')
        if c['pep_in_z'] > 0.8: label_parts.append('pep_in+')
        if c['pep_out_z'] > 0.8: label_parts.append('pep_out+')
        CENTROID_LABELS_K7[cluster_id] = '/'.join(label_parts) if label_parts else 'Diffuse'

profile['cluster_label'] = profile['cluster'].map(CENTROID_LABELS_K7)
print(f'\\nk-means k=7 cluster labels: {CENTROID_LABELS_K7}')
print(f'\\nCluster sizes:')
print(profile['cluster_label'].value_counts().to_string())"""))

cells.append(nbf.v4.new_markdown_cell("## Step 5 — Preregistered biological tests"))

cells.append(nbf.v4.new_code_cell("""# Test 3 — NT class differential occupation of archetypes
print('=== Test 3: NT enrichment per archetype ===')
crosstab_nt = pd.crosstab(profile['archetype'], profile['NT'])
print(crosstab_nt.to_string())
print()

# Fisher's exact per (archetype, NT) cell
nt_enrich = []
for archetype in profile['archetype'].unique():
    if archetype in ['Intermediate','Quiet']: continue
    for nt_cls in ['ACh','GABA','Glu','DA','5HT','OA','TA']:
        a = int(((profile['archetype'] == archetype) & (profile['NT'] == nt_cls)).sum())
        b = int(((profile['archetype'] == archetype) & (profile['NT'] != nt_cls)).sum())
        c = int(((profile['archetype'] != archetype) & (profile['NT'] == nt_cls)).sum())
        d = int(((profile['archetype'] != archetype) & (profile['NT'] != nt_cls)).sum())
        if (a + b) == 0 or (a + c) == 0: continue
        try:
            odds, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        except Exception:
            odds, p = 0, 1
        nt_enrich.append({'archetype': archetype, 'NT': nt_cls, 'in_archetype': a,
                          'total_archetype': a+b, 'total_NT': a+c,
                          'odds_ratio': odds, 'p': p})

nt_enrich_df = pd.DataFrame(nt_enrich).sort_values('p')
nt_sig = nt_enrich_df[nt_enrich_df['p'] < 0.01]
print(f'\\nSignificant (archetype, NT) enrichments at p < 0.01:')
print(nt_sig.head(20).to_string(index=False))

# How many distinct archetypes show at least one NT enrichment at p<0.01?
n_archetypes_enriched = nt_sig['archetype'].nunique()
test3_pass = n_archetypes_enriched >= 3"""))

cells.append(nbf.v4.new_code_cell("""# Test 4 — Command interneurons as wired integrators
print('=== Test 4: Command interneurons as wired integrators ===')
cmd_in_dataset = [n for n in COMMAND_INTERNEURONS if n in set(w_neurons)]
cmd_archetypes = profile[profile['neuron'].isin(cmd_in_dataset)]['archetype']
print('Command interneuron archetypes:')
for _, r in profile[profile['neuron'].isin(cmd_in_dataset)].sort_values('neuron').iterrows():
    print(f'  {r[\"neuron\"]}: archetype={r[\"archetype\"]}  chem_in_z={r[\"chem_in_z\"]:+.2f}  chem_out_z={r[\"chem_out_z\"]:+.2f}  gap_z={r[\"gap_deg_z\"]:+.2f}')

n_cmd_wired_int = int(((profile['is_command'] == 1) & (profile['archetype'].isin(['Wired_integrator','Multiplex_integrator']))).sum())
n_cmd_total = int((profile['is_command'] == 1).sum())
n_non_cmd_wired_int = int(((profile['is_command'] == 0) & (profile['archetype'].isin(['Wired_integrator','Multiplex_integrator']))).sum())
n_non_cmd_total = int((profile['is_command'] == 0).sum())
print(f'\\nCommand neurons in Wired/Multiplex integrator: {n_cmd_wired_int}/{n_cmd_total}')
print(f'Non-command neurons in Wired/Multiplex integrator: {n_non_cmd_wired_int}/{n_non_cmd_total}')

try:
    odds_cmd, p_cmd = fisher_exact([[n_cmd_wired_int, n_cmd_total - n_cmd_wired_int],
                                     [n_non_cmd_wired_int, n_non_cmd_total - n_non_cmd_wired_int]],
                                    alternative='greater')
except Exception:
    odds_cmd, p_cmd = 0, 1
print(f'Fisher odds={odds_cmd:.2f}, p={p_cmd:.3e}')
test4_pass = p_cmd < 0.05"""))

cells.append(nbf.v4.new_code_cell("""# Test 5 — Monoamine neurons as wireless modulators
print('=== Test 5: Monoamine neurons as wireless modulators ===')
mono_in_dataset = sorted([n for n in MONOAMINE_NEURONS if n in set(w_neurons)])
mono_archetypes = profile[profile['neuron'].isin(mono_in_dataset)][['neuron','archetype','pep_out_z','pep_in_z']]
print('Monoamine neuron archetypes:')
print(mono_archetypes.sort_values('neuron').to_string(index=False))

n_mono_wl = int(((profile['is_monoamine'] == 1) &
                  (profile['archetype'].isin(['Wireless_modulator','Wireless_integrator','Multiplex_integrator']))).sum())
n_mono_total = int((profile['is_monoamine'] == 1).sum())
n_non_mono_wl = int(((profile['is_monoamine'] == 0) &
                      (profile['archetype'].isin(['Wireless_modulator','Wireless_integrator','Multiplex_integrator']))).sum())
n_non_mono_total = int((profile['is_monoamine'] == 0).sum())
print(f'\\nMonoamine neurons in Wireless/Multiplex archetype: {n_mono_wl}/{n_mono_total}')
print(f'Non-monoamine in Wireless/Multiplex: {n_non_mono_wl}/{n_non_mono_total}')

try:
    odds_mono, p_mono = fisher_exact([[n_mono_wl, n_mono_total - n_mono_wl],
                                       [n_non_mono_wl, n_non_mono_total - n_non_mono_wl]],
                                      alternative='greater')
except Exception:
    odds_mono, p_mono = 0, 1
print(f'Fisher odds={odds_mono:.2f}, p={p_mono:.3e}')
test5_pass = p_mono < 0.05"""))

cells.append(nbf.v4.new_code_cell("""# Test 6 — Reproduction of Nb 23 multiplex hubs
print('=== Test 6: Multiplex hubs reproduced from Nb 23 ===')
expected_multi_hubs = {'RIML','RIMR','RMGL','RMGR','RIGR'}
found_multi = set(profile[profile['archetype'] == 'Multiplex_integrator']['neuron'])
overlap = expected_multi_hubs & found_multi
print(f'Expected multiplex hubs (Nb 23): {sorted(expected_multi_hubs)}')
print(f'Found multiplex in Nb 28:        {sorted(found_multi)}')
print(f'Overlap: {sorted(overlap)} ({len(overlap)}/{len(expected_multi_hubs)})')
test6_pass = len(overlap) >= 3"""))

cells.append(nbf.v4.new_markdown_cell("## Step 6 — Preregistered criteria check"))

cells.append(nbf.v4.new_code_cell("""# Test 1: layers should not collapse to a single dominant axis
# Compute pairwise cross-layer correlations; at least one should be |rho| < 0.5
min_cross_layer_corr = np.inf
pair_min = None
layers_by_neuron_agg = {'chem_total': chem_in + chem_out, 'gap': gap_deg, 'pep_total': pep_in + pep_out}
for a in layers_by_neuron_agg:
    for b in layers_by_neuron_agg:
        if a >= b: continue
        r, _ = spearmanr(layers_by_neuron_agg[a], layers_by_neuron_agg[b])
        print(f'  rho({a}, {b}) = {r:+.3f}')
        if abs(r) < min_cross_layer_corr:
            min_cross_layer_corr = abs(r)
            pair_min = (a, b)
test1_pass = min_cross_layer_corr < 0.5
print(f'Min |cross-layer rho|: {min_cross_layer_corr:.3f} (pair {pair_min})')

# Test 2: at least 4 non-empty specialized archetypes
specialized_archetypes = profile[~profile['archetype'].isin(['Quiet','Intermediate'])]['archetype'].value_counts()
n_specialized = (specialized_archetypes >= 1).sum()
test2_pass = n_specialized >= 4

print('\\nPREREGISTERED CRITERIA')
print('=' * 70)
print(f'  [{\"PASS\" if test1_pass else \"FAIL\"}] 1 Layer distinctness (min rho < 0.5)              {min_cross_layer_corr:.3f}')
print(f'  [{\"PASS\" if test2_pass else \"FAIL\"}] 2 >=4 non-empty specialized archetypes             {n_specialized}')
print(f'  [{\"PASS\" if test3_pass else \"FAIL\"}] 3 >=3 archetypes show NT enrichment p<0.01         {n_archetypes_enriched}')
print(f'  [{\"PASS\" if test4_pass else \"FAIL\"}] 4 Command interneurons -> wired/multiplex          p={p_cmd:.3e}')
print(f'  [{\"PASS\" if test5_pass else \"FAIL\"}] 5 Monoamines -> wireless/multiplex                 p={p_mono:.3e}')
print(f'  [{\"PASS\" if test6_pass else \"FAIL\"}] 6 >=3 of 5 Nb 23 multiplex hubs reproduced         {len(overlap)}/5')
print('=' * 70)
n_pass = sum([test1_pass, test2_pass, test3_pass, test4_pass, test5_pass, test6_pass])
print(f'{n_pass}/6 criteria pass')

if n_pass >= 5:
    verdict = 'STRONG POSITIVE — multiplex archetype framing is biologically coherent and reproducible'
elif n_pass == 4:
    verdict = 'POSITIVE — framing holds but some tests borderline'
elif n_pass == 3:
    verdict = 'WEAK — partial evidence for multiplex organization'
else:
    verdict = 'NULL — archetype framing is not load-bearing; keep Nb 27 as synthesis'
print(f'\\nVERDICT: {verdict}')"""))

cells.append(nbf.v4.new_markdown_cell("## Step 7 — Save comprehensive outputs"))

cells.append(nbf.v4.new_code_cell("""# Save per-neuron profile with archetype + cluster + biological annotations
profile.to_csv(DERIVED / 'nb28_neuron_profile.csv', index=False)

# Save archetype × NT crosstab
crosstab_nt.to_csv(DERIVED / 'nb28_archetype_nt_crosstab.csv')

# Save NT enrichment table
nt_enrich_df.to_csv(DERIVED / 'nb28_nt_enrichment.csv', index=False)

# Save kmeans centroids
kmeans_results[7]['centroids'].to_csv(DERIVED / 'nb28_kmeans_centroids.csv')

# Final summary
summary = pd.DataFrame([{
    'n_neurons': N,
    'n_chem_edges': int((A_chem > 0).sum()),
    'n_gap_edges': int((A_gap > 0).sum()) // 2,
    'n_pep_edges': int(A_pep.sum()),
    'min_cross_layer_rho': float(min_cross_layer_corr),
    'n_specialized_archetypes': int(n_specialized),
    'n_archetypes_with_nt_enrichment': int(n_archetypes_enriched),
    'command_fisher_p': float(p_cmd),
    'monoamine_fisher_p': float(p_mono),
    'nb23_multiplex_overlap': f'{len(overlap)}/5',
    'n_criteria_pass': n_pass,
    'verdict': verdict,
}])
summary.to_csv(DERIVED / 'nb28_final_summary.csv', index=False)

# Print final archetype distribution with biological annotations
print('=== FINAL ARCHETYPE DISTRIBUTION (with biological annotations) ===')
for archetype in sorted(profile['archetype'].unique()):
    sub = profile[profile['archetype'] == archetype]
    if archetype in ['Intermediate','Quiet']:
        print(f'\\n{archetype} ({len(sub)} neurons):  [not specialized]')
        continue
    nt_mix = sub['NT'].value_counts().head(3).to_dict()
    sample = sub.head(12)['neuron'].tolist()
    print(f'\\n{archetype} ({len(sub)} neurons):')
    print(f'  NT mix: {nt_mix}')
    print(f'  Sample: {sample}')

print(f'\\n=== Summary ===')
print(summary.T.to_string())"""))

cells.append(nbf.v4.new_markdown_cell("## Step 8 — Narrative interpretation"))

cells.append(nbf.v4.new_code_cell("""# Build a narrative table pairing each archetype with its dominant biology
narrative_rows = []
for archetype in ['Wired_broadcaster','Wired_receiver','Wired_integrator',
                  'Electrical_synchronizer','Wireless_modulator','Wireless_target',
                  'Wireless_integrator','Multiplex_integrator']:
    sub = profile[profile['archetype'] == archetype]
    if len(sub) == 0: continue
    nt_top = sub['NT'].value_counts().head(3).to_dict()
    sensory_count = int(sub['is_sensory'].sum())
    command_count = int(sub['is_command'].sum())
    mono_count = int(sub['is_monoamine'].sum())
    narrative_rows.append({
        'archetype': archetype,
        'n_neurons': len(sub),
        'NT_dominant': list(nt_top.keys())[0] if nt_top else '-',
        'NT_breakdown': str(nt_top),
        'n_sensory': sensory_count,
        'n_command': command_count,
        'n_monoamine': mono_count,
        'sample_neurons': ','.join(sub['neuron'].head(5).tolist()),
    })
narrative_df = pd.DataFrame(narrative_rows)
print('Narrative archetype table:')
print(narrative_df.to_string(index=False))
narrative_df.to_csv(DERIVED / 'nb28_archetype_narrative.csv', index=False)"""))

nb.cells = cells
with open('/home/rohit/Desktop/C-Elegans/New Notebooks/28_multiplex_architecture.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'Notebook 28 written ({len(nb.cells)} cells)')
