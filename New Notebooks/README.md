# New Notebooks — C. elegans Connectome + Expression Analysis Pipeline

Complete rebuild of the C. elegans analysis, with preregistered criteria on every test.

## Main finding

**Paper 1 thesis:** Topological rules, refined by gene expression, govern developmental edge arrival in the C. elegans connectome.

- Topology-only AUC: **0.758 [0.746, 0.770]**
- Topology + genes AUC: **0.802 [0.793, 0.811]**
- Delta (genes contribution): **+0.044 [+0.032, +0.056]**

See `PAPER_OUTLINE.md` for the full draft structure.

## Notebook index

| Notebook | Subject | Verdict |
|---|---|---|
| 01 | Clean connectome (Witvliet 2020) | **PASS** (infrastructure) |
| 02 | CeNGEN expression alignment | **PASS** (infrastructure) |
| 03 | Neuron-level gene × motif | NULL (pseudorep) |
| 03b | Class-level gene × motif | **NULL** (0/60k FDR) |
| 04 | Synthetic dynamics + classification | partial null |
| 05 | Developmental rewiring (AUC) | weak null |
| 05b | Gene stability on rewiring | NULL (tied with null) |
| 06 | L-R edge prediction | NULL above contact |
| **07** | **Developmental topological rule** | **POSITIVE AUC 0.76** |
| 08 | Sex dimorphism / cross-dataset | NULL (methods) |
| **09** | **Topology + genes for rewiring** | **KEY POSITIVE AUC 0.80** |
| 10 | Gap junction arrival | NULL (0.60) |
| 11 | Gene stability above topology | (see final summary) |
| **12** | **Peptide wireless connectome** | **POSITIVE (structurally distinct)** |
| **13** | **Signed connectome motifs** | **POSITIVE (6/8 classes)** |
| 14 | Yemini phenotype enrichment | class pos / hub null |
| 15 | Human-ortholog conservation | descriptive null |
| 16 | Topology+genes+peptide | weak positive (<0.01) |
| **17** | **Cross-stage generalization** | **POSITIVE (generalizes)** |
| **18** | **Bootstrap CIs on key AUCs** | **BOTH PASS** |
| 99 | Paper synthesis table | — |

## Reproducing from scratch

```bash
cd "New Notebooks"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ml
for N in 01 02 03 03b 04 05 05b 06 07 08 09 10 11 12 13 14 15 16 17 18 99; do
  # Each notebook file is named like NN_topic.ipynb — find the one with the leading number
  f=$(ls *NN_*.ipynb 2>/dev/null | head -1)  # pseudocode: match by prefix
  [ -z "$f" ] || jupyter nbconvert --to notebook --execute --inplace "$f" --ExecutePreprocessor.timeout=3600
done
```

Each notebook:
1. Imports from `lib/` (shared modules: `connectome.py`, `expression.py`, `reference.py`, `motifs.py`, `lr_compatibility.py`, `paths.py`)
2. Declares preregistered criteria in a top markdown cell
3. Halts with `assert all_pass` if criteria fail (honest nulls)
4. Saves `data_derived/nbXX_final_summary.csv`

## Key library modules

- `lib/paths.py` — canonical project paths
- `lib/connectome.py` — Witvliet 2020 loader (L1–adult), preserves chem/gap layers
- `lib/expression.py` — CeNGEN single-cell thresholded loader + Witvliet neuron alignment
- `lib/reference.py` — Loer & Rand 2022 NT classification
- `lib/motifs.py` — Per-neuron motif features (FFL, cycle3, recip, two-step, clustering) with degree-residualized variants
- `lib/lr_compatibility.py` — Bentley 2016 curated ligand-receptor atlas + canonical binding pairs

## Artifacts under `data_derived/`

- `connectome_adult.npz` — canonical adult adjacency
- `developmental/connectome_{L1_1..adult}.npz` — all 7 developmental stages
- `expression_tpm.npz` — aligned (neuron × gene) TPM matrix
- `motif_features.csv` (per-neuron), `motif_features_per_class.csv` (per CeNGEN class)
- `nt_reference_loer_rand_2022.csv`
- `nb07_pooled_candidates.csv` — 98,718 candidate edge-arrival pairs
- `nb12_peptide_adjacency.npz` — A_peptide matrix
- Plus per-notebook final summaries and intermediate results

## Contact / attribution

This rebuild was done autonomously by Claude Code / Opus 4.7 on 2026-04-17, guided by Rohit Ravi. The rebuild followed a strict discipline of preregistered criteria with explicit halting rules, and transparent documentation of both positive findings and nulls.
