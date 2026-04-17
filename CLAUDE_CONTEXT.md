# Rohit's C-Elegans Research — Context for Claude

**Last updated:** 2026-04-17
**Purpose:** Paste as custom instructions in a claude.ai Project, or upload as a knowledge file, so any web chat has full context on this work without re-explaining.

---

## Who is Rohit

- NYU undergraduate, Data Science major, Philosophy minor. Age 22.
- Career direction: AI that bridges technical and philosophical domains.
- Intellectual interests: neural science, philosophy of mind, consciousness studies, quantum computing, Vedanta.
- Foundations: linear algebra, calculus, probability/statistics. Strong Python/PyTorch.
- Hardware constraint: RTX 4060 Ti 8GB VRAM. For ML work, assume tight memory — bf16/fp16, gradient checkpointing, smaller batches with accumulation.
- Welcomes non-dualist / Vedantic framings when they sharpen technical work. Avoid ideological overlays (e.g., feminist frameworks) in historical/textual analysis unless invited.

## How Rohit wants to collaborate

- **Plan first, execute second** on non-trivial tasks. Scoped approvals are fine ("do Phase A and B together").
- **No sugarcoating on assessments.** Give honest calibration of work quality at the right career stage.
- **Rigor over brevity** for technical/academic work. Full-credit-quality reasoning.
- Prefer **topic/function-based organization** for research data, not pipeline-stage.

---

## The C-Elegans Research Project

### Thesis (Paper1MasterNotebook)

> **Local connectomic topology in *C. elegans* is largely self-determining, while gene expression acts as a modulatory refinement layer rather than a primary architect of wiring.**

Multi-modal triangulation across connectivity, neural dynamics, and behavioral role prediction.

### Key results (executed, reproducible)

| Model | Target | Result |
|---|---|---|
| GNCA (full) | Outgoing synaptic strength | **r = 0.987, MAE = 0.097** |
| Shuffled expression | same | r = 0.934 (topology carries signal) |
| Graph-only | same | r = 0.861 |
| Expression-only | same | r = 0.27 |
| 5-fold gene → motif participation | — | r = [0.25, 0.62, 0.82, 0.86, 0.64] |
| Gene → 3-class behavioral role | — | **92.7% accuracy** |
| CCA (expression ↔ motif space) | top 3 | ρ = [0.875, 0.836, 0.809] |
| Conditional diffusion p(Motif \| Expr) | — | r = 0.397 |
| PhaseD dynamics | gene-shuffle | `gene_causal_by_shuffle: False` (null result) |

### Assessment

**Legit research, unfinished.** Methodology is at early-PhD level (shuffle/null/ablation discipline, multi-modal triangulation, clean paper architecture with explicit limitations section). Above average for a 22yo undergrad — independent hypothesis, real results.

**Strongest publishable angles:**
1. **Main paper** on topology-dominant, gene-modulatory wiring (eLife, PLOS Comp Bio, Network Neuroscience, Nature Communications).
2. **Methods paper** on GNCA for connectome prediction with shuffle controls (NeurIPS/ICLR workshops — GRL, LMRL, NeSy).
3. **Short communication** on gene→behavioral role at 92.7% accuracy.
4. **Honest-null paper** on PhaseD dynamics result.

**Gaps before submission (~2–4 weeks focused work):**
- Section 5 combined (structure+expression) models — empty cells in paper notebook.
- R²=1.0 leakage in motif-masked baselines (permutation also hits 1.0 — confirms leakage).
- Gene enrichment incomplete — crashed on lineage `KeyError: 'Neuron'` / `'Lineage'` (schema drift).
- No bootstrap CIs on N=91 neurons.
- Cross-species Caenorhabditis + parasitic genomes loaded but not analyzed — potential follow-up paper.

### Data sources in the project

- **Connectome**: White (Mind of a Worm) CSVs + xls, Cook 2019 SI 2-10, Witvliet 2020 (8 developmental timepoints L1→adult), Varshney/Chklovskii 2011 (NIHMS496349), Adult + L4 nerve ring neighbors.
- **Gene expression**:
  - CeNGEN (primary): single-cell L4 (.qs R objects), Seurat+monocle .rds (4.5G), Barrett 2022 bulk RNA-seq, aggregated/concordance/gold tables, thresholded TPM.
  - Other scRNA-seq: GSE136049 (Packer/Zhu 2019), Cao et al 2017, GSE98561, GSE126954.
  - Bentley 2016, Yemini neuropeptide atlas, Loer & Rand 2022 neurotransmitters.
- **Genomes**: 29 Caenorhabditis + parasitic species (204 FASTA, 71 annotations). C. elegans WBcel235 NCBI bundle (GCA + GCF).
- **WormBase WS297 release** (~80G): database tars, .wrm blocks, .wib tracks, .wb associations, wormpep.sql, ortholog tables.
- **Neural activity traces** (Kato et al 2015, PNAS 1507110112): 15 CSVs, 102 timepoints × hundreds of neurons.
- **Paper supplements**: NIHMS496349, NIHMS2095963, pnas.1507110112, aam8940 (Cao 2017), 41467_2025, eLife 2023.
- **OpenWorm c302 simulation**: 71 Python files + 691 NeuroML files.

### Project directory layout (post-reorg 2026-04-17)

**Root:** `~/Desktop/C-Elegans/` (symlinked to `/mnt/ssd4tb/Desktop/C-Elegans/`). ~123G, 2247 files.

```
C-Elegans/
├── data/          977 files, 122G     # all biological data by source
│   ├── connectome/{white_MoW,cook2019,witvliet2020,varshney2011,nerve_ring_neighbors}/
│   ├── expression/
│   │   ├── cengen/{single_cell_L4,single_cell_rds,bulk_barrett2022,aggregated,concordance,gold,thresholded,derived}/
│   │   ├── scrna_seq_other/{GSE136049,cao2017,GSE98561,GSE126954}/
│   │   └── {bentley2016,neurotransmitter,imaging}/
│   ├── genomes/
│   │   ├── c_elegans_wbcel235/ncbi_dataset/
│   │   ├── multi_species_fasta/<29 species>/
│   │   ├── multi_species_annot/<29 species>/
│   │   └── {fpkm_controls,sra_expression_tars}/
│   ├── wormbase_release_WS297/        # 80G reference blob
│   ├── lineage/  attributes/  paper_supplements/  third_party_releases/
│
├── analysis/      240 files, 385M     # outputs by phase and topic
│   ├── phase_{A,B,C,D}/                 # by paper section
│   ├── {motif,DE_hubs,gnca,consensus_connectome,markov_hypergraph}/
│   ├── {gene_mapping,connectome_derived,ml_eval,splicing,crispr,ligand_receptor}/
│   └── {parsed_schemas,adjacency,curated,figures}/
│
├── models/         90 files            # {checkpoints,embeddings,logits,nulls,trajectories}
├── simulation/    779 files            # c302_code + cells/networks/synapses/channels NML
├── manifests/      92 files            # {audit,metrics,adjacency_mappings,sim_configs,misc}
├── notebooks/      27 notebooks        # Paper1MasterNotebook.ipynb at top; archive/ categorized
├── utils/paths.py                      # canonical path resolver
├── RESULTS/                            # empty, for paper figures
```

### Canonical file locations (post-reorg)

- Connectome edges canonical: `analysis/gene_mapping/connectome_edges_canonical.csv`
- Neuron metadata canonical: `analysis/gene_mapping/neuron_metadata_canonical.csv`
- Structural targets: `analysis/gene_mapping/structural_targets_{degree_strength,motif_residualized}.csv`
- Expression (neuron mean): `data/expression/cengen/derived/expression_neuron_mean.csv`
- Expression canonical bundle: `data/expression/cengen/derived/expression_canonical/` (matrix.npz + genes.csv + cells.csv)
- GSE136049 raw: `data/expression/scrna_seq_other/GSE136049/`
- Lineage: `data/lineage/NeuronLineage_Part{1,2}.xls`
- Dataset provenance: `manifests/misc/dataset_provenance.json`

### Reorg artifacts

- `.trash_20260417_122139/` — 60 files, 219M — recoverable deletes (dups + 0-byte).
- `_reorg_manifest_20260417_122139.log` — every operation logged.
- Every patched notebook has a `.bak` (Phase C.1) or `.bak2` (Phase C.2/3) backup.

---

## Separate Project: hsde / recursive-research-factory

**Location:** `~/Desktop/Projects/oldproject/recursive-research-factory/hsde/` (NOT part of C-Elegans tree).

Research prototypes — ambitious ideas, unfinished:
- `ASI.ipynb` — "Roadmap to Superintelligence": CA + transformer attention + symbolic rule mining hybrid. Phases 4–5 empty.
- `esde2.ipynb` — Enhanced Symbolic Discovery Engine for AIME/SAT/GSM8K (Phi-3 → SymPy/Z3 → FAISS memory). 6 errors end-to-end.

Publishable angle if pursued: ASI Phases 1–3B as workshop paper (ICLR ME-FoMo, NeurIPS SRI, NeSy).

---

## For future Claude chats

When Rohit mentions:
- **"the paper" / "Paper 1" / "the master notebook"** → `notebooks/Paper1MasterNotebook.ipynb` (C-Elegans topology-vs-gene-expression thesis).
- **"the reorg"** → the 2026-04-17 topic-based reorganization; paths are in `utils/paths.py` and layout section above.
- **"CeNGEN"** → single umbrella at `data/expression/cengen/` — 76 files, 5.7G across 8 subfolders.
- **"the NCA work" / "GNCA"** → `notebooks/archive/nca/NCA.ipynb` and `analysis/gnca/` outputs.
- **"hsde" or "ASI"** → separate project in `~/Desktop/Projects/oldproject/`, not in C-Elegans.

Don't trust old path strings from prior chats — the canonical layout is above. For any file reference, resolve by basename in the reorganized tree.
