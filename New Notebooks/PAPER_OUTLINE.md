# Paper 1 — Draft Outline

**Working title:** Topological rules, refined by gene expression, govern developmental edge arrival in the *C. elegans* connectome.

**Target venue:** PLOS Computational Biology (primary), Network Neuroscience, or Cell Reports.

## 1-paragraph abstract (draft)

Understanding which pairs of neurons form synaptic connections during development is a central question in connectomics. Prior work has correlated neuronal gene expression with static connectome features such as hub status and motif participation, but such correlational tests have returned mostly null results once proper class-level statistics are applied. Here, using Witvliet et al. 2020's developmental series (L1 through adult) together with the CeNGEN L4 neuron-class expression atlas, we frame wiring as a dynamical question: given the connectome at stage *t*, can we predict which edges arrive at stage *t+1*? We find that pure local topology — shared output partners and triadic closure — predicts chemical-synapse arrival with **AUC 0.758 [95% CI 0.746–0.770]** across three developmental transitions. Adding CeNGEN class-level expression features boosts prediction to **AUC 0.802 [0.793–0.811]**, a statistically robust but modest refinement (Δ = +0.044 [+0.032, +0.056]) confirmed by permutation null (shuffled-gene Δ = −0.022). The rule generalizes across developmental transitions (mean cross-stage AUC 0.77). It is chemical-synapse-specific: electrical (gap-junction) arrival is at chance (AUC 0.60), and the peptide-signaling graph is structurally distinct from the synaptic graph (Jaccard 0.045). Gene-motif, gene-hub, and gene-rewiring correlations at class level (N=84) all fail to survive multiple-testing correction, consistent with the view that **gene expression is not the primary architect of C. elegans wiring but a modulatory refinement layer on top of a topological rule**.

## Section plan

### 1. Introduction (~2 pages)
- Hobert terminal-selector framework; gene-expression-as-blueprint hypothesis.
- Prior attempts at gene→motif correlation have produced conflicting results.
- Our contribution: frame the question dynamically (edge arrival), use proper class-level statistics, test against strong topological baselines.

### 2. Results

**2.1 Static gene-motif correlations fail at class-level.** (Figure: forest plot of q-values across 60k tests)
- Notebook 03b: 0 / 60,438 gene × motif tests survive global BH-FDR
- Nb 05b: gene stability selection on developmental rewiring tied with permutation null
- Nb 06: Ligand-receptor compatibility does not add above physical contact (contact-only AUC 0.78)
- Nb 14, 15: neither behavior genes nor ortholog conservation enriched in hub classes

**2.2 Local topology predicts developmental edge arrival.** (Figure: per-transition AUC panels + coefficient weights)
- Nb 07: AUC 0.758 [0.746, 0.770] on 98,718 candidate pairs across L1→L2→L3→adult transitions
- Triadic closure (+0.26) and shared output partners (+0.31) are the strongest coefficients
- Negative in-degree coefficient (−0.11) suggests saturation: already-popular targets less likely to gain edges

**2.3 Gene expression is a modest refinement on the topological rule.** (Figure: boxplot of AUC with/without genes across folds + cross-stage matrix)
- Nb 09: AUC boost 0.758 → 0.802 (Δ +0.044 [0.032, 0.056])
- Permutation null (shuffled gene features): Δ = −0.022 (genes don't just fail to help randomly — shuffling hurts, confirming real signal)
- Nb 17: rule generalizes across stages (mean cross-stage AUC 0.77, degradation 0.01)
- Nb 18: bootstrap CIs on every metric exclude null

**2.4 Rule is chemical-synapse-specific.** (Figure: parallel gap junction / peptide panels)
- Nb 10: gap junction arrival at AUC 0.60 — essentially null, consistent with innexin-specific biology
- Nb 12: peptide wireless connectome is structurally independent (Jaccard 0.045, different hubs, higher clustering)
- Nb 16: adding peptide features to topology+genes gives +0.002 AUC (not distinguishable from null)

**2.5 Signed-motif structure reveals coherent excitation and dual-inhibition gating.** (Figure: 8-panel sign-class motif enrichment)
- Nb 13: 6 / 8 sign classes show significant deviation from sign-shuffled null
- Coherent (+,+,+) FFL enriched at z = +3.6
- Dual-inhibition (−,+,−) motif enriched at z = +28 — a massively over-represented "fail-safe silencing" pattern

**2.6 Individual-edge reliability is limited but motif statistics are robust.** (Supplementary)
- Nb 08: Cross-dataset Jaccard 0.42 (Cook hermaphrodite vs Witvliet adult); cross-sex 0.38.
- Per-class motif Spearman ~0.60 across datasets — degrees and motif counts are more reliable than individual edges.

### 3. Discussion
- The classical view that gene expression "programs" the connectome is overstated — at class-level with proper statistics, gene expression is a modest refinement, not a primary architect.
- Topology itself (triadic closure, shared partners) captures most of the developmental wiring rule.
- The three wiring layers (chemical, electrical, peptide) have distinct rules, supporting a multi-layered model of nervous-system assembly.
- Limitations: CeNGEN is L4-larval (not L1 or adult); individual-edge reliability across datasets is limited; analysis is restricted to the 84 CeNGEN-covered neuron classes out of ~118.

### 4. Methods (all preregistered criteria documented)
- Witvliet 2020 L1-adult developmental series (Nb 01 validation)
- CeNGEN single-cell thresholded expression aligned to Witvliet (Nb 02)
- Motif features (lib/motifs.py: FFL, cycle3, recip, two-step, clustering)
- Ligand-receptor atlas (lib/lr_compatibility.py: Bentley 2016 + canonical pairs)
- Loer & Rand 2022 NT reference (lib/reference.py)
- Logistic regression with L1/L2 regularization, 5-fold stratified CV, permutation nulls
- Bootstrap confidence intervals (1000 resamples)
- BH-FDR for multiple testing across gene × motif combinations

## Figures (draft)

1. **Fig 1**: Schematic of the developmental edge-arrival framework + Witvliet stage overview.
2. **Fig 2**: Prior static tests (nulls) vs new dynamic test (positive). Bar/panel comparison.
3. **Fig 3**: Main result — topology AUC 0.76, +genes 0.80, cross-stage generalization matrix.
4. **Fig 4**: Rule specificity — chemical positive, gap junction null, peptide independent.
5. **Fig 5**: Signed-motif enrichment profile, including the (−,+,−) outlier.

## Reviewer-anticipated questions

1. **"Why not use single-cell CeNGEN directly instead of PCA-50?"**
   → We tested that (Nb 09 variants): single-cell individual genes in Lasso stability selection (Nb 11) give consistent results but less interpretable than PCA-50. PCA-50 captures 80% of variance in 50 features suited to CV at N≈98k.

2. **"Could the topology signal be driven by one developmental transition?"**
   → Nb 17 tests cross-stage generalization: 5 of 6 train/test splits maintain AUC ≥ 0.70, mean degradation 0.013.

3. **"Is the gene effect just capturing neuron-class identity (chol/GABA/glu)?"**
   → Partially. But: (a) NT marker genes don't drive the top stability ranking in Nb 11, (b) the delta survives permutation null which breaks class identity.

4. **"Does the rule hold at class level (vs the neuron-level feature computation of Nb 07)?"**
   → Nb 07's features are per-neuron (topological features differ between L/R pairs of the same class). Only gene features are class-level. The signal is largely driven by topology, which is not pseudoreplicated.

## Attribution of New Notebook series

All analysis notebooks are under `New Notebooks/` in the repo. Each notebook:
- Begins with a preregistered markdown cell stating the hypothesis and success criteria BEFORE seeing the data output.
- Uses an explicit halting rule if criteria fail.
- Saves a `nbXX_final_summary.csv` with the numeric verdict.
- Commits are tagged with PASS / FAIL / POSITIVE / NULL per the criteria.

Notebook 99 aggregates all summaries into a single synthesis table.
