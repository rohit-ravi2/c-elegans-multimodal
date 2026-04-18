# Paper 1 — Draft Outline (v2, post-Nb 20–23)

**Working title:** A three-layer mechanistic decomposition of chemical synapse formation in the *C. elegans* connectome: physical contact, topological refinement, and gene-expression specificity.

**Target venue:** PLOS Computational Biology (primary), Network Neuroscience, or Cell Reports. Potentially Nature Communications given the newer AUC 0.89 headline.

## 1-paragraph abstract (v2)

We decompose chemical synapse formation in the *C. elegans* connectome into three mechanistically distinct and independently-contributing layers. Using Witvliet et al. 2020's developmental series (L1 through adult) paired with the Brittin/Zhen nerve-ring EM contact matrix and the CeNGEN L4 neuron-class expression atlas, we frame edge formation as an arrival task: given the connectome at stage *t*, can we predict which chemical synapses arrive by stage *t+1*? **Physical contact alone predicts arrival at AUC 0.833 [95% CI 0.822–0.843]** — the strongest single determinant. Adding chemical-graph topology (shared output partners, triadic closure) raises this to **AUC 0.874 [0.863–0.884]** (Δ = +0.041 [+0.035, +0.048]). Adding CeNGEN gene expression raises it further to **AUC 0.890 [0.881–0.898]** (Δ = +0.016 [+0.010, +0.021]). All three layers contribute independently, with 95% CIs on every delta excluding zero. The mechanism hierarchy — physical adjacency dominates (0.83), topology adds specificity (+0.04), transcription adds refinement (+0.02) — maps cleanly to developmental biology: axon guidance, activity-dependent plasticity, and terminal-differentiation gene programs. The rule is chemical-synapse-specific: electrical (gap junction) arrival is at chance (AUC 0.60), and the peptide-wireless signaling graph is structurally distinct from the synaptic graph (Jaccard 0.045), with 5 textbook state-integrator neurons (RIM, RMG, RIG) identified as multiplex hubs. Static gene-motif correlations at class level (N=84) universally fail BH-FDR correction, consistent with the view that gene expression is not the primary architect of *C. elegans* wiring but a modulatory refinement on top of contact and topology.

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

**2.2 Physical contact is the dominant predictor.** (Figure: contact-only AUC + distribution of contact areas)
- **Nb 20: Contact-only AUC 0.833** [0.822, 0.843], single strongest predictor of edge arrival.
- Brittin/Zhen nerve-ring EM contact matrix: 161 of the 185 common neurons covered.
- Interpretation: neurons must physically touch in the nerve ring to form a chemical synapse. This anatomical constraint captures most of the wiring signal.

**2.3 Local topology adds specificity above contact.** (Figure: nested model AUC panels)
- **Nb 22: Δ topology above contact = +0.041** [+0.035, +0.048], CI excludes zero.
- Triadic closure (+0.26) and shared output partners (+0.31) are the strongest coefficients (Nb 07).
- Negative in-degree coefficient (−0.11): already-popular targets less likely to gain edges (anti-preferential-attachment).
- Contact + topology → AUC **0.874** [0.863, 0.884].

**2.4 Gene expression adds further refinement.** (Figure: full 3-layer stack + delta CIs)
- **Nb 22: Δ genes above contact+topology = +0.016** [+0.010, +0.021], CI excludes zero.
- Full three-layer model: **AUC 0.890** [0.881, 0.898].
- Permutation null (shuffled gene features): Δ = −0.022.
- Nb 17: rule generalizes across developmental stages (mean cross-stage AUC 0.77, degradation 0.01).

**2.5 Rule is chemical-synapse-specific.** (Figure: parallel gap junction / peptide panels)
- Nb 10: gap junction arrival at AUC 0.60 — essentially null.
- **Nb 19: gap-junctions at stage t do NOT predict chemical arrival at t+1** (Δ +0.008, fails preregistered bar +0.02). The mammalian "electrical scaffold" hypothesis does not obviously apply to C. elegans at this resolution.
- Nb 12: peptide wireless connectome is structurally independent (Jaccard 0.045 with synaptic, different hubs, higher clustering).
- Nb 16: adding peptide features to contact+topology+genes gives +0.002 AUC (not distinguishable from null).

**2.6 Multiplex hubs are textbook state-integrator neurons.** (Figure: scatter plot of synaptic vs peptide degree + labeled multiplex hubs)
- **Nb 23: 5 neurons in top 10% of both graphs — RIM L/R, RMG L/R, RIGR.**
- RIM: classical locomotion state-integrator (Coe 2018).
- RMG: social/pheromone integrator via NPR-1 (Macosko 2009).
- Independent data-driven identification of the biology — convergent validation of Nb12's two-graph separation.

**2.7 Signed-motif structure reveals coherent excitation and dual-inhibition gating.** (Figure: 8-panel sign-class motif enrichment)
- Nb 13: 6 / 8 sign classes show significant deviation from sign-shuffled null.
- Coherent (+,+,+) FFL enriched at z = +3.6.
- Dual-inhibition (−,+,−) motif enriched at z = +28 — a massively over-represented "fail-safe silencing" pattern.
- **Nb 21 honest caveat**: the (−,+,−) motif is so rare (37 total, concentrated in ≤2 neurons at class level) that class-level gene correlation fails due to N limits, not sign-ambiguity.

**2.8 Individual-edge reliability is limited but motif statistics are robust.** (Supplementary)
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
