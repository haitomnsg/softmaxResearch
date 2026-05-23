# Paper Outline

The target is an 8-page main paper (ACL / ICLR / NeurIPS format) plus appendix.
This outline is structured so each section maps to specific experiments and
figures from [04_experimental_plan.md](04_experimental_plan.md).

---

## Working title options

In rough order of how readable they are:

1. **GAM-Softmax: Generalized Adaptive Margins for Sparse Classification**
   *(safe, descriptive, NeurIPS-style)*
2. **Beyond Scalar Margins: Class-Pair, Sample, and Temporal Structure in Adaptive Softmax**
   *(ICLR-style, sets up the contribution explicitly)*
3. **One Margin Is Not Enough: Generalizing Adaptive Sparse Softmax**
   *(ACL/EMNLP-style, more provocative)*

Pick the title that matches the venue you submit to first.

---

## Abstract (~200 words)

**Template:**

> Modern classifiers train softmax with cross-entropy, pushing the target
> probability toward 1 even though test-time only requires it to exceed
> competing classes. Adaptive Sparse Softmax (AS-Softmax) resolves this with
> a scalar margin $\delta$, masking already-correct classes from gradient.
> However, the margin is global — the same $\delta$ for every class pair,
> every input, and every training step. We argue this is provably suboptimal:
> classes form similarity clusters, inputs vary in ambiguity, and training
> dynamics evolve. We propose **GAM-Softmax**, which replaces $\delta$ with a
> structured margin $\delta_{t,j}(x, \tau)$ decomposed multiplicatively into
> a low-rank class-pair component, a learned sample-dependent component, and a
> schedule. We derive a generalized margin bound that recovers AS-Softmax as
> a special case, and we show the low-rank parameterization admits a
> generalization bound scaling with $nk$ rather than $n^2$. Empirically, on
> seven text, image, and audio benchmarks, GAM-Softmax improves accuracy by
> X.X% on average and reduces Expected Calibration Error by Y.Y% relative
> over AS-Softmax, with no significant increase in training cost. We further
> demonstrate a regression extension via binned distance-aware margins.

(Numbers are placeholders until experiments finish.)

---

## 1. Introduction (~1 page)

- *Hook.* The train-test mismatch in softmax + cross-entropy.
- *AS-Softmax recap.* What it fixed; the scalar-margin insight.
- *The observation that motivates this paper.* A scalar margin can't capture
  class-pair asymmetry, sample ambiguity, or temporal dynamics.
- *Contributions* (bulleted):
  1. The GAM-Softmax framework (a strict generalization of AS-Softmax).
  2. A low-rank class-pair margin with $\le 2nk$ parameters and a
     generalization bound.
  3. A learned sample-dependent margin from a small head, with a calibration
     story.
  4. A principled schedule analysis.
  5. Empirical validation across text, image, and audio (and regression in
     the appendix).
- *Figure 1.* The big picture: scalar → class-pair → sample → time, with a
  small accuracy/ECE bar chart showing each step's gain.

---

## 2. Related Work (~0.5 page)

Three clusters, in this order:

- **Sparse softmax variants.** Sparsemax, Entmax, Sparse-Softmax, Power Softmax.
  We're not in this cluster, but we're sparse-via-masking.
- **Margin-based losses.** AM-Softmax, ArcFace, CosFace, SphereFace, X2-Softmax,
  AS-Softmax. We extend the AS-Softmax line; we differ from the angular-margin
  line by working in probability space and not requiring feature normalization.
- **Sample reweighting / curriculum.** Focal loss, label smoothing, curriculum
  learning, hard-example mining. We're spiritually related but use a margin
  primitive, not a weight.

Each cluster: 2–3 sentences, point to where we sit.

---

## 3. Method (~1.5 pages)

### 3.1 Background: AS-Softmax

State the scalar-margin formulation, the mask, the margin bound
$\log(n\delta + 1)$. **Re-derive briefly** so the reader doesn't need the
Lv et al. paper open.

### 3.2 GAM-Softmax

State the general definition with $\delta_{t,j}(x, \tau)$. Show how it
recovers AS-Softmax / softmax as special cases (the "no new method does
strictly less than this one" framing).

### 3.3 Three parameterizations

- Class-pair: low-rank (recommended), plus mention of similarity and
  confusion variants.
- Sample: learned head with anchor regularization.
- Time: linear / cosine schedules.

### 3.4 Composition

Multiplicative composition; trade-off vs additive (additive is an ablation).

### 3.5 Multi-label and regression extensions

Brief; details in appendix.

**Figure 2.** Schematic of the three margin axes and how they compose.

---

## 4. Theoretical Analysis (~1 page)

- **Theorem 1 (Generalized margin bound, T1).** State + 4-line sketch.
- **Theorem 2 (Identifiability and capacity of low-rank margins, T2).**
- **Theorem 3 (Convergence under continuous schedules, T3).** Possibly relax
  to soft mask, footnote that hard mask is empirically similar.
- **Proposition 4 (Loss tracks accuracy, T4).** Short. This is the formal
  counterpart of the famous Lv et al. correlation.

Proofs go in the appendix; the main paper has statements + intuition.

---

## 5. Experiments (~2.5 pages)

### 5.1 Setup

- Datasets, backbones, baselines, metrics, hyperparameter protocol, seeds.
- Half a paragraph each.

### 5.2 Main results

- **Table 1.** Headline table — accuracy on all text + image + audio datasets,
  M3 and M10 vs all baselines.
- **Table 2.** Calibration table — ECE on the same datasets.

### 5.3 Ablations

- **Figure 3.** Rank-$k$ sweep curve for M3.
- **Figure 4.** $\delta(\tau)$ schedule comparison.
- **Figure 5.** Per-direction contribution (M3 alone vs M5 alone vs M6 alone
  vs M10).
- **Table 3.** Composition study (multiplicative vs additive).

### 5.4 Diagnostics

- **Figure 6.** Heatmap of learned $M = UV^\top$ on CIFAR-100, showing
  block structure aligned with superclasses.
- **Figure 7.** $\delta(x)$ distribution histogram + example samples at
  high vs low margin.
- **Figure 8.** Pearson $\rho$(loss, val acc) — show GAM-Softmax matches
  or beats AS-Softmax's $-0.95$.

### 5.5 Robustness

- **Table 4.** Label noise (10% flip) results.
- **Table 5.** Few-shot (10%, 25%, 50% data) results.

---

## 6. Discussion (~0.5 page)

- What worked. What didn't (honest about which direction was weaker).
- Limitations: hyperparameters, dataset-dependence, training cost.
- Failure modes observed: e.g., when sample-head over-fits to a feature
  artifact.

---

## 7. Conclusion (~0.2 page)

Short. Restate the contribution; point to future work (regression at scale,
extension to LM heads, theoretical tightening).

---

## Appendix

- **A.** Full proofs (T1–T5).
- **B.** Hyperparameters and training details.
- **C.** Additional ablations that didn't fit.
- **D.** Regression extension results.
- **E.** Multi-label results.
- **F.** Negative results — directions we tried that didn't work, with a
  paragraph each. (Reviewers value this.)
- **G.** Code / reproducibility notes.

---

## Figure / table inventory (so nothing gets forgotten)

| # | Type | Where | Source experiment |
|---|---|---|---|
| F1 | Conceptual schematic | §1 Intro | n/a |
| F2 | Method overview | §3.2 | n/a |
| F3 | Rank-$k$ sweep curve | §5.3 | Ablation 3 |
| F4 | Schedule comparison | §5.3 | Ablation 5 |
| F5 | Per-direction bar chart | §5.3 | M3, M5, M6, M10 |
| F6 | $M$ heatmap | §5.4 | M3 on CIFAR-100 |
| F7 | $\delta(x)$ distribution | §5.4 | M5 on SST-5 |
| F8 | Loss-accuracy correlation | §5.4 | All main runs |
| T1 | Accuracy table | §5.2 | Phase 2–5 |
| T2 | Calibration table | §5.2 | Phase 2–5 |
| T3 | Composition study | §5.3 | Ablation 2 |
| T4 | Label noise | §5.5 | Robustness |
| T5 | Few-shot | §5.5 | Robustness |

---

## Venue calendar (check before committing)

| Venue | Typical cycle | Deadline (approx.) | Notes |
|---|---|---|---|
| NeurIPS | Annual | mid-May | Most prestigious for ML theory + empirical |
| ICLR | Annual | late September | Slightly more theory-friendly; rolling discussion |
| ACL | Annual | mid-October (Feb cycle), mid-February (June cycle) | NLP focus — lead with text results |
| EMNLP | Annual | mid-June | NLP focus; somewhat more applied than ACL |
| AAAI | Annual | mid-August | Broader AI; sometimes more inclusive |

**Recommendation given timeline:** target ICLR (late Sept deadline) as
primary; ACL Feb cycle as fallback if you slip.

---

## What a paper-quality result looks like in numbers

For the main table, aim for these effect sizes (these are not promises;
they're a sanity check on whether the paper has enough signal):

- Accuracy: **+0.3–0.8%** absolute over AS-Softmax on text core.
- ECE: **−15% to −30%** relative.
- Cross-modality: at least non-regression with possible small gains.
- Robustness (label noise): on average **+0.5%** under 10% noise.
- $\rho$(loss, val acc): match or beat AS-Softmax's $-0.95$.

If you're below half these numbers consistently, the paper needs a different
framing (e.g., calibration-only). If you're hitting these, you have a paper.
