# Reading List

Prioritized literature. Tier 1 should be read before any new code is written;
Tier 2 before drafting the related work section; Tier 3 is reference / depth.

For each, the second line is the one-sentence reason it matters to this
project.

---

## Tier 1 — Read in Weeks 1–2

### Lv et al. (2023). *Adaptive Sparse Softmax: An Effective and Efficient Softmax Variant for Text Classification.* IEEE TASLP.
The paper this project extends. Read the method (§3–§5), the proofs (the
margin bound), and the experimental table. Note the warm-up ratio $r$ and
the AS-Speed trick — these are reusable infrastructure for us.

### Wang et al. (2018). *Additive Margin Softmax for Face Verification.* IEEE Signal Processing Letters.
Closest sibling work — additive margin in angular space. We're the
probability-space analogue. Useful to articulate the difference in the
related-work section.

### Martins & Astudillo (2016). *From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification.* ICML.
First sparse softmax. Read for the projection interpretation (gives us a
clean alternative framing of what sparsity means).

### Peters et al. (2019). *Sparse Sequence-to-Sequence Models.* ACL.
Entmax-$\alpha$ family. Read for the Tsallis-entropy connection (potential
appendix-level theory link).

---

## Tier 2 — Read in Weeks 3–5

### Wang et al. (2017). *NormFace: $\ell_2$ Hypersphere Embedding for Face Verification.* ACM MM.
Background on feature normalization. Relevant to feature-norm-based sample
margins.

### Liu et al. (2017). *SphereFace: Deep Hypersphere Embedding for Face Recognition.* CVPR.
Multiplicative angular margin — the original of the family.

### Deng et al. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR.
Most widely-used angular margin loss. Standard comparator citation.

### Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.
Per-sample reweighting for hard example mining. Spiritually similar to
sample-dependent margins; useful contrast point in related work.

### Szegedy et al. (2016). *Rethinking the Inception Architecture for Computer Vision.* CVPR.
The label smoothing paper. Standard regularizer baseline.

### Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
The calibration paper. Defines ECE, reliability diagrams, and the
overconfidence problem we partly address.

### Zhou et al. (2022). *Learning Towards the Largest Margins.* ICLR.
Class-global margin learning — closest existing work on adaptive class
margins, but only one-axis (no sample, no time). Important to position
against.

### Xie et al. (2024). *X2-Softmax: Margin Adaptive Loss Function.* Expert Systems with Applications.
Angle-adaptive margin in face recognition. Same goal (adaptive margins),
different domain and parameterization.

---

## Tier 3 — Read while writing (Weeks 12–16)

### Bengio et al. (2009). *Curriculum Learning.* ICML.
The original curriculum-learning paper. Cite when motivating $\delta(\tau)$
schedules.

### Hu et al. (2019). *Multi-class Classification without Multi-class Labels.* ICLR.
Background on label structure and pairwise classification.

### Liu et al. (2016). *Large-Margin Softmax Loss for Convolutional Neural Networks.* ICML.
L-Softmax — the ancestor of the angular margin family.

### Cao et al. (2019). *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss.* NeurIPS.
LDAM — class-dependent margin for imbalance. Different motivation (imbalance,
not confusability), but the math is closely related.

### Zhang & Sabuncu (2018). *Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels.* NeurIPS.
Noise-robust loss. Useful for our label-noise robustness section.

### Bartlett et al. (2017). *Spectrally-normalized margin bounds for neural networks.* NeurIPS.
Background on margin-based generalization bounds. Relevant if we go after T5.

### Neyshabur et al. (2018). *A PAC-Bayesian Approach to Spectrally-Normalized Margin Bounds for Neural Networks.* ICLR.
Same; PAC-Bayes framing.

### Bridle (1989). *Probabilistic Interpretation of Feedforward Classification Network Outputs.*
Historical origin of softmax. Worth scanning for the abstract framing.

---

## Tier 3+ — Reference / depth (read selectively)

### Convexity, optimization, gradient flow

- Boyd & Vandenberghe (2004). *Convex Optimization.* — Reference for
  convexity claims about CE loss.
- Su et al. (2014). *A Differential Equation for Modeling Nesterov's Accelerated
  Gradient Method.* — Background on gradient flow as continuous limit.

### Calibration

- Naeini et al. (2015). *Obtaining Well Calibrated Probabilities Using Bayesian Binning.* AAAI. — ECE definition.
- Müller et al. (2019). *When Does Label Smoothing Help?* NeurIPS. — Why label smoothing helps calibration.

### Low-rank / matrix factorization theory

- Candès & Recht (2009). *Exact Matrix Completion via Convex Optimization.* — Low-rank recovery foundations.
- Tropp (2015). *An Introduction to Matrix Concentration Inequalities.* — Tools for T2's identifiability proof.

---

## What to read about Power Softmax (so it's a fair baseline)

The `Beyond_Softmax_Research_Notes.md` is the source. The math is mostly
self-contained, but check:

- Boyd & Vandenberghe §3 on positive-homogeneous functions (Power Softmax is
  one).
- Any recent work on **polynomial softmax** or **monomial softmax** — search
  arxiv for these terms. The idea is folkloric but lightly published.

---

## How to actually read all this

- Tier 1: read in full (3–5 hours each). Take notes.
- Tier 2: read intro + method (1 hour each). Skim experiments.
- Tier 3: scan abstract + figures. Re-read what's relevant when writing.

A spreadsheet of "what I cited and what for" saves time at writing —
make it as you go.

---

## arxiv search queries that surfaced new relevant work

(Run these periodically — new papers appear during your project.)

- `softmax margin classification`
- `adaptive margin loss`
- `class-dependent margin`
- `sparse softmax variant`
- `softmax calibration`
- `margin-based loss generalization bound`
- `temperature schedule softmax`
- `regression softmax binned`

If you find something published *after May 2026*, check whether it scoops
any direction. If it does, the response is almost always "incorporate as a
baseline + reframe to distinguish" — not "give up."
