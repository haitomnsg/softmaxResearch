# Problem Statement, Gaps, and Research Questions

This document is the **scientific contract** of the project. Everything
downstream — method, theory, experiments — must trace back to a research
question stated here.

---

## 1. Problem statement

### 1.1 The core problem

Modern classification networks are trained by minimizing cross-entropy over a
softmax. This couples training to a quantity — the predicted probability $p_t$
of the gold class — that does **not** correspond to what we care about at test
time.

Train objective (per sample):

$$
\min_\theta\ -\log p_t(\theta) \quad\Rightarrow\quad p_t \to 1.
$$

Test objective (per sample):

$$
\hat y = t \iff p_t > p_j\ \forall j \ne t.
$$

The gap between $p_t \to 1$ and $p_t > p_j$ wastes optimization budget,
degrades calibration, and produces a loss curve that does not track
classification accuracy.

### 1.2 What AS-Softmax fixed (and didn't)

AS-Softmax (Lv et al., 2023) closes most of this gap with a scalar margin:

$$
p_t - p_j \ge \delta \quad \forall j \ne t.
$$

It works, but it makes **three structural assumptions** that the literature
has not seriously challenged:

| Assumption | Reality |
|---|---|
| Every class pair $(t, j)$ is equally hard | Classes form similarity clusters; some pairs are inherently confusable |
| Every input $x$ is equally ambiguous | A clearly-positive review needs less confidence than a borderline one |
| The same margin is right at step 1 and step 100,000 | Optimization dynamics change; what's "easy" early is "hard" late |

Each assumption is an empirical artifact, not a derived requirement. The
goal of this project is to relax all three and show that doing so improves
performance.

---

## 2. Five gaps

Gaps are numbered for cross-referencing. Each gap has (i) a one-line
description, (ii) the empirical evidence it's real, (iii) the research
question it implies.

### G1 — Class-pair margin asymmetry

*Description.* The required margin between target class $t$ and confuser
class $j$ should depend on the semantic / geometric relationship between $t$
and $j$.

*Evidence.*
- In sentiment (SST-5), most softmax errors are off-by-one (very-negative ↔
  negative, positive ↔ very-positive). These pairs need stricter margins;
  diagonally-opposite pairs basically never confuse and need none.
- In CIFAR-100, the 20 coarse superclasses partition fine classes into
  semantic groups. Confusions almost never cross superclass boundaries.
- Confusion matrices in the AS-Softmax paper itself show non-uniform error
  distributions, but the method treats them uniformly.

*Research question (RQ1).* Does a structured class-pair margin
$\delta_{t,j}$ outperform a scalar $\delta$, and what is the best
parameterization (similarity-based, confusion-based, learned low-rank)?

### G2 — Sample-level ambiguity

*Description.* Within a fixed class, samples vary in how confidently they
should be classified. The margin should adapt to per-sample ambiguity.

*Evidence.*
- Focal loss works precisely because per-sample importance varies.
- Calibration metrics (ECE) are dominated by overconfidence on ambiguous
  samples and underconfidence on easy ones.
- Aleatoric uncertainty is real and irreducible; insisting on a fixed margin
  for samples that are genuinely $50/50$ is forcing the model to memorize.

*Research question (RQ2).* Does a per-sample margin $\delta(x)$ improve
calibration without sacrificing accuracy, and what's the best source of the
signal — entropy of the prediction, feature norm, or a learned head?

### G3 — Static margin during training

*Description.* The margin should evolve over training. Early epochs need
loose constraints (the model is randomly initialized); late epochs benefit
from stricter constraints once general patterns are learned.

*Evidence.*
- Curriculum learning literature shows monotone improvements from easy-to-hard
  schedules.
- AS-Softmax uses a warm-up ratio $r$ — implicit acknowledgement that the
  scalar margin alone isn't right at $t = 0$.
- Cyclical learning rates work; cyclical margins are an unexplored cousin.

*Research question (RQ3).* What schedule $\delta(t)$ is optimal, and can
the optimum be characterized by the gradient flow dynamics rather than
hyperparameter search?

### G4 — Cross-modality / cross-task generality

*Description.* AS-Softmax has been demonstrated on text, image, and audio
classification. But the *interaction* between modality and margin structure
hasn't been studied — does a learned class-pair margin transfer? Is
low-rank structure stronger in some modalities?

*Evidence.*
- Vision features have known geometric structure (ImageNet → hierarchical).
- Audio features have known temporal structure.
- Text classes often inherit lexical / semantic graphs.
- One-size-fits-all hyperparameters across modalities are suspicious.

*Research question (RQ4).* Does the optimal margin parameterization depend
on modality, and is there a modality-invariant default that ships well?

### G5 — Regression and ordinal targets

*Description.* AS-Softmax is for categorical classification only. But many
real tasks (age estimation, satisfaction scores, bounded regression) have
**ordinal** or **continuous** targets where the "wrong" class is wrong by
varying degrees. The margin should reflect distance.

*Evidence.*
- Ordinal regression literature already exists, but is mostly orthogonal to
  the AS-Softmax line.
- Binned-regression (predict over discretized intervals, take weighted
  average) is a known technique; it fits naturally into our framework if we
  give it a distance-aware margin.

*Research question (RQ5).* Can GAM-Softmax with a distance-aware margin
$\delta_{t,j} = f(|y_t - y_j|)$ outperform standard regression heads on
bounded / ordinal tasks?

---

## 3. Hypotheses

These are the **falsifiable predictions** the experiments will test. Stating
them explicitly forces honest reporting — if a hypothesis dies, we say so.

- **H1 (class-pair).** A low-rank parameterized $\delta_{t,j} = \sigma(u_t^\top v_j)$
  with $u, v \in \mathbb{R}^k$, $k \ll n$, will outperform scalar $\delta$ on
  datasets with $\ge 20$ classes by $\ge 0.5\%$ macro-F1.
- **H2 (sample).** A learned $\delta(x)$ from a small head will reduce
  Expected Calibration Error (ECE) by $\ge 20\%$ relative on at least 3 of
  5 text datasets, while preserving accuracy within 0.3%.
- **H3 (schedule).** A monotonically increasing $\delta(t)$ ("easy-to-hard")
  will match or beat a constant $\delta$ tuned per dataset, removing one
  hyperparameter from practical use.
- **H4 (generality).** The best parameterization on text (likely H1's
  low-rank) will be in the top-2 on at least one of {image, audio} without
  re-tuning.
- **H5 (regression, stretch).** Binned-regression with distance-aware margin
  will beat MSE regression on a bounded task (UTKFace age estimation) by
  $\ge 5\%$ relative MAE.

A hypothesis being wrong is **also a result.** If H1 fails because class-pair
margins help on small-$n$ datasets but not large-$n$, that's a paper-worthy
finding. We just need to plan for it (see [08_risks_and_decisions.md](08_risks_and_decisions.md)).

---

## 4. What's explicitly out of scope

- Modifying $\delta$ to a non-margin formulation (e.g., kernel-based losses).
  → out of scope per supervisor instruction; logged for future work.
- Inventing a new normalization family (Power Softmax etc.).
  → kept as a baseline only, not the contribution.
- Pretraining-objective changes (we work at the classification head only).
- Very-large-scale (>10M parameters at the head, >1B-parameter backbones).
  Single-GPU constraint.
- Online / federated / streaming settings.

These are listed so we (and reviewers) know the contribution surface clearly.
