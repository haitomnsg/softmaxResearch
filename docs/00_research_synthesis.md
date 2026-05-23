# Research Synthesis

A consolidated reading of the three notes in `research/` plus the broader
softmax-variant landscape. This is your *internal* state-of-the-art document —
the literature review in [07_paper_outline.md](07_paper_outline.md) is the
*paper-facing* version.

---

## 1. The three source documents

### 1.1 `Adaptive_Sparse_Softmax_Structured_Notes.md` — Lv et al. (2023)

This is the **anchor paper** for the project.

**Core claim:** Standard softmax + cross-entropy pushes $p_t \to 1$ even though
the test-time criterion only requires $p_t > p_j$ for $j \ne t$. This mismatch
causes (a) endless training on already-correct samples, (b) overfitting, and
(c) a weak loss-accuracy correlation.

**Method:** Introduce a margin $\delta \in (0, 1]$ and a binary mask
$z_i \in \{0, 1\}$ that zeroes out class $i$'s contribution when
$p_t - p_i \ge \delta$:

$$
\tilde{p}_i = \frac{z_i e^{o_i}}{\sum_j z_j e^{o_j}},
\qquad
z_i = \begin{cases} 0 & p_t - p_i \ge \delta,\ i \ne t \\ 1 & \text{otherwise.} \end{cases}
$$

**Key theoretical result:** Required margin in logit space drops from
$o_t - o_{\min} \ge \log(n-1)$ (softmax) to
$o_t - o_{\min} \ge \log(n\delta + 1)$ (AS-Softmax). For large $n$, this is
exponentially easier to satisfy.

**Speed trick (AS-Speed):** As more samples get masked, fewer contribute to
the gradient. Compensate by accumulating more steps per update:
$\text{steps}_{\text{accum}} = \lambda \cdot N_{\text{all}} / (N_{\text{all}} - N_{\text{masked}})$.

**Empirical headline:** Pearson $\rho$(loss, val acc) on SST-5 goes from
$+0.038$ (softmax) to $-0.952$ (AS-Softmax). This is the most striking number
in the paper — it says the loss is finally measuring what we care about.

**Limitations (the door they leave open):**
- $\delta$ is a global scalar.
- $\delta$ + warm-up ratio $r$ require dataset-specific tuning.
- No theory beyond the margin bound.
- Multi-label is extended but not deeply analyzed.

### 1.2 `Generalized_Adaptive_Margins_AS_Softmax.md` — Thakuri (2026-03)

This is the **research proposal that defines this project's direction.**

**Three gaps identified:**

1. **Class-pair asymmetry.** "Very negative" vs "negative" is harder than
   "very negative" vs "very positive". A scalar $\delta$ ignores this.
2. **Sample-level ambiguity.** Different inputs in the same class need
   different confidence requirements.
3. **Temporal mismatch.** Early training needs a different margin than late
   training.

**Proposed generalization:**

$$
p_t - p_j \ge \delta_{t,j}(x, t), \quad \forall j \ne t
$$

with three orthogonal parameterizations: $\delta_{t,j}$ (class-pair),
$\delta(x)$ (sample), $\delta(t)$ (schedule).

**Where it stops:** No formal math, no experiments, no implementation. This
project's job is to fill that in.

### 1.3 `Beyond_Softmax_Research_Notes.md` — first-principles direction

This file is **explicitly out of scope** for the primary plan (per the
supervisor's instruction "do not modify delta or existing AS-Softmax
parameters" and your direction choice today), but we keep it on the radar
because:

- The **Power Softmax family** ($p_i \propto (o_i^+)^\gamma$) is a natural
  comparator: it's a different way to achieve sparsity, without margins.
- The **binned-regression idea** in §5.3 is exactly the bridge we'll need
  for the regression extension (stretch goal).
- The **trigonometric / entmax** families are useful sanity baselines.

**Decision:** Treat Power Softmax as one of our baselines (it deserves to be
in the comparison), but do not pursue it as the main contribution.

---

## 2. The landscape of softmax variants

A taxonomy by *what they change*:

### 2.1 Change the normalization (sparsity / shape)

| Method | Year | Idea | One-line takeaway |
|---|---|---|---|
| Sparsemax | 2016 | Euclidean projection onto simplex | First sparse softmax |
| Entmax-α | 2019 | Tsallis-entropy regularization | Generalizes softmax (α=1) and sparsemax (α=2) |
| Sparse-Softmax | 2022 | Top-k truncation | Keeps top-k logits |
| Power Softmax | (proposed) | $(o^+)^\gamma$ normalization | ReLU-style sparsity in output |

### 2.2 Change the margin (discriminative loss)

| Method | Year | Domain | Idea |
|---|---|---|---|
| Large-Margin Softmax (L-Softmax) | 2016 | Vision | Angular margin |
| SphereFace (A-Softmax) | 2017 | Face | Multiplicative angular margin |
| CosFace | 2018 | Face | Cosine margin |
| AM-Softmax | 2018 | Face | Additive margin in cosine |
| ArcFace | 2019 | Face | Additive angular margin |
| X2-Softmax | 2024 | Face | Margin function of angle |
| **AS-Softmax** | **2023** | **General** | **Probability-space margin + masking** |
| **GAM-Softmax (ours)** | **2026** | **General + regression** | **Structured probability-space margin** |

**Key distinction:** AM/ArcFace/CosFace margins are in **angular/cosine space**
on a hypersphere with $\ell_2$-normalized features — only applicable when you
control the feature geometry. AS-Softmax's margin is in **probability space** —
applies to any classifier. We inherit this advantage.

### 2.3 Change the target (label regularization)

- **Label smoothing** (Szegedy 2016): target distribution is
  $(1-\epsilon)\mathbb{1}_t + \epsilon/n$.
- **Focal loss** (Lin 2017): down-weight easy examples by $(1-p_t)^\gamma$.
  Closest spiritual relative to AS-Softmax — both focus on hard samples — but
  focal loss is dense and continuous, AS-Softmax is sparse and binary.

### 2.4 Change the partition function (efficiency)

- Hierarchical softmax, sampled softmax, RF-softmax, spherical softmax. Out
  of our path; these target scaling to very large $n$ (e.g., language models
  with vocab > 50k), not generalization quality.

---

## 3. Where the field is open

After mapping the landscape, the **specific holes** that GAM-Softmax fills:

**H1.** *No method gives a class-pair margin $\delta_{t,j}$ in probability
space.* AM-Softmax and friends do it angularly, but require feature
normalization. We propose probability-space class-pair margins, which compose
with any classifier.

**H2.** *No method gives sample-dependent margins from input alone.* Focal loss
gives sample-dependent *weights*, but the margin itself is global. We propose
a learned $\delta(x)$ that adapts to input ambiguity.

**H3.** *Margin schedules are either absent or set ad hoc.* AS-Softmax uses a
warm-up ratio $r$ but doesn't analyze schedule shape. We propose principled
schedules (cyclical, gradient-driven) with theoretical justification.

**H4.** *AS-Softmax has been validated on text/image/audio classification but
never on regression.* We bridge this via the binned-regression view.

**H5.** *The theoretical margin bound $\log(n\delta+1)$ is for a scalar
$\delta$.* We need the generalization to $\delta_{t,j}(x,t)$ — likely
involving an "effective average margin" $\bar\delta_{\text{eff}}$.

---

## 4. What this tells us about positioning

For an ACL/EMNLP submission, lead with: *generalized margin + better
calibration + cross-task validity*. NLP reviewers like calibration and
sample-efficiency stories.

For an ICLR/NeurIPS submission, lead with: *theory of structured margins +
empirical breadth*. ML-conf reviewers want bounds, ablations, and
modality-agnostic claims.

Either way, the story is the same: **scalar margin is the wrong abstraction;
here is the right one, with proofs and experiments to back it.**
