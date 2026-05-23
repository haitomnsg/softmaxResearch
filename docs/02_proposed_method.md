# GAM-Softmax: The Proposed Method

This is the technical specification of the loss family. Read this *with*
[03_theoretical_plan.md](03_theoretical_plan.md) — they're sibling docs.

---

## 1. Definition

### 1.1 The general form

Let logits be $o \in \mathbb{R}^n$, target class be $t$. Define a **structured
margin function**

$$
\delta_{t,j}(x, \tau) \in [0, 1], \qquad j \ne t,
$$

where $x$ is the input sample and $\tau$ is the (normalized) training step.

Define the **masking variable**

$$
z_i = \begin{cases}
1 & i = t \\
\mathbb{1}\!\left[p_t - p_i < \delta_{t,i}(x, \tau)\right] & i \ne t.
\end{cases}
$$

The **GAM-Softmax probability** is

$$
\boxed{\;\tilde p_i = \frac{z_i\, e^{o_i}}{\sum_{j=1}^n z_j\, e^{o_j}}\;}
$$

and the loss is $L = -\log \tilde p_t$.

### 1.2 Recovering known methods

| Setting | Recovers |
|---|---|
| $\delta_{t,j}(x,\tau) = 0$ | Standard softmax |
| $\delta_{t,j}(x,\tau) = \delta$ (constant scalar) | AS-Softmax |
| $\delta_{t,j}(x,\tau) = \delta_{t,j}$ (constant matrix) | Class-pair variant only |
| $\delta_{t,j}(x,\tau) = \delta(x)$ (input only) | Sample variant only |
| $\delta_{t,j}(x,\tau) = \delta(\tau)$ (time only) | Schedule variant only |

This is the cleanest statement of why GAM-Softmax is a **strict
generalization** — every prior method is a degenerate case.

---

## 2. Direction 1 — Class-pair margin $\delta_{t,j}$

Three parameterizations, in order of complexity.

### 2.1 Similarity-based (parameter-free)

Compute class embeddings $e_c$ once (from BERT / class-name embeddings / class
prototype features), then:

$$
\delta_{t,j} = \delta_{\min} + (\delta_{\max} - \delta_{\min}) \cdot \frac{1 + \cos(e_t, e_j)}{2}
$$

- **Pros:** zero new parameters, easy to ablate.
- **Cons:** depends on quality of $e_c$.
- **Use when:** $n$ is small-to-moderate, class names are meaningful.

### 2.2 Confusion-based (data-driven, frozen)

Train an AS-Softmax model with scalar $\delta_0$ once. Use the resulting
validation confusion matrix $C$ to set:

$$
\delta_{t,j} = \delta_{\min} + (\delta_{\max} - \delta_{\min}) \cdot \frac{C_{t,j}}{\max_{t',j'} C_{t',j'}}
$$

(Larger $C_{t,j}$ ⇒ pair is more confusable ⇒ bigger margin.)

- **Pros:** directly addresses what the model gets wrong.
- **Cons:** requires a two-pass training; risk of self-reinforcing biases.
- **Use when:** scalar $\delta$ baseline is already trained.

### 2.3 Learnable low-rank (recommended primary)

Parameterize

$$
\delta_{t,j} = \delta_{\min} + (\delta_{\max} - \delta_{\min}) \cdot \sigma(u_t^\top v_j / \sqrt k),
$$

with $u_t, v_j \in \mathbb{R}^k$, $k \in \{4, 8, 16\}$, $u, v$ learned
end-to-end. $\sigma$ is the sigmoid. The $/\sqrt k$ is a transformer-style
scale to keep $\sigma$'s input in a good range.

- **Pros:** $2nk$ parameters total (tiny); learns structure jointly with
  the classifier; admits an "M = UV^T" interpretation that motivates a
  generalization bound (see [03_theoretical_plan.md](03_theoretical_plan.md) T2).
- **Cons:** requires careful init (start with $u, v$ small so $\delta \approx \delta_{\min}$,
  warm-up the classifier first).
- **Use when:** this is the default. Start here.

### 2.4 Diagonal restriction (a useful ablation)

Restrict to $\delta_{t,j} = \delta_t$ (depends only on target class). This
tests whether the *pair* structure matters or just *per-class* margins. Cheap
to run.

---

## 3. Direction 2 — Sample-dependent margin $\delta(x)$

### 3.1 Entropy-based (parameter-free)

Using the current prediction $p(x)$:

$$
\delta(x) = \delta_{\min} + (\delta_{\max} - \delta_{\min}) \cdot \frac{H_{\max} - H(p(x))}{H_{\max}}
$$

where $H_{\max} = \log n$. High entropy (ambiguous prediction) ⇒ small
margin; low entropy (confident prediction) ⇒ large margin.

- **Subtlety:** the margin depends on the prediction it's used to score —
  this is fine because we use it inside the mask, not as a target. But it
  introduces a feedback loop; stop-gradient through $\delta(x)$.

### 3.2 Feature-norm based

If $f(x)$ is the penultimate-layer feature:

$$
\delta(x) = \delta_{\min} + (\delta_{\max} - \delta_{\min}) \cdot \text{clip}\!\left(\|f(x)\| / \nu, 0, 1\right)
$$

with $\nu$ a running average of $\|f(x)\|$. Borrowed intuition from
CosFace/ArcFace.

### 3.3 Learnable head (recommended primary for D2)

Add a small MLP $h_\phi: f(x) \mapsto \mathbb{R}$ then squash:

$$
\delta(x) = \delta_{\min} + (\delta_{\max} - \delta_{\min}) \cdot \sigma(h_\phi(f(x))).
$$

$\phi$ is trained jointly. To avoid the degenerate solution $\delta(x) \to 0$
(which lifts all the masking and just makes things easier), add a regularizer

$$
R = \mu \cdot |\bar\delta(x) - \delta_{\text{anchor}}|
$$

that pulls the *average* margin toward a target $\delta_{\text{anchor}}$
(e.g., the best scalar from AS-Softmax tuning).

---

## 4. Direction 3 — Time-adaptive margin $\delta(\tau)$

Let $\tau \in [0, 1]$ be the normalized step counter.

### 4.1 Linear ramp (recommended primary for D3)

$$
\delta(\tau) = \delta_{\min} + (\delta_{\max} - \delta_{\min}) \cdot \tau
$$

Starts loose, ends strict. Curriculum-style.

### 4.2 Cosine schedule (matches modern LR schedules)

$$
\delta(\tau) = \delta_{\min} + \tfrac{1}{2}(\delta_{\max} - \delta_{\min})(1 - \cos(\pi \tau))
$$

### 4.3 Cyclical

$$
\delta(\tau) = \delta_{\text{base}} + \delta_{\text{amp}} \cdot \sin(2\pi \tau / T_{\text{cycle}}).
$$

### 4.4 Gradient-driven

Adjust $\delta$ each $K$ steps based on the masked fraction:

$$
\delta_{\tau + K} = \delta_\tau + \eta \cdot (r_{\text{masked}}^* - r_{\text{masked}}(\tau)),
$$

targeting a fixed masked ratio $r^*$. This is the most principled but adds a
control loop; treat as an ablation, not a default.

---

## 5. Composition

The full GAM-Softmax uses a **multiplicative composition**:

$$
\delta_{t,j}(x, \tau) = \delta^{\text{cls}}_{t,j} \cdot \delta^{\text{smp}}(x) \cdot \delta^{\text{tim}}(\tau) \cdot c,
$$

where each factor lives in $[0, 1]$ and $c$ is a global scale tuned per
dataset (default $c = 1$). Multiplicative because:
- If any factor says "no margin needed" (= 0), there's no margin. This is
  permissive, which is what we want during warm-up.
- Each factor's role stays interpretable.

Additive composition is an ablation worth running (set
$\delta = \delta_1 + \delta_2 + \delta_3$ then clip).

---

## 6. Multi-label extension

Following AS-Softmax §6, the multi-label loss becomes

$$
L = \log\!\left(1 + \sum_{j \in \Omega_{\text{neg}}} z_j e^{o_j}\right) + \log\!\left(1 + \sum_{t \in \Omega_{\text{pos}}} z_t e^{-o_t}\right),
$$

with the same mask $z$, but defined via the per-pair structured margin
$\delta_{t,j}(x, \tau)$ between positives and negatives.

---

## 7. Regression extension (stretch)

For bounded / ordinal regression with target $y \in [y_{\min}, y_{\max}]$:

1. Discretize into $n$ bins with centers $v_1, \dots, v_n$.
2. Predict logits over bins, apply GAM-Softmax.
3. Predict $\hat y = \sum_i v_i \tilde p_i$.
4. Use a **distance-aware margin** $\delta_{t,j} = g(|v_t - v_j|)$, e.g.,

$$
\delta_{t,j} = \delta_{\min} + (\delta_{\max} - \delta_{\min}) \cdot \tanh(\beta |v_t - v_j|).
$$

This makes the margin small for adjacent bins (which we don't care about
distinguishing) and large for distant bins (where confusion is a real error).

---

## 8. Default recipe ("the experiment you should run on day 1 of Phase 2")

Once baselines are reproduced:

1. **Backbone:** BERT-base (text), ResNet-50 (image), wav2vec2-base (audio).
2. **Loss:** GAM-Softmax with class-pair $\delta_{t,j} = \sigma(u_t^\top v_j / \sqrt k)$,
   $k = 8$, $\delta_{\min} = 0.05$, $\delta_{\max} = 0.4$.
3. **Schedule:** linear ramp on $\delta_{\max}$ for first 30% of training,
   constant afterward.
4. **Sample term off** (set to 1) until Phase 3.
5. **AS-Speed:** on (inherit from AS-Softmax).
6. **Optimizer:** AdamW, LR 2e-5 (text), 1e-3 (vision/audio), warmup 10%.
7. **Eval:** every 500 steps. Track accuracy, macro-F1, ECE, masked
   fraction, $\rho$(loss, val acc).

If this beats scalar AS-Softmax on SST-5 by ≥ 0.3% accuracy and improves ECE,
the project's core hypothesis is alive.

---

## 9. Numerical concerns

- **Stop-gradient through the mask $z$.** It's a hard threshold; treat it as
  not differentiable wrt $\delta$. Gradient flows through the classifier
  logits only.
- **Log-sum-exp stability** is unchanged from AS-Softmax — the masked sum is
  always $\ge e^{o_t}$ so log is well defined.
- **Empty mask edge case** — when all non-target classes are masked,
  $\tilde p_t = 1$ and $L = 0$ exactly. This is the intended behavior.
- **AMP/fp16:** sigmoid of low-rank dot product can saturate. Keep
  $u, v$ at fp32 if you see NaNs.
