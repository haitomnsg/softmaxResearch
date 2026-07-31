# How We Vary the Margin δ

*A from-the-ground-up walkthrough of what δ is, why we vary it, and exactly how
the code does it. Written to be readable without heavy math — equations are kept
but each one is explained in plain words.*

---

## 0. Plain-English version

Forget the symbols for a second. Here's the whole idea as a story.

A classifier is a student taking a multiple-choice test over and over while
learning. For each question there's one right answer and several wrong answers.

**Normal training** makes the student practice *every* wrong answer on *every*
question — including the obviously-wrong ones. That's wasteful: once the student
clearly knows "the answer is a cat, definitely not an airplane," drilling
"it's not an airplane" again teaches nothing and just adds noise.

**Our approach** says: *skip the wrong answers the student has already clearly
ruled out, and spend the effort on the wrong answers it's still tempted by.*

**δ (delta) is the "clearly ruled out" line.** It's how big the confidence gap
has to be before we say "you've got this one, move on." If the student is much
more sure of the right answer than of a particular wrong answer (gap bigger than
δ), we drop that wrong answer from this round of practice.

- A **small δ** = a low bar = "you barely have to be ahead before I let you skip
  it" = lots of wrong answers get dropped = aggressive.
- A **large δ** = a high bar = "you have to be *way* ahead before I let you skip
  it" = few get dropped = gentle.

The novelty in our project is simple to state: **the "clearly ruled out" line
shouldn't be one fixed number for everything.** It makes sense to use a different
line:

- for **different pairs of answers** — cat-vs-dog (easy to confuse) deserves a
  stricter line than cat-vs-airplane;
- for **different questions** — a messy or trick question deserves a gentler line
  than an easy one;
- at **different points in the course** — early on the student is basically
  guessing, so don't let them skip anything yet; later, when they're competent,
  let them skip more.

That's the entire idea. Everything below is just *how* we turn "the line should
adapt" into actual numbers and code. **You can stop reading here and still
understand the project.**

---

## 1. What δ even is

Both AS-Softmax (our baseline) and GAM-Softmax (our method) train a classifier
with cross-entropy, but with one twist: before computing the loss they **throw
away some non-target classes** for each training example, then renormalize the
softmax over only the classes that survive.

The rule for throwing a class away is a **margin threshold δ**:

> For a training example whose correct label is `t`, drop a wrong class `i` if
> the model is already confident enough about `t` relative to `i`:
>
> ```
> p_t − p_i  ≥  δ        →   drop class i
> ```
>
> where `p_t` is the softmax probability of the correct class and `p_i` is the
> probability of wrong class `i`.

Intuition: if the model already assigns the true class a much higher probability
than some wrong class, that wrong class is "easy" — there's nothing left to learn
from it, so we stop pushing its logit down. The loss then concentrates gradient
on the **hard, still-confusable** classes. δ is the cutoff that defines "much
higher."

Two correctness points the codebase is strict about (see
[CLAUDE.md](CLAUDE.md) and [gam_softmax/losses/as_softmax.py](gam_softmax/losses/as_softmax.py)):

1. **The test is on probabilities `p_t − p_i`, not on raw logits `o_t − o_i`.**
   You must run softmax first. Doing it on logits loses the AS-Softmax reduction
   property.
2. **The mask is detached** (`with torch.no_grad()`), so no gradient flows
   *through* the threshold decision. The math assumes a hard on/off mask.

---

## 2. The core idea: δ is not one number

AS-Softmax uses **a single scalar δ** for every example, every class pair, and
every step of training (e.g. `δ = 0.1`).

Our whole research contribution is to replace that one number with a **margin
function** that varies along three axes:

| Axis | Question it answers | Why it should vary |
|------|--------------------|--------------------|
| **Class-pair** (`t, j`) | Should "cat vs dog" use a different cutoff than "cat vs airplane"? | Some class pairs are intrinsically confusable and deserve a tighter margin. |
| **Sample** (`x`) | Should an easy example and a hard/noisy example use the same cutoff? | Hard or noisy samples may need a looser margin so we don't prematurely drop classes. |
| **Time** (`τ`, the training progress) | Should the cutoff be the same at step 0 as at the end of training? | Early on the model is random, so masking aggressively is dangerous; loosen at the start, tighten later. |

So conceptually:

```
scalar δ            →            δ_{t,j}(x, τ)
(AS-Softmax)                     (GAM-Softmax)
```

The rest of this document explains how each axis is actually implemented.

---

## 2.5 Do we vary all three axes on every problem? (No — and that's deliberate)

This is the most common point of confusion, so let's be explicit.

The general formula `δ_{t,j}(x, τ)` *can* depend on all three axes at once, but
**we almost never turn all three on at the same time.** The three axes are
**independent directions you can switch on or off**, and each well-known method is
just GAM-Softmax with most of them switched off:

| What δ depends on | What it reduces to |
|-------------------|--------------------|
| nothing (`δ = 0`) | plain softmax / cross-entropy |
| a single constant | AS-Softmax (the baseline) |
| **class-pair only** `δ_{t,j}` | "class-pair variant" (our **M3**, §4) |
| **sample only** `δ(x)` | "sample variant" (our **M4**, §4.5) |
| **time only** `δ(τ)` | "schedule variant" (the `Schedule`, §5) |
| several at once | a combined variant (our **M6** = M3 × M4) |

So the answer to "do we vary class-pair *and* sample *and* time for every
problem?" is **no**. Our research strategy is the opposite:

1. **Isolate one axis at a time first.** We test "does varying δ *by class-pair
   alone* beat a fixed δ?" before we ever combine axes. That's variant M3 — it
   uses the class-pair axis and a simple time schedule, but does **not** make δ
   depend on the individual sample `x`. Isolating lets us attribute any
   improvement to a specific cause instead of a tangle of three changes.
2. **Only combine axes if the isolated ones each earn their keep.** Combining is
   more parameters, more tuning, and more ways to fail — so we pay that cost only
   after an axis has proven it helps on its own.

**Which axis matters also depends on the problem.** They are not equally useful
everywhere:

- **Class-pair (`t, j`)** matters most when some classes are genuinely confusable
  with each other (fine-grained sentiment like SST-5, or visually similar image
  classes). If all classes are equally distinct, a per-pair margin buys little.
- **Sample (`x`)** matters most when examples differ in difficulty or
  trustworthiness — especially the **noisy-label** setting we pivoted to, where a
  mislabeled example should get a *gentler* margin so we don't confidently drop
  the (actually correct) class. On clean, uniform data it matters less.
- **Time (`τ`)** matters most for **training stability**: it protects the early,
  random phase of training from over-aggressive masking. It's cheap and almost
  always worth including as a warmup, even when it's not the headline
  contribution.

In short: think of the three axes as a **menu**, not a fixed recipe. Each variant
in the plan (M1–M10) picks the axis or axes that target the specific weakness
we're studying — and the codebase is built so each axis is one composable piece
you add or remove (see next section).

---

## 3. The architecture: how varying δ is wired together

We deliberately did **not** write a new loss class per variant. Instead there are
three composable pieces:

```
GAMSoftmaxLoss  ──uses──▶  MarginFunction  ──uses──▶  Schedule
 (the loss)               (varies δ over                (varies δ's
                           class-pair & sample)          upper cap over time)
```

- **`GAMSoftmaxLoss`** ([gam_softmax/losses/gam_softmax.py](gam_softmax/losses/gam_softmax.py))
  is identical to AS-Softmax **except** the scalar δ is replaced by a per-(sample,
  class) tensor that it asks a `MarginFunction` to produce.
- **`MarginFunction`** ([gam_softmax/margins/base.py](gam_softmax/margins/base.py))
  is the abstraction that produces the `(B, n)` margin tensor. Each research
  variant (M1–M10 in the plan) is one subclass.
- **`Schedule`** ([gam_softmax/schedules/base.py](gam_softmax/schedules/base.py))
  produces a single time-varying number — typically the **upper cap** that the
  margin function blends toward. This is the **time axis**.

Adding a new way of varying δ means writing one `MarginFunction` subclass (and
maybe one `Schedule`) plus a YAML config — not touching the trainer or loss.

### The `step_frac` trick (the time axis input)

Schedules need to know "how far through training are we?" but they must not
depend on the absolute number of steps (which differs per dataset/run). So the
trainer passes a **normalized progress** value:

```
step_frac = current_step / total_steps     # always in [0, 1]
```

Every margin function and schedule receives `step_frac`, never the raw step. That
makes a schedule like "ramp up over the first 30% of training" portable across any
run length.

---

## 4. Axis 1 + 2: varying δ by class-pair (and reading it per sample)

This is implemented in
[gam_softmax/margins/classpair_lowrank.py](gam_softmax/margins/classpair_lowrank.py)
(our **M3** variant). The formula is:

```
δ_{t,j} = δ_min + (δ_max(τ) − δ_min) · σ(u_t · v_j / √k)
```

Reading it left to right:

- **`δ_min`** is a floor (e.g. `0.05`) — the margin never goes below this.
- **`δ_max(τ)`** is the time-varying ceiling produced by the `Schedule` (the time
  axis — see §5).
- **`σ(u_t · v_j / √k)`** is a learned per-class-pair score squashed into `[0, 1]`
  by a sigmoid. `u` and `v` are two small learned matrices of shape `(n_classes,
  rank)`; their dot product gives every class pair `(t, j)` its own number. `√k`
  (k = rank) keeps the dot product from blowing up. This is a **low-rank**
  parameterization: instead of learning a full `n × n` margin matrix, we learn
  two thin `n × k` matrices, which is cheap and regularized.

So the sigmoid term sits between 0 and 1 and **interpolates each class pair
somewhere between the floor `δ_min` and the ceiling `δ_max`**. Confusable pairs can
learn a different cutoff than easy pairs — that's the class-pair axis.

The code:

```python
def forward(self, logits, targets, features, step_frac):
    delta_min = float(self.schedule.delta_min)
    delta_max_now = float(self.schedule(step_frac))          # time axis (§5)

    scores = (self.u @ self.v.t()) / math.sqrt(self.rank)    # (n, n) per-pair score
    raw = torch.sigmoid(scores)                              # squash to [0, 1]
    delta_matrix = delta_min + (delta_max_now - delta_min) * raw

    # pick the row for each sample's true class → (B, n)
    return delta_matrix.index_select(0, targets)
```

The last line is where the **sample axis** comes in (lightly, for now): each
sample looks up the margin row that corresponds to *its* true class `t`, giving a
`(B, n)` tensor — one margin per (sample, candidate-class) slot. (A future variant
can make this depend on the sample's features `x` directly; the hook
`features` is already threaded through.)

**Init detail that matters:** `u` and `v` are initialized with tiny values
(`init_std = 0.02`), so at step 0 the sigmoid is ≈ `σ(0) = 0.5` — the margin
starts near the midpoint and near-uniform across pairs. Combined with a schedule
whose `δ_max(0) == δ_min` (see below), the whole margin **collapses to `δ_min` at
step 0** — the deliberate "start loose" recipe.

---

## 4.5 Axis 2 for real: varying δ by *sample* (M4)

§4's "sample axis" is thin — every sample with the same label gets the same row.
The genuine sample axis is
[gam_softmax/margins/sample_confidence.py](gam_softmax/margins/sample_confidence.py)
(**M4**), where δ depends on how the model is currently doing *on that specific
example*:

```
δ_i(τ) = clamp( δ_base(τ) − β(τ) · (2·s_i − 1),  δ_floor, δ_ceil )

s_i = σ( (p̄_t − p_{t,i}) / (std(p_t) · temp) )
```

In words:

- `p_{t,i}` is the probability the model gives to **sample i's training label**
  right now. Low = "the model disagrees with this label."
- `s_i` compares that to the **rest of the batch**. `s_i > 0.5` means "this
  sample is less confident than its peers" — call it **suspicious**. Comparing
  within the batch is what makes the score self-calibrating: early in training
  *everything* has low `p_t`, so an absolute threshold would flag the whole
  dataset.
- Suspicious → δ goes **down** → more classes clear the `p_t − p_j ≥ δ` bar →
  they get masked → **the sample contributes less gradient**.
- Confident → δ goes up → fewer classes masked → the sample keeps training.

### Why this is the axis that matters for noisy labels

Here's the trick, and it's the one genuinely new idea in this variant.

**δ is allowed to go negative.** Nothing in the math requires a margin to be
positive — `p_t − p_j` lives in `[−1, 1]`, so a δ of, say, `−0.4` is a perfectly
meaningful threshold. It says: *drop class j even if it is currently beating the
training label by up to 0.4.*

Now consider a **mislabeled** training example. Its "correct" label is wrong, so
the model — which has learned the real pattern from thousands of correctly
labeled examples — keeps assigning it a low `p_t`. With a fixed positive δ, none
of the competing classes ever clear the bar, so **every** competitor stays in
the loss and the model gets pushed, over and over, to fit the wrong label. That
is exactly how networks memorize label noise.

With a negative δ, all those competitors get masked instead. The kept set
collapses to just the target, the renormalized cross-entropy over a single
surviving class is ~0, and **the sample quietly drops out of training.**

That means M4 makes the noisy-label literature's "small-loss trick" (throw away
high-loss samples, they're probably mislabeled) a **special case of a margin** —
no separate sample-selection heuristic, no extra loss term, no second network.
It's the same one-line threshold, with the threshold allowed to be negative.

### The two guardrails

- **`reject_warmup_frac` (default 0.3):** β is held at **0** for the first 30% of
  training. Early on the model is near-random, so `p_t` says nothing about which
  labels are wrong — rejecting samples then would just discard data at random.
  β then ramps to its full value.
- **`delta_ceiling`:** the correction is symmetric (mean ≈ 0 across the batch),
  which would also let *confident* samples get a **tighter** margin than
  AS-Softmax uses — quietly reverting them to dense cross-entropy and muddying
  the comparison. Setting `delta_ceiling` to the schedule's `δ_max` makes the
  axis **one-sided**: suspicious samples get loosened, confident ones sit exactly
  where AS-Softmax would put them. Any difference between the M4 and AS-Softmax
  rows is then attributable to one thing only.

### Composing it with the class-pair axis (M6)

`SampleConfidenceMargin` takes an optional `base_margin`. Pass it a
`ClassPairLowRankMargin` and the per-sample correction modulates the class-pair
*matrix* instead of a scalar — δ then varies along **all three axes at once**.
That's variant **M6**, and it needed no new class, just a nested `base:` block in
the YAML.

---

## 5. Axis 3: varying δ over time (the Schedule)

The time axis lives in the `Schedule`. The default is
[gam_softmax/schedules/linear.py](gam_softmax/schedules/linear.py):

```python
class LinearSchedule(Schedule):
    def __call__(self, step_frac):
        if step_frac >= self.warmup_frac:
            return self.delta_max                       # constant after warmup
        t = step_frac / self.warmup_frac
        return self.delta_min + (self.delta_max - self.delta_min) * t
```

In words: the **upper cap** `δ_max(τ)` ramps linearly from `δ_min` up to `δ_max`
over the first `warmup_frac` of training (e.g. the first 30%), then stays flat at
`δ_max`.

Why ramp up? Early in training the model is essentially random, so `p_t − p_i` is
noise. If we masked aggressively from step 0 we'd drop classes for the wrong
reasons. Starting the cap at `δ_min` means **almost nothing gets masked early**
(loose); as the model becomes trustworthy the cap rises and masking gets more
aggressive (tight). This is the "start loose, tighten later" curriculum.

There's also a `ConstantSchedule` (cap never changes) for ablations and to recover
plain AS-Softmax behavior.

> **Note on the division of labor:** the schedule only ever returns the *cap*. The
> margin function (§4) is what blends `δ_min` into its own output. Keeping `δ_min`
> on the schedule object means the floor and the ceiling can't accidentally drift
> apart — `ClassPairLowRankMargin` reads `schedule.delta_min` directly.

---

## 6. Where δ is actually used: the loss

Once the margin function has produced the `(B, n)` δ tensor, the loss applies it
exactly like AS-Softmax — just with a per-slot threshold instead of a scalar
([gam_softmax/losses/gam_softmax.py](gam_softmax/losses/gam_softmax.py)):

```python
delta = self.margin_fn(logits, targets, features, step_frac)   # (B, n)

with torch.no_grad():
    probs = torch.softmax(logits, dim=-1)
    p_t = probs.gather(1, targets.unsqueeze(1))                # (B, 1)
    gap = p_t - probs                                          # (B, n)
    mask_out = gap >= delta.detach()                           # drop where gap ≥ δ
    # ... never mask the target class itself ...
    keep = ~mask_out

# renormalize cross-entropy over only the kept classes
masked_logits = logits.masked_fill(~keep, float("-inf"))
log_probs = torch.log_softmax(masked_logits, dim=-1)
nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
loss = nll.mean()
```

Note `delta.detach()` in the comparison — the **keep/drop decision is hard and
non-differentiable**, exactly as in AS-Softmax. This is why, by default, the
margin's parameters `u, v` get **no gradient** and stay at their random init.

### Making δ learnable (the STE path)

If we want the class-pair structure to actually *train* (variant **M3'**), we set
`learnable_margin=True`. The loss then uses a **straight-through estimator**:

- **Forward:** still uses the hard kept-set, so the loss value is numerically
  identical and still recovers AS-Softmax.
- **Backward:** routes gradient into δ (and hence into `u, v`) through a smooth
  sigmoid surrogate of the keep decision, `σ((δ − gap) / ste_temp)`.

```python
keep_soft = torch.sigmoid((delta - gap) / self.ste_temp)
keep_ste  = keep.float() + (keep_soft - keep_soft.detach())   # value=hard, grad=soft
```

`ste_temp` controls how sharp the surrogate is: smaller = closer to the true hard
step, larger = smoother gradient. This is the one knob that turns "δ varies but is
fixed-random" into "δ varies and is learned."

---

## 7. How you set all this from a config

One YAML = one experiment. Here is the learnable M3 config
([configs/m3_classpair_lowrank/sst5_learnable.yaml](configs/m3_classpair_lowrank/sst5_learnable.yaml)),
annotated to show which knob controls which axis:

```yaml
loss:
  type: gam_softmax
  learnable_margin: true        # turn on the STE gradient path → δ trains
  ste_temp: 0.1                 # surrogate sharpness
  margin:
    type: classpair_lowrank     # the class-pair axis (§4)
    rank: 8                     # low-rank size k
    init_std: 0.02              # start near-uniform (σ≈0.5) at step 0
    schedule:                   # the time axis (§5)
      type: linear
      delta_min: 0.05           # floor
      delta_max: 0.4            # ceiling
      warmup_frac: 0.3          # ramp the cap over the first 30% of training
```

Run it the usual way:

```bash
python experiments/run.py --config configs/m3_classpair_lowrank/sst5_learnable.yaml
```

---

## 8. One-paragraph summary

We replace AS-Softmax's single scalar margin δ with a **margin function**
`δ_{t,j}(x, τ)` that varies along three axes. The **class-pair** axis is a learned
low-rank score `σ(u_t·v_j/√k)` (M3); the **sample** axis scores each example by
its batch-relative confidence and, critically, lets δ go **negative** so a
suspect example's competitors are all masked and it drops out of the loss (M4 —
this makes the "small-loss trick" a special case of a margin); the **time** axis
is a `Schedule` that ramps the margin's cap from a floor `δ_min` up to a ceiling
`δ_max` over a warmup fraction. The axes compose (M6 = class-pair × sample ×
time) via a nested `base:` block rather than a new loss class. The loss uses this
δ tensor in *probability* space (`p_t − p_i ≥ δ`) with a detached hard mask, and
— when `learnable_margin=True` — uses a straight-through estimator so the
class-pair parameters actually train. When δ is constant, the whole thing reduces
exactly to AS-Softmax.

**What the experiments said about all this:** see
[FINAL_REPORT.md](FINAL_REPORT.md). The short version is that the *machinery*
works exactly as designed, and the *hypothesis* it was built to test did not
survive contact with the data.
