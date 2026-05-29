# The Project, Explained Simply

*Read this before the presentation. No math, plain words. It explains everything in `PRESENTATION.md`.*

---

## What problem are we solving?

When a computer learns to sort things into groups (like sorting movie reviews into
"very bad → very good"), we normally train it to be **100% sure** about every single answer.

That's like forcing a student to score a perfect 100 on every question. It wastes effort,
and it actually makes the student *worse* at the real test.

A smarter idea already exists, called **AS-Softmax**. It says:

> "Don't aim for 100%. Just make the right answer **win by a safe margin**, then stop."

Like a student who only needs to pass comfortably, not ace everything. This trains faster
and works better.

---

## What is OUR idea?

AS-Softmax uses **one single margin** (one "winning gap") for everything.

We think that's too simple. **A smart gap should change depending on the situation:**

- **Some answers are easy to tell apart** (cat vs. airplane) → small gap is fine.
- **Some are hard to tell apart** (cat vs. dog) → need a bigger gap.
- **Easy examples vs. confusing examples** → should be treated differently.
- **Early in training vs. late in training** → the gap can change over time.

Our method is called **GAM-Softmax**. In one sentence:

> **Instead of one fixed gap for everything, use a smart gap that adjusts to the situation.**

That's the whole bet. The project is about proving that this smart gap actually works better.

---

## What have we actually done?

Think of it in three parts:

### Part 1 — We built the machine ✅
We built all the tools: the different learning methods (9 of them), the training system,
the data, and a big set of automatic tests that check our math is correct.

**This part is finished and works.** The tests confirm our method is built correctly.

### Part 2 — We checked the basics work ✅
Before testing our new idea, we made sure the *known* method (AS-Softmax) behaves as
expected. It did: it beat the plain old method, just like the original paper said.

**So our setup is trustworthy.** Good results aren't a fluke or a bug.

### Part 3 — We tested our actual idea 🟡 (this is where it gets interesting)
We tried our "smart gap" on two different datasets:

- **On the first dataset (movie reviews / SST-5):** our method **tied** with AS-Softmax.
  No improvement. But we figured out *why* — that dataset is too easy. **Every** method
  scored about the same (~52%), so there was no room for anyone to win. It's like racing
  on a track so short everyone finishes at the same time.

- **On a harder dataset (news topics / 20 Newsgroups):** our method **slightly beat**
  AS-Softmax (+0.6%). And something interesting happened: **our method kept getting better
  while the others got worse from over-studying.** That "doesn't burn out as fast" behavior
  is a real, repeatable difference — even when the final scores are close.

---

## The honest truth (don't hide this in the talk)

Here's the catch we discovered:

> The method we're trying to improve (AS-Softmax) **doesn't always beat the plain old
> simple method** in our experiments.

On clean, simple data, the boring old method is actually really hard to beat. Fancy
margin methods like ours usually shine on **harder** problems — lots of categories, messy
labels, uneven data. We were testing on data that's a bit too easy and clean.

So: **the code is fine. The idea isn't disproven. We were just testing in the wrong arena.**

---

## What's next?

We hit a decision point and **paused on purpose** to choose a direction (instead of burning
more computer time guessing). The options:

1. **Change the story** — focus on the thing our method clearly *does* better: it doesn't
   over-study / burn out as fast.
2. **Test on harder problems** — where smart margins are known to help.
3. **Try a different "smart gap" axis** — maybe the gap should change by example or over
   time, not by category.
4. **Re-word our main claim** — match what we promise to what we can actually prove.

---

## If you only remember 5 sentences

1. Normal AI training wastes effort by forcing 100% confidence; a smarter method (AS-Softmax)
   just asks the right answer to win by a safe margin.
2. **Our idea (GAM-Softmax): make that margin *smart* — adjust it to the situation instead
   of using one fixed number.**
3. We **built and tested the whole system**, and confirmed the basics work correctly.
4. Our smart-gap idea **tied on easy data** (no room to win) but showed a **small win plus
   slower burnout on harder data** — promising, not yet proven.
5. The real lesson: **we need to test on harder problems**, because easy clean data can't
   tell any of these methods apart — that's our next move.

---

*Full numbers and details are in [PRESENTATION.md](PRESENTATION.md). The developer's
step-by-step map is in [STATUS_AND_NEXT_STEPS.md](STATUS_AND_NEXT_STEPS.md).*
