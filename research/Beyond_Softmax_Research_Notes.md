# Research Notes: Beyond Softmax
## New Activation Functions for Classification and Regression

**Author:** Your Name  
**Date:** April 20, 2026  
**Purpose:** Reading Material for Personal Use  

---

# Table of Contents

1. Introduction  
2. What Softmax Is and Why It Dominates  
   - 2.1 Why Softmax Succeeded  
   - 2.2 Limitations of Softmax  
3. Historical Precedent: Sigmoid to ReLU  
   - 3.1 Sigmoid  
   - 3.2 ReLU  
   - 3.3 The Lesson  
   - 3.4 The Question  
4. The Power Softmax Family  
   - 4.1 Definition  
   - 4.2 Why This Makes Sense  
   - 4.3 Mathematical Properties to Verify  
   - 4.4 Gradient Derivation  
   - 4.5 Open Questions About Power Softmax  
5. Beyond Classification: Regression  
   - 5.1 Why Softmax Doesn’t Work for Regression  
   - 5.2 Existing Bridge: Mixture Density Networks  
   - 5.3 Idea: Softmax Over Bins  
   - 5.4 Can Power Softmax Do Better Here?  
   - 5.5 Alternative: Direct Regression Without Binning  
6. Other Function Families to Consider  
   - 6.1 The Exponential-Power Hybrid  
   - 6.2 The Rational Softmax  
   - 6.3 The Entmax Family (Existing Work)  
   - 6.4 The Trigonometric Softmax  
7. Theoretical Questions a Mathematician Could Answer  
8. Experimental Plan  
9. Next Steps  
10. Open Questions for My Supervisor  
11. Preliminary References  

---

# 1. Introduction

This document summarizes my current thinking about how to approach the problem my supervisor gave me: invent a completely new activation function that can outperform Softmax in some domain, possibly regression.

The key points from our conversation were:

- Do not modify delta or existing AS-Softmax parameters
- Think from first principles as a mathematician
- Look at what the transition from Sigmoid to ReLU taught us
- Consider regression as an alternative domain to classification

---

# 2. What Softmax Is and Why It Dominates

The Softmax function is defined as:

$$
p_i = \frac{e^{o_i}}{\sum_{j=1}^{n} e^{o_j}}, \quad i = 1, \dots, n
$$

where:

- \( o \in \mathbb{R}^n \) are logits (raw neural network outputs)
- \( p \) is a probability distribution over \( n \) classes

---

## 2.1 Why Softmax Succeeded

### 1. Differentiability

The gradient is elegant and easy to compute:

$$
\frac{\partial p_i}{\partial o_j} = p_i(\delta_{ij} - p_j)
$$

### 2. Natural Probability Interpretation

Outputs lie in the interval:

$$
0 < p_i < 1
$$

and satisfy:

$$
\sum_i p_i = 1
$$

### 3. Compatibility with Cross-Entropy Loss

For target class \( t \):

$$
L = -\log p_t
$$

The gradient simplifies to:

$$
\frac{\partial L}{\partial o_j} =
\begin{cases}
p_t - 1, & j = t \\
p_j, & j \ne t
\end{cases}
$$

### 4. Convexity in Logits

The loss is convex in \( o \), which improves optimization stability.

---

## 2.2 Limitations of Softmax

Despite its success, Softmax has several known weaknesses:

1. **Extreme confidence**  
   Even moderately large logits can produce:

   $$
   p_t \approx 1
   $$

   leading to overconfidence and overfitting.

2. **No sparsity**  
   Every class receives a positive probability, even highly unlikely classes.

3. **Exponential computation cost**  
   Computing exponentials is expensive and can overflow numerically.

4. **Train-test mismatch**  
   Training pushes:

   $$
   p_t \to 1
   $$

   while testing only requires:

   $$
   p_t > p_j \quad \forall j \ne t
   $$

---

# 3. Historical Precedent: Sigmoid to ReLU

---

## 3.1 Sigmoid

The sigmoid function is:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### Problems

- Vanishing gradients for large \( |x| \)
- Outputs are not zero-centered

---

## 3.2 ReLU

The ReLU function is:

$$
\text{ReLU}(x) = \max(0, x)
$$

### Advantages

- No vanishing gradient for positive \( x \)
- Computational simplicity
- Naturally induces sparsity

---

## 3.3 The Lesson

A simpler piecewise-linear function replaced a smooth exponential function because it solved a practical optimization problem.

---

## 3.4 The Question

Can we find a similarly simpler function to replace Softmax in the output layer?

Potential goals include:

- Sparsity
- Better calibration
- Faster computation
- More natural support for regression

---

# 4. The Power Softmax Family

---

## 4.1 Definition

Define:

$$
p_i =
\frac{\max(0, o_i)^\gamma}
{\sum_{j=1}^{n} \max(0, o_j)^\gamma},
\quad \gamma \ge 0
$$

where:

$$
x^+ = \max(0, x)
$$

denotes the positive part.

---

## 4.2 Why This Makes Sense

### 1. Computational Simplicity

Powers are generally cheaper than exponentials.

### 2. Natural Sparsity

If:

$$
o_i \le 0
$$

then:

$$
p_i = 0
$$

exactly.

### 3. Tunable Sharpness

The parameter \( \gamma \) controls response sharpness.

### 4. Familiar Special Cases

- \( \gamma = 0 \): Uniform distribution over positive logits
- \( \gamma = 1 \): Normalized positive part
- \( \gamma = 2 \): Squared normalized positive part
- \( \gamma \to \infty \): Argmax behavior

---

## 4.3 Mathematical Properties to Verify

### Non-negativity

Guaranteed by construction.

### Normalization

Outputs sum to 1.

### Monotonicity

If:

$$
o_i > o_j
$$

then:

$$
p_i > p_j
$$

for \( \gamma > 0 \).

### Differentiability

Differentiable almost everywhere except at:

$$
o_i = 0
$$

similar to ReLU.

---

## 4.4 Gradient Derivation

Let:

$$
S = \sum_j (o_j^+)^\gamma
$$

Then:

$$
p_i = \frac{(o_i^+)^\gamma}{S}
$$

For \( o_i > 0 \):

$$
\frac{\partial p_i}{\partial o_i}
=
\frac{\gamma o_i^{\gamma - 1}}{S}
-
p_i \cdot
\frac{\gamma o_i^{\gamma - 1}}{S}
$$

For \( o_i \le 0 \):

$$
p_i = 0
$$

and the gradient is zero.

For cross-entropy loss:

$$
L = -\log p_t
$$

the gradient becomes:

$$
\frac{\partial L}{\partial o_i}
=
\begin{cases}
\frac{\gamma o_i^{\gamma - 1}}{S}
\cdot
\frac{p_t - 1}{p_t},
& i = t,\ o_i > 0
\\[10pt]
\frac{\gamma o_i^{\gamma - 1}}{S},
& i \ne t,\ o_i > 0
\\[10pt]
0,
& o_i \le 0
\end{cases}
$$

---

## 4.5 Open Questions About Power Softmax

1. What is the optimal value of \( \gamma \)?
2. Does sparsity improve generalization?
3. Is the gradient stable for \( \gamma < 1 \)?
4. How does calibration compare to Softmax?

---

# 5. Beyond Classification: Regression

---

## 5.1 Why Softmax Doesn’t Work for Regression

Softmax outputs a probability distribution over discrete classes, while regression requires continuous outputs.

---

## 5.2 Existing Bridge: Mixture Density Networks

Mixture density networks use Softmax for mixing coefficients:

$$
p(y|x)
=
\sum_{k=1}^{K}
\pi_k(x)
\cdot
\mathcal{N}(y \mid \mu_k(x), \sigma_k^2(x))
$$

where:

$$
\pi_k(x) = \text{softmax}(o_k)
$$

---

## 5.3 Idea: Softmax Over Bins

Discretize the target space into bins and predict probabilities over bins.

Prediction:

$$
\hat{y}
=
\sum_{i=1}^{n}
v_i p_i
$$

where \( v_i \) are bin centers.

---

## 5.4 Can Power Softmax Do Better Here?

Potential advantages:

- Sharper distributions
- Sparse predictions
- Better uncertainty calibration

Power Softmax with \( \gamma > 1 \) naturally produces sharper distributions.

---

## 5.5 Alternative: Direct Regression Without Binning

Consider:

$$
\hat{y}
=
\frac{\sum_i o_i^+ w_i}
{\sum_j o_j^+}
$$

where:

- \( w_i \) are learnable weights
- \( o_i^+ = \max(0, o_i) \)

This avoids discretization entirely.

---

# 6. Other Function Families to Consider

---

## 6.1 The Exponential-Power Hybrid

$$
p_i =
\frac{e^{o_i^p}}
{\sum_j e^{o_j^p}},
\quad p \ge 1
$$

Properties:

- \( p = 1 \): Standard Softmax
- \( p > 1 \): Exaggerates differences
- \( 0 < p < 1 \): Compresses differences

---

## 6.2 The Rational Softmax

Simplest version:

$$
p_i =
\frac{o_i^+}
{\sum_j o_j^+}
$$

Alternative form:

$$
p_i =
\frac{o_i^+ + \epsilon}
{\sum_j (o_j^+ + \epsilon)}
$$

where \( \epsilon \) prevents division by zero.

---

## 6.3 The Entmax Family

Entmax generalizes Softmax via Tsallis entropy:

$$
\text{Entmax}_\alpha(o)
=
\arg\max_{p \in \Delta}
\left(
p^T o + H_\alpha(p)
\right)
$$

where:

$$
H_\alpha(p)
=
\frac{1}{\alpha - 1}
\left(
1 - \sum_i p_i^\alpha
\right)
$$

Special cases:

- \( \alpha = 1 \): Softmax
- \( \alpha = 2 \): Sparsemax

---

## 6.4 The Trigonometric Softmax

$$
p_i =
\frac{\cos(o_i) + 1}
{\sum_j (\cos(o_j) + 1)}
$$

Potentially useful for periodic targets such as angles or time.

---

# 7. Theoretical Questions a Mathematician Could Answer

---

## 7.1 Generalization Bounds

Can Power Softmax provide tighter generalization guarantees than Softmax?

---

## 7.2 Convergence Rates

Could different gradient magnitudes improve optimization speed?

Possible relation:

$$
\|\nabla L_{\text{pow}}\|
\le
\|\nabla L_{\text{exp}}\|
$$

in certain regimes.

---

## 7.3 Information-Theoretic Interpretation

Softmax maximizes entropy under linear constraints.

Could Power Softmax correspond to maximizing Tsallis entropy?

---

## 7.4 The Limits \( \gamma \to 0 \) and \( \gamma \to \infty \)

- \( \gamma \to 0 \): Uniform distribution over positive logits
- \( \gamma \to \infty \): Argmax behavior

Can the transition phase be characterized mathematically?

---

# 8. Experimental Plan

---

## 8.1 Classification Experiments

### Datasets

- SST-5
- CLINC150
- CoNLL-2003

### Baselines

- Softmax
- AS-Softmax
- Sparsemax

### Proposed Method

- Power Softmax with varying \( \gamma \)

### Metrics

- Accuracy
- F1 Score
- Calibration Error
- Training Time

---

## 8.2 Regression Experiments

### Task

Bounded regression in:

$$
[0, 1]
$$

### Method

- Discretize outputs into bins
- Apply Power Softmax
- Use weighted-average prediction

### Baselines

- Linear Regression
- Neural Networks with Linear Output

### Metrics

- MSE
- MAE
- Calibration Quality

---

# 9. Next Steps

1. Implement Power Softmax in PyTorch
2. Test on a small dataset
3. Compare \( \gamma = 1 \) against standard Softmax
4. Run a full \( \gamma \)-sweep
5. Report findings to supervisor

---

# 10. Open Questions for My Supervisor

1. Is the Power Softmax direction promising?
2. Should the focus remain on classification or move toward regression?
3. How much theory is expected?
4. Which datasets should be prioritized?
5. Which domains matter most?

---

# 11. Preliminary References

1. Lv, Q. et al. (2023). *Adaptive Sparse Softmax*. IEEE TASLP.  
2. Martins, A. & Astudillo, R. (2016). *From Softmax to Sparsemax*. ICML.  
3. Peters, B. et al. (2019). *Sparse Sequence-to-Sequence Models*. ACL.  
4. Szegedy, C. et al. (2016). *Rethinking the Inception Architecture*. CVPR.  
