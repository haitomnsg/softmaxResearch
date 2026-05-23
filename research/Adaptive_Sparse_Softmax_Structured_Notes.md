# Adaptive Sparse Softmax: An Effective and Efficient Softmax Variant

## Authors

- Qi Lv
- Lei Geng
- Ziqiang Cao
- Min Cao
- Sujian Li
- Wenjie Li
- Guohong Fu

---

## Abstract

Softmax with cross-entropy loss is the standard configuration for modern neural classification models. However, the target probability of the gold class is theoretically pushed toward 1 during training, even though classification only requires the correct class to rank above the others during testing.

This mismatch leads to:

- Endless optimization
- Overfitting
- Inefficient learning on already-correct samples

To address this issue, the authors propose **Adaptive Sparse Softmax (AS-Softmax)**, which discards classes whose probabilities are sufficiently smaller than the target class during training.

The method:

- Focuses learning on hard negative classes
- Matches training objectives with testing objectives
- Reduces overfitting
- Accelerates training through adaptive gradient accumulation

Experiments across text, image, and audio classification tasks show:

- Better classification performance
- Stronger loss-performance correlation
- Approximately **1.2× training speedup**

---

# 1. Introduction

Softmax is widely used as the final activation function in neural classification models.

The standard Softmax probability is:

genui{"math_block_widget_always_prefetch_v2":{"content":"p_i=\\frac{e^{o_i}}{\\sum_{j=1}^{n}e^{o_j}}"}}

where:

- \( o_i \) is the logit for class \( i \)
- \( p_i \) is the predicted probability

---

## Problems with Standard Softmax

### 1. Endless Training

Softmax continually pushes:

\[
p_t \to 1
\]

even when the model already classifies correctly.

---

### 2. Train-Test Objective Mismatch

Training objective:

\[
p_t \to 1
\]

Testing only requires:

\[
p_t > p_j, \quad \forall j \ne t
\]

---

### 3. Overfitting

The model continues learning from easy samples that are already confidently correct.

---

## Example

### Case A

\[
\{0.4, 0.15, 0.15, 0.15, 0.15\}
\]

Correct classification with clear separation.

### Case B

\[
\{0.49, 0.5, 0.004, 0.003, 0.003\}
\]

Incorrect classification despite lower cross-entropy loss.

---

# 2. Related Work

The paper reviews two major categories of Softmax improvements:

---

## 2.1 Improving Efficiency

Methods include:

- Hierarchical Softmax
- D-Softmax
- Sparse-Softmax
- RF-Softmax
- Sampled Softmax
- Spherical Softmax

---

## 2.2 Improving Effectiveness

Methods include:

- AM-Softmax
- SphereFace
- Label Smoothing
- Sparsemax
- Entmax

---

# 3. Background: Standard Softmax

For output vector:

\[
o \in \mathbb{R}^n
\]

Softmax computes:

genui{"math_block_widget_always_prefetch_v2":{"content":"p_i=\\frac{e^{o_i}}{\\sum_{j=1}^{n}e^{o_j}}"}}

---

## Cross-Entropy Loss

For target class \( t \):

\[
L = -\log(p_t)
\]

Equivalent form:

\[
L = \log\left(\sum_{j=1}^{n} e^{o_j}\right) - o_t
\]

---

## Gradient

\[
\frac{\partial L}{\partial o_j}
=
\begin{cases}
p_j - 1, & j=t \\
p_j, & j \ne t
\end{cases}
\]

---

## Required Margin in Softmax

The paper cites:

\[
o_t - o_{\min} \ge \log(n-1)
\]

This becomes unnecessarily large for high-dimensional classification.

---

# 4. Adaptive Sparse Softmax (AS-Softmax)

## Core Idea

Instead of forcing the target probability toward 1, AS-Softmax only requires:

genui{"math_block_widget_always_prefetch_v2":{"content":"p_t-p_{i\\neq t}\\geq\\delta"}}

where:

- \( \delta \in (0,1] \)
- \( \delta \) is a margin hyperparameter

---

## Binary Masking Variable

Define:

\[
z_i =
\begin{cases}
0, & \text{if } p_t - p_i \ge \delta \text{ and } i \ne t \\
1, & \text{otherwise}
\end{cases}
\]

Easy negative classes are removed from training.

---

## AS-Softmax Probability

\[
\tilde{p}_i
=
\frac{z_i e^{o_i}}
{\sum_{j=1}^{n} z_j e^{o_j}}
\]

---

## Main Advantages

### 1. Focus on Hard Negatives

Only difficult competing classes contribute to gradients.

---

### 2. Reduced Overfitting

Easy samples gradually produce zero loss.

---

### 3. Better Alignment with Testing

The objective directly encourages correct ranking.

---

# 5. Mathematical Analysis

## Theoretical Margin Requirement

The authors prove:

\[
o_t - o_{\min}
\ge
\log(n\delta + 1)
\]

Compared with standard Softmax:

\[
o_t - o_{\min}
\ge
\log(n-1)
\]

This is significantly easier to satisfy.

---

## Proof Sketch

From:

\[
p_t - p_{\min} \ge \delta
\]

they derive:

\[
e^{o_t} \ge (n\delta + 1)e^{o_{\min}}
\]

Applying logarithms gives:

\[
o_t - o_{\min}
\ge
\log(n\delta + 1)
\]

---

# 6. Multi-Label Extension

The paper extends AS-Softmax to multi-label classification.

---

## Original Multi-Label Loss

\[
L
=
\log\left(1+\sum_{i\in\Omega_{neg}}e^{o_i}\right)
+
\log\left(1+\sum_{t\in\Omega_{pos}}e^{-o_t}\right)
\]

---

## Multi-Label Margin Constraints

\[
p_t^{\min} - p_i \ge \delta
\]

and

\[
p_t - p_i^{\max} \ge \delta
\]

---

## Final Multi-Label AS-Softmax Loss

\[
L
=
\log\left(
1+\sum_{i\in\Omega_{neg}} z_i e^{o_i}
\right)
+
\log\left(
1+\sum_{t\in\Omega_{pos}} z_t e^{-o_t}
\right)
\]

---

# 7. Adaptive Gradient Accumulation (AS-Speed)

Because AS-Softmax masks easy samples, the effective batch size decreases over time.

The authors propose:

\[
\text{steps}_{accum}
=
\lambda
\cdot
\frac{N_{all}}
{N_{all}-N_{masked}}
\]

where:

- \( \lambda \) controls acceleration
- \( N_{masked} \) is the number of masked samples

---

## Benefits

- Faster training
- Better GPU utilization
- Dynamic optimization scheduling

---

# 8. Experimental Setup

## Datasets

### Text Classification

- SST5
- CLINC150
- CoNLL2003
- SIGHAN2015
- Eurlex
- WOS-46985

### Image Classification

- WikiArt

### Audio Classification

- Chest_falsetto

---

## Baselines

- Softmax
- T-Softmax
- Sparse-Softmax
- Sparsemax
- Entmax
- Label Smoothing
- AM-Softmax

---

# 9. Main Results

## Classification Performance

AS-Softmax consistently outperformed standard Softmax across most tasks.

Key findings:

- Better accuracy
- Better F1 scores
- Better calibration
- Less overfitting

---

## Speed Improvements

AS-Speed achieved approximately:

\[
1.2\times
\]

training acceleration.

---

## Correlation Between Loss and Accuracy

AS-Softmax produced significantly stronger negative Pearson correlations between loss and validation accuracy.

Example:

- Standard Softmax on SST5:

\[
0.038
\]

- AS-Softmax on SST5:

\[
-0.952
\]

This indicates better alignment between optimization and actual classification quality.

---

# 10. Key Insights

## 1. Hard Sample Learning

AS-Softmax naturally focuses on difficult samples.

---

## 2. Better Generalization

Reducing optimization on easy samples helps mitigate overfitting.

---

## 3. Sparse Training Dynamics

Many samples eventually contribute zero loss.

---

## 4. Interpretability

The probability margin distribution directly reflects the training objective.

---

# 11. Limitations

The paper identifies several limitations:

- Additional hyperparameters:
  - \( \delta \)
  - warm-up ratio \( r \)

- Requires tuning per dataset

- Performance depends on task difficulty

---

# 12. Future Work

The authors propose future directions including:

- Adaptive selection of \( \delta \)
- Better theoretical analysis
- Improved dynamic scheduling
- Extension to additional modalities

---

# 13. Conclusion

Adaptive Sparse Softmax introduces a simple but effective alternative to standard Softmax.

Main contributions include:

- Margin-based masking of easy classes
- Better train-test alignment
- Reduced overfitting
- Adaptive acceleration strategy

The method demonstrates strong empirical performance across text, image, and audio classification tasks while remaining easy to implement.

---

# References

1. Qi Lv et al. *Adaptive Sparse Softmax: An Effective and Efficient Softmax Variant.*
2. Martins & Astudillo. *From Softmax to Sparsemax.*
3. Peters et al. *Sparse Sequence-to-Sequence Models.*
4. Szegedy et al. *Rethinking the Inception Architecture.*
5. Liu et al. *AM-Softmax and SphereFace.*
