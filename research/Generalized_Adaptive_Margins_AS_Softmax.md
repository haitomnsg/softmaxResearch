# Beyond Fixed Margins
## Generalizing Adaptive Sparse Softmax

**Author:** Sandesh Thakuri  
**Mentor:** Dr. Yagya Raj Pandeya  
**Institution:** Kathmandu University  
**Department:** Department of Artificial Intelligence  
**Date:** March 4, 2026  

---

# Table of Contents

1. Research Context: The AS-Softmax Paper  
2. Research Gap: Three Unanswered Questions  
3. Proposed Research: Generalized Adaptive Margins  
4. Direction 1: Class-Dependent Margins  
5. Direction 2: Sample-Dependent Margins  
6. Direction 3: Time-Adaptive Margins  
7. Research Questions  
8. Research Gap: Why This is Novel  
9. Experimental Plan  
10. Expected Contributions  
11. Timeline  
12. Open Questions for Discussion  
13. References  

---

# 1. Research Context: The AS-Softmax Paper

## Original Contribution (Lv et al., 2023)

The original AS-Softmax paper identified a fundamental mismatch in standard Softmax training objectives:

- **Training objective:**  
  $$
  p_t \to 1
  $$

- **Test-time objective:**  
  $$
  p_t > p_j
  \quad \forall j \ne t
  $$

To address this mismatch, the authors proposed **Adaptive Sparse Softmax (AS-Softmax)**:

$$
p_t - p_{i \ne t} \ge \delta,
\quad \delta \in (0,1]
$$

### Key Insight

Easy samples are automatically discarded once the condition is satisfied.

### Empirical Findings

- Better performance
- Faster convergence
- Strong correlation between loss and accuracy:

  $$
  \rho \approx -0.95
  $$

### Limitation

The margin \( \delta \) is a **global scalar**, meaning:

- Same margin for all class pairs
- Same margin for all samples
- Same margin throughout training

---

# 2. Research Gap: Three Unanswered Questions

---

## Question 1: Are All Class Pairs Equally Difficult?

Example in sentiment analysis:

- Distinguishing **“very negative”** vs **“negative”** is harder than
- Distinguishing **“very negative”** vs **“very positive”**

Should both use the same margin \( \delta \)?

---

## Question 2: Are All Samples Equally Ambiguous?

A clearly positive review is easier than a neutral review.

Should the model require equal confidence for both?

---

## Question 3: Should Margins Remain Static During Training?

Training evolves over time:

- Early stages learn general patterns
- Later stages focus on difficult examples

Should the margin adapt during optimization?

---

### Core Observation

The fixed margin \( \delta \) is mathematically convenient, but not necessarily optimal.

---

# 3. Proposed Research: Generalized Adaptive Margins

## Core Idea

Replace the scalar margin \( \delta \) with a more expressive mathematical object.

---

## Three Directions of Generalization

### 1. Class-Dependent Margins

$$
\delta_{t,j}
$$

Margin depends on the relationship between target class \( t \) and non-target class \( j \).

---

### 2. Sample-Dependent Margins

$$
\delta(x)
$$

Margin depends on the input sample \( x \).

---

### 3. Time-Adaptive Margins

$$
\delta(t)
$$

Margin changes during training.

---

## General Formulation

$$
p_t - p_j \ge \delta_{t,j}(x,t),
\quad \forall j \ne t
$$

---

# 4. Direction 1: Class-Dependent Margins

## Motivation

Some class pairs are inherently harder to separate than others.

---

## Mathematical Formulation

$$
p_t - p_j \ge \delta_{t,j},
\quad \delta_{t,j} \in (0,1]
$$

---

## Possible Definitions of \( \delta_{t,j} \)

### 1. Semantic Similarity

Using embedding similarity:

$$
\delta_{t,j}
=
\delta_{\min}
+
(\delta_{\max} - \delta_{\min})
\cdot
\text{sim}(e_t, e_j)
$$

where:

- \( e_t, e_j \) are BERT/Word2Vec embeddings
- \( \text{sim}(\cdot,\cdot) \) is cosine similarity

---

### 2. Confusion-Based Margins

Define margins proportional to confusion scores from a pretrained model:

$$
\delta_{t,j}
\propto
\text{Confusion}(t,j)
$$

---

### 3. Learnable Low-Rank Factorization

Parameterize:

$$
\delta_{t,j}
=
\sigma(u_t^T v_j)
$$

where:

- \( u_t, v_j \in \mathbb{R}^k \)
- \( k \ll n \)

This yields a parameter-efficient representation.

---

## Key Mathematical Insight

The margin matrix:

$$
M \in \mathbb{R}^{n \times n}
$$

may possess low-rank structure because classes naturally form clusters.

---

# 5. Direction 2: Sample-Dependent Margins

## Motivation

Not all samples within the same class are equally ambiguous.

---

## Mathematical Formulation

$$
p_t - p_j \ge \delta(x),
\quad \forall j \ne t
$$

where:

$$
\delta(x)
$$

depends on the input sample.

---

## Implementation Approaches

### 1. Confidence-Based Margins

Define:

$$
\delta(x)
=
\sigma(\text{entropy}(p(x)))
$$

Interpretation:

- Higher uncertainty
- Smaller required margin

---

### 2. Learnable Margin Function

Use a small neural network:

$$
h_\phi(x) \to \delta(x)
$$

and jointly train with the classifier.

---

### 3. Feature-Norm Based Margins

Inspired by SphereFace and CosFace:

$$
\delta(x)
\propto
\|f(x)\|
$$

where:

$$
f(x)
$$

is the feature vector.

---

## Key Research Question

Can sample-dependent margins improve calibration quality?

---

# 6. Direction 3: Time-Adaptive Margins

## Motivation

Training requirements evolve over time.

---

## Mathematical Formulation

$$
p_t - p_j \ge \delta(t),
\quad \forall j \ne t
$$

---

## Possible Schedules

### 1. Increasing Schedule

Start easy and gradually become stricter:

$$
\delta(t)
=
\delta_{\min}
+
(\delta_{\max} - \delta_{\min})
\frac{t}{T}
$$

---

### 2. Decreasing Schedule

Start strict and gradually relax:

$$
\delta(t)
=
\delta_{\max}
-
(\delta_{\max} - \delta_{\min})
\frac{t}{T}
$$

This resembles AS-Softmax warm-up strategies.

---

### 3. Cyclical Schedule

Explore multiple training regimes:

$$
\delta(t)
=
\delta_{\text{base}}
+
\delta_{\text{amp}}
\sin\left(
\frac{2\pi t}{T_{\text{cycle}}}
\right)
$$

---

## Key Question

Can the optimal schedule be derived from gradient flow dynamics?

---

# 7. Research Questions

---

## RQ1: Class-Dependent Margins

Can class-pair specific margins improve performance, especially for datasets with semantic or hierarchical structure?

---

## RQ2: Sample-Dependent Margins

Do adaptive sample margins improve probability calibration?

---

## RQ3: Time-Adaptive Margins

What scheduling strategy is theoretically and empirically optimal?

---

## RQ4: Theoretical Foundations

Can we derive tighter generalization bounds based on the structure of the margin matrix?

---

# 8. Research Gap: Why This is Novel

| Approach | Class-Pair | Sample-Dependent | Time-Adaptive |
|---|---|---|---|
| AS-Softmax (2023) | No | No | Warm-up only |
| AM-Softmax (2018) | No | No | No |
| X2-Softmax (2024) | Angular-space only | No | No |
| Largest Margins (2022) | Class-global only | No | No |
| **Our Proposal** | **Yes** | **Yes** | **Yes** |

---

## Key Novel Contributions

1. First proposal of class-pair margins in probability space
2. First systematic exploration of sample-dependent margins
3. First systematic study of adaptive margin schedules

---

# 9. Experimental Plan

---

## Datasets

Following AS-Softmax benchmarks:

- SST-5  
  *(sentiment classification, 5 classes)*

- CLINC150  
  *(intent classification, 151 classes)*

- CoNLL-2003  
  *(NER, 9 classes)*

- SIGHAN2015  
  *(Chinese spelling correction, 5201 classes)*

---

## Baselines

- Standard Softmax
- AS-Softmax
- X2-Softmax
- Proposed adaptive variants

---

## Metrics

### Classification Metrics

- Accuracy
- F1 Score

### Calibration Metrics

- Expected Calibration Error (ECE)

### Optimization Metrics

- Training time
- Convergence speed
- Masked sample ratio per class

---

# 10. Expected Contributions

---

## 1. Mathematical Contributions

- Generalized AS-Softmax framework
- Analysis of low-rank margin structures
- Study of metric properties

---

## 2. Empirical Contributions

- Improved classification performance
- Better handling of hierarchical class structures

---

## 3. Practical Contributions

- Parameter-efficient implementations
- Guidelines for margin scheduling

---

## 4. Theoretical Contributions

- Margin-dependent generalization bounds

---

# 11. Timeline

| Phase | Activities |
|---|---|
| Weeks 1–2 | Reproduce AS-Softmax results; literature review |
| Weeks 3–4 | Implement semantic similarity margins |
| Weeks 5–6 | Experiments on SST-5 and CLINC150 |
| Weeks 7–8 | Implement low-rank learnable margins |
| Weeks 9–10 | Explore sample-dependent and time-adaptive variants |
| Weeks 11–12 | Theoretical analysis and first draft writing |

---

## Deliverables

- PyTorch implementation repository
- Technical report / conference paper
- Mathematical appendix with proofs

---

# 12. Open Questions for Discussion

---

## Priority

Which direction should be explored first?

Current intuition:

> Class-dependent margins appear most straightforward.

---

## Scope

Should the work focus only on text classification, or extend to image/audio domains?

---

## Theory

What level of mathematical rigor is expected?

Possible directions:

- Generalization bounds
- Convergence proofs
- Optimization theory

---

## Collaboration

Should the AS-Softmax authors be contacted?

---

## Target Venue

Potential publication venues:

- ACL
- EMNLP
- ICLR

---

# 13. References

1. Lv, Q., Geng, L., Cao, Z., et al. (2023).  
   *Adaptive Sparse Softmax: An Effective and Efficient Softmax Variant for Text Classification.*  
   IEEE Transactions on Audio, Speech and Language Processing.

2. X2-Softmax: Margin Adaptive Loss Function.  
   *Expert Systems with Applications*, 2024.

3. Zhou, et al. (2022).  
   *Learning Towards the Largest Margins.*  
   International Conference on Learning Representations (ICLR).

4. Wang, F., et al. (2018).  
   *Additive Margin Softmax for Face Verification.*  
   IEEE Signal Processing Letters.
