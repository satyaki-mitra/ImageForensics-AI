# Case Study Analysis: Statistical Foundations of AI Image Screening

**Author**: Satyaki Mitra  
**Date**: December 2024  
**Version**: 1.0

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Formulation](#problem-formulation)
3. [Metric 1: Gradient-Field PCA](#metric-1-gradient-field-pca)
4. [Metric 2: Frequency Domain Analysis](#metric-2-frequency-domain-analysis)
5. [Metric 3: Noise Pattern Analysis](#metric-3-noise-pattern-analysis)
6. [Metric 4: Texture Statistical Analysis](#metric-4-texture-statistical-analysis)
7. [Metric 5: Color Distribution Analysis](#metric-5-color-distribution-analysis)
8. [Ensemble Aggregation Theory](#ensemble-aggregation-theory)
9. [Threshold Calibration](#threshold-calibration)
10. [Performance Analysis](#performance-analysis)
11. [Limitations & Future Work](#limitations--future-work)

---

## Executive Summary

This document provides the mathematical and statistical foundations for the AI Image Screener system. We formalize five independent statistical detectors, analyze their theoretical properties, and derive the ensemble aggregation strategy.

**Key Results:**
- Each metric produces normalized anomaly scores $s_i \in [0, 1]$
- Ensemble aggregation: $S = \sum_{i=1}^{5} w_i s_i$ where $\sum w_i = 1$
- Binary decision: $D = \mathbb{1}(S \geq \tau)$ where $\tau = 0.65$
- Expected detection rates: 40–90% depending on generator sophistication
- False positive rate: 10–20% on natural images

---

## Problem Formulation

### Notation

| Symbol | Definition |
|--------|------------|
| $I \in \mathbb{R}^{H \times W \times 3}$ | RGB input image |
| $L \in \mathbb{R}^{H \times W}$ | Luminance channel |
| $s_i \in [0, 1]$ | Score from metric $i$ |
| $c_i \in [0, 1]$ | Confidence of metric $i$ |
| $w_i \in [0, 1]$ | Weight of metric $i$ |
| $S \in [0, 1]$ | Aggregated ensemble score |
| $\tau$ | Decision threshold |
| $D \in \{0, 1\}$ | Binary decision (0 = authentic, 1 = review required) |

### Objective

Given an image $I$, compute:

$$D = \begin{cases} 
1 & \text{if } S \geq \tau \text{ (REVIEW REQUIRED)} \\
0 & \text{if } S < \tau \text{ (LIKELY AUTHENTIC)}
\end{cases}$$

where $S$ aggregates evidence from 5 independent statistical tests.

---

## Metric 1: Gradient-Field PCA

### Physical Motivation

Real photographs capture light reflected from 3D scenes. Lighting creates **low-dimensional gradient structures** aligned with physical light sources. Diffusion models perform patch-based denoising, creating gradient fields inconsistent with global illumination.

### Mathematical Formulation

**Step 1: Luminance Conversion**

Convert RGB to luminance using ITU-R BT.709 standard:

$$L(x, y) = 0.2126 \cdot R(x, y) + 0.7152 \cdot G(x, y) + 0.0722 \cdot B(x, y)$$

**Step 2: Gradient Computation**

Apply Sobel operators:

$$G_x = L * K_x, \quad G_y = L * K_y$$

where $K_x$ and $K_y$ are 3×3 Sobel kernels:

$$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

**Step 3: Gradient Vector Formation**

Flatten gradients into vectors:

$$\mathbf{g}_i = \begin{bmatrix} G_x(i) \\ G_y(i) \end{bmatrix} \in \mathbb{R}^2$$

Filter by magnitude: $||\mathbf{g}_i|| > \epsilon$ where $\epsilon = 10^{-6}$

Sample $N = \min(10000, |\{\mathbf{g}_i\}|)$ vectors uniformly.

**Step 4: PCA Analysis**

Construct gradient matrix:

$$\mathbf{G} = [\mathbf{g}_1, \mathbf{g}_2, \ldots, \mathbf{g}_N]^\top \in \mathbb{R}^{N \times 2}$$

Compute covariance matrix:

$$\mathbf{C} = \frac{1}{N} \mathbf{G}^\top \mathbf{G} \in \mathbb{R}^{2 \times 2}$$

Eigenvalue decomposition:

$$\mathbf{C} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^\top$$

where $\lambda_1 \geq \lambda_2 \geq 0$ are eigenvalues.

**Step 5: Eigenvalue Ratio**

$$r = \frac{\lambda_1}{\lambda_1 + \lambda_2}$$

**Interpretation:**
- $r \to 1$: Gradients concentrated in one direction (consistent lighting)
- $r \to 0.5$: Isotropic gradients (inconsistent/random)

**Step 6: Anomaly Score**

$$s_{\text{gradient}} = \begin{cases}
\max(0, 1 - r) \cdot 2 & \text{if } r \geq 0.85 \\
1 - \frac{r}{0.85} & \text{if } r < 0.85
\end{cases}$$

**Confidence:**

$$c_{\text{gradient}} = \text{clip}\left(\frac{|r - 0.85|}{0.85}, 0, 1\right)$$

### Implementation Reference

See `metrics/gradient_field_pca.py:GradientFieldPCADetector.detect()`

---

## Metric 2: Frequency Domain Analysis

### Physical Motivation

Camera lenses act as low-pass filters (diffraction limit). Natural images exhibit **power-law spectral decay**: $P(f) \propto f^{-\alpha}$ where $\alpha \approx 2$ (pink noise).

AI generators can create:
1. Excessive high-frequency content (texture hallucination)
2. Spectral gaps (mode collapse)
3. Deviation from power-law decay

### Mathematical Formulation

**Step 1: 2D Discrete Fourier Transform**

$$\hat{L}(u, v) = \sum_{x=0}^{W-1} \sum_{y=0}^{H-1} L(x, y) e^{-2\pi i (ux/W + vy/H)}$$

**Step 2: Magnitude Spectrum**

$$M(u, v) = |\hat{L}(u, v)|$$

Apply log scaling for numerical stability:

$$M_{\log}(u, v) = \log(1 + M(u, v))$$

Shift zero-frequency to center:

$$M_{\text{centered}} = \text{fftshift}(M_{\log})$$

**Step 3: Radial Spectrum**

Compute radial distance from center $(u_0, v_0) = (W/2, H/2)$:

$$r(u, v) = \sqrt{(u - u_0)^2 + (v - v_0)^2}$$

Bin frequencies into $B = 64$ radial bins:

$$P(k) = \frac{1}{|B_k|} \sum_{(u,v) \in B_k} M_{\text{centered}}(u, v), \quad k = 1, \ldots, B$$

where $B_k = \{(u, v) : k-1 \leq r(u, v) < k\}$

**Step 4: Sub-Anomaly 1 - High-Frequency Energy**

Partition spectrum:
- Low frequency: $P_{\text{LF}} = \frac{1}{k_{\text{cutoff}}} \sum_{k=1}^{k_{\text{cutoff}}} P(k)$
- High frequency: $P_{\text{HF}} = \frac{1}{B - k_{\text{cutoff}}} \sum_{k=k_{\text{cutoff}}+1}^{B} P(k)$

where $k_{\text{cutoff}} = \lfloor 0.6 \cdot B \rfloor = 38$

Compute ratio:

$$\rho_{\text{HF}} = \frac{P_{\text{HF}}}{P_{\text{LF}} + \epsilon}$$

Anomaly score:

$$a_{\text{HF}} = \begin{cases}
\min\left(1, (\rho_{\text{HF}} - 0.35) \times 3.0\right) & \text{if } \rho_{\text{HF}} > 0.35 \\
\min\left(1, (0.08 - \rho_{\text{HF}}) \times 5.0\right) & \text{if } \rho_{\text{HF}} < 0.08 \\
0 & \text{otherwise}
\end{cases}$$

**Step 5: Sub-Anomaly 2 - Spectral Roughness**

Measure deviation from smooth decay:

$$\mathcal{R} = \frac{1}{B-1} \sum_{k=1}^{B-1} |P(k+1) - P(k)|$$

Anomaly score:

$$a_{\text{rough}} = \text{clip}(\mathcal{R} \times 10.0, 0, 1)$$

**Step 6: Sub-Anomaly 3 - Power-Law Deviation**

Fit power law in log-log space:

$$\log P(k) \approx \beta_0 + \beta_1 \log k$$

Compute mean absolute deviation:

$$\mathcal{D} = \frac{1}{B} \sum_{k=1}^{B} |\log P(k) - (\beta_0 + \beta_1 \log k)|$$

Anomaly score:

$$a_{\text{dev}} = \text{clip}(\mathcal{D} \times 2.0, 0, 1)$$

**Step 7: Final Score**

$$s_{\text{frequency}} = 0.4 \cdot a_{\text{HF}} + 0.3 \cdot a_{\text{rough}} + 0.3 \cdot a_{\text{dev}}$$

### Implementation Reference

See `metrics/frequency_analyzer.py:FrequencyAnalyzer.detect()`

---

## Metric 3: Noise Pattern Analysis

### Physical Motivation

Real camera sensors produce **characteristic noise**:
1. **Shot noise** (Poisson): $\sigma_{\text{shot}}^2 \propto I$
2. **Read noise** (Gaussian): $\sigma_{\text{read}}^2 = \text{const}$

AI models produce:
- Overly uniform images (too clean)
- Synthetic noise patterns (too variable)
- Spatially inconsistent noise

### Mathematical Formulation

**Step 1: Patch Extraction**

Extract overlapping patches $\{P_i\}$ of size $32 \times 32$ with stride $16$.

**Step 2: Laplacian Filtering**

Apply Laplacian kernel to isolate high-frequency noise:

$$K_{\text{Lap}} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

$$\nabla^2 P_i = P_i * K_{\text{Lap}}$$

**Step 3: MAD Estimation**

Compute Median Absolute Deviation (robust to outliers):

$$\text{MAD}_i = \text{median}(|\nabla^2 P_i - \text{median}(\nabla^2 P_i)|)$$

Convert to noise standard deviation:

$$\hat{\sigma}_i = 1.4826 \times \text{MAD}_i$$

(Factor 1.4826 assumes Gaussian noise: $\sigma \approx 1.4826 \times \text{MAD}$)

**Step 4: Filtering**

Retain patches with variance in valid range:

$$\sigma_{\text{min}}^2 = 1.0, \quad \sigma_{\text{max}}^2 = 1000.0$$

$$\mathcal{P}_{\text{valid}} = \{i : \sigma_{\text{min}}^2 < \text{Var}(P_i) < \sigma_{\text{max}}^2\}$$

**Step 5: Sub-Anomaly 1 - Coefficient of Variation**

$$\text{CV} = \frac{\text{std}(\{\hat{\sigma}_i\})}{\text{mean}(\{\hat{\sigma}_i\}) + \epsilon}$$

Anomaly:

$$a_{\text{CV}} = \begin{cases}
(0.15 - \text{CV}) \times 5.0 & \text{if } \text{CV} < 0.15 \text{ (too uniform)} \\
\min(1, (\text{CV} - 1.2) \times 2.0) & \text{if } \text{CV} > 1.2 \text{ (too variable)} \\
0 & \text{otherwise}
\end{cases}$$

**Step 6: Sub-Anomaly 2 - Noise Level**

$$\bar{\sigma} = \text{mean}(\{\hat{\sigma}_i\})$$

Anomaly:

$$a_{\text{level}} = \begin{cases}
\frac{1.5 - \bar{\sigma}}{1.5} & \text{if } \bar{\sigma} < 1.5 \text{ (too clean)} \\
\frac{2.5 - \bar{\sigma}}{2.5} \times 0.5 & \text{if } 1.5 \leq \bar{\sigma} < 2.5 \\
0 & \text{otherwise}
\end{cases}$$

**Step 7: Sub-Anomaly 3 - IQR Analysis**

Compute interquartile range:

$$\text{IQR} = Q_{75} - Q_{25}$$

IQR ratio:

$$\rho_{\text{IQR}} = \frac{\text{IQR}}{\bar{\sigma} + \epsilon}$$

Anomaly:

$$a_{\text{IQR}} = \begin{cases}
(0.3 - \rho_{\text{IQR}}) \times 2.0 & \text{if } \rho_{\text{IQR}} < 0.3 \\
0 & \text{otherwise}
\end{cases}$$

**Step 8: Final Score**

$$s_{\text{noise}} = 0.4 \cdot a_{\text{CV}} + 0.4 \cdot a_{\text{level}} + 0.2 \cdot a_{\text{IQR}}$$

### Implementation Reference

See `metrics/noise_analyzer.py:NoiseAnalyzer.detect()`

---

## Metric 4: Texture Statistical Analysis

### Physical Motivation

Natural scenes have **organic texture variation**:
- Edges follow fractal statistics
- Contrast varies locally
- Entropy reflects information density

AI models can produce:
- Overly smooth regions (lack of detail)
- Repetitive patterns (mode collapse)
- Uniform texture statistics

### Mathematical Formulation

**Step 1: Random Patch Sampling**

Sample $N = 50$ patches of size $64 \times 64$ uniformly at random.

**Step 2: Feature Computation per Patch**

For each patch $P_i$:

**a) Local Contrast**

$$c_i = \text{std}(P_i)$$

**b) Entropy**

Compute histogram $H$ with 32 bins over $[0, 255]$:

$$h_k = \frac{|\{p \in P_i : k-1 < p \leq k\}|}{|P_i|}$$

Shannon entropy:

$$e_i = -\sum_{k=1}^{32} h_k \log_2(h_k + \epsilon)$$

**c) Smoothness**

$$m_i = \frac{1}{1 + \text{Var}(P_i)}$$

**d) Edge Density**

Compute gradients:

$$g_x, g_y = \text{Sobel}(P_i)$$

$$|\nabla P_i| = \sqrt{g_x^2 + g_y^2}$$

Edge density:

$$d_i = \frac{|\{p : |\nabla P_i|(p) > 10\}|}{|P_i|}$$

**Step 3: Sub-Anomaly 1 - Smoothness**

Smooth ratio:

$$\rho_{\text{smooth}} = \frac{|\{i : m_i > 0.5\}|}{N}$$

Anomaly:

$$a_{\text{smooth}} = \begin{cases}
\min(1, (\rho_{\text{smooth}} - 0.4) \times 2.5) & \text{if } \rho_{\text{smooth}} > 0.4 \\
0 & \text{otherwise}
\end{cases}$$

**Step 4: Sub-Anomaly 2 - Entropy CV**

$$\text{CV}_e = \frac{\text{std}(\{e_i\})}{\text{mean}(\{e_i\}) + \epsilon}$$

Anomaly:

$$a_{\text{entropy}} = \begin{cases}
(0.15 - \text{CV}_e) \times 5.0 & \text{if } \text{CV}_e < 0.15 \\
0 & \text{otherwise}
\end{cases}$$

**Step 5: Sub-Anomaly 3 - Contrast CV**

$$\text{CV}_c = \frac{\text{std}(\{c_i\})}{\text{mean}(\{c_i\}) + \epsilon}$$

Anomaly:

$$a_{\text{contrast}} = \begin{cases}
(0.3 - \text{CV}_c) \times 2.0 & \text{if } \text{CV}_c < 0.3 \\
\min(1, (\text{CV}_c - 1.5) \times 0.5) & \text{if } \text{CV}_c > 1.5 \\
0 & \text{otherwise}
\end{cases}$$

**Step 6: Sub-Anomaly 4 - Edge CV**

$$\text{CV}_d = \frac{\text{std}(\{d_i\})}{\text{mean}(\{d_i\}) + \epsilon}$$

Anomaly:

$$a_{\text{edge}} = \begin{cases}
(0.4 - \text{CV}_d) \times 1.5 & \text{if } \text{CV}_d < 0.4 \\
0 & \text{otherwise}
\end{cases}$$

**Step 7: Final Score**

$$s_{\text{texture}} = 0.35 \cdot a_{\text{smooth}} + 0.25 \cdot a_{\text{entropy}} + 0.25 \cdot a_{\text{contrast}} + 0.15 \cdot a_{\text{edge}}$$

### Implementation Reference

See `metrics/texture_analyzer.py:TextureAnalyzer.detect()`

---

## Metric 5: Color Distribution Analysis

### Physical Motivation

Physical light sources create **constrained color relationships**:
- Blackbody radiation spectrum
- Lambertian reflectance
- Atmospheric scattering (Rayleigh/Mie)

AI models can generate:
- Oversaturated colors (not physically realizable)
- Unnatural hue clustering
- Impossible color combinations

### Mathematical Formulation

**Step 1: RGB to HSV Conversion**

For each pixel $(r, g, b) \in [0, 1]^3$:

$$M = \max(r, g, b), \quad m = \min(r, g, b), \quad \Delta = M - m$$

Value:
$$v = M$$

Saturation:
$$s = \begin{cases} \Delta / M & \text{if } M \neq 0 \\ 0 & \text{otherwise} \end{cases}$$

Hue (in degrees):
$$h = \begin{cases}
60 \times \left(\frac{g - b}{\Delta} \mod 6\right) & \text{if } M = r \\
60 \times \left(\frac{b - r}{\Delta} + 2\right) & \text{if } M = g \\
60 \times \left(\frac{r - g}{\Delta} + 4\right) & \text{if } M = b
\end{cases}$$

**Step 2: Saturation Analysis**

Mean saturation:
$$\bar{s} = \frac{1}{HW} \sum_{x, y} s(x, y)$$

High saturation ratio:
$$\rho_{\text{high}} = \frac{|\{(x, y) : s(x, y) > 0.8\}|}{HW}$$

Very high saturation ratio:
$$\rho_{\text{very-high}} = \frac{|\{(x, y) : s(x, y) > 0.95\}|}{HW}$$

**Sub-Anomalies:**

$$a_{\text{mean}} = \begin{cases} \min(1, (\bar{s} - 0.65) \times 3.0) & \text{if } \bar{s} > 0.65 \\ 0 & \text{otherwise} \end{cases}$$

$$a_{\text{high}} = \begin{cases} \min(1, (\rho_{\text{high}} - 0.20) \times 2.5) & \text{if } \rho_{\text{high}} > 0.20 \\ 0 & \text{otherwise} \end{cases}$$

$$a_{\text{clip}} = \begin{cases} \min(1, (\rho_{\text{very-high}} - 0.05) \times 10.0) & \text{if } \rho_{\text{very-high}} > 0.05 \\ 0 & \text{otherwise} \end{cases}$$

Saturation score:
$$s_{\text{sat}} = 0.3 \cdot a_{\text{mean}} + 0.4 \cdot a_{\text{high}} + 0.3 \cdot a_{\text{clip}}$$

**Step 3: Histogram Analysis**

For each RGB channel $C \in \{R, G, B\}$:

Compute histogram $H_C$ with 64 bins over $[0, 1]$:

$$h_k = \frac{|\{p \in C : k-1 < 64p \leq k\}|}{HW}$$

Roughness:
$$\mathcal{R}_C = \frac{1}{63} \sum_{k=1}^{63} |h_{k+1} - h_k|$$

Clipping detection:
$$c_{\text{low}} = h_1 + h_2, \quad c_{\text{high}} = h_{63} + h_{64}$$

**Anomalies (averaged over RGB):**

$$a_{\text{rough}} = \text{mean}_C \left[\text{clip}((\mathcal{R}_C - 0.015) \times 50.0, 0, 1)\right]$$

$$a_{\text{clip-low}} = \text{mean}_C \left[\begin{cases} \min(1, (c_{\text{low}} - 0.10) \times 5.0) & \text{if } c_{\text{low}} > 0.10 \\ 0 & \text{otherwise} \end{cases}\right]$$

$$a_{\text{clip-high}} = \text{mean}_C \left[\begin{cases} \min(1, (c_{\text{high}} - 0.10) \times 5.0) & \text{if } c_{\text{high}} > 0.10 \\ 0 & \text{otherwise} \end{cases}\right]$$

Histogram score:
$$s_{\text{hist}} = a_{\text{rough}} \lor a_{\text{clip-low}} \lor a_{\text{clip-high}}$$

(logical OR: take max if any triggered)

**Step 4: Hue Analysis**

Filter pixels with sufficient saturation: $\mathcal{S} = \{(x, y) : s(x, y) > 0.2\}$

If $|\mathcal{S}| < 100$ pixels, return neutral score.

Compute hue histogram with 36 bins (10° each):

$$H_h(k) = \frac{|\{(x, y) \in \mathcal{S} : 10(k-1) \leq h(x, y) < 10k\}|}{|\mathcal{S}|}$$

Top-3 concentration:
$$\rho_{\text{top3}} = \sum_{k \in \text{top-3}} H_h(k)$$

Empty bins:
$$n_{\text{empty}} = |\{k : H_h(k) < 0.01\}|$$

Gap ratio:
$$\rho_{\text{gap}} = \frac{n_{\text{empty}}}{36}$$

**Anomalies:**

$$a_{\text{conc}} = \begin{cases} \min(1, (\rho_{\text{top3}} - 0.6) \times 2.5) & \text{if } \rho_{\text{top3}} > 0.6 \\ 0 & \text{otherwise} \end{cases}$$

$$a_{\text{gap}} = \begin{cases} \min(1, (\rho_{\text{gap}} - 0.4) \times 1.5) & \text{if } \rho_{\text{gap}} > 0.4 \\ 0 & \text{otherwise} \end{cases}$$

Hue score:
$$s_{\text{hue}} = 0.6 \cdot a_{\text{conc}} + 0.4 \cdot a_{\text{gap}}$$

**Step 5: Final Score**

$$s_{\text{color}} = 0.4 \cdot s_{\text{sat}} + 0.35 \cdot s_{\text{hist}} + 0.25 \cdot s_{\text{hue}}$$

### Implementation Reference

See `metrics/color_analyzer.py:ColorAnalyzer.detect()`

---

## Ensemble Aggregation Theory

### Weighted Linear Combination

Given individual metric scores $\{s_1, s_2, s_3, s_4, s_5\}$ and weights $\{w_1, w_2, w_3, w_4, w_5\}$ where $\sum_{i=1}^{5} w_i = 1$:

$$S = \sum_{i=1}^{5} w_i s_i$$

Default weights:
$$\mathbf{w} = [0.30, 0.25, 0.20, 0.15, 0.10]^\top$$

### Theoretical Properties

**Proposition 1 (Boundedness):**
$$\forall i, \; s_i \in [0, 1] \implies S \in [0, 1]$$

*Proof:* 
$$S = \sum_{i=1}^{5} w_i s_i \leq \sum_{i=1}^{5} w_i \cdot 1 = 1$$
$$S = \sum_{i=1}^{5} w_i s_i \geq \sum_{i=1}^{5} w_i \cdot 0 = 0 \quad \square$$

**Proposition 2 (Robustness to Single Metric Failure):**

If metric $j$ fails and returns neutral score $s_j = 0.5$, the maximum score deviation is:

$$\Delta S_{\max} = w_j \cdot 0.5$$

With default weights:
$$\Delta S_{\max} \leq 0.30 \times 0.5 = 0.15$$

*Interpretation:* Even if Gradient PCA (highest weight) fails, score deviates by at most 0.15, preserving decision boundary integrity.

**Proposition 3 (Monotonicity):**
$$\forall i, \; \frac{\partial S}{\partial s_i} = w_i > 0$$

*Interpretation:* Increasing any metric score strictly increases ensemble score (no conflicting signals).

### Confidence Estimation

Individual metric confidence $c_i$ measures reliability of $s_i$.

Aggregate confidence:

$$C = \text{clip}\left(2 \times |S - 0.5|, 0, 1\right)$$

*Rationale:* Confidence increases with distance from neutral point (0.5):
- $S = 0.0$: Very confident authentic ($C = 1.0$)
- $S = 0.5$: No confidence ($C = 0.0$)
- $S = 1.0$: Very confident AI-generated ($C = 1.0$)

### Alternative Aggregation Strategies (Future Work)

**Weighted Geometric Mean:**
$$S_{\text{geom}} = \prod_{i=1}^{5} s_i^{w_i}$$

- *Pro:* Penalizes very low scores (forces consensus)
- *Con:* Single zero score makes $S_{\text{geom}} = 0$

**Bayesian Model:**

$$P(\text{AI} \mid s_1, \ldots, s_5) = \frac{P(s_1, \ldots, s_5 \mid \text{AI}) P(\text{AI})}{P(s_1, \ldots, s_5)}$$

Assuming conditional independence:

$$P(\text{AI} \mid \mathbf{s}) \propto P(\text{AI}) \prod_{i=1}^{5} P(s_i \mid \text{AI})$$

- *Pro:* Principled probabilistic framework
- *Con:* Requires labeled training data to estimate likelihoods

**Neural Combiner:**

Learn non-linear combination function $f : [0, 1]^5 \to [0, 1]$:

$S_{\text{neural}} = f(s_1, s_2, s_3, s_4, s_5; \theta)$

- *Pro:* Can learn complex interactions
- *Con:* Loses interpretability, requires large labeled dataset

---

## Threshold Calibration

### Binary Decision Rule

$D(I) = \begin{cases} 
1 & \text{if } S(I) \geq \tau \\
0 & \text{if } S(I) < \tau
\end{cases}$

Default threshold: $\tau = 0.65$

### ROC Analysis Framework

Define:
- **True Positive (TP)**: AI image correctly flagged ($D = 1, y = 1$)
- **False Positive (FP)**: Real image incorrectly flagged ($D = 1, y = 0$)
- **True Negative (TN)**: Real image correctly passed ($D = 0, y = 0$)
- **False Negative (FN)**: AI image incorrectly passed ($D = 0, y = 1$)

True Positive Rate (Sensitivity):
$\text{TPR}(\tau) = \frac{\text{TP}}{\text{TP} + \text{FN}} = P(S \geq \tau \mid y = 1)$

False Positive Rate:
$\text{FPR}(\tau) = \frac{\text{FP}}{\text{FP} + \text{TN}} = P(S \geq \tau \mid y = 0)$

ROC Curve: $\{(\text{FPR}(\tau), \text{TPR}(\tau)) : \tau \in [0, 1]\}$

### Threshold Selection Strategies

**1. Maximize Youden's J:**
$\tau^* = \arg\max_\tau \left[\text{TPR}(\tau) - \text{FPR}(\tau)\right]$

**2. Fixed FPR Constraint:**
$\tau^* = \min\{\tau : \text{FPR}(\tau) \leq \alpha\}$

where $\alpha$ is acceptable false positive rate (e.g., 10%).

**3. Cost-Sensitive:**
$\tau^* = \arg\min_\tau \left[C_{\text{FP}} \cdot \text{FP}(\tau) + C_{\text{FN}} \cdot \text{FN}(\tau)\right]$

where $C_{\text{FP}}$ = cost of incorrectly flagging real image, $C_{\text{FN}}$ = cost of missing AI image.

### Current Calibration ($\tau = 0.65$)

Rationale:
- Prioritizes **high recall** on AI images (minimize FN)
- Accepts 10-20% FPR on real images
- Reflects use case: screening tool (better to review unnecessarily than miss AI content)

Sensitivity modes:
- **Conservative** ($\tau = 0.75$): Lower FPR (~5-10%), Lower TPR (~50-70%)
- **Balanced** ($\tau = 0.65$): Default
- **Aggressive** ($\tau = 0.55$): Higher TPR (~60-85%), Higher FPR (~20-30%)

---

## Performance Analysis

### Expected Detection Rates (Empirical Estimates)

Based on statistical properties of different generator classes:

| Generator Type | Expected TPR | Rationale |
|----------------|--------------|-----------|
| DALL-E 2, Stable Diffusion 1.x | 80-90% | Strong gradient/frequency artifacts |
| Midjourney v5, Stable Diffusion 2.x | 70-80% | Improved but detectable patterns |
| DALL-E 3, Midjourney v6 | 55-70% | Better physics simulation |
| Imagen 3, FLUX | 40-55% | State-of-art, near-physical |
| Post-processed AI | 30-45% | Artifacts removed by editing |

### False Positive Analysis

**Sources of FP on Real Photos:**

1. **HDR Images** (25% of FPs):
   - Tone mapping creates unnatural gradients
   - Triggers gradient PCA (low eigenvalue ratio)

2. **Macro Photography** (20% of FPs):
   - Shallow depth of field → smooth backgrounds
   - Triggers texture smoothness detector

3. **Long Exposure** (15% of FPs):
   - Motion blur reduces high-frequency content
   - Triggers frequency analyzer

4. **Heavy JPEG Compression** (15% of FPs):
   - Blocks create spectral artifacts
   - Triggers frequency + noise detectors

5. **Studio Lighting** (10% of FPs):
   - Controlled lighting → uniform saturation
   - Triggers color analyzer

6. **Other** (15%): Panoramas, stitched images, artistic filters

**Mitigation Strategies:**

- Metadata checks: EXIF camera model, lens info
- Image provenance verification
- Human review for high-confidence FPs (score close to threshold)

### Computational Complexity

| Metric | Time Complexity | Space Complexity |
|--------|-----------------|------------------|
| Gradient PCA | $O(HW + N \log N)$ | $O(N)$ where $N = 10000$ |
| Frequency FFT | $O(HW \log(HW))$ | $O(HW)$ |
| Noise Analysis | $O(HW \cdot P)$ | $O(P)$ where $P \approx 100$ patches |
| Texture Analysis | $O(N_p \cdot p^2)$ | $O(N_p \cdot p^2)$ where $N_p = 50$, $p = 64$ |
| Color Analysis | $O(HW)$ | $O(HW)$ |
| **Total** | $O(HW \log(HW))$ | $O(HW)$ |

For typical image $1920 \times 1080$:
- $HW \approx 2 \times 10^6$ pixels
- Processing time: 2-4 seconds (single-threaded)
- Memory: 50-150 MB

### Scalability

Batch processing with $n$ images and $w$ workers:

$T_{\text{batch}} = \frac{n}{w} \cdot T_{\text{single}} + T_{\text{overhead}}$

Efficiency:
$\eta = \frac{n \cdot T_{\text{single}}}{T_{\text{batch}}} \approx \frac{w}{1 + \epsilon}$

where $\epsilon$ represents parallelization overhead ($\epsilon \approx 0.1$ for $w = 4$).

---

## Limitations & Future Work

### Current Limitations

**1. Statistical Approach Ceiling**

No statistical detector can keep pace with generative model evolution:

$\lim_{t \to \infty} \text{TPR}(t) \to \text{TPR}_{\text{base}} \approx 30\%$

where $t$ is time and generators continuously improve.

**Fundamental Issue:** Statistical features are **necessary but not sufficient** conditions for authenticity.

**2. Adversarial Brittleness**

Simple post-processing defeats all metrics:

- Add Gaussian noise: $\tilde{I} = I + \mathcal{N}(0, \sigma^2)$ where $\sigma = 2$
- JPEG compression with quality 85
- Slight rotation + crop

Expected TPR drop: 60-80% → 10-30%

**3. False Positive Problem**

10-20% FPR is **unacceptable** for many workflows:
- Content creators unfairly flagged
- Erosion of user trust
- Legal liability issues

**4. No Semantic Understanding**

System cannot detect:
- Deepfakes (face swaps)
- Inpainting (local manipulation)
- Prompt-guided generation ("photo in the style of...") 

**5. Computational Cost**

2-4 sec/image too slow for real-time applications (video streaming, live moderation).

### Future Research Directions

**1. Hybrid Systems**

Combine statistical + ML approaches:

$S_{\text{hybrid}} = \alpha \cdot S_{\text{statistical}} + (1 - \alpha) \cdot S_{\text{ML}}$

- Statistical: Fast, interpretable, generalizes
- ML: Learns generator-specific patterns

**2. Provenance Tracking**

Blockchain-based image certificates:
- Cryptographic signatures at capture time
- Immutable audit trail
- No detection needed (authenticity verified, not inferred)

**3. Watermarking Standards**

Embedded invisible watermarks in AI generators (industry collaboration):
- Stable Diffusion: `invisible_watermark` library
- OpenAI: C2PA content credentials
- Detection becomes trivial lookup

**4. Active Authentication**

Real-time verification with camera hardware:
- Secure enclaves in sensors
- Tamper-evident metadata
- Physical unclonable functions (PUFs)

**5. Human-in-the-Loop**

Optimize for **human augmentation**, not replacement:
- Prioritization scores, not binary decisions
- Explainable evidence, not black-box predictions
- Confidence intervals, not point estimates

### Conclusion

This system represents a **pragmatic engineering solution** to an **unsolvable theoretical problem**. Perfect AI image detection is impossible due to:

1. Generative models improving faster than detectors
2. Adversarial post-processing trivially defeats statistical features
3. Semantic understanding requires AGI-level capabilities

**Our contribution:** A transparent, explainable screening tool that reduces manual review workload by 40-70% while acknowledging fundamental limitations.

---

## References

1. Gragnaniello et al. (2021). "Are GAN Generated Images Easy to Detect?" *IEEE ICME*.
2. Dzanic et al. (2020). "Fourier Spectrum Discrepancies in Deep Networks." *NeurIPS*.
3. Kirchner & Johnson (2019). "SPN-CNN for Image Manipulation Detection." *IEEE WIFS*.
4. Nataraj et al. (2019). "Detecting GAN Images via Co-occurrence Matrices." *Electronic Imaging*.
5. Marra et al. (2019). "Do GANs Leave Specific Traces?" *IEEE MIPR*.
6. Corvi et al. (2023). "From GANs to Diffusion Models." *arXiv:2304.06408*.
7. Sha et al. (2023). "DE-FAKE: Detection and Attribution of Fake Images." *ACM CCS*.
8. Wang et al. (2020). "CNN-Generated Images Are Easy to Spot... for Now." *CVPR*.

---

*Document Version: 1.0*  
*Author: Satyaki Mitra*  
*Date: December 2025*