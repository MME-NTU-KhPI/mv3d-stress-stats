# Mathematical and Algorithmic Specifications

This document provides the formal mathematical formulations and step-by-step algorithms implemented in `mv3d-stress_ensemble_statistics`.

---

## Table of Contents
1. [Stochastic Framework & Conceptual Model](#1-stochastic-framework--conceptual-model)
2. [Algorithm 1: Direct Vectorized Tensor Transformation & Invariants](#2-algorithm-1-direct-vectorized-tensor-transformation--invariants)
3. [Algorithm 2: Anisotropic Coordinate Transformation](#3-algorithm-2-anisotropic-coordinate-transformation)
4. [Algorithm 3: Law of Total Variance Decomposition](#4-algorithm-3-law-of-total-variance-decomposition)
5. [Algorithm 4: Ensemble Probability Density & CDF Mixture](#5-algorithm-4-ensemble-probability-density--cdf-mixture)
6. [Algorithm 5: Stress-Invariant Joint Probability & Contour Superlevel Sets](#6-algorithm-5-stress-invariant-joint-probability--contour-superlevel-sets)
7. [Algorithm 6: Copula Dependence Modeling vs. Independent Marginal Bias](#7-algorithm-6-copula-dependence-modeling-vs-independent-marginal-bias)
8. [Algorithm 7: 3D Spatial Autocorrelation & Effective Sample Size ($N_{\text{eff}}$)](#8-algorithm-7-3d-spatial-autocorrelation--effective-sample-size-n_texttext-eff)

---

## 1. Stochastic Framework & Conceptual Model

Let:
* $R \in \{1, \ldots, M\}$ denote the random microstructure realization with probability distribution $p(R=r) = w_r$, where $\sum_{r=1}^M w_r = 1$.
* $X \in \Omega_r$ denote a spatial coordinate inside RVE realization $r$ sampled according to spatial density $p(X \mid R=r)$.
* $\boldsymbol{\sigma}^{(r)}(x) \in \mathbb{R}^{3 \times 3}_{\text{sym}}$ denote the local symmetric Cauchy stress tensor at point $x$.

The complete stochastic object is:
$$\boldsymbol{\Sigma} = \boldsymbol{\sigma}(X, R)$$

### Dual Levels of Probability Distributions
1. **Conditional Spatial Distribution**: Describes spatial heterogeneity within realization $r$ for any scalar quantity $Q(x) = g(\boldsymbol{\sigma}^{(r)}(x))$:
   $$p(Q \mid R=r)$$
2. **Unconditional Ensemble Distribution**: Weighted mixture across all realizations:
   $$p(Q) = \sum_{r=1}^M w_r p(Q \mid R=r)$$

---

## 2. Algorithm 1: Direct Vectorized Tensor Transformation & Invariants

Derived quantities $Q = g(\boldsymbol{\sigma})$ are evaluated directly on the raw joint stress tensor samples to preserve cross-component covariance without assuming component independence.

### 2.1 Input Representation
Stress is stored as a 6-component vector array of shape `(..., 6)`:
$$\boldsymbol{\sigma} = \begin{bmatrix} \sigma_x & \sigma_y & \sigma_z & \tau_{xy} & \tau_{yz} & \tau_{xz} \end{bmatrix}^T$$

Symmetric $3 \times 3$ tensor mapping:
$$\mathbf{T}(\boldsymbol{\sigma}) = \begin{pmatrix} \sigma_x & \tau_{xy} & \tau_{xz} \\ \tau_{xy} & \sigma_y & \tau_{yz} \\ \tau_{xz} & \tau_{yz} & \sigma_z \end{pmatrix}$$

### 2.2 Mathematical Formulations
1. **First Invariant ($I_1$) & Hydrostatic Stress ($\sigma_h$)**:
   $$I_1 = \operatorname{tr}(\boldsymbol{\sigma}) = \sigma_x + \sigma_y + \sigma_z, \qquad \sigma_h = \frac{I_1}{3}$$

2. **Deviatoric Stress Tensor ($\mathbf{s}$)**:
   $$\mathbf{s} = \boldsymbol{\sigma} - \sigma_h \mathbf{I} = \begin{bmatrix} \sigma_x - \sigma_h & \sigma_y - \sigma_h & \sigma_z - \sigma_h & \tau_{xy} & \tau_{yz} & \tau_{xz} \end{bmatrix}^T$$

3. **Second Deviatoric Invariant ($J_2$) & von Mises Stress ($\sigma_{\text{VM}}$)**:
   $$J_2 = \frac{1}{2} \mathbf{s} : \mathbf{s} = \frac{1}{6}\left[ (\sigma_x - \sigma_y)^2 + (\sigma_y - \sigma_z)^2 + (\sigma_z - \sigma_x)^2 \right] + \tau_{xy}^2 + \tau_{yz}^2 + \tau_{xz}^2$$
   $$\sigma_{\text{VM}} = \sqrt{3 J_2}$$

4. **Third Deviatoric Invariant ($J_3$)**:
   $$J_3 = \det(\mathbf{s}) = s_x s_y s_z + 2 \tau_{xy}\tau_{yz}\tau_{xz} - s_x \tau_{yz}^2 - s_y \tau_{xz}^2 - s_z \tau_{xy}^2$$

5. **Principal Stresses ($\sigma_1 \ge \sigma_2 \ge \sigma_3$)**:
   Eigenvalues of $\mathbf{T}(\boldsymbol{\sigma})$ computed via LAPACK symmetric eigensolver `eigvalsh`:
   $$\det(\mathbf{T}(\boldsymbol{\sigma}) - \lambda \mathbf{I}) = 0 \implies \sigma_1 = \lambda_{\max}, \quad \sigma_2 = \lambda_{\text{mid}}, \quad \sigma_3 = \lambda_{\min}$$

6. **Lode Angle ($\theta$) & Lode Parameter ($\mu$)**:
   $$\cos(3\theta) = \frac{3\sqrt{3}}{2} \frac{J_3}{J_2^{3/2}}, \qquad \theta \in \left[0, \frac{\pi}{3}\right]$$
   $$\mu = \frac{2\sigma_2 - \sigma_1 - \sigma_3}{\sigma_1 - \sigma_3}, \qquad \mu \in [-1, 1]$$
   *Numerical Protection*: When $J_2 < \epsilon$ ($\epsilon = 10^{-12}$), the ratio is clipped to $[-1, 1]$, with degenerate states defaulting to $\theta = \pi/6$ (pure shear / neutral).

7. **Stress Triaxiality ($\eta$)**:
   $$\eta = \frac{\sigma_h}{\max(\sigma_{\text{VM}}, \epsilon)}$$

---

## 3. Algorithm 2: Anisotropic Coordinate Transformation

The framework does not assume material isotropy ($p(\sigma_x) \neq p(\sigma_y)$). When transforming from global coordinates to local material axes:

$$\boldsymbol{\sigma}_{\text{local}} = \mathbf{R}^T \boldsymbol{\sigma}_{\text{global}} \mathbf{R}$$

where $\mathbf{R} \in \mathrm{SO}(3)$ is the orthogonal transformation matrix whose columns are the local basis unit vectors.

### Vectorized Implementation via Einstein Summation
For a batch of tensors `(..., 3, 3)` and rotation matrix $\mathbf{R}$:
$$(\boldsymbol{\sigma}_{\text{local}})_{kl} = R_{ik} (\boldsymbol{\sigma}_{\text{global}})_{ij} R_{jl} \quad \implies \quad \texttt{np.einsum('...ik,...ij,...jl->...kl', R, sigma, R)}$$

---

## 4. Algorithm 3: Law of Total Variance Decomposition

Quantifies the proportion of observed variability arising from internal spatial heterogeneity versus microstructure realization uncertainty.

```
                      Total Stress Variability: Var(Q)
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
 Within-RVE Variance: V_within                      Between-RVE Variance: V_between
E_R[Var(Q|R)] (Spatial Heterogeneity)            Var_R(E[Q|R]) (Realization Uncertainty)
```

### Computational Steps
1. **Spatial Moments per Realization $r$**:
   $$\mu_r = E_X[Q \mid R=r] = \frac{1}{N_r} \sum_{i=1}^{N_r} Q_i^{(r)}$$
   $$s_r^2 = \operatorname{Var}_X(Q \mid R=r) = \frac{1}{N_r} \sum_{i=1}^{N_r} (Q_i^{(r)} - \mu_r)^2$$

2. **Ensemble Homogenized Mean**:
   $$\bar{\mu} = E_R[E_X[Q \mid R]] = \sum_{r=1}^M w_r \mu_r$$

3. **Within-RVE Variance**:
   $$V_{\text{within}} = E_R[\operatorname{Var}_X(Q \mid R)] = \sum_{r=1}^M w_r s_r^2$$

4. **Between-RVE Variance**:
   $$V_{\text{between}} = \operatorname{Var}_R(E_X[Q \mid R]) = \sum_{r=1}^M w_r (\mu_r - \bar{\mu})^2$$

5. **Variance Fractions**:
   $$V_{\text{total}} = V_{\text{within}} + V_{\text{between}}$$
   $$\eta_{\text{within}} = \frac{V_{\text{within}}}{V_{\text{total}}}, \qquad \eta_{\text{between}} = \frac{V_{\text{between}}}{V_{\text{total}}}$$

---

## 5. Algorithm 4: Ensemble Probability Density & CDF Mixture

### Problem with Naive Concatenation
Concatenating all spatial points $\{Q_i^{(r)}\}$ across realizations implicitly assumes that each realization has identical spatial sample counts and equal physical volume. If sample counts differ, realizations with finer meshes artificially dominate the statistics.

### Weighted Mixture Algorithm
1. **Global Dynamic Range**:
   $$q_{\min} = \min_{r=1\ldots M} \left(\min_{x} Q^{(r)}(x)\right), \qquad q_{\max} = \max_{r=1\ldots M} \left(\max_{x} Q^{(r)}(x)\right)$$
2. **Common Partition**:
   Divide $[q_{\min}, q_{\max}]$ into $K$ intervals with bin edges $b_0 < b_1 < \dots < b_K$ and centers $c_k = \frac{b_{k-1} + b_k}{2}$, width $\Delta q_k = b_k - b_{k-1}$.
3. **Per-Realization Density Normalization**:
   For each realization $r$, compute histogram counts $n_r(k)$ and normalize to PDF:
   $$h_r(k) = \frac{n_r(k)}{N_r \Delta q_k}, \qquad \sum_{k=1}^K h_r(k) \Delta q_k = 1$$
4. **Ensemble Weighted Mixture**:
   $$h_{\text{ensemble}}(k) = \sum_{r=1}^M w_r h_r(k)$$
5. **Cumulative Distribution Function (CDF)**:
   $$F_{\text{ensemble}}(k) = \sum_{j=1}^k h_{\text{ensemble}}(j) \Delta q_j$$
6. **Ensemble Kernel Density Estimation (KDE)**:
   $$\hat{p}_{\text{ensemble}}(q) = \sum_{r=1}^M w_r \left[ \frac{1}{N_r b_r} \sum_{i=1}^{N_r} \mathcal{K}\left( \frac{q - Q_i^{(r)}}{b_r} \right) \right]$$

---

## 6. Algorithm 5: Stress-Invariant Joint Probability & Contour Superlevel Sets

For multi-axial failure and yield analysis, the stress state is represented in invariant coordinate spaces such as $(p, q)$, $(q, \theta)$, or $(p, q, \theta)$, where $p = -\sigma_h$ and $q = \sigma_{\text{VM}}$.

```
Hydrostatic Pressure p ──┐
                         ├──> Joint Estimator ──> p(p, q) ──> Inversion ──> Superlevel Contours {f >= c_alpha}
Equivalent Stress q    ──┘
```

### Contour Superlevel Set Algorithm
To identify the stress-state domain enclosing probability content $\alpha \in (0, 1)$ (e.g. 50%, 90%, 95%):

1. Compute 2D ensemble density grid $f_{i,j} \approx p(p_i, q_j)$ with cell areas $\Delta A_{i,j} = \Delta p_i \Delta q_j$.
2. Flatten $f_{i,j}$ and sort in descending order: $f_{(1)} \ge f_{(2)} \ge \dots \ge f_{(P)}$.
3. Form the cumulative probability sequence:
   $$C_m = \sum_{k=1}^m f_{(k)} \Delta A_{(k)}$$
4. Locate smallest index $m^*$ such that $C_{m^*} \ge \alpha$.
5. The density contour threshold is $c_\alpha = f_{(m^*)}$.
6. The probabilistic stress boundary is the isoline $\{ (p, q) : f(p, q) = c_\alpha \}$.

---

## 7. Algorithm 6: Copula Dependence Modeling vs. Independent Marginal Bias

### The Independence Pitfall (AGENTS.md Section 7)
Given 6 stress components $(\sigma_x, \sigma_y, \sigma_z, \tau_{xy}, \tau_{yz}, \tau_{xz})$, knowing the marginal distributions $p(\sigma_x), \ldots, p(\tau_{xz})$ is **insufficient** to determine the distribution of nonlinear derived quantities like von Mises:
$$\sigma_{\text{VM}} = \sqrt{\frac{1}{2}\left[(\sigma_x - \sigma_y)^2 + (\sigma_y - \sigma_z)^2 + (\sigma_z - \sigma_x)^2\right] + 3(\tau_{xy}^2 + \tau_{yz}^2 + \tau_{xz}^2)}$$

If $\sigma_x$ and $\sigma_y$ are positively correlated, $(\sigma_x - \sigma_y)^2$ is small. Assuming independence artificially inflates $(\sigma_x - \sigma_y)^2$, producing severe positive bias in equivalent stress!

### Gaussian Copula Reconstruction Pipeline
1. **Marginal Uniform Scores via Empirical Ranks**:
   $$U_{i,j} = \frac{\operatorname{rank}(\sigma_{i,j})}{N + 1}, \quad j = 1, \ldots, 6$$
2. **Normal Score Transformation**:
   $$Z_{i,j} = \Phi^{-1}(U_{i,j})$$
3. **Correlation Matrix in Gaussian Space**:
   $$\mathbf{C} = \operatorname{corr}(\mathbf{Z}) \in \mathbb{R}^{6 \times 6}$$
   Project $\mathbf{C}$ to nearest positive semi-definite matrix if needed via eigenvalue thresholding.
4. **Monte Carlo Sampling**:
   Sample $\mathbf{Z}^* \sim \mathcal{N}(\mathbf{0}, \mathbf{C})$.
   Convert to uniforms: $\mathbf{U}^* = \Phi(\mathbf{Z}^*)$.
5. **Marginal Quantile Inversion**:
   $$\sigma_{i,j}^* = F_j^{-1}(U_{i,j}^*)$$
6. **Bias Evaluation**:
   Compare $\mu_{\text{raw}}$, $\mu_{\text{copula}}$, and $\mu_{\text{independent}}$:
   $$\Delta_{\text{bias}} = \frac{\mu_{\text{independent}} - \mu_{\text{raw}}}{\mu_{\text{raw}}} \times 100\%$$

---

## 8. Algorithm 7: 3D Spatial Autocorrelation & Effective Sample Size ($N_{\text{eff}}$)

Integration points in a continuum simulation are spatially correlated through elliptic PDEs and constitutive equations:
$$N_{\text{points}} \neq N_{\text{effective}}$$

### 3D Spatial Autocorrelation via Wiener–Khinchin Theorem
For a 3D scalar field $Q(x, y, z)$ on grid $(N_x, N_y, N_z)$:

1. **Centering & Zero-Padding**:
   $$q(x, y, z) = Q(x, y, z) - \bar{Q}$$
   Pad $q$ to size $(2N_x, 2N_y, 2N_z)$ with zeros to prevent periodic wrap-around.
   Create mask $\mathbf{M}$ of ones on domain, zero elsewhere.
2. **FFT-Based Convolution**:
   $$\hat{q} = \operatorname{FFT}_{3D}(q_{\text{pad}}), \qquad \hat{M} = \operatorname{FFT}_{3D}(\mathbf{M}_{\text{pad}})$$
   $$\operatorname{Cov}(\mathbf{r}) = \operatorname{Re}\left( \operatorname{IFFT}_{3D}(|\hat{q}|^2) \right)$$
   $$\operatorname{Counts}(\mathbf{r}) = \operatorname{Re}\left( \operatorname{IFFT}_{3D}(|\hat{M}|^2) \right)$$
3. **Normalized Autocorrelation**:
   $$R(\mathbf{r}) = \frac{\operatorname{Cov}(\mathbf{r})}{\operatorname{Counts}(\mathbf{r}) \cdot \operatorname{Var}(Q)}$$
4. **Radial Averaging**:
   Bin lags by radial distance $r = \|\mathbf{r}\|_2$ to obtain 1D function $R(r)$.

### Effective Sample Size & Confidence Intervals
1. **Correlation Length ($\lambda$)**:
   Distance where autocorrelation drops to $1/e$:
   $$R(\lambda) = \frac{1}{e} \approx 0.3679$$
2. **Correlation Volume ($V_c$) & Domain Volume ($V$)**:
   $$V_c = \frac{4}{3}\pi \lambda^3, \qquad V = N_x N_y N_z \Delta x^3$$
3. **Effective Sample Size**:
   $$N_{\text{eff}} = \operatorname{clip}\left( \frac{V}{V_c}, 1.0, N_{\text{points}} \right)$$
4. **Adjusted Standard Error & 95% Confidence Interval**:
   $$\mathrm{SE}_{\text{naive}} = \frac{s}{\sqrt{N_{\text{points}}}}, \qquad \mathrm{SE}_{\text{effective}} = \frac{s}{\sqrt{N_{\text{eff}}}}$$
   $$\mathrm{CI}_{95\%} = \left[ \bar{Q} - 1.96 \cdot \mathrm{SE}_{\text{effective}}, \; \bar{Q} + 1.96 \cdot \mathrm{SE}_{\text{effective}} \right]$$
