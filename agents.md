# Statistical Analysis of Stress Fields in Ensembles of RVE Simulations

## 1. Purpose

This document defines a general statistical framework for analyzing stress fields obtained from multiple realizations of representative volume elements (RVEs).

The main objective is to combine two sources of variability:

1. **Spatial variability** inside an individual RVE.
2. **Realization-to-realization variability** caused by different statistically equivalent microstructures.

The framework must remain **material-model agnostic** and must not assume isotropy. In particular, the statistical machinery must also be applicable to anisotropic, heterogeneous, textured, composite, porous, polycrystalline, and otherwise direction-dependent materials.

The implementation is intended to use the `matviz3d-py` library as the HDF5/MatViz3D data-access layer.

---

# 2. Conceptual Model

Let

- \(r=1,\ldots,M\) — RVE realization;
- \(x\in\Omega_r\) — spatial point / integration point / voxel;
- \(\boldsymbol{\sigma}^{(r)}(x)\) — local Cauchy stress tensor.

The complete stochastic object is

\[
\boldsymbol{\Sigma}=\boldsymbol{\sigma}(X,R),
\]

where

\[
R \sim p(R)
\]

represents the random microstructure realization and

\[
X\sim p(X\mid R)
\]

represents a spatially sampled location inside the RVE.

This gives two distinct sources of variability:

\[
\boxed{
\text{spatial heterogeneity}
+
\text{microstructure-realization variability}
}
\]

They must not be conflated.

---

# 3. Stress Tensor Representation

The MatViz3D `results` dataset uses the following component enumeration:

```python
enum tensor_components {
    ID,
    X, Y, Z,
    UX, UY, UZ,
    SX, SY, SZ,
    SXY, SYZ, SXZ,
    EpsX, EpsY, EpsZ,
    EpsXY, EpsYZ, EpsXZ,
    USUM,
    SEQV,
    EpsEQV
};
```

The six stress components are therefore

\[
\boldsymbol{\sigma}
=
\begin{bmatrix}
\sigma_x &
\sigma_y &
\sigma_z &
\tau_{xy} &
\tau_{yz} &
\tau_{xz}
\end{bmatrix}.
\]

Equivalently,

\[
\boldsymbol{\sigma}
=
\begin{pmatrix}
\sigma_x & \tau_{xy} & \tau_{xz}\\
\tau_{xy} & \sigma_y & \tau_{yz}\\
\tau_{xz} & \tau_{yz} & \sigma_z
\end{pmatrix}.
\]

The statistical framework must work with the full tensor and must not impose an isotropic constitutive assumption.

---

# 4. Two Levels of Probability Distributions

## 4.1 Conditional spatial distribution

For a fixed RVE realization \(r\), define

\[
p(Q\mid R=r)
\]

for a scalar quantity

\[
Q(x)=g(\boldsymbol{\sigma}^{(r)}(x)).
\]

Examples include:

- individual stress components;
- hydrostatic stress;
- deviatoric stress measures;
- von Mises equivalent stress;
- principal stresses;
- stress invariants;
- Lode-related parameters;
- arbitrary user-defined scalar quantities.

This distribution describes **spatial heterogeneity inside one RVE**.

---

## 4.2 Ensemble distribution

If all RVE realizations are equally probable,

\[
p(R=r)=\frac{1}{M}.
\]

The unconditional distribution is

\[
\boxed{
p(Q)
=
\frac{1}{M}
\sum_{r=1}^{M}
p(Q\mid R=r)
}
\]

or, for arbitrary realization weights \(w_r\),

\[
\boxed{
p(Q)
=
\sum_{r=1}^{M}
w_r\,p(Q\mid R=r),
\qquad
\sum_r w_r=1.
}
\]

This is the correct conceptual method for combining spatial and realization variability.

The implementation should therefore **not simply concatenate all spatial samples from all RVEs** unless equal-volume/equal-sampling assumptions have been explicitly verified.

---

# 5. Variance Decomposition

For a scalar quantity \(Q\), the law of total variance gives

\[
\boxed{
\operatorname{Var}(Q)
=
E_R[\operatorname{Var}(Q\mid R)]
+
\operatorname{Var}_R(E[Q\mid R]).
}
\]

This provides a physically meaningful decomposition.

## 5.1 Within-RVE variance

\[
V_{\mathrm{within}}
=
E_R[\operatorname{Var}(Q\mid R)].
\]

This quantifies spatial heterogeneity.

## 5.2 Between-RVE variance

\[
V_{\mathrm{between}}
=
\operatorname{Var}_R(E[Q\mid R]).
\]

This quantifies variability caused by different microstructure realizations.

## 5.3 Total variance

\[
V_{\mathrm{total}}
=
V_{\mathrm{within}}
+
V_{\mathrm{between}}.
\]

The relative contributions can be reported as

\[
\eta_{\mathrm{within}}
=
\frac{V_{\mathrm{within}}}{V_{\mathrm{total}}},
\]

\[
\eta_{\mathrm{between}}
=
\frac{V_{\mathrm{between}}}{V_{\mathrm{total}}}.
\]

This allows statements such as:

> A specified fraction of the observed stress variability is associated with spatial heterogeneity, while the remainder is associated with microstructure realization variability.

---

# 6. Direct Tensor-to-Quantity Transformation

For every spatial sample,

\[
\boldsymbol{\sigma}^{(r)}(x)
\rightarrow
Q^{(r)}(x)
=
g(\boldsymbol{\sigma}^{(r)}(x)).
\]

For example, the von Mises stress is

\[
\sigma_{\mathrm{VM}}
=
\sqrt{\frac32\mathbf{s}:\mathbf{s}},
\]

where

\[
\mathbf{s}
=
\boldsymbol{\sigma}
-
\frac13\operatorname{tr}(\boldsymbol{\sigma})\mathbf I.
\]

Equivalently,

\[
\sigma_{\mathrm{VM}}
=
\sqrt{
\frac12
\left[
(\sigma_x-\sigma_y)^2+
(\sigma_y-\sigma_z)^2+
(\sigma_z-\sigma_x)^2
\right]
+
3(\tau_{xy}^2+\tau_{yz}^2+\tau_{xz}^2)
}.
\]

The important implementation principle is:

> Compute derived quantities directly from the joint stress samples whenever the original samples are available.

Do not reconstruct a derived distribution only from six independently estimated marginal PDFs unless the dependence structure is explicitly modeled.

---

# 7. Joint Distribution Versus Marginal Distributions

Suppose the six stress components are

\[
\boldsymbol{\sigma}
=
(\sigma_x,\sigma_y,\sigma_z,\tau_{xy},\tau_{yz},\tau_{xz}).
\]

The six marginal PDFs

\[
p(\sigma_x),\ldots,p(\tau_{xz})
\]

do not uniquely define

\[
p(\boldsymbol{\sigma}).
\]

The dependence structure between components is important.

For example,

\[
\operatorname{Cov}(\sigma_x,\sigma_y)
\]

and more generally the full dependence structure may strongly affect the distribution of

\[
\sigma_{\mathrm{VM}}
=
g(\boldsymbol{\sigma}).
\]

Therefore the preferred data path is

\[
\boxed{
\text{raw joint stress samples}
\rightarrow
g(\boldsymbol{\sigma})
\rightarrow
\text{distribution of }Q
}
\]

rather than

\[
\text{six marginal PDFs}
\rightarrow
\text{assumed independence}
\rightarrow
Q.
\]

---

# 8. If Only Marginal PDFs Are Available

If the original joint samples are unavailable, a Monte Carlo reconstruction can be performed.

### Simplest approximation

Sample independently:

\[
\sigma_i^{(k)}\sim p_i(\sigma_i).
\]

However, this assumes independence and should be treated as an explicit modeling assumption.

### Improved approach

Use marginal distributions together with a dependence model, for example a copula:

\[
p(\boldsymbol{\sigma})
=
C\left(
F_1(\sigma_1),\ldots,F_6(\sigma_6)
\right)
\prod_i p_i(\sigma_i).
\]

Possible dependence models include:

- Gaussian copula;
- Student-\(t\) copula;
- empirical / nonparametric dependence models.

The reconstructed joint samples can then be transformed into any desired scalar or tensor characteristic.

---

# 9. Material Anisotropy

The framework is explicitly **not based on isotropy**.

No assumption such as

\[
p(\sigma_x)=p(\sigma_y)=p(\sigma_z)
\]

or

\[
E_x=E_y=E_z
\]

is required.

Likewise, the stress statistics may depend on:

- crystallographic texture;
- preferred orientation;
- anisotropic elasticity;
- anisotropic plasticity;
- phase morphology;
- directional loading;
- material interfaces;
- spatially correlated microstructure.

The coordinate system used for the statistical analysis must therefore be explicitly tracked.

If local material coordinate systems are available, both options should be supported:

1. analysis in the **global coordinate system**;
2. transformation into a **local material coordinate system** before statistical analysis.

These are different analyses and must not be silently mixed.

---

# 10. Stress Invariants

A general tensor analysis module should provide invariant quantities in addition to individual components.

Define

\[
I_1=\operatorname{tr}(\boldsymbol{\sigma}).
\]

The hydrostatic stress is

\[
\sigma_h=\frac{I_1}{3}.
\]

Define

\[
\mathbf{s}
=
\boldsymbol{\sigma}
-
\frac{I_1}{3}\mathbf I.
\]

Then

\[
J_2=\frac12\mathbf{s}:\mathbf{s},
\]

and

\[
J_3=\det(\mathbf{s}).
\]

The von Mises equivalent stress is

\[
\sigma_{\mathrm{VM}}=\sqrt{3J_2}.
\]

Principal stresses are obtained from the eigenvalues

\[
\sigma_1,\sigma_2,\sigma_3
\]

of the symmetric stress tensor.

---

# 11. Lode-Related Quantities

A general implementation may also calculate the Lode angle from

\[
\cos(3\theta)
=
\frac{3\sqrt{3}}{2}
\frac{J_3}{J_2^{3/2}}.
\]

Numerical protection must be used near

\[
J_2=0.
\]

The implementation should define and document its exact Lode-angle convention because several conventions are used in the literature.

The important point is that \(J_2\), \(J_3\), principal stresses, and Lode parameters are calculated from the **full local stress tensor**, not inferred from marginal component distributions.

---

# 12. Probability in Stress-Invariant Space

For advanced analysis, instead of only calculating

\[
p(\sigma_{\mathrm{VM}})
\]

we can estimate a joint density such as

\[
p(p,q,\theta),
\]

where

\[
p=-\frac13I_1,
\]

\[
q=\sqrt{3J_2}.
\]

Possible representations include:

\[
p(p,q),
\]

\[
p(q,\theta),
\]

or

\[
p(p,q,\theta).
\]

Kernel density estimation (KDE), multidimensional histograms, or other density estimators can be used.

This provides a probabilistic description of the stress-state population rather than a single scalar equivalent-stress distribution.

---

# 13. Probabilistic Stress / Yield Surface Analysis

The same framework can be used for probabilistic yield-surface investigations.

For an ensemble of local stress states,

\[
\left\{
(p,q,\theta)^{(r)}(x)
\right\},
\]

estimate

\[
f(p,q,\theta).
\]

Density contours or superlevel sets can then be defined by

\[
f(p,q,\theta)=c.
\]

Different values of \(c\) produce different probability-density regions.

This should be interpreted as a **probabilistic stress-state region** unless an explicit failure/yield criterion has been established. A density contour by itself is not automatically a yield surface.

---

# 14. Spatial Correlation and Effective Sample Size

A critical issue is that integration points inside an RVE are generally **not statistically independent**.

If an RVE contains 15,625 spatial samples, this does not necessarily mean that it provides 15,625 independent observations.

Spatial correlation should therefore be considered when estimating:

- confidence intervals;
- uncertainty of PDF estimates;
- uncertainty of mean values;
- KDE bandwidth;
- effective sample size.

The framework should distinguish:

\[
N_{\mathrm{points}}
\]

from

\[
N_{\mathrm{effective}}.
\]

Possible future extensions include estimating spatial autocorrelation lengths and effective sample size from the stress field or derived scalar field.

---

# 15. Ensemble Histogram Algorithm

For a scalar quantity \(Q\), use common bins for all realizations.

For realization \(r\):

\[
h_r(q_k)
\approx
p(q_k\mid R=r).
\]

Then calculate

\[
\boxed{
h_{\mathrm{ensemble}}(q_k)
=
\sum_r w_r h_r(q_k).
}
\]

For equally weighted realizations:

\[
h_{\mathrm{ensemble}}(q_k)
=
\frac1M\sum_r h_r(q_k).
\]

This preserves the intended realization weighting.

The implementation should support:

- fixed user-defined bins;
- automatically determined global range;
- number of bins;
- linear bins;
- logarithmic bins where physically appropriate;
- normalized PDF;
- CDF;
- empirical quantiles.

---

# 16. KDE Algorithm

For each RVE:

\[
\hat p_r(q)
=
\frac{1}{Nh}
\sum_{i=1}^{N}
K\left(
\frac{q-q_i}{h}
\right).
\]

Then:

\[
\boxed{
\hat p(q)
=
\sum_r w_r\hat p_r(q).
}
\]

Again, the ensemble KDE should be an average of realization-level densities rather than a blind concatenation of all spatial points.

For correlated spatial data, bandwidth and uncertainty estimates must be interpreted carefully.

---

# 17. Recommended Software Architecture

Use `matviz3d-py` for data access and create a separate statistical-analysis layer.

Suggested structure:

```text
matviz3d-py/
│
├── pymv3d/
│   ├── ...
│   └── HDF5 / MatViz3D data access
│
└── statistical analysis layer
    ├── stress.py
    ├── invariants.py
    ├── distributions.py
    ├── ensemble.py
    ├── kde.py
    └── correlation.py
```

Alternatively, keep the analysis in a separate repository/package:

```text
matviz3d-statistics/
```

This avoids coupling the core HDF5 reader to higher-level statistical methods.

---

# 18. Proposed Python API

The main abstraction should be an ensemble:

```python
ensemble = RVEEnsemble(files)
```

Example:

```python
files = [
    "ansys_angle000.00_r0.hdf5",
    "ansys_angle000.00_r1.hdf5",
    "ansys_angle000.00_r2.hdf5",
]

ensemble = RVEEnsemble(files)
```

Analysis:

```python
result = ensemble.analyze(
    set_index=0,
    load_step=1,
    quantity="von_mises",
    bins=200,
)
```

Potential result structure:

```python
result.rve_stats
result.histograms
result.ensemble_pdf
result.bin_centers
result.variance_decomposition
```

---

# 19. Supported Scalar Quantities

The first implementation should support at least:

```text
SX
SY
SZ
SXY
SYZ
SXZ

mean_stress
hydrostatic_stress

j2
j3

von_mises

principal_1
principal_2
principal_3

triaxiality
lode_angle
```

The design should also allow custom functions:

```python
quantity=lambda stress: ...
```

where

```python
stress.shape == (..., 6)
```

and the function returns

```python
quantity.shape == stress.shape[:-1]
```

This makes the algorithm independent of any particular material model or constitutive law.

---

# 20. Vectorized Tensor Processing

The implementation should use NumPy vectorization.

Input:

```python
stress.shape == (Nx, Ny, Nz, 6)
```

or

```python
stress.shape == (N, 6)
```

Output:

```python
von_mises.shape == (Nx, Ny, Nz)
```

No Python loop over individual spatial points should be required.

This is important for large RVE datasets and future GPU implementations.

---

# 21. Spatial Versus Ensemble Statistics

For each RVE, calculate:

\[
\mu_r=E_X[Q\mid R=r]
\]

\[
s_r^2=\operatorname{Var}_X(Q\mid R=r).
\]

Store at least:

```text
mean
std
min
max
q01
q05
q25
q50
q75
q95
q99
```

Across realizations, calculate:

```text
mean of RVE means
std of RVE means
quantiles of RVE means
within-RVE variance
between-RVE variance
total variance
```

This creates a clear distinction between:

```text
distribution of local values
```

and

```text
distribution of RVE-level response
```

---

# 22. Local Distribution Versus Homogenized Response

Two statistically different questions must be supported.

## Question A — Local spatial statistics

> What is the probability of finding a local stress state of value \(Q\) at a randomly selected point in a randomly selected RVE?

This gives

\[
p(Q).
\]

## Question B — RVE-to-RVE response uncertainty

> How much does the homogenized response change between statistically equivalent RVE realizations?

For each realization calculate a macroscopic quantity:

\[
\bar{\boldsymbol{\sigma}}^{(r)}
\]

and then construct

\[
p(\bar{\boldsymbol{\sigma}}).
\]

These distributions must not be mixed.

---

# 23. Loading Conditions

The statistical model should also support multiple loading conditions.

Let

\[
L
\]

denote the loading condition.

The general object becomes

\[
\boxed{
p(Q\mid L)
=
\sum_r
p(Q\mid R=r,L)p(R=r\mid L).
}
\]

For an ensemble with equal realization weights:

\[
p(Q\mid L)
=
\frac1M
\sum_r
p(Q\mid R=r,L).
\]

Possible loading parameters include:

- strain tensor;
- stress tensor;
- loading direction;
- angle;
- strain magnitude;
- time/load step;
- boundary-condition type.

The software should therefore treat `load_step` and `set_index` as explicit analysis dimensions rather than hard-coded assumptions.

---

# 24. Orientation and Anisotropy

For anisotropic materials, the orientation of the tensor is part of the data.

If local coordinate systems are available, the software should support tensor transformation:

\[
\boldsymbol{\sigma}_{local}
=
\mathbf R^T
\boldsymbol{\sigma}_{global}
\mathbf R.
\]

However, this transformation must only be performed when the analysis explicitly requests it.

Default behavior should preserve the original coordinate system.

This avoids accidentally removing physically meaningful anisotropy.

---

# 25. Reproducibility

Every statistical result should record:

```text
input files
RVE identifiers
load case
load step
quantity definition
coordinate system
sampling method
histogram bins
KDE method
KDE bandwidth
weights
software version
matviz3d-py version
```

The statistical result should therefore be reproducible from the original HDF5 datasets.

---

# 26. Initial Implementation Plan

## Phase 1 — Data access

Use `matviz3d-py` to load:

```text
RVE
  └── load case
      └── stress field
```

Return a vectorized stress array:

```python
(..., 6)
```

---

## Phase 2 — Tensor calculations

Implement:

```python
stress_tensor()
deviatoric_stress()
invariants()
principal_stresses()
von_mises()
triaxiality()
lode_angle()
```

---

## Phase 3 — Single-RVE statistics

Implement:

```python
describe()
histogram()
cdf()
quantiles()
kde()
```

for a single realization.

---

## Phase 4 — Ensemble statistics

Implement:

```python
RVEEnsemble
```

with:

```python
per_rve_statistics()
ensemble_histogram()
ensemble_kde()
variance_decomposition()
```

---

## Phase 5 — Multidimensional distributions

Implement:

```python
joint_histogram()
joint_kde()
```

for:

\[
(p,q),
\quad
(q,\theta),
\quad
(p,q,\theta).
\]

---

## Phase 6 — Uncertainty and spatial correlation

Add:

```python
effective_sample_size()
spatial_autocorrelation()
confidence_intervals()
```

where appropriate.

---

# 27. Target End-to-End Workflow

The intended final workflow is:

```text
                    HDF5 RVE files
                           │
                           ▼
                    matviz3d-py
                           │
                           ▼
                  stress tensor fields
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       individual RVE              RVE ensemble
              │                         │
              ▼                         ▼
       tensor quantities          weighted mixture
              │                         │
              ▼                         ▼
       spatial PDF               ensemble PDF
              │                         │
              └────────────┬────────────┘
                           ▼
                 variance decomposition
                           │
                           ▼
             multidimensional distributions
                           │
                           ▼
                p(p,q), p(q,θ), p(p,q,θ)
                           │
                           ▼
             probabilistic stress-state
                    characterization
```

---

# 28. Important Statistical Principle

The central principle of the implementation is:

\[
\boxed{
\text{Do not confuse spatial samples with independent RVE realizations.}
}
\]

An RVE containing many thousands of integration points provides detailed information about **spatial heterogeneity**, but it does not automatically provide thousands of independent microstructure realizations.

Therefore:

\[
N_{\mathrm{points}}
\neq
N_{\mathrm{independent\ realizations}}.
\]

The two statistical levels must remain explicit in both the mathematical model and the software API.

---

# 29. Generality of the Framework

The framework is intentionally independent of:

- isotropic versus anisotropic material behavior;
- elastic versus plastic constitutive behavior;
- polycrystalline versus composite microstructure;
- a specific FEM solver;
- a particular scalar equivalent-stress definition;
- a particular probability distribution;
- a particular density-estimation technique.

The input is fundamentally:

\[
\boxed{
\boldsymbol{\sigma}^{(r)}(x;L)
}
\]

and the user defines the quantity of interest:

\[
\boxed{
Q=g(\boldsymbol{\sigma},L,x,r).
}
\]

This makes the framework suitable for general statistical analysis of heterogeneous material simulations.

---

# 30. Recommended First Prototype

The first prototype should use the uploaded file

```text
ansys_angle000.00_r0.hdf5
```

as one realization and validate the pipeline on:

1. `SX`;
2. `SEQV` / independently calculated von Mises;
3. hydrostatic stress;
4. \(J_2\);
5. \(J_3\);
6. principal stresses.

After validation, add

```text
r1
r2
...
rM
```

and verify:

\[
p(Q)
=
\frac1M
\sum_r p(Q\mid R=r).
\]

Only after this validation should KDE, copula reconstruction, multidimensional density estimation, and probabilistic yield-surface analysis be added.
