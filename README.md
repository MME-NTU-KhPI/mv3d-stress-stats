# mv3d-stress_ensemble_statistics

A rigorous, material-model agnostic Python framework for analyzing stress fields in ensembles of representative volume elements (RVEs) simulated with [MatViz3D](https://github.com/MME-NTU-KhPI/MatViz3D) / ANSYS.

---

## 1. Key Objectives

Combines two distinct levels of variability without conflation:
1. **Spatial variability** within an individual RVE realization:
   $$p(Q \mid R=r)$$
2. **Realization-to-realization variability** between microstructures:
   $$p(Q) = \sum_{r=1}^M w_r p(Q \mid R=r)$$

### Law of Total Variance
Decomposes observed variability for any scalar quantity $Q$:
$$\operatorname{Var}(Q) = \underbrace{E_R[\operatorname{Var}(Q \mid R)]}_{V_{\text{within}}\text{ (spatial heterogeneity)}} + \underbrace{\operatorname{Var}_R(E[Q \mid R])}_{V_{\text{between}}\text{ (realization uncertainty)}}$$

$$\eta_{\text{within}} = \frac{V_{\text{within}}}{V_{\text{total}}}, \qquad \eta_{\text{between}} = \frac{V_{\text{between}}}{V_{\text{total}}}$$

---

## 2. Architecture & Modules

```text
mv3d_stress_stats/
├── reader.py       # HDF5 data access layer (pymv3d / h5py integration, nodal/voxel grids)
├── tensors.py      # 6-component <-> 3x3 symmetric tensor, coordinate rotation R.T @ sigma @ R
├── invariants.py   # I1, J2, J3, von Mises, principal stresses, Lode angle, triaxiality
├── single_rve.py   # Single-RVE spatial summary statistics, histograms, CDF, KDE
├── ensemble.py     # RVEEnsemble, variance decomposition, mixture PDF/CDF/KDE, macro response
├── joint.py        # 2D/3D joint invariant distributions p(p,q), probability contours
├── copula.py       # Gaussian copula reconstruction vs independent marginals bias analysis
├── correlation.py  # 3D spatial autocorrelation R(r), correlation length, effective sample size N_eff
└── plotting.py     # Matplotlib publication plots (PDFs, variance breakdown, joint density)
```

---

## 3. Quick Start

### Basic Analysis Example

```python
from mv3d_stress_stats import RVEEnsemble, RVESimulation

# 1. Define ensemble of RVE realizations
ensemble = RVEEnsemble(
    sources=["ansys_angle000.00_r0.hdf5", "ansys_angle000.00_r1.hdf5"],
    weights=[0.5, 0.5],
)

# 2. Analyze scalar quantity across ensemble
result = ensemble.analyze(
    quantity="von_mises",  # or "SX", "j2", "j3", "principal_1", "lode_angle", or custom lambda
    set_index=0,
    load_step=1,
    bins=100,
    kde=True,
)

# 3. Inspect variance decomposition
vd = result.variance_decomposition
print(vd.summary())

# 4. Access per-RVE statistics DataFrame
df_stats = result.to_dataframe()
print(df_stats)
```

---

## 4. Visualizations & Prototype Results

Running the end-to-end prototype pipeline ([`examples/run_prototype.py`](examples/run_prototype.py)) validates all 6 phases of analysis on `ansys_angle000.00_r0.hdf5` and generates high-resolution figures in [`figures/`](figures/):

### Summary Dashboard
![Summary Dashboard](figures/summary_dashboard.png)

---

### Detailed Figure Breakdown

#### A. Ensemble Probability Density Function (`figures/ensemble_pdf.png`)
![Ensemble PDF](figures/ensemble_pdf.png)
* **Red line**: Unconditional ensemble mixture PDF $p(Q) = \sum_r w_r p(Q \mid R=r)$.
* **Dashed navy line**: Ensemble Kernel Density Estimate (KDE).
* **Faint gray curves**: Individual RVE spatial distributions $p(Q \mid R=r)$.
* Preserves intended realization weights without naive concatenation of spatial points.

---

#### B. Variance Decomposition (`figures/variance_decomposition.png`)
![Variance Decomposition](figures/variance_decomposition.png)
* Separates **Within-RVE variance** ($V_{\text{within}} = E_R[\operatorname{Var}(Q \mid R)]$, spatial heterogeneity) from **Between-RVE variance** ($V_{\text{between}} = \operatorname{Var}_R(E[Q \mid R])$, microstructure realization uncertainty).
* Enables quantitative statements such as: *"67.0% of observed stress variability originates from spatial heterogeneity, while 33.0% stems from realization-to-realization variability."*

---

#### C. Invariant Stress-State Joint Distribution (`figures/joint_invariant_distribution.png`)
![Joint Distribution](figures/joint_invariant_distribution.png)
* Joint probability density in stress-invariant space: $p(p, q)$, where $p = -\frac{1}{3}I_1$ (hydrostatic pressure) and $q = \sigma_{\text{VM}}$ (von Mises stress).
* White contour overlays indicate $50\%$, $90\%$, and $95\%$ probability content superlevel sets for probabilistic yield surface and failure domain characterization.

---

#### D. Spatial Autocorrelation & Effective Sample Size (`figures/spatial_autocorrelation.png`)
![Spatial Autocorrelation](figures/spatial_autocorrelation.png)
* Computed via 3D FFT (Wiener–Khinchin theorem).
* Identifies the spatial correlation length $\lambda$ (distance where $R(r)$ decays to $1/e$).
* Corrects sample size: integration points in an RVE are not independent ($N_{\text{points}} \neq N_{\text{effective}}$). In the benchmark, $15,625$ integration points yield $N_{\text{effective}} \approx 2,280$, producing statistically rigorous confidence intervals on the mean stress.

---

## 5. Running the Pipeline & Unit Tests

### Execute Prototype Analysis & Generate Figures
```bash
python examples/run_prototype.py
```

### Run Full Test Suite
```bash
pytest -v
```

All 22 unit tests validate:
- 6-component $\leftrightarrow$ $3 \times 3$ symmetric tensor conversions and coordinate frame rotations ($\mathbf{R}^T \boldsymbol{\sigma} \mathbf{R}$)
- Stress invariants ($I_1, J_2, J_3$), principal stresses ($\sigma_1 \ge \sigma_2 \ge \sigma_3$), and Lode angle conventions
- Analytical validation of the Law of Total Variance
- Direct ANSYS validation against `ansys_angle000.00_r0.hdf5` (`von_mises` matches ANSYS `SEQV` with 0 numerical error)
- Copula dependence modeling and spatial autocorrelation
