"""
Multidimensional stress-state probability distributions and invariant spaces.

Estimates joint probability densities in invariant spaces:
- p(p, q): pressure vs. equivalent stress
- p(q, theta): equivalent stress vs. Lode angle
- p(p, q, theta): complete 3D invariant representation
- Superlevel sets / density contour levels for probabilistic stress characterization.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats

from mv3d_stress_stats.invariants import (
    pressure,
    hydrostatic_stress,
    von_mises,
    lode_angle,
    resolve_quantity,
)
from mv3d_stress_stats.single_rve import RVESimulation


@dataclass
class JointDistribution2D:
    """2D joint distribution estimate (histogram or KDE)."""
    x_name: str
    y_name: str
    x_bins: np.ndarray      # 1D array of bin edges for X
    y_bins: np.ndarray      # 1D array of bin edges for Y
    x_centers: np.ndarray   # 1D array of bin centers for X
    y_centers: np.ndarray   # 1D array of bin centers for Y
    density: np.ndarray     # 2D array of shape (len(x_centers), len(y_centers))
    per_rve_densities: Optional[List[np.ndarray]] = None
    kde_density: Optional[np.ndarray] = None       # Optional 2D array evaluated via KDE
    per_rve_kdes: Optional[List[np.ndarray]] = None
    is_normalized: bool = False
    macro_x_values: Optional[np.ndarray] = None    # per-RVE Q_macro for X if normalized
    macro_y_values: Optional[np.ndarray] = None    # per-RVE Q_macro for Y if normalized

    def find_contour_level(self, probability: float, use_kde: bool = False) -> float:
        """Find density threshold c such that integral_{f >= c} f dA = probability.
        
        Args:
            probability: Desired enclosed probability content (e.g. 0.50, 0.90, 0.95).
            use_kde: If True, uses kde_density instead of histogram density.
            
        Returns:
            Density threshold value c.
        """
        target_density = self.kde_density if (use_kde and self.kde_density is not None) else self.density
        dx = np.diff(self.x_bins)
        dy = np.diff(self.y_bins)
        area_element = np.outer(dx, dy)
        
        # Flatten density and area
        f_flat = target_density.ravel()
        da_flat = area_element.ravel()
        
        # Sort descending by density
        sorter = np.argsort(f_flat)[::-1]
        f_sorted = f_flat[sorter]
        da_sorted = da_flat[sorter]
        
        cum_prob = np.cumsum(f_sorted * da_sorted)
        total_p = cum_prob[-1] if cum_prob[-1] > 0 else 1.0
        cum_prob /= total_p
        
        idx = np.searchsorted(cum_prob, probability)
        idx = min(idx, len(f_sorted) - 1)
        return float(f_sorted[idx])


def compute_joint_histogram_2d(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    bins: Union[int, Tuple[int, int]] = (50, 50),
    range_x: Optional[Tuple[float, float]] = None,
    range_y: Optional[Tuple[float, float]] = None,
    density: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D histogram of paired spatial samples."""
    x = np.asarray(x_vals).ravel()
    y = np.asarray(y_vals).ravel()
    
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    
    if isinstance(bins, int):
        n_bx, n_by = bins, bins
    else:
        n_bx, n_by = bins
        
    hist_range = None
    if range_x is not None and range_y is not None:
        hist_range = [range_x, range_y]
        
    h, x_edges, y_edges = np.histogram2d(
        x, y,
        bins=[n_bx, n_by],
        range=hist_range,
        density=density,
    )
    return h, x_edges, y_edges


def compute_joint_kde_2d(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    bandwidth: Optional[Union[float, str]] = None,
    max_samples: Optional[int] = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Compute 2D Kernel Density Estimation evaluated on a 2D grid.
    
    Args:
        x_vals: 1D array of X samples.
        y_vals: 1D array of Y samples.
        x_eval: 1D array of evaluation points along X (e.g. x_centers).
        y_eval: 1D array of evaluation points along Y (e.g. y_centers).
        bandwidth: Bandwidth method ('scott', 'silverman', or a scalar).
        max_samples: If the dataset has more than max_samples, subsample for efficient evaluation.
                     None disables subsampling.
        seed: Random seed for subsampling reproducibility.
        
    Returns:
        2D density array of shape (len(x_eval), len(y_eval)).
    """
    x = np.asarray(x_vals).ravel()
    y = np.asarray(y_vals).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    
    if len(x) == 0:
        return np.zeros((len(x_eval), len(y_eval)), dtype=np.float64)
        
    if max_samples is not None and len(x) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=max_samples, replace=False)
        x_fit = x[idx]
        y_fit = y[idx]
    else:
        x_fit = x
        y_fit = y
        
    try:
        kde = stats.gaussian_kde(np.vstack([x_fit, y_fit]), bw_method=bandwidth)
    except np.linalg.LinAlgError:
        # Near-degenerate variance: add minuscule jitter to avoid singular matrix
        rng = np.random.default_rng(seed)
        jx = 1e-8 * (np.ptp(x_fit) if np.ptp(x_fit) > 0 else 1.0)
        jy = 1e-8 * (np.ptp(y_fit) if np.ptp(y_fit) > 0 else 1.0)
        x_fit = x_fit + rng.normal(0, jx, size=len(x_fit))
        y_fit = y_fit + rng.normal(0, jy, size=len(y_fit))
        kde = stats.gaussian_kde(np.vstack([x_fit, y_fit]), bw_method=bandwidth)
        
    X, Y = np.meshgrid(x_eval, y_eval, indexing="ij")
    positions = np.vstack([X.ravel(), Y.ravel()])
    kde_grid = kde(positions).reshape(len(x_eval), len(y_eval))
    return kde_grid


def compute_ensemble_joint_2d(
    simulations: List[RVESimulation],
    weights: np.ndarray,
    x_quantity: Union[str, Callable[[np.ndarray], np.ndarray]] = "pressure",
    y_quantity: Union[str, Callable[[np.ndarray], np.ndarray]] = "von_mises",
    bins: Union[int, Tuple[int, int]] = (50, 50),
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    kde: bool = False,
    kde_bandwidth: Optional[Union[float, str]] = None,
    max_kde_samples: Optional[int] = 10000,
    normalize: Union[bool, Tuple[bool, bool]] = False,
    normalize_min_zscore: float = 3.0,
) -> JointDistribution2D:
    """Compute ensemble 2D joint distribution on a common grid.
    
    Combines spatial joint realizations across the ensemble using weighted mixtures:
        p(X, Y) = sum_r w_r * p(X, Y | R=r)
    
    Args:
        simulations: List of RVESimulation objects.
        weights: Weights w_r summing to 1.0.
        x_quantity: 'pressure', 'hydrostatic_stress', 'von_mises', 'lode_angle', or custom callable.
        y_quantity: Second variable for joint space.
        bins: Number of bins in each direction (scalar or (nb_x, nb_y)).
        x_range: Global range for X. If None, auto-detected across all realizations.
        y_range: Global range for Y. If None, auto-detected across all realizations.
        kde: If True, computes common-grid 2D KDE for each realization and ensemble mixture.
        kde_bandwidth: Optional KDE bandwidth parameter ('scott', 'silverman', or float).
        max_kde_samples: Max spatial samples per RVE used for KDE fitting (speeds up large grids).
        normalize: If True, normalizes both variables by their respective macro values
                   eta_x = X_local / X_macro and eta_y = Y_local / Y_macro.
                   Can also be a tuple (norm_x, norm_y) to normalize only one axis
                   (e.g. normalize=(False, True) for (lode_angle, von_mises)).
        normalize_min_zscore: Statistical significance guard. Raises ValueError if macro value
                              is indistinguishable from zero given spatial sampling noise.
        
    Returns:
        JointDistribution2D instance.
    """
    x_fn = resolve_quantity(x_quantity)
    y_fn = resolve_quantity(y_quantity)
    x_name = x_quantity if isinstance(x_quantity, str) else getattr(x_quantity, "__name__", "custom_x")
    y_name = y_quantity if isinstance(y_quantity, str) else getattr(y_quantity, "__name__", "custom_y")

    weights = np.asarray(weights, dtype=np.float64)
    if not np.isclose(np.sum(weights), 1.0):
        weights = weights / np.sum(weights)

    if isinstance(normalize, (tuple, list)):
        norm_x, norm_y = bool(normalize[0]), bool(normalize[1])
    else:
        norm_x, norm_y = bool(normalize), bool(normalize)
    is_normalized = norm_x or norm_y

    # 1. Extract raw samples
    x_raw_list: List[np.ndarray] = []
    y_raw_list: List[np.ndarray] = []
    for sim in simulations:
        x = x_fn(sim.stress).ravel()
        y = y_fn(sim.stress).ravel()
        valid = np.isfinite(x) & np.isfinite(y)
        x_raw_list.append(x[valid])
        y_raw_list.append(y[valid])

    # 2. Optional per-realization normalization by homogenized macro values
    macro_x: Optional[np.ndarray] = None
    macro_y: Optional[np.ndarray] = None
    if norm_x or norm_y:
        macro_stresses = np.array([sim.homogenized_stress() for sim in simulations])  # (M, 6)

    x_pairs = list(x_raw_list)
    y_pairs = list(y_raw_list)

    if norm_x:
        macro_x = x_fn(macro_stresses)
        for i, m in enumerate(macro_x):
            n_pts = len(x_raw_list[i])
            local_std = float(np.std(x_raw_list[i])) if n_pts else 0.0
            se = local_std / np.sqrt(n_pts) if n_pts > 0 else 0.0
            z = abs(m) / se if se > 0 else np.inf
            if z < normalize_min_zscore:
                raise ValueError(
                    f"Cannot normalize X variable '{x_name}': realization {i}'s macro value ({m:.3e}) "
                    f"is not statistically distinguishable from zero given its sampling noise "
                    f"(z={z:.2f} < {normalize_min_zscore}). This typically occurs when the quantity "
                    "is signed and vanishes by symmetry. Set normalize=False or normalize=(False, True)."
                )
        x_pairs = [x_vals / macro_x[i] for i, x_vals in enumerate(x_raw_list)]

    if norm_y:
        macro_y = y_fn(macro_stresses)
        for i, m in enumerate(macro_y):
            n_pts = len(y_raw_list[i])
            local_std = float(np.std(y_raw_list[i])) if n_pts else 0.0
            se = local_std / np.sqrt(n_pts) if n_pts > 0 else 0.0
            z = abs(m) / se if se > 0 else np.inf
            if z < normalize_min_zscore:
                raise ValueError(
                    f"Cannot normalize Y variable '{y_name}': realization {i}'s macro value ({m:.3e}) "
                    f"is not statistically distinguishable from zero given its sampling noise "
                    f"(z={z:.2f} < {normalize_min_zscore}). This typically occurs when the quantity "
                    "is signed and vanishes by symmetry. Set normalize=False or normalize=(True, False)."
                )
        y_pairs = [y_vals / macro_y[i] for i, y_vals in enumerate(y_raw_list)]

    # 3. Determine common grid ranges
    x_mins = [float(np.min(x)) for x in x_pairs if len(x) > 0]
    x_maxs = [float(np.max(x)) for x in x_pairs if len(x) > 0]
    y_mins = [float(np.min(y)) for y in y_pairs if len(y) > 0]
    y_maxs = [float(np.max(y)) for y in y_pairs if len(y) > 0]

    if x_range is None:
        gx_min = min(x_mins) if x_mins else 0.0
        gx_max = max(x_maxs) if x_maxs else 1.0
        if gx_min == gx_max:
            gx_min -= 1.0
            gx_max += 1.0
        x_range = (gx_min, gx_max)

    if y_range is None:
        gy_min = min(y_mins) if y_mins else 0.0
        gy_max = max(y_maxs) if y_maxs else 1.0
        if gy_min == gy_max:
            gy_min -= 1.0
            gy_max += 1.0
        y_range = (gy_min, gy_max)

    if isinstance(bins, int):
        nb_x, nb_y = bins, bins
    else:
        nb_x, nb_y = bins

    x_edges = np.linspace(x_range[0], x_range[1], nb_x + 1)
    y_edges = np.linspace(y_range[0], y_range[1], nb_y + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # 4. Compute 2D Histograms (normalized to density=True)
    per_rve_hists: List[np.ndarray] = []
    ensemble_density = np.zeros((nb_x, nb_y), dtype=np.float64)

    for w, (x, y) in zip(weights, zip(x_pairs, y_pairs)):
        h, _, _ = np.histogram2d(
            x, y,
            bins=[x_edges, y_edges],
            density=True,
        )
        per_rve_hists.append(h)
        ensemble_density += w * h

    # 5. Optional 2D Kernel Density Estimation (per realization and weighted mixture)
    per_rve_kdes: Optional[List[np.ndarray]] = None
    ensemble_kde_density: Optional[np.ndarray] = None

    if kde:
        per_rve_kdes = []
        ensemble_kde_density = np.zeros((nb_x, nb_y), dtype=np.float64)
        for w, (x, y) in zip(weights, zip(x_pairs, y_pairs)):
            k = compute_joint_kde_2d(
                x, y,
                x_eval=x_centers,
                y_eval=y_centers,
                bandwidth=kde_bandwidth,
                max_samples=max_kde_samples,
            )
            per_rve_kdes.append(k)
            ensemble_kde_density += w * k

    ret_x_name = f"{x_name}/Q_macro" if norm_x else x_name
    ret_y_name = f"{y_name}/Q_macro" if norm_y else y_name

    return JointDistribution2D(
        x_name=ret_x_name,
        y_name=ret_y_name,
        x_bins=x_edges,
        y_bins=y_edges,
        x_centers=x_centers,
        y_centers=y_centers,
        density=ensemble_density,
        per_rve_densities=per_rve_hists,
        kde_density=ensemble_kde_density,
        per_rve_kdes=per_rve_kdes,
        is_normalized=is_normalized,
        macro_x_values=macro_x,
        macro_y_values=macro_y,
    )
