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
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats

from mv3d_stress_stats.invariants import (
    pressure,
    hydrostatic_stress,
    von_mises,
    lode_angle,
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

    def find_contour_level(self, probability: float) -> float:
        """Find density threshold c such that integral_{f >= c} f dA = probability.
        
        Args:
            probability: Desired enclosed probability content (e.g. 0.50, 0.90, 0.95).
            
        Returns:
            Density threshold value c.
        """
        dx = np.diff(self.x_bins)
        dy = np.diff(self.y_bins)
        area_element = np.outer(dx, dy)
        
        # Flatten density and area
        f_flat = self.density.ravel()
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


def compute_ensemble_joint_2d(
    simulations: List[RVESimulation],
    weights: np.ndarray,
    x_quantity: str = "pressure",
    y_quantity: str = "von_mises",
    bins: Union[int, Tuple[int, int]] = (50, 50),
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
) -> JointDistribution2D:
    """Compute ensemble 2D joint distribution on common grid.
    
    Args:
        simulations: List of RVESimulation objects.
        weights: Weights w_r summing to 1.0.
        x_quantity: 'pressure', 'hydrostatic_stress', 'von_mises', 'lode_angle', etc.
        y_quantity: Second variable for joint space.
        bins: Number of bins in each direction.
        x_range: Global range for X. If None, auto-detected across all realizations.
        y_range: Global range for Y. If None, auto-detected across all realizations.
        
    Returns:
        JointDistribution2D instance.
    """
    pairs = []
    x_mins, x_maxs, y_mins, y_maxs = [], [], [], []
    
    for sim in simulations:
        x = sim.get_quantity(x_quantity).ravel()
        y = sim.get_quantity(y_quantity).ravel()
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        pairs.append((x, y))
        x_mins.append(float(np.min(x)))
        x_maxs.append(float(np.max(x)))
        y_mins.append(float(np.min(y)))
        y_maxs.append(float(np.max(y)))
        
    # Global ranges
    if x_range is None:
        gx_min, gx_max = min(x_mins), max(x_maxs)
        if gx_min == gx_max:
            gx_min -= 1.0
            gx_max += 1.0
        x_range = (gx_min, gx_max)
        
    if y_range is None:
        gy_min, gy_max = min(y_mins), max(y_maxs)
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
    
    per_rve_hists: List[np.ndarray] = []
    ensemble_density = np.zeros((nb_x, nb_y), dtype=np.float64)
    
    for w, (x, y) in zip(weights, pairs):
        h, _, _ = np.histogram2d(
            x, y,
            bins=[x_edges, y_edges],
            density=True,
        )
        per_rve_hists.append(h)
        ensemble_density += w * h
        
    return JointDistribution2D(
        x_name=x_quantity,
        y_name=y_quantity,
        x_bins=x_edges,
        y_bins=y_edges,
        x_centers=x_centers,
        y_centers=y_centers,
        density=ensemble_density,
        per_rve_densities=per_rve_hists,
    )
