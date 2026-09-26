"""
Spatial autocorrelation and effective sample size estimation.

Implements AGENTS.md Section 14:
Distinguishes N_points from N_effective due to spatial autocorrelation
inside an individual RVE, and computes correlation-adjusted confidence intervals.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class SpatialCorrelationResult:
    """Results of spatial autocorrelation and effective sample size analysis."""
    n_points: int
    n_effective: float
    correlation_length: float       # Distance where R(r) drops to 1/e
    integral_length_scale: float    # Integral of R(r) dr
    sample_mean: float
    sample_std: float
    standard_error_naive: float     # s / sqrt(N_points)
    standard_error_effective: float # s / sqrt(N_effective)
    ci_naive_95: Tuple[float, float]
    ci_effective_95: Tuple[float, float]
    lags: np.ndarray                # Radial distance bins
    autocorrelation_radial: np.ndarray # R(r) values


def compute_spatial_autocorrelation_3d(
    field_3d: np.ndarray,
    dx: float = 1.0,
    max_lag_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radially averaged 3D spatial autocorrelation function using FFT.
    
    Args:
        field_3d: 3D numpy array of shape (Nx, Ny, Nz).
        dx: Grid spacing (assumed uniform voxel spacing, default 1.0).
        max_lag_fraction: Maximum radial lag fraction of domain size to consider.
        
    Returns:
        (lags, radial_r):
            lags: 1D array of radial lag distances.
            radial_r: 1D array of autocorrelation values R(r) in [ -1, 1 ].
    """
    f = np.asarray(field_3d, dtype=np.float64)
    if f.ndim != 3:
        raise ValueError(f"field_3d must have 3 dimensions, got shape {f.shape}")
        
    nx, ny, nz = f.shape
    f_mean = np.mean(f)
    f_var = np.var(f)
    
    if f_var <= 1e-20:
        # Uniform field: perfect correlation
        lags = np.linspace(0, max(nx, ny, nz) * dx * max_lag_fraction, 50)
        return lags, np.ones_like(lags)
        
    dev = f - f_mean
    
    # Zero-padding for linear convolution via FFT
    px, py, pz = 2 * nx, 2 * ny, 2 * nz
    f_pad = np.zeros((px, py, pz), dtype=np.float64)
    f_pad[:nx, :ny, :nz] = dev
    
    # Overlap count mask
    mask = np.zeros((px, py, pz), dtype=np.float64)
    mask[:nx, :ny, :nz] = 1.0
    
    # FFT
    f_fft = np.fft.fftn(f_pad)
    m_fft = np.fft.fftn(mask)
    
    cov = np.real(np.fft.ifftn(f_fft * np.conj(f_fft)))
    counts = np.real(np.fft.ifftn(m_fft * np.conj(m_fft)))
    
    # Valid non-zero overlaps
    valid = counts > 0.5
    r_3d = np.zeros_like(cov)
    r_3d[valid] = (cov[valid] / counts[valid]) / f_var
    
    # Shift zero lag to center
    r_shifted = np.fft.fftshift(r_3d)
    
    # Coordinate grids centered at zero
    gx = (np.arange(px) - px // 2) * dx
    gy = (np.arange(py) - py // 2) * dx
    gz = (np.arange(pz) - pz // 2) * dx
    
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    dist = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    
    # Maximum lag
    max_dist = min(nx, ny, nz) * dx * max_lag_fraction
    mask_dist = dist <= max_dist
    
    dist_valid = dist[mask_dist]
    r_valid = r_shifted[mask_dist]
    
    # Bin by radial distance
    n_bins = int(min(nx, ny, nz) * max_lag_fraction)
    n_bins = max(n_bins, 10)
    
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    bin_indices = np.digitize(dist_valid, bin_edges) - 1
    radial_r = np.zeros(n_bins, dtype=np.float64)
    
    for i in range(n_bins):
        in_bin = r_valid[bin_indices == i]
        if len(in_bin) > 0:
            radial_r[i] = np.mean(in_bin)
        else:
            radial_r[i] = 0.0
            
    # Guarantee R(0) = 1.0 at origin
    if len(radial_r) > 0:
        radial_r[0] = 1.0
        
    return bin_centers, radial_r


def estimate_effective_sample_size(
    field_or_values: np.ndarray,
    dx: float = 1.0,
    confidence_level: float = 0.95,
) -> SpatialCorrelationResult:
    """Estimate effective sample size N_effective and spatial correlation statistics.
    
    Args:
        field_or_values: 3D spatial grid (Nx, Ny, Nz) or 1D array of values.
        dx: Grid spacing.
        confidence_level: Desired confidence level for confidence intervals (default 0.95).
        
    Returns:
        SpatialCorrelationResult dataclass.
    """
    arr = np.asarray(field_or_values, dtype=np.float64)
    
    if arr.ndim == 3:
        n_points = int(arr.size)
        f_mean = float(np.mean(arr))
        f_std = float(np.std(arr))
        
        lags, radial_r = compute_spatial_autocorrelation_3d(arr, dx=dx)
        
        # 1. Correlation length lambda where R(r) drops below 1/e
        decay_target = 1.0 / np.e
        drop_idx = np.where(radial_r < decay_target)[0]
        if len(drop_idx) > 0:
            idx = drop_idx[0]
            if idx > 0:
                # Linear interpolation
                r0, r1 = radial_r[idx - 1], radial_r[idx]
                l0, l1 = lags[idx - 1], lags[idx]
                if abs(r1 - r0) > 1e-12:
                    corr_length = float(l0 + (decay_target - r0) / (r1 - r0) * (l1 - l0))
                else:
                    corr_length = float(l0)
            else:
                corr_length = float(lags[0])
        else:
            corr_length = float(lags[-1])
            
        # 2. Integral length scale L = int_0^{r_cut} max(R(r), 0) dr
        dr = np.diff(lags)
        pos_r = np.maximum(radial_r[:-1], 0.0)
        integral_scale = float(np.sum(pos_r * dr))
        
        # 3. Correlation volume and N_effective
        # For an isotropic 3D field, correlation volume V_c approx 4/3 * pi * lambda^3
        # Domain volume V = Nx * Ny * Nz * dx^3
        v_domain = n_points * (dx ** 3)
        v_corr = max((4.0 / 3.0) * np.pi * (corr_length ** 3), dx ** 3)
        n_eff = float(np.clip(v_domain / v_corr, 1.0, float(n_points)))
        
    else:
        # 1D flattened array (no grid geometry available)
        vals = arr.ravel()
        n_points = len(vals)
        f_mean = float(np.mean(vals))
        f_std = float(np.std(vals))
        lags = np.array([0.0])
        radial_r = np.array([1.0])
        corr_length = 0.0
        integral_scale = 0.0
        n_eff = float(n_points)
        
    # Standard errors and confidence intervals
    se_naive = f_std / np.sqrt(n_points) if n_points > 0 else 0.0
    se_eff = f_std / np.sqrt(n_eff) if n_eff > 0 else 0.0
    
    z_crit = float(stats.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))
    ci_naive = (f_mean - z_crit * se_naive, f_mean + z_crit * se_naive)
    ci_eff = (f_mean - z_crit * se_eff, f_mean + z_crit * se_eff)
    
    return SpatialCorrelationResult(
        n_points=n_points,
        n_effective=n_eff,
        correlation_length=corr_length,
        integral_length_scale=integral_scale,
        sample_mean=f_mean,
        sample_std=f_std,
        standard_error_naive=se_naive,
        standard_error_effective=se_eff,
        ci_naive_95=ci_naive,
        ci_effective_95=ci_eff,
        lags=lags,
        autocorrelation_radial=radial_r,
    )
