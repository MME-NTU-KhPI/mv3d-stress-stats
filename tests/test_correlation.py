import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from mv3d_stress_stats.correlation import (
    compute_spatial_autocorrelation_3d,
    estimate_effective_sample_size,
)


def test_spatial_autocorrelation_white_noise():
    # Uncorrelated noise: N_eff should be close to N_points
    rng = np.random.default_rng(42)
    grid = rng.normal(0, 1, size=(20, 20, 20))
    res = estimate_effective_sample_size(grid, dx=1.0)
    
    assert res.n_points == 8000
    # In white noise, R(r) drops immediately at lag > 0
    assert res.correlation_length <= 2.0
    # Effective sample size should be a substantial fraction of total points
    assert res.n_effective > 500


def test_spatial_autocorrelation_correlated_field():
    # Correlated field created via Gaussian filter smoothing
    rng = np.random.default_rng(42)
    raw = rng.normal(0, 1, size=(24, 24, 24))
    # Gaussian blur with sigma = 3 voxels
    smooth_grid = gaussian_filter(raw, sigma=3.0)
    
    res = estimate_effective_sample_size(smooth_grid, dx=1.0)
    assert res.n_points == 24**3
    # Correlation length should be around 3 to 6 voxels
    assert res.correlation_length >= 2.5
    # Effective sample size must be significantly lower than N_points
    assert res.n_effective < res.n_points * 0.1
    # Effective standard error must be strictly larger than naive standard error
    assert res.standard_error_effective > res.standard_error_naive
    # Effective CI must be wider than naive CI
    ci_eff_width = res.ci_effective_95[1] - res.ci_effective_95[0]
    ci_naive_width = res.ci_naive_95[1] - res.ci_naive_95[0]
    assert ci_eff_width > ci_naive_width
