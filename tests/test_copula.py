import numpy as np
import pytest
from mv3d_stress_stats.copula import (
    fit_gaussian_copula,
    sample_copula,
    sample_independent,
    evaluate_copula_impact,
)


def test_copula_correlation_preservation():
    # Generate 6 correlated stress variables
    rng = np.random.default_rng(101)
    true_cov = np.array([
        [1.0, 0.8, 0.2, 0.0, 0.0, 0.0],
        [0.8, 1.0, 0.3, 0.0, 0.0, 0.0],
        [0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.1, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    raw_samples = rng.multivariate_normal(mean=np.zeros(6), cov=true_cov, size=10000)
    
    corr_mat, marginals = fit_gaussian_copula(raw_samples)
    # Check that estimated correlation matrix is close to true_cov
    np.testing.assert_allclose(corr_mat, true_cov, atol=0.08)
    
    # Sample from copula and check empirical correlation
    synth = sample_copula(corr_mat, marginals, n_samples=10000, seed=42)
    synth_corr = np.corrcoef(synth, rowvar=False)
    np.testing.assert_allclose(synth_corr, true_cov, atol=0.08)


def test_evaluate_copula_impact():
    # When SX and SY are strongly positively correlated (e.g. corr = 0.95):
    # (SX - SY)^2 is much smaller than when SX and SY are independent!
    # Therefore independent sampling will artificially OVERESTIMATE von Mises stress!
    rng = np.random.default_rng(202)
    n = 15000
    z = rng.normal(100.0, 10.0, n)
    sx = z + rng.normal(0, 1.0, n)
    sy = z + rng.normal(0, 1.0, n)
    sz = z + rng.normal(0, 1.0, n)
    stress = np.zeros((n, 6))
    stress[:, 0] = sx
    stress[:, 1] = sy
    stress[:, 2] = sz
    
    res = evaluate_copula_impact(stress, quantity="von_mises", n_synthetic=15000, seed=42)
    
    # Raw VM should be small because components track each other
    # Copula VM should closely match raw VM
    assert res.copula_mean == pytest.approx(res.raw_mean, rel=0.15)
    
    # Independent sampling breaks correlation, so (SX - SY)^2 is much larger,
    # causing independent mean von Mises to be significantly higher than raw!
    assert res.indep_mean > res.raw_mean * 1.5
