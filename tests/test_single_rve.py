import numpy as np
import pytest
from mv3d_stress_stats.single_rve import (
    RVESimulation,
    compute_statistics,
    compute_histogram,
    compute_cdf,
    compute_kde,
)


def test_compute_statistics():
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    stats = compute_statistics(data)
    assert stats.mean == pytest.approx(30.0)
    assert stats.min == pytest.approx(10.0)
    assert stats.max == pytest.approx(50.0)
    assert stats.q50 == pytest.approx(30.0)
    assert stats.num_samples == 5


def test_compute_histogram_normalization():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=100.0, scale=15.0, size=5000)
    hist, edges, centers = compute_histogram(data, bins=50, density=True)
    # Integral of PDF dx must be 1.0
    dx = np.diff(edges)
    integral = np.sum(hist * dx)
    assert integral == pytest.approx(1.0, rel=1e-3)


def test_compute_cdf_and_kde():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0, scale=1.0, size=2000)
    
    # CDF
    x_eval = np.linspace(-3.0, 3.0, 100)
    _, cdf_vals = compute_cdf(data, eval_points=x_eval)
    assert cdf_vals[0] < 0.05
    assert cdf_vals[-1] > 0.95
    # CDF must be monotonically non-decreasing
    assert np.all(np.diff(cdf_vals) >= 0.0)

    # KDE
    kde_vals = compute_kde(data, eval_points=x_eval)
    dx = x_eval[1] - x_eval[0]
    kde_integral = np.sum(kde_vals * dx)
    assert kde_integral == pytest.approx(1.0, rel=0.05)


def test_rve_simulation_in_memory():
    # Construct an in-memory RVE with 100 points
    stress_field = np.zeros((100, 6))
    stress_field[:, 0] = np.linspace(80.0, 120.0, 100)
    
    rve = RVESimulation(stress_field, realization_id="test_in_mem")
    assert rve.num_points == 100
    st = rve.describe("SX")
    assert st.mean == pytest.approx(100.0)
    assert st.min == pytest.approx(80.0)
    assert st.max == pytest.approx(120.0)
