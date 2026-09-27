import numpy as np
import pytest
from mv3d_stress_stats.joint import (
    compute_joint_histogram_2d,
    compute_ensemble_joint_2d,
)
from mv3d_stress_stats.single_rve import RVESimulation


def test_joint_histogram_2d():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(0, 1, 5000)
    
    h, x_edges, y_edges = compute_joint_histogram_2d(x, y, bins=(20, 20), density=True)
    # Integral of 2D PDF over area element dx * dy must equal 1.0
    dx = np.diff(x_edges)
    dy = np.diff(y_edges)
    area = np.outer(dx, dy)
    total_prob = np.sum(h * area)
    assert total_prob == pytest.approx(1.0, rel=0.01)


def test_ensemble_joint_2d():
    # 2 synthetic RVEs
    s1 = np.zeros((1000, 6))
    s1[:, 0] = np.random.normal(100, 10, 1000) # SX
    s1[:, 1] = np.random.normal(50, 5, 1000)   # SY
    
    s2 = np.zeros((1000, 6))
    s2[:, 0] = np.random.normal(120, 12, 1000)
    s2[:, 1] = np.random.normal(60, 6, 1000)
    
    sim1 = RVESimulation(s1, realization_id="sim1")
    sim2 = RVESimulation(s2, realization_id="sim2")
    
    weights = np.array([0.5, 0.5])
    joint = compute_ensemble_joint_2d(
        [sim1, sim2],
        weights=weights,
        x_quantity="pressure",
        y_quantity="von_mises",
        bins=(25, 25),
    )
    
    assert joint.density.shape == (25, 25)
    # Find contour level for 50% probability
    c50 = joint.find_contour_level(0.50)
    assert c50 > 0.0
