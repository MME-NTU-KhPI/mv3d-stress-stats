import numpy as np
import pytest
from mv3d_stress_stats.joint import (
    compute_joint_histogram_2d,
    compute_joint_kde_2d,
    compute_ensemble_joint_2d,
)
from mv3d_stress_stats.plotting import plot_joint_density_2d
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


def test_joint_kde_2d():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 2000)
    y = rng.normal(0, 1, 2000)
    x_eval = np.linspace(-3, 3, 30)
    y_eval = np.linspace(-3, 3, 30)
    
    kde_grid = compute_joint_kde_2d(x, y, x_eval=x_eval, y_eval=y_eval)
    assert kde_grid.shape == (30, 30)
    assert np.all(kde_grid >= 0.0)
    
    # Check approximate Riemann integral over [-3, 3] x [-3, 3]
    dx = x_eval[1] - x_eval[0]
    dy = y_eval[1] - y_eval[0]
    integral = np.sum(kde_grid) * dx * dy
    # Covers ~99% of normal distribution
    assert integral == pytest.approx(0.99, abs=0.05)


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
        kde=True,
    )
    
    assert joint.density.shape == (25, 25)
    assert joint.kde_density is not None
    assert joint.kde_density.shape == (25, 25)
    assert len(joint.per_rve_kdes) == 2
    
    # Find contour level for 50% probability
    c50_hist = joint.find_contour_level(0.50, use_kde=False)
    c50_kde = joint.find_contour_level(0.50, use_kde=True)
    assert c50_hist > 0.0
    assert c50_kde > 0.0

    # Test plotting with KDE
    ax = plot_joint_density_2d(joint, use_kde=True)
    assert "KDE" in ax.get_title()


def test_ensemble_joint_2d_normalization():
    # 2 synthetic RVEs with nonzero macro pressure and von Mises
    rng = np.random.default_rng(123)
    s1 = np.zeros((1000, 6))
    s1[:, 0] = rng.normal(100, 10, 1000)
    s1[:, 1] = rng.normal(50, 5, 1000)
    s1[:, 2] = rng.normal(30, 3, 1000)
    
    s2 = np.zeros((1000, 6))
    s2[:, 0] = rng.normal(200, 20, 1000)
    s2[:, 1] = rng.normal(100, 10, 1000)
    s2[:, 2] = rng.normal(60, 6, 1000)
    
    sim1 = RVESimulation(s1, realization_id="sim1")
    sim2 = RVESimulation(s2, realization_id="sim2")
    
    # 1. Full normalization (both coordinates)
    joint_norm = compute_ensemble_joint_2d(
        [sim1, sim2],
        weights=np.array([0.5, 0.5]),
        x_quantity="pressure",
        y_quantity="von_mises",
        bins=(30, 30),
        normalize=True,
        kde=True,
    )
    assert joint_norm.is_normalized is True
    assert joint_norm.x_name == "pressure/Q_macro"
    assert joint_norm.y_name == "von_mises/Q_macro"
    assert joint_norm.macro_x_values is not None
    assert joint_norm.macro_y_values is not None
    # Since values are normalized by macro scale, the centers should be centered near 1.0 (or -1.0 for signed p)
    assert np.min(joint_norm.y_centers) < 1.0 < np.max(joint_norm.y_centers)

    # 2. Partial normalization: normalize=(False, True)
    joint_partial = compute_ensemble_joint_2d(
        [sim1, sim2],
        weights=np.array([0.5, 0.5]),
        x_quantity="lode_angle",
        y_quantity="von_mises",
        bins=(30, 30),
        normalize=(False, True),
    )
    assert joint_partial.is_normalized is True
    assert joint_partial.x_name == "lode_angle"
    assert joint_partial.y_name == "von_mises/Q_macro"


def test_ensemble_joint_2d_normalize_zero_guard():
    # Pure shear loading: macro SX = 0
    rng = np.random.default_rng(42)
    s = np.zeros((1000, 6))
    s[:, 0] = rng.normal(0, 10, 1000)     # SX mean ~ 0
    s[:, 3] = rng.normal(100, 5, 1000)    # SXY mean = 100
    
    sim = RVESimulation(s, realization_id="shear_sim")
    
    with pytest.raises(ValueError, match="Cannot normalize X variable 'sx'"):
        compute_ensemble_joint_2d(
            [sim],
            weights=np.array([1.0]),
            x_quantity="sx",
            y_quantity="von_mises",
            normalize=True,
        )
