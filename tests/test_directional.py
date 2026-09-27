import numpy as np
import pytest
from mv3d_stress_stats.directional import (
    principal_directions,
    principal_direction_angles,
    compute_circular_statistics,
    compute_directional_histogram,
)


def test_principal_directions_analytical():
    # 1. Uniaxial tension in X: [100, 0, 0, 0, 0, 0]
    s_x = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    vals, vecs = principal_directions(s_x)
    assert vals[0] == pytest.approx(100.0)
    assert vals[1] == pytest.approx(0.0)
    assert vals[2] == pytest.approx(0.0)
    # v1 must align with X axis [1, 0, 0]
    np.testing.assert_allclose(np.abs(vecs[:, 0]), [1.0, 0.0, 0.0], atol=1e-6)

    # 2. Pure shear in XY: [0, 0, 0, 50, 0, 0]
    # Principal stresses for pure shear tau are +tau, 0, -tau
    s_xy = np.array([0.0, 0.0, 0.0, 50.0, 0.0, 0.0])
    vals_xy, vecs_xy = principal_directions(s_xy)
    assert vals_xy[0] == pytest.approx(50.0)
    assert vals_xy[1] == pytest.approx(0.0)
    assert vals_xy[2] == pytest.approx(-50.0)
    # Tensile principal direction v1 should be at 45 deg in XY plane
    v1 = vecs_xy[:, 0]
    assert abs(v1[0]) == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-5)
    assert abs(v1[1]) == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-5)
    assert v1[2] == pytest.approx(0.0, abs=1e-6)


def test_principal_direction_angles():
    # Batch array of 3 states: tension X, tension Y, pure shear XY
    stresses = np.array([
        [80.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # X tension -> 0 deg
        [0.0, 80.0, 0.0, 0.0, 0.0, 0.0],   # Y tension -> 90 deg
        [0.0, 0.0, 0.0, 50.0, 0.0, 0.0],   # XY shear  -> 45 deg
    ])
    
    angles_deg = principal_direction_angles(stresses, plane="xy", degrees=True)
    assert angles_deg[0] == pytest.approx(0.0, abs=1e-4) or angles_deg[0] == pytest.approx(180.0, abs=1e-4)
    assert angles_deg[1] == pytest.approx(90.0, abs=1e-4)
    assert angles_deg[2] == pytest.approx(45.0, abs=1e-4)

    # Test XZ plane
    s_z = np.array([0.0, 0.0, 80.0, 0.0, 0.0, 0.0])
    ang_xz = principal_direction_angles(s_z, plane="xz", degrees=True)
    assert ang_xz == pytest.approx(90.0, abs=1e-4)


def test_circular_statistics():
    # Angles strictly aligned at 30 deg
    angles = np.full(100, np.radians(30.0))
    mean_dir, r, disp, circ_std = compute_circular_statistics(angles)
    assert np.degrees(mean_dir) == pytest.approx(30.0, abs=1e-4)
    assert r == pytest.approx(1.0, abs=1e-4)
    assert disp == pytest.approx(0.0, abs=1e-4)
    assert circ_std == pytest.approx(0.0, abs=1e-4)

    # Uniformly distributed angles over [0, pi) -> isotropic dispersion
    rng = np.random.default_rng(42)
    uniform_angles = np.linspace(0.0, np.pi, 1000, endpoint=False)
    _, r_unif, disp_unif, _ = compute_circular_statistics(uniform_angles)
    assert r_unif == pytest.approx(0.0, abs=0.02)
    assert disp_unif == pytest.approx(1.0, abs=0.02)


def test_directional_histogram_generation():
    # Synthetic stressed points oriented predominantly along 45 degrees
    rng = np.random.default_rng(999)
    n = 2000
    # Create stress array with pure shear plus small perturbations
    base_shear = np.tile([0.0, 0.0, 0.0, 50.0, 0.0, 0.0], (n, 1))
    noise = rng.normal(0, 2.0, size=(n, 6))
    stress_field = base_shear + noise

    dh = compute_directional_histogram(
        stress_field,
        plane="xy",
        n_bins=36,
        weights="von_mises",
        symmetric=True,
    )

    assert dh.symmetric is True
    assert dh.weighted is True
    assert len(dh.bin_centers) == 36
    assert len(dh.frequencies) == 36
    # Mean orientation should be near 45 degrees
    assert dh.mean_direction_degrees == pytest.approx(45.0, abs=2.0)
    # High directional concentration (R > 0.95)
    assert dh.mean_resultant_length > 0.95
    assert dh.circular_dispersion < 0.05
