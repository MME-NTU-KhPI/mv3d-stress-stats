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


def test_pair_plane_sections_calculation():
    rng = np.random.default_rng(42)
    n = 1500
    stress = np.zeros((n, 6))
    stress[:, 0] = rng.normal(100.0, 10.0, n)  # SX
    stress[:, 1] = rng.normal(50.0, 5.0, n)    # SY
    stress[:, 2] = rng.normal(20.0, 2.0, n)    # SZ
    stress[:, 3] = rng.normal(10.0, 2.0, n)    # SXY

    from mv3d_stress_stats.directional import (
        compute_pair_plane_sections,
        compute_all_pair_plane_sections,
    )
    from mv3d_stress_stats.plotting import (
        plot_pair_plane_sections,
        plot_pair_planes_grid,
        plot_pair_planes_angular_comparison,
        plot_pair_plane_polar_heatmap,
    )

    res = compute_pair_plane_sections(stress, plane=("sx", "sy"), n_angles=180)
    assert res.plane_label == "(SX, SY)"
    assert len(res.angles_deg) == 180
    assert len(res.radii) == 5
    assert res.section_densities.shape == (5, 180)
    assert res.conditional_densities.shape == (5, 180)
    assert len(res.angular_pdf) == 180
    assert np.all(res.section_densities >= 0.0)
    assert res.is_centered is True
    assert res.polar_grid_radii is not None
    assert len(res.polar_grid_radii) == 60
    assert res.polar_grid_density is not None
    assert res.polar_grid_density.shape == (60, 180)

    # Check conditional density integrates to ~1 over angles
    for k in range(len(res.radii)):
        integ = np.trapezoid(res.conditional_densities[k, :], res.angles_rad)
        assert integ == pytest.approx(1.0, rel=0.01)

    # Test centered normalization (relative fluctuation amplitudes)
    res_norm = compute_pair_plane_sections(stress, plane=("sx", "sy"), centered=True, normalize=True)
    assert res_norm.is_normalized is True
    assert res_norm.is_centered is True
    assert res_norm.macro_radius > 0
    assert np.all(res_norm.radii > 0)

    # Test uncentered normalization (raw radius centered around 1.0)
    res_uncentered_norm = compute_pair_plane_sections(stress, plane=("sx", "sy"), centered=False, normalize=True)
    assert res_uncentered_norm.is_normalized is True
    assert res_uncentered_norm.is_centered is False
    assert np.min(res_uncentered_norm.radii) < 1.0 < np.max(res_uncentered_norm.radii)

    # Test compute all canonical planes
    all_res = compute_all_pair_plane_sections(stress)
    assert "(SX, SY)" in all_res
    assert "(SY, SZ)" in all_res
    assert "(SX, SXY)" in all_res
    assert len(all_res) == 6

    # Test plotting functions execute without error
    ax1 = plot_pair_plane_sections(res, density_type="joint")
    assert ax1 is not None

    ax2 = plot_pair_plane_sections(res, density_type="conditional")
    assert ax2 is not None

    fig_grid, _ = plot_pair_planes_grid(all_res, ncols=3)
    assert fig_grid is not None

    ax_comp = plot_pair_planes_angular_comparison(all_res)
    assert ax_comp is not None

    # Test polar heatmap with horizontal reference lines
    ax_hm = plot_pair_plane_polar_heatmap(res, show_section_lines=True, show_mean_line=True)
    assert ax_hm is not None


def test_pair_plane_ray_sections():
    rng = np.random.default_rng(42)
    n = 1500
    stress = np.zeros((n, 6))
    stress[:, 0] = rng.normal(100.0, 10.0, n)  # SX
    stress[:, 1] = rng.normal(50.0, 5.0, n)    # SY
    stress[:, 2] = rng.normal(20.0, 2.0, n)    # SZ
    stress[:, 3] = rng.normal(10.0, 2.0, n)    # SXY

    from mv3d_stress_stats.directional import (
        PairPlaneRaySectionResult,
        compute_pair_plane_ray_sections,
        compute_all_pair_plane_ray_sections,
    )
    from mv3d_stress_stats.plotting import (
        plot_pair_plane_ray_sections,
        plot_pair_planes_ray_grid,
        plot_pair_plane_ray_dashboard,
    )

    ray_res = compute_pair_plane_ray_sections(
        stress,
        plane=("sx", "sy"),
        angles_deg=[0.0, 30.0, 45.0, 60.0, 90.0],
        normalize=True,
        centered=False,
    )

    assert isinstance(ray_res, PairPlaneRaySectionResult)
    assert ray_res.plane_label == "(SX, SY)"
    assert len(ray_res.angles_deg) == 5
    assert len(ray_res.radii) == 150
    assert ray_res.ray_densities.shape == (5, 150)
    assert np.all(ray_res.ray_densities >= 0.0)
    assert len(ray_res.angle_labels) == 5
    assert len(ray_res.peak_radii) == 5
    assert len(ray_res.peak_densities) == 5
    assert ray_res.is_normalized is True
    assert ray_res.is_centered is False

    # Check that peak for (SX=100, SY=50) is near theta=30 deg (since atan2(50, 100) = 26.5 deg)
    # The 30 deg ray density should be higher than 90 deg ray density (since SY/SX != inf)
    peak_30 = float(np.max(ray_res.ray_densities[1, :])) # 30 deg
    peak_90 = float(np.max(ray_res.ray_densities[4, :])) # 90 deg
    assert peak_30 > peak_90

    # Test compute all canonical planes
    all_ray_res = compute_all_pair_plane_ray_sections(stress, angles_deg=[0, 30, 45, 60, 90])
    assert len(all_ray_res) == 6
    assert "(SX, SY)" in all_ray_res
    assert "(SY, SZ)" in all_ray_res
    assert "(SX, SZ)" in all_ray_res

    # Test plotting functions
    ax_ray = plot_pair_plane_ray_sections(ray_res)
    assert ax_ray is not None

    fig_grid, axes = plot_pair_planes_ray_grid(all_ray_res, ncols=3)
    assert fig_grid is not None
    assert axes.shape == (2, 3)

    fig_dash = plot_pair_plane_ray_dashboard(ray_res)
    assert fig_dash is not None


def test_pair_plane_cartesian_sections_calculation():
    rng = np.random.default_rng(42)
    n = 1000
    stress = np.zeros((n, 6))
    stress[:, 0] = rng.normal(100.0, 10.0, n)  # SX
    stress[:, 1] = rng.normal(50.0, 5.0, n)    # SY
    stress[:, 2] = rng.normal(20.0, 2.0, n)    # SZ
    stress[:, 3] = rng.normal(10.0, 2.0, n)    # SXY

    from mv3d_stress_stats.directional import (
        PairPlaneCartesianSectionResult,
        compute_pair_plane_cartesian_sections,
        compute_all_pair_plane_cartesian_sections,
    )
    from mv3d_stress_stats.plotting import (
        plot_pair_plane_cartesian_sections,
        plot_pair_planes_cartesian_grid,
        plot_pair_plane_cartesian_dashboard,
    )

    cart_res = compute_pair_plane_cartesian_sections(
        stress,
        plane=("sx", "sy"),
        slice_axis="y",
        percentiles=[10, 25, 50, 75, 90],
        normalize=True,
        include_2d_surface=True,
    )

    assert isinstance(cart_res, PairPlaneCartesianSectionResult)
    assert cart_res.plane_label == "(SX, SY)"
    assert cart_res.slice_axis == "y"
    assert cart_res.slice_component == "SY"
    assert cart_res.profile_component == "SX"
    assert len(cart_res.slice_levels) == 5
    assert len(cart_res.slice_labels) == 5
    assert len(cart_res.profile_grid) == 150
    assert cart_res.section_densities.shape == (5, 150)
    assert cart_res.conditional_densities.shape == (5, 150)
    assert np.all(cart_res.section_densities >= 0.0)
    # Crucial property of Option 3: ALL quantile cross-sections must be active and non-zero
    assert np.all(cart_res.peak_densities > 0.0)
    assert cart_res.is_normalized is True

    # Check that conditional densities integrate to 1.0 along the profile grid
    for k in range(5):
        area = np.trapezoid(cart_res.conditional_densities[k, :], cart_res.profile_grid)
        assert area == pytest.approx(1.0, rel=1e-2)

    # Check 2D surface was generated
    assert cart_res.grid_x is not None
    assert cart_res.grid_y is not None
    assert cart_res.density_surface is not None
    assert cart_res.density_surface.shape == (60, 60)

    # Test compute all canonical planes
    all_cart_res = compute_all_pair_plane_cartesian_sections(stress, percentiles=[10, 25, 50, 75, 90])
    assert len(all_cart_res) == 6
    assert "(SX, SY)" in all_cart_res
    assert "(SY, SZ)" in all_cart_res
    assert "(SX, SZ)" in all_cart_res

    # Test plotting functions
    ax_cart = plot_pair_plane_cartesian_sections(cart_res)
    assert ax_cart is not None

    fig_grid, axes = plot_pair_planes_cartesian_grid(all_cart_res, ncols=3)
    assert fig_grid is not None
    assert axes.shape == (2, 3)

    fig_dash = plot_pair_plane_cartesian_dashboard(cart_res)
    assert fig_dash is not None


