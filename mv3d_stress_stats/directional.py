"""
Directional stress analysis and rose diagram / directional histogram algorithms.

Analyzes the spatial orientation of principal stress axes, circular dispersion,
stress-weighted angular concentrations, and directional anisotropy in 3D microstructures.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy import stats

from mv3d_stress_stats.tensors import to_tensor_3x3
from mv3d_stress_stats.invariants import von_mises


@dataclass
class DirectionalHistogram:
    """Container for directional histogram / rose diagram data and circular statistics."""
    bin_centers: np.ndarray          # Polar angle centers in radians [0, 2*pi] or [0, pi]
    bin_edges: np.ndarray            # Polar angle edges in radians
    frequencies: np.ndarray          # Normalized density or relative frequency per bin
    mean_direction: float            # Circular mean orientation in radians [0, pi)
    mean_direction_degrees: float    # Circular mean orientation in degrees [0, 180)
    mean_resultant_length: float     # R in [0, 1] (1 = perfect alignment, 0 = isotropic)
    circular_dispersion: float       # 1 - R in [0, 1]
    circular_std_degrees: float      # Mardia-Jupp circular standard deviation in degrees
    symmetric: bool                  # True if bidirectional rose diagram [0, 2*pi]
    weighted: bool                   # True if stress-weighted
    weight_quantity: str             # Name of weight quantity (e.g. "von_mises", "uniform")
    plane: str                       # Projection plane: "xy", "xz", "yz"


def principal_directions(stress_6: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized calculation of sorted principal stresses and 3D unit eigenvectors.
    
    Args:
        stress_6: Cauchy stress array of shape (..., 6) with components
                  [SX, SY, SZ, SXY, SYZ, SXZ].
                  
    Returns:
        eigenvalues: Sorted principal stresses (..., 3) with sigma_1 >= sigma_2 >= sigma_3.
        eigenvectors: 3D unit eigenvectors of shape (..., 3, 3) where
                      eigenvectors[..., :, 0] is v1 (corresponding to sigma_1),
                      eigenvectors[..., :, 1] is v2 (corresponding to sigma_2),
                      eigenvectors[..., :, 2] is v3 (corresponding to sigma_3).
    """
    stress_arr = np.asarray(stress_6, dtype=np.float64)
    tensor = to_tensor_3x3(stress_arr)  # (..., 3, 3)
    
    # eigh returns ascending eigenvalues (lambda_0 <= lambda_1 <= lambda_2)
    # and column i of vecs is the eigenvector for lambda_i
    eigvals, eigvecs = np.linalg.eigh(tensor)
    
    # Flip along last axis to arrange descending: sigma_1 >= sigma_2 >= sigma_3
    sorted_vals = np.flip(eigvals, axis=-1)
    sorted_vecs = np.flip(eigvecs, axis=-1)
    
    return sorted_vals, sorted_vecs


def principal_direction_angles(
    stress_6: np.ndarray,
    plane: str = "xy",
    which_principal: int = 1,
    degrees: bool = False,
) -> np.ndarray:
    """Calculate the in-plane orientation angle of a principal stress axis.
    
    Because stress tensors are headless axial lines (v and -v represent the
    same state of stress), the returned angle is mapped modulo pi into [0, pi).
    
    Args:
        stress_6: Cauchy stress array of shape (..., 6).
        plane: Projection plane: "xy" (azimuth), "xz", or "yz".
        which_principal: 1 for maximum tensile principal stress (v1),
                         2 for intermediate (v2),
                         3 for minimum / compressive (v3).
        degrees: If True, returns angles in degrees [0, 180).
                 If False, returns angles in radians [0, pi).
                 
    Returns:
        np.ndarray of angles matching batch dimensions (...,).
    """
    _, eigvecs = principal_directions(stress_6)
    vec_idx = which_principal - 1
    v = eigvecs[..., :, vec_idx]  # (..., 3)
    
    plane = plane.lower()
    if plane == "xy":
        # Azimuthal angle in XY plane: atan2(vy, vx)
        phi = np.arctan2(v[..., 1], v[..., 0])
    elif plane == "xz":
        # Angle in XZ plane: atan2(vz, vx)
        phi = np.arctan2(v[..., 2], v[..., 0])
    elif plane == "yz":
        # Angle in YZ plane: atan2(vz, vy)
        phi = np.arctan2(v[..., 2], v[..., 1])
    else:
        raise ValueError(f"Unknown plane: '{plane}'. Must be 'xy', 'xz', or 'yz'.")
        
    # Axial modulo pi: maps [-pi, pi] to [0, pi)
    theta = phi % np.pi
    
    if degrees:
        return np.degrees(theta)
    return theta


def compute_circular_statistics(
    angles_rad: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """Calculate circular statistics for axial data (angles modulo pi).
    
    Uses double-angle formulation (2*theta) to account for bidirectional symmetry.
    
    Returns:
        mean_theta: Mean orientation angle in radians [0, pi).
        mean_resultant_length: R in [0, 1].
        circular_dispersion: 1 - R in [0, 1] (0 = perfect alignment, 1 = isotropic).
        circular_std_rad: Mardia-Jupp circular standard deviation in radians.
    """
    angles = np.asarray(angles_rad, dtype=np.float64).ravel()
    angles_mod = angles % np.pi
    
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        w = np.maximum(w, 0.0)
        total_w = np.sum(w)
        if total_w <= 0.0:
            w = np.ones_like(angles_mod)
            total_w = float(len(angles_mod))
        else:
            w = w / total_w
        c = np.sum(w * np.cos(2.0 * angles_mod))
        s = np.sum(w * np.sin(2.0 * angles_mod))
    else:
        c = float(np.mean(np.cos(2.0 * angles_mod)))
        s = float(np.mean(np.sin(2.0 * angles_mod)))
        
    r = float(np.sqrt(c * c + s * s))
    r = min(max(r, 0.0), 1.0)
    
    mean_theta = float(0.5 * np.arctan2(s, c) % np.pi)
    dispersion = float(1.0 - r)
    
    # Mardia & Jupp circular standard deviation for axial data
    eps_r = max(r, 1e-12)
    circ_std = float(0.5 * np.sqrt(-2.0 * np.log(eps_r)))
    
    return mean_theta, r, dispersion, circ_std


def compute_directional_histogram(
    stress_6: np.ndarray,
    plane: str = "xy",
    which_principal: int = 1,
    n_bins: int = 36,
    weights: Optional[Union[str, np.ndarray]] = "von_mises",
    symmetric: bool = True,
    density: bool = True,
) -> DirectionalHistogram:
    """Compute directional histogram / rose diagram of principal stress orientations.
    
    Args:
        stress_6: Cauchy stress array (..., 6).
        plane: Projection plane ("xy", "xz", or "yz").
        which_principal: Principal stress vector index (1 for v1 / max tension).
        n_bins: Number of angular bins across [0, 2*pi] (if symmetric) or [0, pi].
        weights: Optional weighting for bins:
                 - "von_mises": Weighted by local von Mises equivalent stress.
                 - "principal_1": Weighted by max(sigma_1, 0).
                 - np.ndarray: Custom user-defined per-voxel weights.
                 - None / "uniform": Unweighted spatial point frequency.
        symmetric: If True, produces symmetric bidirectional rose diagram in [0, 2*pi]
                   with f(theta) = f(theta + pi).
        density: If True, normalizes frequencies such that integral over angle is 1.0.
                 If False, returns relative probabilities summing to 1.0.
                 
    Returns:
        DirectionalHistogram dataclass containing bin centers, frequencies, and circular stats.
    """
    angles = principal_direction_angles(stress_6, plane=plane, which_principal=which_principal, degrees=False)
    angles_flat = angles.ravel()
    
    # Resolve weights
    weight_name = "uniform"
    w_arr: Optional[np.ndarray] = None
    
    if isinstance(weights, str):
        if weights.lower() in ("vm", "von_mises"):
            w_arr = von_mises(stress_6).ravel()
            weight_name = "von_mises"
        elif weights.lower() in ("principal_1", "p1", "sigma_1"):
            vals, _ = principal_directions(stress_6)
            w_arr = np.maximum(vals[..., 0].ravel(), 0.0)
            weight_name = "principal_1"
        elif weights.lower() in ("none", "uniform"):
            w_arr = None
            weight_name = "uniform"
        else:
            raise ValueError(f"Unknown weight mode: '{weights}'")
    elif weights is not None:
        w_arr = np.asarray(weights, dtype=np.float64).ravel()
        weight_name = "custom"
        
    mean_dir, r_len, disp, circ_std = compute_circular_statistics(angles_flat, weights=w_arr)
    
    if symmetric:
        # Full circle [0, 2*pi] with duplicated lobes for headless stress axes
        all_angles = np.concatenate([angles_flat, (angles_flat + np.pi) % (2.0 * np.pi)])
        if w_arr is not None:
            all_weights = np.concatenate([w_arr, w_arr])
        else:
            all_weights = None
            
        bin_edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
        counts, _ = np.histogram(all_angles, bins=bin_edges, weights=all_weights, density=density)
    else:
        # Half circle [0, pi]
        bin_edges = np.linspace(0.0, np.pi, n_bins + 1)
        counts, _ = np.histogram(angles_flat, bins=bin_edges, weights=w_arr, density=density)
        
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    return DirectionalHistogram(
        bin_centers=bin_centers,
        bin_edges=bin_edges,
        frequencies=counts,
        mean_direction=mean_dir,
        mean_direction_degrees=float(np.degrees(mean_dir)),
        mean_resultant_length=r_len,
        circular_dispersion=disp,
        circular_std_degrees=float(np.degrees(circ_std)),
        symmetric=symmetric,
        weighted=(w_arr is not None),
        weight_quantity=weight_name,
        plane=plane,
    )


@dataclass
class PairPlaneSectionResult:
    """Angular PDF sections in a 2D stress component pair plane."""
    plane: Tuple[str, str]            # e.g. ('sx', 'sy')
    plane_label: str                  # e.g. "(SX, SY)"
    angles_deg: np.ndarray            # 1D array of evaluation angles in degrees [-180, 180]
    angles_rad: np.ndarray            # 1D array of evaluation angles in radians [-pi, pi]
    angular_pdf: np.ndarray           # 1D marginal PDF of theta = arctan2(y, x) (in rad^-1)
    radii: np.ndarray                 # 1D array of radial stress levels r_k (Pa or normalized)
    section_densities: np.ndarray     # 2D array of shape (len(radii), len(angles_deg)): joint PDF f(r, theta) (in Pa^-2)
    conditional_densities: np.ndarray # 2D array of shape (len(radii), len(angles_deg)): conditional p(theta | R=r) (in rad^-1)
    r_labels: List[str]               # Labels for each radius, e.g. ["q10", "q25", "q50", "q75", "q90"]
    r_mean: float                     # Mean stress radius
    r_std: float                      # Standard deviation of stress radius
    is_centered: bool = True          # True if centered around macroscopic mean (fluctuation field)
    is_normalized: bool = False
    macro_radius: Optional[float] = None
    macro_center: Optional[Tuple[float, float]] = None
    polar_grid_radii: Optional[np.ndarray] = None   # High-resolution radial grid for heatmap
    polar_grid_density: Optional[np.ndarray] = None # Shape (len(polar_grid_radii), len(angles_deg))


@dataclass
class PairPlaneRaySectionResult:
    """Container for radial ray sections of 2D joint stress PDF at fixed angles.
    
    Evaluates f(r*cos(theta), r*sin(theta)) vs r for fixed ray angles theta
    (e.g. 0°, 30°, 45°, 60°, 90°). For normalized stresses (X/R_macro, Y/R_macro),
    angles theta represent physical component ratios Y/X = tan(theta).
    """
    plane: Tuple[str, str]            # e.g. ('sx', 'sy')
    plane_label: str                  # e.g. "(SX, SY)"
    angles_deg: np.ndarray            # 1D array of ray angles in degrees
    angles_rad: np.ndarray            # 1D array in radians
    radii: np.ndarray                 # 1D array of radial evaluation points [0, r_max]
    ray_densities: np.ndarray         # Shape (len(angles_deg), len(radii)): joint PDF f(r; theta)
    angle_labels: List[str]           # Labels e.g. ["θ=0° (SY=0)", "θ=30° (SY/SX=0.58)", ...]
    is_centered: bool = False
    is_normalized: bool = True
    macro_radius: Optional[float] = None
    macro_center: Optional[Tuple[float, float]] = None
    macro_angle_deg: Optional[float] = None
    peak_radii: Optional[np.ndarray] = None
    peak_densities: Optional[np.ndarray] = None


@dataclass
class PairPlaneCartesianSectionResult:
    """Container for parallel Cartesian sections of 2D joint stress PDF.
    
    Evaluates parallel cross-sections f(X, y_k) vs X (if slice_axis='y') or f(x_k, Y) vs Y (if slice_axis='x')
    at planar levels across the 2D stress probability landscape.
    """
    plane: Tuple[str, str]            # e.g. ('sx', 'sy')
    plane_label: str                  # e.g. "(SX, SY)"
    slice_axis: str                   # 'y' (cuts at fixed Y) or 'x' (cuts at fixed X)
    slice_component: str              # Name of sliced component (e.g. 'SY')
    profile_component: str            # Name of profile axis component (e.g. 'SX')
    slice_levels: np.ndarray          # 1D array of fixed slice values (e.g. q10..q90)
    slice_labels: List[str]           # Human-readable labels e.g. ["q10: SY=0.38", ...]
    profile_grid: np.ndarray          # 1D array of evaluation points along the profile axis
    section_densities: np.ndarray     # Shape (len(slice_levels), len(profile_grid)): joint PDF f(x, y_k)
    conditional_densities: np.ndarray # Shape (len(slice_levels), len(profile_grid)): conditional p(x | y_k)
    peak_positions: np.ndarray        # Peak location on profile axis for each slice
    peak_densities: np.ndarray        # Peak PDF value for each slice
    is_normalized: bool = True
    macro_center: Optional[Tuple[float, float]] = None
    macro_radius: Optional[float] = None
    grid_x: Optional[np.ndarray] = None           # Optional 2D grid for contour plotting
    grid_y: Optional[np.ndarray] = None
    density_surface: Optional[np.ndarray] = None  # Shape (len(grid_y), len(grid_x))


COMPONENT_INDICES: Dict[str, int] = {
    "sx": 0, "sy": 1, "sz": 2,
    "sxy": 3, "syz": 4, "sxz": 5,
    "txx": 0, "tyy": 1, "tzz": 2,
    "txy": 3, "tyz": 4, "txz": 5,
}


def _resolve_component_name_and_index(name: Union[str, int]) -> Tuple[str, int]:
    canonical = ["sx", "sy", "sz", "sxy", "syz", "sxz"]
    if isinstance(name, int):
        if 0 <= name < 6:
            return canonical[name], name
        raise ValueError(f"Stress component index must be 0..5, got {name}")
    key = str(name).strip().lower()
    if key in COMPONENT_INDICES:
        idx = COMPONENT_INDICES[key]
        return canonical[idx], idx
    raise ValueError(f"Unknown stress component '{name}'. Supported: sx, sy, sz, sxy, syz, sxz")


def _extract_pair_plane_data(
    source: Any,
    plane: Tuple[Union[str, int], Union[str, int]],
    weights: Optional[Sequence[float]] = None,
    centered: bool = False,
    normalize: bool = False,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    np.ndarray,
    Tuple[str, str],
    str,
    Optional[float],
    Optional[Tuple[float, float]],
]:
    """Helper to unpack realizations, weights, and (X, Y) stress samples."""
    if hasattr(source, "_get_realizations") and hasattr(source, "weights"):
        sims = source._get_realizations()
        w_arr = np.asarray(source.weights, dtype=np.float64)
    elif isinstance(source, (list, tuple)) and len(source) > 0 and hasattr(source[0], "stress"):
        sims = list(source)
        if weights is not None:
            w_arr = np.asarray(weights, dtype=np.float64)
        else:
            w_arr = np.ones(len(sims), dtype=np.float64) / len(sims)
    elif hasattr(source, "stress"):
        sims = [source]
        w_arr = np.array([1.0], dtype=np.float64)
    elif isinstance(source, np.ndarray):
        from mv3d_stress_stats.single_rve import RVESimulation
        sims = [RVESimulation(source)]
        w_arr = np.array([1.0], dtype=np.float64)
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")
        
    if not np.isclose(np.sum(w_arr), 1.0):
        w_arr = w_arr / np.sum(w_arr)
        
    c_x, idx_x = _resolve_component_name_and_index(plane[0])
    c_y, idx_y = _resolve_component_name_and_index(plane[1])
    plane_label = f"({c_x.upper()}, {c_y.upper()})"
    
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    macro_radii: List[float] = []
    macro_centers: List[Tuple[float, float]] = []
    
    for sim in sims:
        s_arr = sim.stress.reshape(-1, 6)
        x_raw = s_arr[:, idx_x]
        y_raw = s_arr[:, idx_y]
        valid = np.isfinite(x_raw) & np.isfinite(y_raw)
        x_val = x_raw[valid]
        y_val = y_raw[valid]
        
        mx = float(np.mean(x_val)) if len(x_val) > 0 else 0.0
        my = float(np.mean(y_val)) if len(y_val) > 0 else 0.0
        macro_centers.append((mx, my))
        
        if centered:
            x_val = x_val - mx
            y_val = y_val - my
            
        if normalize:
            macro = sim.homogenized_stress()
            macro_x, macro_y = float(macro[idx_x]), float(macro[idx_y])
            r_macro = float(np.sqrt(macro_x**2 + macro_y**2))
            macro_radii.append(r_macro)
            if r_macro > 0:
                x_val = x_val / r_macro
                y_val = y_val / r_macro
                
        x_list.append(x_val)
        y_list.append(y_val)
        
    mean_macro_r = float(np.mean(macro_radii)) if macro_radii else None
    mean_macro_center = (
        float(np.mean([mc[0] for mc in macro_centers])),
        float(np.mean([mc[1] for mc in macro_centers]))
    ) if macro_centers else None
    
    return x_list, y_list, w_arr, (c_x, c_y), plane_label, mean_macro_r, mean_macro_center


def compute_pair_plane_sections(
    source: Any,
    plane: Tuple[Union[str, int], Union[str, int]] = ("sx", "sy"),
    weights: Optional[Sequence[float]] = None,
    n_angles: int = 180,
    angles_deg: Optional[np.ndarray] = None,
    radii: Optional[Sequence[float]] = None,
    n_radii: int = 5,
    percentiles: Sequence[float] = (10, 25, 50, 75, 90),
    centered: bool = True,
    normalize: bool = False,
    kde_bandwidth: Optional[Union[float, str]] = None,
    n_polar_radii: int = 60,
    max_samples: Optional[int] = 5000,
    seed: int = 42,
) -> PairPlaneSectionResult:
    """Compute angular PDF sections in a 2D stress component plane.
    
    Evaluates the probability density of stress states in the pair plane (X, Y)
    as a function of polar angle theta = arctan2(Y, X) across radial stress sections
    r_k = sqrt(X^2 + Y^2).
    
    When `centered=True` (default), the stress components are centered around each
    realization's macroscopic mean (Delta X = X - X_mean, Delta Y = Y - Y_mean),
    isolating the microstructural spatial fluctuations. The fluctuation angle
    theta = arctan2(Delta Y, Delta X) spans the full [-180, +180] degrees interval,
    and constant radii form horizontal lines across all angles.
    
    Args:
        source: Single RVESimulation, raw stress array (..., 6), sequence of
                RVESimulations, or an RVEEnsemble instance.
        plane: Pair of component names or indices, e.g. ('sx', 'sy'), ('sy', 'sz'),
               ('sx', 'sxy'), ('sz', 'sxz'), etc.
        weights: Realization weights w_r. If None, equal weights are used.
        n_angles: Number of angular evaluation points in [-180, 180] degrees.
        angles_deg: Optional explicit array of angles in degrees.
        radii: Explicit radial values r_k for section evaluation. If None,
               quantiles of the radial distribution (from `percentiles`) are used.
        n_radii: Number of radial quantiles if `radii` is None and `percentiles` not given.
        percentiles: Percentiles of r used for radial sections (default: 10, 25, 50, 75, 90).
        centered: If True (default), centers components around the realization mean
                  (Delta X = X - mean(X), Delta Y = Y - mean(Y)), yielding spatial fluctuations.
                  If False, analyzes raw stress components.
        normalize: If True, normalizes components by the plane macro radius
                   R_macro = sqrt(X_macro^2 + Y_macro^2).
        kde_bandwidth: Bandwidth method or scalar for Gaussian KDE.
        n_polar_radii: Number of radial grid points for high-resolution 2D polar heatmap (default: 60).
        max_samples: Max spatial samples per RVE for KDE fitting (speeds up computation).
        seed: Random seed for subsampling reproducibility.
        
    Returns:
        PairPlaneSectionResult dataclass.
    """
    x_list, y_list, w_arr, (c_x, c_y), plane_label, mean_macro_r, mean_macro_center = (
        _extract_pair_plane_data(source, plane, weights=weights, centered=centered, normalize=normalize)
    )
    
    # 3. Setup angle grid
    if angles_deg is None:
        angles_deg = np.linspace(-180.0, 180.0, n_angles)
    angles_deg = np.asarray(angles_deg, dtype=np.float64)
    angles_rad = np.radians(angles_deg)
    
    # 4. Determine radial section levels
    pooled_r = np.concatenate([np.sqrt(x**2 + y**2) for x, y in zip(x_list, y_list) if len(x) > 0])
    if len(pooled_r) == 0:
        raise ValueError("No valid finite stress samples found for pair plane.")
        
    r_mean = float(np.mean(pooled_r))
    r_std = float(np.std(pooled_r))
    
    if radii is not None:
        sec_radii = np.asarray(radii, dtype=np.float64)
        r_labels = [f"r={r:.2e}" for r in sec_radii]
    else:
        sec_radii = np.percentile(pooled_r, percentiles)
        r_labels = [f"q{int(p)}" for p in percentiles]
        
    n_sec = len(sec_radii)
    ensemble_sections = np.zeros((n_sec, len(angles_deg)), dtype=np.float64)
    ensemble_ang_pdf = np.zeros(len(angles_deg), dtype=np.float64)
    
    # Setup high-resolution polar grid for heatmap
    polar_grid_radii: Optional[np.ndarray] = None
    ensemble_polar_grid: Optional[np.ndarray] = None
    if n_polar_radii is not None and n_polar_radii > 0:
        r_min_grid = 0.0 if centered else float(np.percentile(pooled_r, 0.5))
        r_max_grid = float(np.percentile(pooled_r, 99.5))
        polar_grid_radii = np.linspace(r_min_grid, r_max_grid, n_polar_radii)
        ensemble_polar_grid = np.zeros((len(polar_grid_radii), len(angles_deg)), dtype=np.float64)
        
    # 5. Fit 2D and circular KDEs per realization and accumulate with weights
    for w, x, y in zip(w_arr, x_list, y_list):
        if len(x) == 0:
            continue
            
        if max_samples is not None and len(x) > max_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(x), size=max_samples, replace=False)
            x_fit, y_fit = x[idx], y[idx]
        else:
            x_fit, y_fit = x, y
            
        # 2D KDE for radial sections f(r*cos(theta), r*sin(theta))
        try:
            kde_2d = stats.gaussian_kde(np.vstack([x_fit, y_fit]), bw_method=kde_bandwidth)
        except np.linalg.LinAlgError:
            rng = np.random.default_rng(seed)
            jx = 1e-8 * (np.ptp(x_fit) if np.ptp(x_fit) > 0 else 1.0)
            jy = 1e-8 * (np.ptp(y_fit) if np.ptp(y_fit) > 0 else 1.0)
            kde_2d = stats.gaussian_kde(
                np.vstack([x_fit + rng.normal(0, jx, size=len(x_fit)),
                           y_fit + rng.normal(0, jy, size=len(y_fit))]),
                bw_method=kde_bandwidth,
            )
            
        # Evaluate discrete radial sections
        sec_pts = np.vstack([
            np.outer(sec_radii, np.cos(angles_rad)).ravel(),
            np.outer(sec_radii, np.sin(angles_rad)).ravel(),
        ])
        sec_dens = kde_2d(sec_pts).reshape(n_sec, len(angles_deg))
        ensemble_sections += w * sec_dens
        
        # Evaluate high-resolution polar grid for heatmap
        if polar_grid_radii is not None and ensemble_polar_grid is not None:
            grid_pts = np.vstack([
                np.outer(polar_grid_radii, np.cos(angles_rad)).ravel(),
                np.outer(polar_grid_radii, np.sin(angles_rad)).ravel(),
            ])
            grid_dens = kde_2d(grid_pts).reshape(len(polar_grid_radii), len(angles_deg))
            ensemble_polar_grid += w * grid_dens
            
        # Circular KDE for 1D marginal angular PDF p(theta)
        theta = np.arctan2(y_fit, x_fit)
        std_th = float(np.std(theta)) if len(theta) > 1 else 1.0
        if std_th <= 0.0:
            std_th = 0.1
        bw = std_th * (len(theta) ** (-0.2)) if kde_bandwidth is None else (
            float(kde_bandwidth) if isinstance(kde_bandwidth, (int, float)) else std_th * (len(theta) ** (-0.2))
        )
        margin = 3.0 * bw
        left_wrap = theta[theta > np.pi - margin] - 2.0 * np.pi
        right_wrap = theta[theta < -np.pi + margin] + 2.0 * np.pi
        theta_wrapped = np.concatenate([theta, left_wrap, right_wrap])
        try:
            kde_ang = stats.gaussian_kde(theta_wrapped, bw_method=bw / (np.std(theta_wrapped) or 1.0))
            p_ang = kde_ang(angles_rad) * (len(theta_wrapped) / len(theta))
        except np.linalg.LinAlgError:
            p_ang = np.ones_like(angles_rad) / (2.0 * np.pi)
        ensemble_ang_pdf += w * p_ang
        
    # 6. Compute conditional densities along each section
    conditional_densities = np.zeros_like(ensemble_sections)
    for k in range(n_sec):
        int_val = np.trapezoid(ensemble_sections[k, :], angles_rad)
        if int_val > 0:
            conditional_densities[k, :] = ensemble_sections[k, :] / int_val
            
    return PairPlaneSectionResult(
        plane=(c_x, c_y),
        plane_label=plane_label,
        angles_deg=angles_deg,
        angles_rad=angles_rad,
        angular_pdf=ensemble_ang_pdf,
        radii=sec_radii,
        section_densities=ensemble_sections,
        conditional_densities=conditional_densities,
        r_labels=r_labels,
        r_mean=r_mean,
        r_std=r_std,
        is_centered=centered,
        is_normalized=normalize,
        macro_radius=mean_macro_r,
        macro_center=mean_macro_center,
        polar_grid_radii=polar_grid_radii,
        polar_grid_density=ensemble_polar_grid,
    )


CANONICAL_PAIR_PLANES = [
    ("sx", "sy"),
    ("sy", "sz"),
    ("sx", "sz"),
    ("sx", "sxy"),
    ("sy", "syz"),
    ("sz", "sxz"),
]


def compute_all_pair_plane_sections(
    source: Any,
    planes: Optional[Sequence[Tuple[Union[str, int], Union[str, int]]]] = None,
    weights: Optional[Sequence[float]] = None,
    n_angles: int = 180,
    radii: Optional[Sequence[float]] = None,
    n_radii: int = 5,
    percentiles: Sequence[float] = (10, 25, 50, 75, 90),
    centered: bool = True,
    normalize: bool = False,
    kde_bandwidth: Optional[Union[float, str]] = None,
    n_polar_radii: int = 60,
    max_samples: Optional[int] = 5000,
) -> Dict[str, PairPlaneSectionResult]:
    """Compute angular sections across multiple pair planes.
    
    Default planes: (SX, SY), (SY, SZ), (SX, SZ), (SX, SXY), (SY, SYZ), (SZ, SXZ).
    """
    if planes is None:
        target_planes = CANONICAL_PAIR_PLANES
    else:
        target_planes = list(planes)
        
    results: Dict[str, PairPlaneSectionResult] = {}
    for pl in target_planes:
        res = compute_pair_plane_sections(
            source=source,
            plane=pl,
            weights=weights,
            n_angles=n_angles,
            radii=radii,
            n_radii=n_radii,
            percentiles=percentiles,
            centered=centered,
            normalize=normalize,
            kde_bandwidth=kde_bandwidth,
            n_polar_radii=n_polar_radii,
            max_samples=max_samples,
        )
        results[res.plane_label] = res
    return results


def compute_pair_plane_ray_sections(
    source: Any,
    plane: Tuple[Union[str, int], Union[str, int]] = ("sx", "sy"),
    angles_deg: Sequence[float] = (0.0, 30.0, 45.0, 60.0, 90.0),
    weights: Optional[Sequence[float]] = None,
    n_radii: int = 150,
    radii: Optional[Sequence[float]] = None,
    r_max: Optional[float] = None,
    centered: bool = False,
    normalize: bool = True,
    kde_bandwidth: Optional[Union[float, str]] = None,
    max_samples: Optional[int] = 5000,
    seed: int = 42,
) -> PairPlaneRaySectionResult:
    """Compute radial ray sections of 2D joint stress PDF at fixed angles.
    
    Evaluates f(r*cos(theta), r*sin(theta)) along radial rays from origin for fixed
    angles theta in angles_deg (default: 0°, 30°, 45°, 60°, 90°).
    
    Args:
        source: Single RVESimulation, raw stress array (..., 6), sequence of
                RVESimulations, or an RVEEnsemble instance.
        plane: Pair of component names or indices, e.g. ('sx', 'sy'), ('sy', 'sz'), etc.
        angles_deg: Sequence of ray angles in degrees (default: 0, 30, 45, 60, 90).
        weights: Realization weights w_r. If None, equal weights are used.
        n_radii: Number of radial points from 0 to r_max (default: 150).
        radii: Explicit radial evaluation grid. If None, linspace(0, r_max, n_radii) is used.
        r_max: Maximum radial evaluation distance. If None, 99.8th percentile * 1.15 is used.
        centered: If True, centers around macroscopic mean (fluctuations). If False (default for
                  normalized planes), keeps raw coordinates so angles reflect physical stress ratios.
        normalize: If True (default), scales components by plane macro radius R_macro.
        kde_bandwidth: Bandwidth method or float for Gaussian KDE.
        max_samples: Max spatial samples per RVE for KDE fitting.
        seed: Random seed for subsampling reproducibility.
        
    Returns:
        PairPlaneRaySectionResult dataclass.
    """
    x_list, y_list, w_arr, (c_x, c_y), plane_label, mean_macro_r, mean_macro_center = (
        _extract_pair_plane_data(source, plane, weights=weights, centered=centered, normalize=normalize)
    )
    
    pooled_r = np.concatenate([np.sqrt(x**2 + y**2) for x, y in zip(x_list, y_list) if len(x) > 0])
    if len(pooled_r) == 0:
        raise ValueError("No valid finite stress samples found for pair plane.")
        
    if radii is not None:
        eval_radii = np.asarray(radii, dtype=np.float64)
    else:
        if r_max is None:
            r_max_val = float(np.percentile(pooled_r, 99.8) * 1.15)
            if normalize and r_max_val < 1.3:
                r_max_val = 1.3
        else:
            r_max_val = float(r_max)
        eval_radii = np.linspace(0.0, r_max_val, n_radii)
        
    angles_deg_arr = np.asarray(angles_deg, dtype=np.float64)
    angles_rad_arr = np.radians(angles_deg_arr)
    n_rays = len(angles_deg_arr)
    n_pts = len(eval_radii)
    
    c1_name = c_x.upper()
    c2_name = c_y.upper()
    angle_labels: List[str] = []
    for deg in angles_deg_arr:
        rad = np.radians(deg)
        if np.isclose(deg, 0.0):
            ratio_str = f"{c2_name}=0"
        elif np.isclose(deg, 90.0):
            ratio_str = f"{c1_name}=0"
        elif np.isclose(deg, 45.0):
            ratio_str = f"{c2_name}={c1_name}"
        else:
            ratio = np.tan(rad)
            ratio_str = f"{c2_name}/{c1_name}={ratio:.2f}"
        angle_labels.append(rf"$\theta={deg:g}^\circ$ ({ratio_str})")
        
    R_mesh, Th_mesh = np.meshgrid(eval_radii, angles_rad_arr)
    eval_pts = np.vstack([
        (R_mesh * np.cos(Th_mesh)).ravel(),
        (R_mesh * np.sin(Th_mesh)).ravel(),
    ])
    
    ensemble_ray_densities = np.zeros((n_rays, n_pts), dtype=np.float64)
    
    for w, x, y in zip(w_arr, x_list, y_list):
        if len(x) == 0:
            continue
            
        if max_samples is not None and len(x) > max_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(x), size=max_samples, replace=False)
            x_fit, y_fit = x[idx], y[idx]
        else:
            x_fit, y_fit = x, y
            
        try:
            kde_2d = stats.gaussian_kde(np.vstack([x_fit, y_fit]), bw_method=kde_bandwidth)
        except np.linalg.LinAlgError:
            rng = np.random.default_rng(seed)
            jx = 1e-8 * (np.ptp(x_fit) if np.ptp(x_fit) > 0 else 1.0)
            jy = 1e-8 * (np.ptp(y_fit) if np.ptp(y_fit) > 0 else 1.0)
            kde_2d = stats.gaussian_kde(
                np.vstack([x_fit + rng.normal(0, jx, size=len(x_fit)),
                           y_fit + rng.normal(0, jy, size=len(y_fit))]),
                bw_method=kde_bandwidth,
            )
            
        dens = kde_2d(eval_pts).reshape(n_rays, n_pts)
        ensemble_ray_densities += w * dens
        
    macro_angle = None
    if mean_macro_center is not None:
        macro_angle = float(np.degrees(np.arctan2(mean_macro_center[1], mean_macro_center[0])))
        
    peak_idx = np.argmax(ensemble_ray_densities, axis=1)
    peak_radii = eval_radii[peak_idx]
    peak_densities = np.max(ensemble_ray_densities, axis=1)
    
    return PairPlaneRaySectionResult(
        plane=(c_x, c_y),
        plane_label=plane_label,
        angles_deg=angles_deg_arr,
        angles_rad=angles_rad_arr,
        radii=eval_radii,
        ray_densities=ensemble_ray_densities,
        angle_labels=angle_labels,
        is_centered=centered,
        is_normalized=normalize,
        macro_radius=mean_macro_r,
        macro_center=mean_macro_center,
        macro_angle_deg=macro_angle,
        peak_radii=peak_radii,
        peak_densities=peak_densities,
    )


def compute_all_pair_plane_ray_sections(
    source: Any,
    planes: Optional[Sequence[Tuple[Union[str, int], Union[str, int]]]] = None,
    angles_deg: Sequence[float] = (0.0, 30.0, 45.0, 60.0, 90.0),
    weights: Optional[Sequence[float]] = None,
    n_radii: int = 150,
    radii: Optional[Sequence[float]] = None,
    r_max: Optional[float] = None,
    centered: bool = False,
    normalize: bool = True,
    kde_bandwidth: Optional[Union[float, str]] = None,
    max_samples: Optional[int] = 5000,
) -> Dict[str, PairPlaneRaySectionResult]:
    """Compute radial ray sections across multiple pair planes at fixed angles.
    
    Default planes: (SX, SY), (SY, SZ), (SX, SZ), (SX, SXY), (SY, SYZ), (SZ, SXZ).
    """
    if planes is None:
        target_planes = CANONICAL_PAIR_PLANES
    else:
        target_planes = list(planes)
        
    results: Dict[str, PairPlaneRaySectionResult] = {}
    for pl in target_planes:
        res = compute_pair_plane_ray_sections(
            source=source,
            plane=pl,
            angles_deg=angles_deg,
            weights=weights,
            n_radii=n_radii,
            radii=radii,
            r_max=r_max,
            centered=centered,
            normalize=normalize,
            kde_bandwidth=kde_bandwidth,
            max_samples=max_samples,
        )
        results[res.plane_label] = res
    return results


def compute_pair_plane_cartesian_sections(
    source: Any,
    plane: Tuple[Union[str, int], Union[str, int]] = ("sx", "sy"),
    slice_axis: str = "y",
    slice_levels: Optional[Sequence[float]] = None,
    percentiles: Sequence[float] = (10, 25, 50, 75, 90),
    weights: Optional[Sequence[float]] = None,
    n_points: int = 150,
    normalize: bool = True,
    kde_bandwidth: Optional[Union[float, str]] = None,
    include_2d_surface: bool = True,
    n_surface_grid: int = 60,
    max_samples: Optional[int] = 5000,
    seed: int = 42,
) -> PairPlaneCartesianSectionResult:
    """Compute parallel Cartesian cross-sections of 2D joint stress PDF.
    
    Takes parallel planar cuts through the 2D joint density landscape f(X, Y).
    If slice_axis='y' (default), cuts along horizontal lines Y = y_k at specified
    levels (default: quantiles q10, q25, q50, q75, q90 of Y) and evaluates
    f(X, y_k) as a function of X.
    If slice_axis='x', cuts along vertical lines X = x_k and evaluates f(x_k, Y) vs Y.
    
    Args:
        source: RVESimulation, stress array, list of simulations, or RVEEnsemble.
        plane: Pair of components, e.g. ('sx', 'sy'), ('sy', 'sz'), etc.
        slice_axis: 'y' to slice along Y and plot vs X; 'x' to slice along X and plot vs Y.
        slice_levels: Explicit values for the sliced component. If None, quantiles from `percentiles` are used.
        percentiles: Percentiles for slice levels (default: 10, 25, 50, 75, 90).
        weights: Realization weights w_r.
        n_points: Number of points along profile grid (default: 150).
        normalize: If True (default), normalizes components by plane macro radius R_macro.
        kde_bandwidth: Bandwidth method or float for Gaussian KDE.
        include_2d_surface: If True, evaluates 2D grid surface for contour visualization.
        n_surface_grid: Resolution of 2D surface grid (default: 60).
        max_samples: Max spatial samples per RVE for KDE fitting.
        seed: Random seed.
        
    Returns:
        PairPlaneCartesianSectionResult dataclass.
    """
    x_list, y_list, w_arr, (c_x, c_y), plane_label, mean_macro_r, mean_macro_center = (
        _extract_pair_plane_data(source, plane, weights=weights, centered=False, normalize=normalize)
    )
    
    pooled_x = np.concatenate([x for x in x_list if len(x) > 0])
    pooled_y = np.concatenate([y for y in y_list if len(y) > 0])
    if len(pooled_x) == 0:
        raise ValueError("No valid finite stress samples found for pair plane.")
        
    axis_key = slice_axis.strip().lower()
    if axis_key not in ("x", "y"):
        raise ValueError(f"slice_axis must be 'x' or 'y', got '{slice_axis}'")
        
    if axis_key == "y":
        slice_data = pooled_y
        prof_data = pooled_x
        slice_name = c_y.upper()
        prof_name = c_x.upper()
    else:
        slice_data = pooled_x
        prof_data = pooled_y
        slice_name = c_x.upper()
        prof_name = c_y.upper()
        
    if slice_levels is not None:
        eval_levels = np.asarray(slice_levels, dtype=np.float64)
        slice_labels = [f"{slice_name}={lvl:.3f}" if normalize else f"{slice_name}={lvl:.2e}" for lvl in eval_levels]
    else:
        eval_levels = np.percentile(slice_data, percentiles)
        slice_labels = []
        for p, lvl in zip(percentiles, eval_levels):
            val_str = f"{lvl:.3f}" if normalize else f"{lvl:.2e}"
            slice_labels.append(f"q{int(p)}: {slice_name}={val_str}")
            
    n_slices = len(eval_levels)
    
    p_min = float(np.percentile(prof_data, 0.5))
    p_max = float(np.percentile(prof_data, 99.5))
    margin = 0.08 * (p_max - p_min if p_max > p_min else 1.0)
    profile_grid = np.linspace(p_min - margin, p_max + margin, n_points)
    
    grid_x, grid_y, density_surface = None, None, None
    if include_2d_surface:
        gx_min = float(np.percentile(pooled_x, 0.5)) - 0.08 * (np.ptp(pooled_x) or 1.0)
        gx_max = float(np.percentile(pooled_x, 99.5)) + 0.08 * (np.ptp(pooled_x) or 1.0)
        gy_min = float(np.percentile(pooled_y, 0.5)) - 0.08 * (np.ptp(pooled_y) or 1.0)
        gy_max = float(np.percentile(pooled_y, 99.5)) + 0.08 * (np.ptp(pooled_y) or 1.0)
        grid_x = np.linspace(gx_min, gx_max, n_surface_grid)
        grid_y = np.linspace(gy_min, gy_max, n_surface_grid)
        density_surface = np.zeros((n_surface_grid, n_surface_grid), dtype=np.float64)
        
    ensemble_sections = np.zeros((n_slices, n_points), dtype=np.float64)
    
    if axis_key == "y":
        P_mesh, L_mesh = np.meshgrid(profile_grid, eval_levels)
        eval_pts = np.vstack([P_mesh.ravel(), L_mesh.ravel()])
    else:
        L_mesh, P_mesh = np.meshgrid(eval_levels, profile_grid)
        eval_pts = np.vstack([L_mesh.T.ravel(), P_mesh.T.ravel()])
        
    for w, x, y in zip(w_arr, x_list, y_list):
        if len(x) == 0:
            continue
            
        if max_samples is not None and len(x) > max_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(x), size=max_samples, replace=False)
            x_fit, y_fit = x[idx], y[idx]
        else:
            x_fit, y_fit = x, y
            
        try:
            kde_2d = stats.gaussian_kde(np.vstack([x_fit, y_fit]), bw_method=kde_bandwidth)
        except np.linalg.LinAlgError:
            rng = np.random.default_rng(seed)
            jx = 1e-8 * (np.ptp(x_fit) if np.ptp(x_fit) > 0 else 1.0)
            jy = 1e-8 * (np.ptp(y_fit) if np.ptp(y_fit) > 0 else 1.0)
            kde_2d = stats.gaussian_kde(
                np.vstack([x_fit + rng.normal(0, jx, size=len(x_fit)),
                           y_fit + rng.normal(0, jy, size=len(y_fit))]),
                bw_method=kde_bandwidth,
            )
            
        dens = kde_2d(eval_pts).reshape(n_slices, n_points)
        ensemble_sections += w * dens
        
        if include_2d_surface and grid_x is not None and grid_y is not None and density_surface is not None:
            Gx, Gy = np.meshgrid(grid_x, grid_y)
            surf_pts = np.vstack([Gx.ravel(), Gy.ravel()])
            density_surface += w * kde_2d(surf_pts).reshape(n_surface_grid, n_surface_grid)
            
    conditional_densities = np.zeros_like(ensemble_sections)
    for k in range(n_slices):
        area = np.trapezoid(ensemble_sections[k, :], profile_grid)
        if area > 0:
            conditional_densities[k, :] = ensemble_sections[k, :] / area
            
    peak_idx = np.argmax(ensemble_sections, axis=1)
    peak_positions = profile_grid[peak_idx]
    peak_densities = np.max(ensemble_sections, axis=1)
    
    return PairPlaneCartesianSectionResult(
        plane=(c_x, c_y),
        plane_label=plane_label,
        slice_axis=axis_key,
        slice_component=slice_name,
        profile_component=prof_name,
        slice_levels=eval_levels,
        slice_labels=slice_labels,
        profile_grid=profile_grid,
        section_densities=ensemble_sections,
        conditional_densities=conditional_densities,
        peak_positions=peak_positions,
        peak_densities=peak_densities,
        is_normalized=normalize,
        macro_center=mean_macro_center,
        macro_radius=mean_macro_r,
        grid_x=grid_x,
        grid_y=grid_y,
        density_surface=density_surface,
    )


def compute_all_pair_plane_cartesian_sections(
    source: Any,
    planes: Optional[Sequence[Tuple[Union[str, int], Union[str, int]]]] = None,
    slice_axis: str = "y",
    percentiles: Sequence[float] = (10, 25, 50, 75, 90),
    weights: Optional[Sequence[float]] = None,
    n_points: int = 150,
    normalize: bool = True,
    kde_bandwidth: Optional[Union[float, str]] = None,
    include_2d_surface: bool = False,
    max_samples: Optional[int] = 5000,
) -> Dict[str, PairPlaneCartesianSectionResult]:
    """Compute parallel Cartesian cross-sections across multiple pair planes.
    
    Default planes: (SX, SY), (SY, SZ), (SX, SZ), (SX, SXY), (SY, SYZ), (SZ, SXZ).
    """
    if planes is None:
        target_planes = CANONICAL_PAIR_PLANES
    else:
        target_planes = list(planes)
        
    results: Dict[str, PairPlaneCartesianSectionResult] = {}
    for pl in target_planes:
        res = compute_pair_plane_cartesian_sections(
            source=source,
            plane=pl,
            slice_axis=slice_axis,
            percentiles=percentiles,
            weights=weights,
            n_points=n_points,
            normalize=normalize,
            kde_bandwidth=kde_bandwidth,
            include_2d_surface=include_2d_surface,
            max_samples=max_samples,
        )
        results[res.plane_label] = res
    return results


