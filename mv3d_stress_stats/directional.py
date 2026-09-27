"""
Directional stress analysis and rose diagram / directional histogram algorithms.

Analyzes the spatial orientation of principal stress axes, circular dispersion,
stress-weighted angular concentrations, and directional anisotropy in 3D microstructures.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np

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
