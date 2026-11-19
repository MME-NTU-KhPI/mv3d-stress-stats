"""
Stress invariants, equivalent stresses, principal stresses, and Lode parameters.

Fully vectorized across arbitrary batch dimensions (..., 6).
Direct tensor-to-quantity evaluation preserves inter-component correlation structure.
"""

from __future__ import annotations
from typing import Callable, Union
import numpy as np

from mv3d_stress_stats.tensors import (
    to_tensor_3x3,
    trace_stress,
    hydrostatic_stress,
    deviatoric_stress,
)


def i1(stress_6: np.ndarray) -> np.ndarray:
    """First stress invariant I1 = tr(sigma) = SX + SY + SZ."""
    return trace_stress(stress_6)


def mean_stress(stress_6: np.ndarray) -> np.ndarray:
    """Mean / Hydrostatic stress sigma_m = I1 / 3."""
    return hydrostatic_stress(stress_6)


def pressure(stress_6: np.ndarray) -> np.ndarray:
    """Pressure p = -sigma_h = -I1 / 3."""
    return -hydrostatic_stress(stress_6)


def j2(stress_6: np.ndarray) -> np.ndarray:
    """Second invariant of the deviatoric stress tensor J2.
    
    J2 = 1/2 * s : s
       = 1/6 * [(SX - SY)^2 + (SY - SZ)^2 + (SZ - SX)^2] + SXY^2 + SYZ^2 + SXZ^2
    """
    stress_6 = np.asarray(stress_6)
    sx = stress_6[..., 0]
    sy = stress_6[..., 1]
    sz = stress_6[..., 2]
    sxy = stress_6[..., 3]
    syz = stress_6[..., 4]
    sxz = stress_6[..., 5]
    
    val = (
        (1.0 / 6.0) * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
        + sxy ** 2
        + syz ** 2
        + sxz ** 2
    )
    return np.maximum(val, 0.0)


def j3(stress_6: np.ndarray) -> np.ndarray:
    """Third invariant of the deviatoric stress tensor J3 = det(s).
    
    Computed directly from deviatoric components:
        s_x * s_y * s_z + 2 * s_xy * s_yz * s_xz - s_x * s_yz^2 - s_y * s_xz^2 - s_z * s_xy^2
    """
    s = deviatoric_stress(stress_6)
    sx = s[..., 0]
    sy = s[..., 1]
    sz = s[..., 2]
    sxy = s[..., 3]
    syz = s[..., 4]
    sxz = s[..., 5]
    
    val = (
        sx * sy * sz
        + 2.0 * sxy * syz * sxz
        - sx * (syz ** 2)
        - sy * (sxz ** 2)
        - sz * (sxy ** 2)
    )
    return val


def von_mises(stress_6: np.ndarray) -> np.ndarray:
    """von Mises equivalent stress sigma_VM = sqrt(3 * J2)."""
    return np.sqrt(3.0 * j2(stress_6))


def principal_stresses(stress_6: np.ndarray) -> np.ndarray:
    """Calculate principal stresses (eigenvalues of Cauchy stress tensor).
    
    Returns:
        np.ndarray of shape (..., 3) containing [sigma_1, sigma_2, sigma_3]
        sorted in descending order: sigma_1 >= sigma_2 >= sigma_3.
    """
    p, r, theta = _principal_parts(stress_6)
    s1 = p + r * np.cos(theta)
    s3 = p + r * np.cos(theta + 2.0 * np.pi / 3.0)
    s2 = 3.0 * p - s1 - s3
    return np.stack([s1, s2, s3], axis=-1)


def _principal_parts(stress_6: np.ndarray):
    """(p, r, θ) with σ_k = p + r·cos(θ + 2π(k-1)/3), descending for θ∈[0,π/3]."""
    s = np.asarray(stress_6, dtype=np.float64)
    sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]
    txy, tyz, txz = s[..., 3], s[..., 4], s[..., 5]
    # Closed form (trigonometric solution of the characteristic cubic) instead
    # of np.linalg.eigvalsh: same result to ~1e-12 of the stress magnitude but
    # ~20x faster for millions of 3x3 tensors (localization statistics).
    p = (sx + sy + sz) / 3.0
    dx, dy, dz = sx - p, sy - p, sz - p
    off2 = txy * txy + tyz * tyz + txz * txz
    j2_ = 0.5 * (dx * dx + dy * dy + dz * dz) + off2
    j3_ = (dx * dy * dz + 2.0 * txy * tyz * txz
           - dx * tyz * tyz - dy * txz * txz - dz * txy * txy)
    sq = np.sqrt(j2_)
    r = (2.0 / np.sqrt(3.0)) * sq
    # cos(3θ) = (3√3/2)·J3 / J2^(3/2); J2·√J2 instead of ** 1.5 (pow is ~6x slower)
    den = j2_ * sq
    safe = den > 0.0
    c3 = np.zeros_like(den)
    np.divide((1.5 * np.sqrt(3.0)) * j3_, den, out=c3, where=safe)
    np.clip(c3, -1.0, 1.0, out=c3)
    theta = np.arccos(c3) / 3.0
    return p, r, theta


def principal_1(stress_6: np.ndarray) -> np.ndarray:
    """Maximum principal stress sigma_1 (sigma_1 >= sigma_2 >= sigma_3)."""
    p, r, theta = _principal_parts(stress_6)
    return p + r * np.cos(theta)


def principal_2(stress_6: np.ndarray) -> np.ndarray:
    """Intermediate principal stress sigma_2 (sigma_1 >= sigma_2 >= sigma_3)."""
    p, r, theta = _principal_parts(stress_6)
    return p + r * np.cos(theta - 2.0 * np.pi / 3.0)


def principal_3(stress_6: np.ndarray) -> np.ndarray:
    """Minimum principal stress sigma_3 (sigma_1 >= sigma_2 >= sigma_3)."""
    p, r, theta = _principal_parts(stress_6)
    return p + r * np.cos(theta + 2.0 * np.pi / 3.0)


def lode_angle(stress_6: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Calculate Lode angle theta from:
        cos(3*theta) = (3 * sqrt(3) / 2) * (J3 / J2^(3/2))
        
    Convention:
        theta in [0, pi/3] (0 rad to ~60 deg).
        theta = 0 corresponds to axisymmetric tension / triaxial extension.
        theta = pi/6 corresponds to pure shear.
        theta = pi/3 corresponds to axisymmetric compression.
        
    Args:
        stress_6: Stress array (..., 6).
        eps: Threshold below which J2 is treated as zero to prevent division by zero.
        
    Returns:
        Lode angle theta in radians with shape matching batch dimensions (...,).
    """
    j2_val = j2(stress_6)
    j3_val = j3(stress_6)
    
    denom = j2_val ** 1.5
    valid_mask = denom > eps
    
    # Pre-allocate result with pi / 6 (pure shear / neutral default)
    theta = np.full_like(j2_val, np.pi / 6.0)
    
    if np.any(valid_mask):
        ratio = (1.5 * np.sqrt(3.0)) * (j3_val[valid_mask] / denom[valid_mask])
        ratio = np.clip(ratio, -1.0, 1.0)
        theta[valid_mask] = (1.0 / 3.0) * np.arccos(ratio)
        
    return theta


def lode_parameter(stress_6: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Calculate normalized Lode parameter mu:
        mu = (2 * sigma_2 - sigma_1 - sigma_3) / (sigma_1 - sigma_3)
        
    mu in [-1, 1]:
        mu = -1 for axisymmetric tension
        mu = 0 for pure shear
        mu = +1 for axisymmetric compression
    """
    p = principal_stresses(stress_6)
    s1 = p[..., 0]
    s2 = p[..., 1]
    s3 = p[..., 2]
    
    diff = s1 - s3
    valid_mask = diff > eps
    
    mu = np.zeros_like(s1)
    if np.any(valid_mask):
        val = (2.0 * s2[valid_mask] - s1[valid_mask] - s3[valid_mask]) / diff[valid_mask]
        mu[valid_mask] = np.clip(val, -1.0, 1.0)
        
    return mu


def triaxiality(stress_6: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Stress triaxiality eta = sigma_h / sigma_VM.
    
    Protected against division by zero when sigma_VM < eps.
    """
    sh = hydrostatic_stress(stress_6)
    svm = von_mises(stress_6)
    valid_mask = svm > eps
    
    eta = np.zeros_like(sh)
    eta[valid_mask] = sh[valid_mask] / svm[valid_mask]
    return eta


# Mapping of standard scalar quantity names to functions
QUANTITY_REGISTRY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    # Components
    "sx": lambda s: np.asarray(s)[..., 0],
    "sy": lambda s: np.asarray(s)[..., 1],
    "sz": lambda s: np.asarray(s)[..., 2],
    "sxy": lambda s: np.asarray(s)[..., 3],
    "syz": lambda s: np.asarray(s)[..., 4],
    "sxz": lambda s: np.asarray(s)[..., 5],
    # Invariants & measures
    "i1": i1,
    "trace": i1,
    "mean_stress": mean_stress,
    "hydrostatic_stress": hydrostatic_stress,
    "pressure": pressure,
    "j2": j2,
    "j3": j3,
    "von_mises": von_mises,
    "seqv": von_mises,
    "principal_1": principal_1,
    "principal_2": principal_2,
    "principal_3": principal_3,
    "triaxiality": triaxiality,
    "lode_angle": lode_angle,
    "lode_parameter": lode_parameter,
}


def resolve_quantity(
    quantity: Union[str, Callable[[np.ndarray], np.ndarray]]
) -> Callable[[np.ndarray], np.ndarray]:
    """Resolve a quantity string or callable to a vectorized function.
    
    Args:
        quantity: Name in QUANTITY_REGISTRY (case-insensitive) or a callable taking
                  (..., 6) array and returning (...,) array.
                  
    Returns:
        Callable[[np.ndarray], np.ndarray]
    """
    if callable(quantity):
        return quantity
        
    key = str(quantity).strip().lower()
    if key in QUANTITY_REGISTRY:
        return QUANTITY_REGISTRY[key]
        
    supported = ", ".join(sorted(QUANTITY_REGISTRY.keys()))
    raise ValueError(
        f"Unknown quantity '{quantity}'. Supported standard quantities: {supported}, "
        "or pass a custom callable(stress_6) -> scalar."
    )
