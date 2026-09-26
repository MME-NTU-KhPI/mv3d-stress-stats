"""
Tensor operations for 3D Cauchy stress representations.

Supports:
- 6-component Mandel/Voigt-like vector format: [SX, SY, SZ, SXY, SYZ, SXZ]
- 3x3 symmetric tensor representation
- Coordinate rotations: R^T * sigma * R
- Deviatoric and hydrostatic decompositions
"""

from __future__ import annotations
import numpy as np


def to_tensor_3x3(stress_6: np.ndarray) -> np.ndarray:
    """Convert a (..., 6) stress array to a (..., 3, 3) symmetric stress tensor.
    
    The 6 components are ordered as:
        [SX, SY, SZ, SXY, SYZ, SXZ]
        
    Returns:
        np.ndarray of shape (..., 3, 3) with symmetric matrix entries:
            [[SX,  SXY, SXZ],
             [SXY, SY,  SYZ],
             [SXZ, SYZ, SZ ]]
    """
    stress_6 = np.asarray(stress_6)
    if stress_6.shape[-1] != 6:
        raise ValueError(f"Last axis must have length 6, got shape {stress_6.shape}")
        
    sx = stress_6[..., 0]
    sy = stress_6[..., 1]
    sz = stress_6[..., 2]
    sxy = stress_6[..., 3]
    syz = stress_6[..., 4]
    sxz = stress_6[..., 5]
    
    # Construct (..., 3, 3)
    out_shape = stress_6.shape[:-1] + (3, 3)
    tensor = np.empty(out_shape, dtype=stress_6.dtype)
    
    tensor[..., 0, 0] = sx
    tensor[..., 0, 1] = sxy
    tensor[..., 0, 2] = sxz
    
    tensor[..., 1, 0] = sxy
    tensor[..., 1, 1] = sy
    tensor[..., 1, 2] = syz
    
    tensor[..., 2, 0] = sxz
    tensor[..., 2, 1] = syz
    tensor[..., 2, 2] = sz
    
    return tensor


def from_tensor_3x3(tensor_3x3: np.ndarray) -> np.ndarray:
    """Convert a (..., 3, 3) symmetric stress tensor to (..., 6) vector format.
    
    Output order: [SX, SY, SZ, SXY, SYZ, SXZ].
    Shear components are averaged: (tensor[..., i, j] + tensor[..., j, i]) / 2
    """
    tensor_3x3 = np.asarray(tensor_3x3)
    if tensor_3x3.shape[-2:] != (3, 3):
        raise ValueError(f"Last two axes must be (3, 3), got shape {tensor_3x3.shape}")
        
    sx = tensor_3x3[..., 0, 0]
    sy = tensor_3x3[..., 1, 1]
    sz = tensor_3x3[..., 2, 2]
    sxy = 0.5 * (tensor_3x3[..., 0, 1] + tensor_3x3[..., 1, 0])
    syz = 0.5 * (tensor_3x3[..., 1, 2] + tensor_3x3[..., 2, 1])
    sxz = 0.5 * (tensor_3x3[..., 0, 2] + tensor_3x3[..., 2, 0])
    
    return np.stack([sx, sy, sz, sxy, syz, sxz], axis=-1)


def rotate_stress(
    stress: np.ndarray,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """Rotate stress tensor into a new coordinate system:
        sigma_rot = R.T @ sigma @ R
        
    Args:
        stress: Array of shape (..., 6) or (..., 3, 3).
        rotation_matrix: Transformation matrix R of shape (3, 3) or matching batch shape (..., 3, 3).
                         The columns of R are the unit basis vectors of the new coordinate system
                         expressed in the old system.
                         
    Returns:
        Rotated stress array matching the input format (6 or 3x3).
    """
    stress_arr = np.asarray(stress)
    R = np.asarray(rotation_matrix)
    
    is_6comp = stress_arr.shape[-1] == 6
    if is_6comp:
        tensor = to_tensor_3x3(stress_arr)
    elif stress_arr.shape[-2:] == (3, 3):
        tensor = stress_arr
    else:
        raise ValueError(f"Invalid stress shape: {stress_arr.shape}")
        
    # sigma_rot = R.T @ tensor @ R
    if R.ndim == 2:
        if R.shape != (3, 3):
            raise ValueError(f"Rotation matrix must be (3, 3), got {R.shape}")
        # R.T @ tensor @ R: (R.T)_{ki} = R_{ik}
        rot_tensor = np.einsum("ik,...ij,jl->...kl", R, tensor, R)
    else:
        rot_tensor = np.einsum("...ik,...ij,...jl->...kl", R, tensor, R)
        
    if is_6comp:
        return from_tensor_3x3(rot_tensor)
    return rot_tensor


def trace_stress(stress_6: np.ndarray) -> np.ndarray:
    """First stress invariant I1 = tr(sigma) = SX + SY + SZ.
    
    Input shape: (..., 6)
    Output shape: (...,)
    """
    stress_6 = np.asarray(stress_6)
    return stress_6[..., 0] + stress_6[..., 1] + stress_6[..., 2]


def hydrostatic_stress(stress_6: np.ndarray) -> np.ndarray:
    """Hydrostatic stress sigma_h = I1 / 3 = (SX + SY + SZ) / 3.
    
    Input shape: (..., 6)
    Output shape: (...,)
    """
    return trace_stress(stress_6) / 3.0


def deviatoric_stress(stress_6: np.ndarray) -> np.ndarray:
    """Deviatoric stress tensor s = sigma - sigma_h * I.
    
    Input shape: (..., 6)
    Output shape: (..., 6) with s_x, s_y, s_z, s_xy, s_yz, s_xz.
    """
    stress_6 = np.asarray(stress_6)
    s_h = hydrostatic_stress(stress_6)[..., np.newaxis]
    s = stress_6.copy()
    s[..., 0:3] -= s_h
    return s
