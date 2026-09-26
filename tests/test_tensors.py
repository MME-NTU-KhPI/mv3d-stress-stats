import numpy as np
import pytest
from mv3d_stress_stats.tensors import (
    to_tensor_3x3,
    from_tensor_3x3,
    rotate_stress,
    trace_stress,
    hydrostatic_stress,
    deviatoric_stress,
)


def test_tensor_roundtrip():
    # Shape (N, 6)
    stress_6 = np.array([
        [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
        [-5.0, 15.0, 0.0, 0.5, -1.0, 2.5],
    ])
    tensor_3x3 = to_tensor_3x3(stress_6)
    assert tensor_3x3.shape == (2, 3, 3)
    # Check symmetry
    np.testing.assert_allclose(tensor_3x3, tensor_3x3.swapaxes(-1, -2))
    
    # Back to 6
    recovered = from_tensor_3x3(tensor_3x3)
    np.testing.assert_allclose(recovered, stress_6)


def test_hydrostatic_and_deviatoric():
    # Pure hydrostatic: SX = SY = SZ = 100, no shear
    p_hydro = np.array([100.0, 100.0, 100.0, 0.0, 0.0, 0.0])
    assert hydrostatic_stress(p_hydro) == pytest.approx(100.0)
    assert trace_stress(p_hydro) == pytest.approx(300.0)
    
    dev = deviatoric_stress(p_hydro)
    np.testing.assert_allclose(dev, np.zeros(6), atol=1e-12)

    # General stress
    s = np.array([10.0, 20.0, 60.0, 4.0, 5.0, 6.0])
    s_h = hydrostatic_stress(s)
    assert s_h == pytest.approx(30.0)
    dev_s = deviatoric_stress(s)
    # Trace of deviatoric stress must be identically zero
    assert trace_stress(dev_s) == pytest.approx(0.0, abs=1e-12)


def test_rotate_stress():
    # Uniaxial stress along X: [100, 0, 0, 0, 0, 0]
    s_x = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # 90-degree rotation about Z axis: X -> Y, Y -> -X
    # R transforms old basis to new:
    # e1' = e2 (0, 1, 0), e2' = -e1 (-1, 0, 0), e3' = e3 (0, 0, 1)
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    # sigma' = R.T @ sigma @ R
    # In new frame, loading along old X is along new Y: SY = 100
    s_rot = rotate_stress(s_x, R)
    np.testing.assert_allclose(s_rot[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(s_rot[1], 100.0, atol=1e-10)
    np.testing.assert_allclose(s_rot[2:], 0.0, atol=1e-10)
